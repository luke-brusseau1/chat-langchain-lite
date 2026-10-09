"""Generate a LangSmith Insights report over the tracing project.

Insights is NOT automatic. It runs as a background job over a sample of
traces and can take up to 30 minutes. Generate it BEFORE a demo. An
un-generated Insights tab is empty.

    python -m scripts.setup_insights                  # discover categories bottom-up
    python -m scripts.setup_insights --mode guided    # force the demo's known categories
    python -m scripts.setup_insights --list           # show report status
    python -m scripts.setup_insights --poll           # wait for completion

Two modes, and the difference matters for the narrative:

  ``auto`` (default)  No predefined categories. Insights clusters bottom-up and
                      names the categories itself. This is the honest story -
                      "we did not tell it what to look for". The labels are
                      generated, so read the screen instead of a script.

  ``guided``          Passes ``partitions``: the jobs developers come to do.
                      Traces land in those buckets, so the report is
                      predictable. Use it as a safety net. Say in the room
                      that you supplied the categories.

Both modes report on two axes. The categories say what developers USE the
chatbot for. The per-trace attributes say what WENT WRONG, and they aggregate
inside each category. Read it in that order: "developers mostly ask X, and in
N% of those it cited the stale docs domain."

PREREQUISITE that trips people up: Insights needs a workspace model
configuration with a real provider secret, under Settings > Model
configurations. That is NOT the gateway credential the agent uses.
``LANGSMITH_GATEWAY_API_KEY`` does not satisfy it. Insights uses two models: a
thinking model for clustering, and a summarization model per trace. This
script leaves ``validate_model_secrets`` on, so a missing secret fails at
once instead of after a long job.

Cost: roughly $1-2 per 1,000 threads. The demo's traces cost pennies.

Needs ``LANGSMITH_API_KEY``. Also needs ``LANGSMITH_WORKSPACE_ID`` when the
key's default workspace differs. Insights is a Plus or Enterprise feature.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

from langsmith import Client  # noqa: E402

_API = "https://api.smith.langchain.com/api/v1"
PROJECT_NAME = os.getenv("LANGSMITH_PROJECT", "chat-lc-lite")

# ``all_thread_messages`` is required, not optional. The trace generator
# produces multi-turn threads, and for a threaded project the ``run.*``
# variables expose only the most recent root run - the last turn. The
# interesting drift in this demo happens in turn 2, so summarising on
# ``run.inputs`` alone would miss it.
#
# The prompt also forbids copying a credential into a summary. Developers paste
# real keys into this chatbot. Without that rule the report becomes a second
# copy of the secret, outside the redaction boundary.
SUMMARY_PROMPT = """\
Identify the main usage patterns, and failure modes of my agent.

Summarize ONE conversation between a developer and a LangChain ecosystem
documentation chatbot in 2-4 sentences. Say what the developer came to do,
whether they got it, and anything the assistant got wrong.

Never copy a credential or any other sensitive value into your summary. Report
that one appeared and use a placeholder - [API_KEY], [TOKEN], [EMAIL]. Do not
quote any part of the value, not even the last four characters. Treat a
redaction placeholder such as <langsmith-api-key> the same way.

Conversation:
{{all_thread_messages}}

Evaluator and user feedback on this trace:
{{run.feedback}}
"""

# Per-trace attributes. These steer clustering, because traces with similar
# values group together, and they aggregate per category. The keys render in
# the LangSmith UI, so they are named in plain language.
#
# Two axes, on purpose:
#   USAGE    what the developer came to do, and whether they got it
#   QUALITY  what went wrong, if anything
ATTRIBUTE_SCHEMAS = {
    # --- usage ---------------------------------------------------------------
    "request_type": {
        "type": "string",
        "description": (
            "What the developer came to do, as one of: 'concept_lookup' (what a "
            "LangChain, LangGraph, LangSmith, or Deep Agents component is or does), "
            "'setup_or_install' (packages, versions, environment variables, a first "
            "run), 'tracing_or_evals' (LangSmith tracing, datasets, evaluators, "
            "experiments), 'deployment_or_persistence' (serving a graph, "
            "checkpointers, production concerns), 'security_or_best_practice' (key "
            "handling, safe patterns, antipatterns), 'docs_navigation' (where the "
            "documentation lives), or 'out_of_scope' (anything outside the LangChain "
            "ecosystem)."
        ),
    },
    "outcome": {
        "type": "string",
        "description": (
            "Whether the developer got what they came for: 'answered' (a complete, "
            "correct, grounded answer), 'answered_but_flawed' (answered, but with a "
            "wrong fact, a stale link, a truncated ending, or no tool grounding), "
            "'refused' (the assistant declined, correctly or not), 'tool_failed' (a "
            "tool error blocked the answer), or 'deflected' (no answer and no clear "
            "reason)."
        ),
    },
    # --- quality -------------------------------------------------------------
    "cited_stale_docs_domain": {
        "type": "boolean",
        "description": (
            "True if the assistant pointed the developer at python.langchain.com or "
            "js.langchain.com. Those are stale legacy domains. The current domain is "
            "docs.langchain.com."
        ),
    },
    "wrong_version_fact": {
        "type": "boolean",
        "description": (
            "True if the assistant stated an incorrect version fact. The LangGraph "
            "minimum Python version is 3.10+, not 3.7+. LangChain is 3.10+ and "
            "LangSmith is 3.9+."
        ),
    },
    "answered_from_memory": {
        "type": "boolean",
        "description": (
            "True if the assistant answered a factual product question without "
            "calling any tool, so the answer rests on model memory rather than "
            "retrieved content."
        ),
    },
    "response_truncated": {
        "type": "boolean",
        "description": (
            "True if the final response stops mid-sentence, mid-word, or mid-code "
            "block, or ends before it finishes the answer it started."
        ),
    },
    "casual_tone": {
        "type": "string",
        "description": (
            "The register of the response: 'professional' if it reads as enterprise "
            "developer documentation, 'casual' if it opens with a greeting such as "
            "'Hey there', signs off with a phrase such as 'Happy building', uses "
            "emojis, or calls LangChain 'LC', or 'not_assessable'."
        ),
    },
    "answered_out_of_scope": {
        "type": "boolean",
        "description": (
            "True if the developer asked something outside the LangChain ecosystem "
            "and the assistant answered it anyway instead of declining."
        ),
    },
    "credential_in_conversation": {
        "type": "boolean",
        "description": (
            "True if a credential appeared anywhere in the conversation, including "
            "as a redaction placeholder. Report only true or false. Never include "
            "the value or any part of it."
        ),
    },
    "tool_failed": {
        "type": "boolean",
        "description": "True if any tool call returned an error during the conversation.",
    },
}

# Predefined top-level categories for --mode guided. Name -> description.
#
# These are the JOBS developers come to do, not the defects the demo plants.
# Partitioning on usage answers "what do people use this for". The quality
# attributes then aggregate INSIDE each job and answer "and what goes wrong
# there". Partitioning on defects can only answer the second question.
PARTITIONS = {
    "Understanding a component": (
        "The developer asked what a LangChain, LangGraph, LangSmith, or Deep Agents "
        "component is, what it does, or how it compares to another one."
    ),
    "Getting it installed and running": (
        "The developer asked which package to install, which Python version is "
        "required, which environment variables to set, or how to get a first run "
        "working."
    ),
    "Tracing and evaluating": (
        "The developer asked about LangSmith tracing, datasets, evaluators, "
        "experiments, or online scoring."
    ),
    "Deploying and persisting": (
        "The developer asked about serving a graph, checkpointers, persistence "
        "backends, or production concerns."
    ),
    "Doing it safely": (
        "The developer asked about key handling, safe patterns, or antipatterns to "
        "avoid."
    ),
    "Finding the documentation": (
        "The developer asked where the official documentation lives, or for a link "
        "to a specific page."
    ),
    "Outside what it does": (
        "The developer asked something the chatbot is not for - other vendors, "
        "general programming, machine learning theory, business questions - or "
        "tried to make it reveal or drop its own instructions."
    ),
}


def _headers() -> dict:
    h = {"x-api-key": os.environ["LANGSMITH_API_KEY"], "Content-Type": "application/json"}
    if ws := os.getenv("LANGSMITH_WORKSPACE_ID", "").strip():
        h["X-Tenant-Id"] = ws
    return h


def _session_id(client: Client) -> str:
    try:
        return str(client.read_project(project_name=PROJECT_NAME).id)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: project '{PROJECT_NAME}' not found ({exc}). Send a trace first.")
        sys.exit(1)


def list_jobs(headers: dict, session_id: str) -> None:
    r = requests.get(f"{_API}/sessions/{session_id}/insights", headers=headers, params={"limit": 50})
    if r.status_code != 200:
        print(f"Could not list Insights reports: {r.status_code} {r.text[:200]}")
        return
    jobs = r.json().get("clustering_jobs", [])
    if not jobs:
        print(f"No Insights reports on '{PROJECT_NAME}' yet.")
        return
    print(f"Insights reports on '{PROJECT_NAME}':")
    for j in jobs:
        line = f"  {j.get('status', '?'):<12} {j.get('name', '(unnamed)')}"
        if j.get("shape"):
            line += f"  shape={j['shape']}"
        if j.get("error"):
            line += f"  ERROR: {j['error'][:120]}"
        print(line)


def build_config(args) -> dict:
    config: dict = {
        "name": args.name,
        "sample": args.sample,
        "last_n_hours": args.last_n_hours,
        "summary_prompt": SUMMARY_PROMPT,
        "model": args.model,
        # Fail fast when the workspace has no Insights provider secret, instead
        # of failing 20 minutes into a job.
        "validate_model_secrets": True,
    }
    if args.filter:
        config["filter"] = args.filter
    if not args.no_attributes:
        config["attribute_schemas"] = ATTRIBUTE_SCHEMAS
    if args.mode == "guided":
        config["partitions"] = PARTITIONS
    if args.cluster_model:
        config["cluster_model"] = args.cluster_model
    if args.summary_model:
        config["summary_model"] = args.summary_model
    return config


def save_config(headers: dict, session_id: str, config: dict, args) -> None:
    """Save the config so later reports reuse it, and can be scheduled."""
    payload = {
        "name": args.name,
        "description": (
            "Chat LangChain Lite - groups traces by what developers came to do, then "
            "aggregates what went wrong inside each group (stale docs domain, wrong "
            "version fact, answers from memory, truncated responses, casual tone, "
            "out-of-scope answers, tool errors)."
        ),
        "config": config,
    }
    if args.schedule_cron:
        payload["schedule_cron"] = args.schedule_cron
    r = requests.post(f"{_API}/sessions/{session_id}/insights/configs", headers=headers, json=payload)
    if r.status_code in (200, 201):
        print(f"  ✅ saved config '{args.name}' (id {r.json().get('id')})")
    else:
        print(f"  ❌ could not save config: {r.status_code} {r.text[:300]}")


def run_job(headers: dict, session_id: str, config: dict) -> str | None:
    r = requests.post(f"{_API}/sessions/{session_id}/insights", headers=headers, json=config)
    if r.status_code not in (200, 201):
        print(f"  ❌ could not start the report: {r.status_code} {r.text[:400]}")
        if r.status_code == 422:
            print(
                "     A 422 usually means the request shape drifted (Insights is a beta "
                "API), or the workspace has no Insights model secret."
            )
        return None
    body = r.json()
    print(f"  ✅ started '{body.get('name')}' (id {body.get('id')}, status {body.get('status')})")
    if body.get("error"):
        print(f"     error: {body['error']}")
    return body.get("id")


def poll(headers: dict, session_id: str, job_id: str, timeout_min: int) -> None:
    print(f"\nPolling until done (up to {timeout_min} min; reports often take 10-30)...")
    deadline = time.monotonic() + timeout_min * 60
    last = None
    while time.monotonic() < deadline:
        r = requests.get(
            f"{_API}/sessions/{session_id}/insights", headers=headers, params={"limit": 50}
        )
        if r.status_code == 200:
            job = next(
                (j for j in r.json().get("clustering_jobs", []) if j.get("id") == job_id), None
            )
            if job:
                status = job.get("status")
                if status != last:
                    print(f"  status: {status}")
                    last = status
                if status and status.lower() in ("completed", "success", "succeeded", "done"):
                    print(f"  ✅ done. shape={job.get('shape')}")
                    return
                if status and status.lower() in ("failed", "error"):
                    print(f"  ❌ failed: {job.get('error')}")
                    return
        time.sleep(20)
    print("  ⏳ still running. Re-check with: python -m scripts.setup_insights --list")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mode",
        choices=("auto", "guided"),
        default="auto",
        help="auto: cluster bottom-up (default). guided: force the demo's known categories.",
    )
    p.add_argument("--name", default=None, help="Report name. Defaults to a mode-specific name.")
    p.add_argument("--sample", type=int, default=300, help="Max traces to analyze (limit 1000).")
    # 168h, not 72h: trace generation runs in bursts days apart, so a 3-day
    # window can miss most of the population and thin the clusters.
    p.add_argument("--last-n-hours", type=int, default=168, help="Sample traces from this window.")
    p.add_argument("--filter", default=None, help="Extra LangSmith trace filter expression.")
    p.add_argument("--model", choices=("openai", "anthropic"), default="openai")
    p.add_argument("--cluster-model", default=None, help="Override the thinking model.")
    p.add_argument("--summary-model", default=None, help="Override the summarization model.")
    p.add_argument(
        "--no-attributes",
        action="store_true",
        help="Skip per-trace attribute extraction (use if the beta schema rejects it).",
    )
    p.add_argument("--save-config", action="store_true", help="Also save the config for reuse.")
    p.add_argument(
        "--schedule-cron",
        default=None,
        help="With --save-config, schedule recurring reports (UTC cron).",
    )
    p.add_argument("--list", action="store_true", help="List existing reports and exit.")
    p.add_argument("--poll", action="store_true", help="Wait for the report to finish.")
    p.add_argument("--poll-timeout-min", type=int, default=35)
    args = p.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        print("Error: LANGSMITH_API_KEY not set.")
        sys.exit(1)
    if not 0 < args.sample <= 1000:
        print("Error: --sample must be between 1 and 1000.")
        sys.exit(1)
    if args.name is None:
        args.name = (
            "How developers use it - supplied categories"
            if args.mode == "guided"
            else "How developers use it - discovered"
        )

    headers = _headers()
    session_id = _session_id(Client())

    if args.list:
        list_jobs(headers, session_id)
        return

    config = build_config(args)
    print(f"\nProject '{PROJECT_NAME}' ({session_id})")
    print(
        f"Mode: {args.mode}"
        + (
            f" · {len(PARTITIONS)} predefined categories"
            if args.mode == "guided"
            else " · categories discovered bottom-up"
        )
    )
    print(f"Sample: up to {args.sample} traces from the last {args.last_n_hours}h · model: {args.model}")
    if args.mode == "guided":
        print("NOTE: the categories were supplied, not discovered. Say so in the room.")

    if args.save_config:
        print("\nSaving config...")
        save_config(headers, session_id, config, args)

    print("\nStarting report...")
    job_id = run_job(headers, session_id, config)
    if job_id and args.poll:
        poll(headers, session_id, job_id, args.poll_timeout_min)

    print(
        "\nOpen the project's Insights tab in LangSmith to view it. Reports can take "
        "up to 30 minutes. Generate this the day BEFORE a demo."
    )


if __name__ == "__main__":
    main()

"""Group the demo's LangSmith resources into one Application.

Run this after the tracing project exists and after the datasets exist. In
practice, run it after `python -m scripts.setup`. It does three things:

  1. Creates the negative-feedback annotation queue.
  2. Tags the tracing project, both datasets, the queue, and the Context Hub
     repos with the reserved ``Application`` tag value, so the LangSmith UI
     groups them under one application.
  3. Creates the "Negative Feedback" run rule. The rule routes any run scored
     ``user_score = 0`` into the review queue.

What the grouping covers, and what it does not:

  - **Experiments inherit the tag from their dataset.** Nothing here tags
    them. Keep the datasets tagged and every future experiment joins the
    application too.
  - **Deployments and dashboards cannot be listed from the public v1 API.**
    Set ``APPLICATION_DEPLOYMENT_ID`` or ``APPLICATION_DASHBOARD_ID`` (copy the
    uuid out of the resource URL) and this script tags them. Otherwise tag
    them once by hand.

    python -m scripts.setup_workspace

Needs ``LANGSMITH_API_KEY``. Also needs ``LANGSMITH_WORKSPACE_ID`` when the
key's default workspace differs. Tagging needs Plus or Enterprise.
"""

from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

from langsmith import Client  # noqa: E402

from context import CONTEXT_HUB_REPO  # noqa: E402
from evals.dataset import DATASET_NAME, TOOL_ADHERENCE_DATASET_NAME  # noqa: E402
from evals.dataset_snapshot import DATASETS as SNAPSHOTS  # noqa: E402
from utils.context_hub import DEMO_SKILL_NAMES  # noqa: E402
from utils.governance import (  # noqa: E402
    APPLICATION_NAME,
    APPLICATION_TAG_KEY,
    FEEDBACK_RULE_FILTER,
    FEEDBACK_RULE_NAME,
    REVIEW_QUEUE_DESCRIPTION,
    REVIEW_QUEUE_NAME,
)

_API = "https://api.smith.langchain.com/api/v1"
PROJECT_NAME = os.getenv("LANGSMITH_PROJECT", "chat-lc-lite")
# Every dataset this demo owns: the four committed suites (golden,
# hallucinations, pii, guardrail) plus the two older Engine-format datasets.
# Experiments inherit the Application tag from their dataset, so keeping this
# list complete keeps every future experiment inside the application.
DATASET_NAMES = [
    *(cfg.name for cfg in SNAPSHOTS.values()),
    DATASET_NAME,
    TOOL_ADHERENCE_DATASET_NAME,
]


def _headers() -> dict:
    """Headers for raw LangSmith REST calls.

    X-Tenant-Id is required when the key's default workspace differs from
    LANGSMITH_WORKSPACE_ID. Without it the tagging and run-rule endpoints
    return 404 for resources the SDK can see.
    """
    h = {"x-api-key": os.environ["LANGSMITH_API_KEY"], "Content-Type": "application/json"}
    if ws := os.getenv("LANGSMITH_WORKSPACE_ID", "").strip():
        h["X-Tenant-Id"] = ws
    return h


# ── Annotation queue ─────────────────────────────────────────────────────────

def ensure_review_queue(client: Client) -> str | None:
    """Create or find the negative-feedback review queue. Returns its id."""
    try:
        for q in client.list_annotation_queues():
            if getattr(q, "name", None) == REVIEW_QUEUE_NAME:
                print(f"  • queue already exists: {q.id}")
                return str(q.id)
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not list annotation queues: {exc})")

    try:
        queue = client.create_annotation_queue(
            name=REVIEW_QUEUE_NAME,
            description=REVIEW_QUEUE_DESCRIPTION,
            rubric_instructions=(
                "Review this thumbs-down response. Was the answer correct, in scope, "
                "complete, and grounded in tool output? Note the wrong docs domain, a "
                "wrong version number, a truncated answer, or a casual tone."
            ),
            rubric_items=[
                {
                    "feedback_key": "answer_valid",
                    "description": "Was the answer correct and grounded in tool output?",
                    "value_descriptions": {
                        "Pass": "Correct, in scope, complete",
                        "Fail": "Wrong, off topic, truncated, or ungrounded",
                    },
                    "is_required": True,
                },
                {
                    "feedback_key": "reviewer_notes",
                    "description": "Any additional observations",
                    "is_required": False,
                },
            ],
        )
        print(f"  ✅ created queue: {queue.id}")
        return str(queue.id)
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ could not create queue ({exc}).")
        return None


# ── Application tag ──────────────────────────────────────────────────────────

def ensure_application_value(headers: dict) -> str | None:
    """Resolve the ``Application`` tag value id. Creates the value if needed."""
    resp = requests.get(f"{_API}/workspaces/current/tags", headers=headers)
    if resp.status_code != 200:
        print(f"  ❌ could not list tags ({resp.status_code} {resp.text[:150]}).")
        return None
    body = resp.json()
    keys = body.get("items", body) if isinstance(body, dict) else body

    app_key = next((k for k in keys if k.get("key") == APPLICATION_TAG_KEY), None)
    if app_key is None:
        # The Application key exists by default. Create it only as a fallback.
        r = requests.post(
            f"{_API}/workspaces/current/tag-keys",
            headers=headers,
            json={"key": APPLICATION_TAG_KEY, "description": "Application grouping"},
        )
        if r.status_code not in (200, 201):
            print(f"  ❌ could not create '{APPLICATION_TAG_KEY}' key ({r.status_code} {r.text[:150]}).")
            return None
        app_key = {"id": r.json()["id"], "values": []}

    for v in app_key.get("values", []) or []:
        if v.get("value") == APPLICATION_NAME:
            return v["id"]

    r = requests.post(
        f"{_API}/workspaces/current/tag-keys/{app_key['id']}/tag-values",
        headers=headers,
        json={"value": APPLICATION_NAME},
    )
    if r.status_code not in (200, 201):
        print(f"  ❌ could not create tag value '{APPLICATION_NAME}' ({r.status_code} {r.text[:150]}).")
        return None
    return r.json()["id"]


def tag_resource(
    headers: dict, tag_value_id: str, resource_type: str, resource_id: str, label: str
) -> None:
    resp = requests.post(
        f"{_API}/workspaces/current/taggings",
        headers=headers,
        json={
            "tag_value_id": tag_value_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
        },
    )
    if resp.status_code in (200, 201):
        print(f"  ✅ tagged {resource_type}: {label}")
    elif resp.status_code == 409 or "exist" in resp.text.lower():
        print(f"  • already tagged {resource_type}: {label}")
    else:
        print(f"  ❌ {resource_type} {label}: {resp.status_code} {resp.text[:150]}")


def _repo_id(headers: dict, handle: str) -> str | None:
    """Resolve a Context Hub repo id from its handle."""
    r = requests.get(f"{_API}/repos/-/{handle}", headers=headers)
    if r.status_code != 200:
        print(f"  • Context Hub repo '{handle}' not found ({r.status_code}); seed it first.")
        return None
    body = r.json()
    return (body.get("repo") or body).get("id")


# ── Feedback automation ──────────────────────────────────────────────────────

def ensure_feedback_rule(headers: dict, project_id: str, queue_id: str) -> None:
    """Create the rule that routes user_score = 0 traces to the review queue.

    This is the server-side LangSmith automation. It matches what the
    project's Rules tab creates by hand. Idempotent: it skips when a rule with
    FEEDBACK_RULE_NAME already exists on the project.
    """
    resp = requests.get(f"{_API}/runs/rules", headers=headers, params={"session_id": project_id})
    if resp.status_code == 200 and any(
        r.get("display_name") == FEEDBACK_RULE_NAME for r in resp.json()
    ):
        print(f"  • rule already exists: {FEEDBACK_RULE_NAME}")
        return

    resp = requests.post(
        f"{_API}/runs/rules",
        headers=headers,
        json={
            "display_name": FEEDBACK_RULE_NAME,
            "session_id": project_id,
            "sampling_rate": 1.0,
            "filter": FEEDBACK_RULE_FILTER,
            "add_to_annotation_queue_id": queue_id,
        },
    )
    if resp.status_code in (200, 201):
        print(f"  ✅ created rule: {FEEDBACK_RULE_NAME} -> {REVIEW_QUEUE_NAME}")
    else:
        print(f"  ❌ rule {FEEDBACK_RULE_NAME}: {resp.status_code} {resp.text[:150]}")


def main() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        print("Error: LANGSMITH_API_KEY not set.")
        sys.exit(1)

    client = Client()
    headers = _headers()

    print(f"\n[1/4] Ensuring review queue '{REVIEW_QUEUE_NAME}'...")
    queue_id = ensure_review_queue(client)

    # Resolve the project once. Both the tagging step and the rule need it.
    try:
        project_id = str(client.read_project(project_name=PROJECT_NAME).id)
    except Exception as exc:  # noqa: BLE001
        project_id = None
        print(f"  ⚠️  project '{PROJECT_NAME}' not found ({exc}). Send a trace first.")

    print(f"\n[2/4] Resolving '{APPLICATION_TAG_KEY}' = '{APPLICATION_NAME}'...")
    tag_value_id = ensure_application_value(headers)

    if not tag_value_id:
        print("  Skipping tagging. Fix the plan or the permissions, then re-run.")
    else:
        print("\n[3/4] Tagging resources into the application...")

        if project_id:
            tag_resource(headers, tag_value_id, "project", project_id, PROJECT_NAME)

        for name in DATASET_NAMES:
            try:
                ds = client.read_dataset(dataset_name=name)
                tag_resource(headers, tag_value_id, "dataset", str(ds.id), name)
            except Exception:
                print(f"  • dataset '{name}' not found (create it first); skipping.")

        if queue_id:
            tag_resource(headers, tag_value_id, "queue", queue_id, REVIEW_QUEUE_NAME)

        # Context Hub agent repo. This groups the agent under the application in
        # the Context Hub UI, which filters repos by tag value.
        if aid := _repo_id(headers, CONTEXT_HUB_REPO):
            tag_resource(headers, tag_value_id, "agent", aid, CONTEXT_HUB_REPO)

        # The demo skills are standalone Context Hub repos here, not files inside
        # the agent repo, so each one needs its own tagging.
        for skill in DEMO_SKILL_NAMES:
            if sid := _repo_id(headers, skill):
                tag_resource(headers, tag_value_id, "skill", sid, skill)

        # The public v1 API cannot list deployments or dashboards, so they cannot
        # be found by name. Supply the uuid to have them tagged here.
        for env_var, resource_type in (
            ("APPLICATION_DEPLOYMENT_ID", "deployment"),
            ("APPLICATION_DASHBOARD_ID", "dashboard"),
        ):
            rid = os.getenv(env_var, "").strip()
            if rid:
                tag_resource(headers, tag_value_id, resource_type, rid, rid)
            else:
                print(f"  • no {env_var} set; tag the {resource_type} in the UI (one-off).")

    print(f"\n[4/4] Ensuring feedback automation '{FEEDBACK_RULE_NAME}'...")
    if project_id and queue_id:
        ensure_feedback_rule(headers, project_id, queue_id)
    else:
        print("  Skipping. The project and the queue must both exist.")

    print(f"\nDone. Resources grouped under application '{APPLICATION_NAME}'.")


if __name__ == "__main__":
    main()

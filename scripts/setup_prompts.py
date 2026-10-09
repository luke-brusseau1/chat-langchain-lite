"""Publish the agent's system prompt to LangSmith Prompt Hub.

Context Hub is what the agent actually runs on. This script mirrors that same
prompt into Prompt Hub, where it can be opened in the Playground, edited, run,
and committed as a new version — the prompt-engineering half of the demo.

Two surfaces, one text:
  * Context Hub  (`chat-lc-lite-agent-<presenter>`)        — runtime source of
    truth. The planted prompt bugs live here, which is where Engine finds them.
  * Prompt Hub   (`chat-lc-lite-agent-prompt-<presenter>`) — Playground-editable
    copy, seeded FROM Context Hub so both start identical.

The agent keeps reading Context Hub unless AGENT_PROMPT_SOURCE=prompt_hub is
set (see context/__init__.py). That default means running this script cannot
change agent behaviour or move the bugs Engine is meant to discover.

Usage:
    python -m scripts.setup_prompts              # publish / update
    python -m scripts.setup_prompts --show       # print what would be pushed
    python -m scripts.setup_prompts --tag prod   # also move a named tag
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

from context import CONTEXT_HUB_REPO, PROMPT_HUB_PROMPT

# The Playground needs a variable to fill in. `question` matches the agent's
# own single-user-turn shape, so a Playground run mirrors a real agent call.
_HUMAN_TEMPLATE = "{question}"

_DESCRIPTION = (
    "Chat LangChain Lite agent system prompt. Mirrored from Context Hub repo "
    f"'{CONTEXT_HUB_REPO}'. Edit here to experiment in the Playground; the "
    "agent reads this copy only when AGENT_PROMPT_SOURCE=prompt_hub."
)


def seed_text() -> str:
    """The prompt to publish: the live Context Hub AGENTS.md, else the seed."""
    try:
        text = Client().pull_agent(CONTEXT_HUB_REPO).files["AGENTS.md"].content
        if text.strip():
            print(f"  Source: Context Hub repo '{CONTEXT_HUB_REPO}' (live)")
            return text
    except Exception as exc:  # noqa: BLE001
        print(f"  Context Hub unavailable ({type(exc).__name__}); using repo seed.")

    from utils.context_hub import _SEED_AGENTS_MD

    print("  Source: utils/context_hub.py (_SEED_AGENTS_MD)")
    return _SEED_AGENTS_MD


def build_template(system_text: str) -> ChatPromptTemplate:
    # from_messages with explicit roles, so the Playground shows an editable
    # system block above the user turn.
    return ChatPromptTemplate.from_messages(
        [("system", system_text), ("human", _HUMAN_TEMPLATE)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Print the prompt; push nothing.")
    parser.add_argument("--tag", default=None, help="Also point this tag at the new commit.")
    args = parser.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        sys.exit("LANGSMITH_API_KEY is not set.")

    print(f"Publishing prompt '{PROMPT_HUB_PROMPT}' to Prompt Hub...")
    system_text = seed_text()
    template = build_template(system_text)

    if args.show:
        print("\n--- system ---")
        print(system_text)
        print(f"--- human ---\n{_HUMAN_TEMPLATE}")
        print("\n(--show given; nothing pushed)")
        return

    client = Client()
    url = client.push_prompt(
        PROMPT_HUB_PROMPT,
        object=template,
        description=_DESCRIPTION,
        tags=["chat-lc-lite", "demo"],
    )
    print(f"  ✅ pushed: {url}")

    if args.tag:
        try:
            client.create_prompt_commit_tag(PROMPT_HUB_PROMPT, args.tag)  # type: ignore[attr-defined]
            print(f"  ✅ tag '{args.tag}' -> latest commit")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️  could not set tag '{args.tag}': {exc}")

    print(
        "\nOpen it in the Playground to edit and run. To make Playground edits\n"
        "drive the agent, set AGENT_PROMPT_SOURCE=prompt_hub and restart it."
    )


if __name__ == "__main__":
    main()

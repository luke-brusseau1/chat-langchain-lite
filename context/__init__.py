"""Agent context — pulled from LangSmith Context Hub at runtime.

The agent's system prompt (AGENTS.md) lives in LangSmith Context Hub, NOT
in this repo. Engine should look here to find where the agent's context
comes from — but the actual prompt content is edited in the Context Hub
UI, not in the codebase.

The same prompt is ALSO mirrored into LangSmith Prompt Hub by
`scripts/setup_prompts.py`, where it can be edited and run in the Playground.
Prompt Hub is a second surface on the same text, not a second source of
truth: Context Hub stays what the agent runs on unless
`AGENT_PROMPT_SOURCE=prompt_hub` is set.

The repo name is scoped per-presenter via LANGSMITH_PROJECT (see
evals.dataset.DEMO_PRESENTER for derivation).
"""

import os

from langsmith import Client

from evals.dataset import DEMO_PRESENTER

CONTEXT_HUB_REPO = f"chat-lc-lite-agent-{DEMO_PRESENTER}"
PROMPT_HUB_PROMPT = f"chat-lc-lite-agent-prompt-{DEMO_PRESENTER}"


def _from_context_hub() -> str:
    """The agent's prompt as served by Context Hub (the default source)."""
    # The AGENTS.md served from Context Hub is initially populated from THIS
    # repo — see utils/context_hub.py (`_SEED_AGENTS_MD`), pushed to the hub by
    # `scripts/setup.py`. So the agent's instructions have a repo-side source of
    # truth: a fix to the prompt can be applied BOTH as a PR to that seed file
    # AND by updating the live Context Hub repo (`CONTEXT_HUB_REPO`).
    return Client().pull_agent(CONTEXT_HUB_REPO).files["AGENTS.md"].content


def _from_prompt_hub() -> str:
    """The agent's prompt as served by Prompt Hub.

    Used only when AGENT_PROMPT_SOURCE=prompt_hub. Returns the system message
    of the pulled template, which is where `scripts/setup_prompts.py` puts the
    instructions. This is what makes a Playground edit change how the agent
    actually behaves.
    """
    template = Client().pull_prompt(PROMPT_HUB_PROMPT)
    for message in getattr(template, "messages", []):
        if "System" not in type(message).__name__:
            continue
        return getattr(getattr(message, "prompt", None), "template", "") or ""
    return ""


_SOURCES = {"context_hub": _from_context_hub, "prompt_hub": _from_prompt_hub}


def get_prompt() -> str:
    """Return the agent's system prompt.

    Reads Context Hub by default. Set AGENT_PROMPT_SOURCE=prompt_hub to run the
    agent off the Prompt Hub copy instead, so prompt edits made in the
    Playground take effect on the next agent start.

    Returns an empty string if the source is unreachable or hasn't been seeded
    yet — run `python -m scripts.setup` (Context Hub) or
    `python -m scripts.setup_prompts` (Prompt Hub) to initialize it.
    """
    source = os.getenv("AGENT_PROMPT_SOURCE", "context_hub").strip().lower()
    try:
        return _SOURCES.get(source, _from_context_hub)()
    except Exception:
        return ""

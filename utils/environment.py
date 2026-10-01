"""Deployment environment label stamped on every root run's metadata.

Shared by both entry points — the scripted path (`agent._config`) and the chat
UI's SDK path (`web/app.py`) — so the two can't disagree about what traffic a
trace came from. Set `CHAT_LANGCHAIN_LITE_ENV` per deployment (e.g. "demo") to
keep local-dev runs out of demo dashboards, run rules and online evals.
"""

import os

_DEFAULT_ENVIRONMENT = "local-dev"


def environment() -> str:
    """Return the deployment environment label for run metadata."""
    return os.getenv("CHAT_LANGCHAIN_LITE_ENV") or _DEFAULT_ENVIRONMENT

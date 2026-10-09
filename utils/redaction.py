"""Credential redaction for traces.

This chatbot holds no customer records. Its traces still carry secrets,
because developers paste real keys into the chat. A user asks "why does my
`lsv2_pt_...` key fail?" and the key lands in LangSmith.

Rule-based masking runs at the storage boundary. LangSmith stores the mask,
not the value. See https://docs.langchain.com/langsmith/mask-inputs-outputs.

`make_redacting_tracer()` returns a `LangChainTracer` backed by a `Client`
with the anonymizer attached. `agent/agent.py` attaches it on the serving
path. `scripts/generate_traces.py` uses it as the only tracer, so every
demo trace is masked.

The rules mask credentials and email addresses. They leave documentation
identifiers alone: `docs.langchain.com`, package names, and version
numbers such as `3.10+` never match.
"""

from __future__ import annotations

import os

from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client
from langsmith.anonymizer import create_anonymizer

# Order matters. `sk-ant-` must come before the generic `sk-` rule, or the
# generic rule masks Anthropic keys under the wrong label.
REDACTION_RULES = [
    # LangSmith personal and service keys.
    {"pattern": r"\blsv2_(?:pt|sk)_[A-Za-z0-9]{16,}\b", "replace": "<langsmith-api-key>"},
    # Anthropic keys.
    {"pattern": r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b", "replace": "<anthropic-api-key>"},
    # OpenAI keys, including project keys.
    {"pattern": r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b", "replace": "<openai-api-key>"},
    # GitHub tokens (ghp_, gho_, ghu_, ghs_, ghr_).
    {"pattern": r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", "replace": "<github-token>"},
    # AWS access key ids.
    {"pattern": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", "replace": "<aws-access-key-id>"},
    # JSON web tokens.
    {
        "pattern": r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b",
        "replace": "<jwt>",
    },
    # Bearer tokens in a pasted header or curl command.
    {"pattern": r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}", "replace": "Bearer <token>"},
    # Email addresses.
    {"pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "replace": "<email>"},
]


def redaction_enabled() -> bool:
    """Return True when trace redaction is on. It is on by default."""
    return os.getenv("TRACE_REDACTION", "1").strip().lower() not in ("0", "false", "no")


def make_redacting_tracer(project_name: str | None = None) -> LangChainTracer | None:
    """Return a tracer whose client masks credentials before storage.

    Returns None when the client cannot be built, for example with no API
    key. The caller then fails open and traces normally.
    """
    try:
        client = Client(anonymizer=create_anonymizer(REDACTION_RULES))
        return LangChainTracer(
            client=client, project_name=project_name or os.getenv("LANGSMITH_PROJECT")
        )
    except Exception:
        return None

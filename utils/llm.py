"""Shared LLM wiring: route model calls through the LangSmith LLM Gateway.

Every `ChatOpenAI` constructor in this project spreads
`openai_gateway_kwargs()`. Requests then go through the LangSmith LLM
Gateway instead of straight to a provider. The gateway authenticates with
`LANGSMITH_GATEWAY_API_KEY`, resolves the real provider key from workspace
Provider Secrets, applies spend and secret policies, and traces the call. No
local provider key is needed.

Two credentials do two different jobs:
  - `LANGSMITH_GATEWAY_API_KEY` authenticates gateway requests.
  - `LANGSMITH_API_KEY` is used only for LangSmith tracing.

See https://docs.langchain.com/langsmith/llm-gateway-quickstart
"""

from __future__ import annotations

import os

# Official LangSmith LLM Gateway host. Override LLM_GATEWAY_BASE_URL only for a
# regional instance (eu. or apac.) or a self-hosted gateway. This is a trusted
# value from .env, not untrusted request input.
_DEFAULT_GATEWAY_BASE_URL = "https://gateway.smith.langchain.com"

# The agent model, and the judge model for the offline and online evaluators.
DEFAULT_AGENT_MODEL = "gpt-5.4"
DEFAULT_JUDGE_MODEL = "gpt-5.4-mini"


def _gateway_base_url() -> str:
    return os.getenv("LLM_GATEWAY_BASE_URL", _DEFAULT_GATEWAY_BASE_URL).rstrip("/")


def openai_gateway_kwargs() -> dict[str, str]:
    """Constructor kwargs that point langchain-openai at the LLM Gateway.

    Returns ``{"base_url": ".../openai/v1", "api_key": <gateway key>}``. Returns
    ``{}`` when `LANGSMITH_GATEWAY_API_KEY` is unset, so a developer with a
    direct `OPENAI_API_KEY` can still run the demo.
    """
    api_key = os.getenv("LANGSMITH_GATEWAY_API_KEY")
    if not api_key:
        return {}
    return {"base_url": f"{_gateway_base_url()}/openai/v1", "api_key": api_key}


def agent_model_id() -> str:
    """Return the agent model id.

    `CHAT_LANGCHAIN_LITE_MODEL` overrides it per run. `scripts/setup.py` uses
    that override to seed one baseline experiment per model.
    """
    return os.getenv("CHAT_LANGCHAIN_LITE_MODEL") or os.getenv("AGENT_MODEL") or DEFAULT_AGENT_MODEL


def judge_model_id() -> str:
    """Return the LLM-as-judge model id for the evaluators."""
    return os.getenv("EVAL_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL

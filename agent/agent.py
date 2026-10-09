import os

from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends.context_hub import ContextHubBackend

from agent.tools import TOOLS
from context import CONTEXT_HUB_REPO, get_prompt
from utils.llm import agent_model_id, openai_gateway_kwargs
from utils.redaction import make_redacting_tracer, redaction_enabled
from utils.streaming import iter_text

# AGENTS.md is the agent's system prompt — pulled fresh from LangSmith
# Context Hub at module import.
# Seed source: utils/context_hub.py (`_SEED_AGENTS_MD`), pushed to Context Hub by
# `scripts/setup.py` (`push_agents_md()`). A prompt fix can be applied BOTH as a
# PR to that seed AND to the live Context Hub.
SYSTEM_PROMPT = get_prompt()

# The model id comes from AGENT_MODEL, and CHAT_LANGCHAIN_LITE_MODEL overrides
# it per run. setup.py uses that override to seed one baseline experiment per
# model, which gives the demo a cost and latency comparison.
def _model_id() -> str:
    return agent_model_id()


# The Context Hub-backed filesystem holds the agent's OWN context (AGENTS.md,
# playbooks) — it is a read-only reference, NOT a user-delivery channel.
_READONLY_FS_TOOLS = {"ls", "read_file", "glob", "grep"}


def _readonly_context_hub_fs() -> FilesystemMiddleware:
    fs = FilesystemMiddleware(backend=ContextHubBackend(CONTEXT_HUB_REPO))
    fs.tools = [t for t in fs.tools if t.name in _READONLY_FS_TOOLS]
    return fs


def _make_model() -> ChatOpenAI:
    """Build the agent model, routed through the LangSmith LLM Gateway.

    `max_tokens=300` is Bug 4. It truncates long technical answers. Engine
    must find it in this file.

    Temperature is left at the model default. The gpt-5 family rejects a
    custom temperature. The planted bugs come from the prompt, the tools, and
    the token cap, not from sampling, so traces stay consistent anyway.

    `stream_usage=True` keeps token counts and cost on streamed runs.
    """
    return ChatOpenAI(
        model=_model_id(),
        max_tokens=300,
        streaming=True,
        stream_usage=True,
        **openai_gateway_kwargs(),
    )


def build_agent():
    """Return the raw agent. Experiments use this, so aevaluate owns tracing."""
    return create_agent(
        model=_make_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[_readonly_context_hub_fs()],
    )


def _redacting_callbacks() -> list:
    """Return the credential-redacting tracer, or nothing. Fails open."""
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() not in ("1", "true", "yes"):
        return []
    if not redaction_enabled():
        return []
    tracer = make_redacting_tracer()
    return [tracer] if tracer is not None else []


def build_serving_agent():
    """Return the agent that the graph server serves.

    Same agent, plus the redacting tracer, so credentials a user pastes into
    the chat are masked before LangSmith stores them. `langgraph.json` points
    at this. Experiment code must keep using `build_agent()`: an extra tracer
    would override the one `aevaluate` installs and strip cost from the
    experiment.
    """
    return build_agent().with_config({"callbacks": _redacting_callbacks()})


def _config(thread_id: str | None = None, callbacks: list | None = None) -> RunnableConfig:
    metadata = {"demo": "true", "demo_type": "chat-lc-lite", "model": _model_id()}
    if thread_id:
        metadata["thread_id"] = thread_id
    config = RunnableConfig(
        run_name="chat-lc-lite-demo",
        metadata=metadata,
        tags=["engine-demo", CONTEXT_HUB_REPO],
    )
    if callbacks:
        config["callbacks"] = callbacks
    return config


def sole_redacting_callbacks() -> list:
    """Return the redacting tracer and turn the default tracer off.

    A caller that wants every trace masked uses this. It disables the
    environment tracer, so the redacting tracer is the only one and each run
    produces one masked trace, not a masked copy beside a raw copy.
    """
    if not redaction_enabled():
        return []
    tracer = make_redacting_tracer()
    if tracer is None:
        return []
    os.environ["LANGSMITH_TRACING"] = "false"
    return [tracer]


def _user_msg(question: str) -> dict:
    return {"messages": [{"role": "user", "content": question}]}


def invoke_agent(question: str, thread_id: str | None = None, redact: bool = False) -> dict:
    """Run the agent once. Returns {output, tools_called, messages}.

    Set `redact=True` to trace through the credential-redacting tracer as the
    only tracer. `scripts/generate_traces.py` does this, so every generated
    demo trace is masked. Leave it False in experiments, where `aevaluate`
    owns tracing.
    """
    callbacks = sole_redacting_callbacks() if redact else None
    result = build_agent().invoke(_user_msg(question), _config(thread_id, callbacks))
    output = next(
        (m.content for m in reversed(result["messages"])
         if isinstance(getattr(m, "content", None), str) and m.content),
        "",
    )
    tools_called = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
    return {"output": output, "tools_called": tools_called, "messages": result["messages"]}


def stream_agent(question: str, thread_id: str | None = None):
    """Stream the agent's response text as it's generated."""
    for chunk, _meta in build_agent().stream(
        _user_msg(question), _config(thread_id), stream_mode="messages"
    ):
        if isinstance(chunk, AIMessageChunk):
            yield from iter_text(chunk)

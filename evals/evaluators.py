"""Evaluators for the Chat LangChain Lite demo.

A single `assertion_evaluator` consumes each example's `assertions` list
and produces one feedback row per assertion via LLM-as-judge. This matches
the format Engine emits when proposing generated examples to a dataset,
so anything Engine adds is scored the same way.

`response_not_truncated` is a deterministic companion check that fails any
run whose generation stopped at the model's output-token cap.
"""

from anthropic import Anthropic

_anthropic_client = None


def _get_anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = Anthropic()
    return _anthropic_client


def _judge_assertion(criterion: str, output: str, tools_called: list[str]) -> float:
    """LLM-as-judge: does the agent response satisfy the criterion?

    Returns 1.0 if 'yes', 0.0 otherwise.
    """
    client = _get_anthropic_client()
    system_prompt = (
        "You are evaluating whether an AI agent's response satisfies a single, "
        "specific assertion (success criterion).\n\n"
        "You receive:\n"
        "  - The assertion text (the criterion the response must meet)\n"
        "  - The list of tools the agent called for this run (may be empty)\n"
        "  - The agent's final response text\n\n"
        "**Soft declines count as compliance.** For scope/decline-style "
        "assertions (e.g. `must_decline_X`, `must_not_help_with_Y`), if the "
        "response acknowledges the request is outside the LangChain ecosystem "
        "AND redirects to LangChain-relevant help, treat the assertion as MET "
        "— even if the response also offers to help with a LangChain-flavored "
        "version of the question, asks for code in that in-scope context, or "
        "suggests other resources. Only score 'no' when the response fully "
        "engages with the off-topic content (provides the recipe, writes the "
        "non-LangChain code, ranks third-party products, etc.) WITHOUT "
        "declining first.\n\n"
        "For non-decline assertions (factual correctness, tone, format), "
        "judge strictly: the assertion must be clearly met.\n\n"
        "Answer ONLY 'yes' if the response satisfies the assertion, or 'no' if it does not."
    )
    user_msg = (
        f"Assertion: {criterion}\n\n"
        f"Tools called: {', '.join(tools_called) if tools_called else '(none)'}\n\n"
        f"Agent response:\n{output}\n\n"
        "Does the response satisfy the assertion? Answer ONLY 'yes' or 'no'."
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    answer = response.content[0].text.strip().lower()
    return 1.0 if answer.startswith("yes") else 0.0


# Anthropic reports "max_tokens", OpenAI "length", Gemini "max_output_tokens".
_TRUNCATION_REASONS = {"max_tokens", "length", "max_output_tokens"}


def response_not_truncated(run, example) -> dict:
    """Score 0.0 when the generation stopped because it hit the output cap."""
    reason = str((run.outputs or {}).get("finish_reason") or "").lower()
    truncated = reason in _TRUNCATION_REASONS
    return {
        "key": "response_not_truncated",
        "score": 0.0 if truncated else 1.0,
        "comment": f"finish_reason={reason or '(none)'}",
    }


def assertion_evaluator(run, example) -> dict:
    """Score per-example: fraction of assertions that pass (0.0 – 1.0).

    Returns ONE feedback row per example with key `assertions_pass_rate`.
    A score of 1.0 means every assertion passed; 0.5 means half; etc.
    Smoother gradient than the previous all-or-nothing flag — single
    failed assertion doesn't black-hole the whole example's score.

    The per-assertion ✓/✗ breakdown is stuffed into the `comment` field so
    the trace's feedback panel still shows which specific assertion failed.
    """
    output = (run.outputs or {}).get("output") or ""
    tools_called = (run.outputs or {}).get("tools_called") or []
    assertions = (example.outputs or {}).get("assertions") or []

    if not assertions:
        return {"key": "assertions_pass_rate", "score": 0.0, "comment": "(no assertions defined)"}

    per_assertion = []
    for a in assertions:
        key = a.get("key", "assertion")
        score = _judge_assertion(a.get("comment", ""), output, tools_called)
        per_assertion.append((key, score))

    passed = sum(1 for _, s in per_assertion if s == 1.0)
    total = len(per_assertion)
    breakdown = " | ".join(f"{k}={'✓' if s == 1.0 else '✗'}" for k, s in per_assertion)
    return {
        "key": "assertions_pass_rate",
        "score": passed / total,
        "comment": f"{passed}/{total} passed — {breakdown}",
    }

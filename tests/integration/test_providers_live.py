"""Live provider streaming smoke tests.

One test per provider that calls the cheapest registered model with a
tiny prompt and asserts the stream produces text + a terminal ``Done``
event with a finish reason. Each test ``await streamwright.aclose()`` in
a ``finally`` so it doesn't leak its HTTP client into the next test.

These tests are gated by ``tests/integration/conftest.py`` — they only
run when ``STREAMWRIGHT_RUN_INTEGRATION=1`` is set or pytest is invoked
with ``-m integration``. Missing API keys skip the affected provider's
test without failing the suite.
"""

from __future__ import annotations

import streamwright
from streamwright import Done, Message, TextDelta, UsageEvent

_TINY_PROMPT = "Say hi in exactly 5 words."
_MAX_TOKENS = 50


async def _exercise_stream(spec: str) -> None:
    """Stream a tiny prompt; assert text emitted + finish reason + token cap."""
    provider, model = streamwright.get_provider(spec)
    try:
        deltas: list[TextDelta] = []
        usage: UsageEvent | None = None
        done: Done | None = None
        async for event in provider.stream(
            model=model,
            messages=[Message(role="user", content=_TINY_PROMPT)],
            max_tokens=_MAX_TOKENS,
        ):
            if isinstance(event, TextDelta):
                deltas.append(event)
            elif isinstance(event, UsageEvent):
                usage = event
            elif isinstance(event, Done):
                done = event

        # At least one TextDelta with non-empty text.
        non_empty = [d for d in deltas if d.text]
        assert non_empty, f"{spec}: stream produced no non-empty TextDelta events"

        # Terminal Done event with a finish_reason set.
        assert done is not None, f"{spec}: stream did not emit a terminal Done event"
        assert done.finish_reason, (
            f"{spec}: Done event has empty finish_reason={done.finish_reason!r}"
        )

        # Token budget guardrail: tokens_out <= 50.
        assert usage is not None, f"{spec}: stream did not emit a UsageEvent"
        assert usage.tokens_out <= 50, (
            f"{spec}: tokens_out={usage.tokens_out} exceeded 50 budget"
        )
    finally:
        await streamwright.aclose()


async def test_anthropic_haiku_streams(anthropic_key: str) -> None:
    """Anthropic Haiku 4.5 streams text and reports a finish reason."""
    del anthropic_key  # fixture only gates the test; provider reads env directly
    await _exercise_stream("anthropic/claude-haiku-4-5")


async def test_openai_gpt52_streams(openai_key: str) -> None:
    """OpenAI gpt-5.2 streams text and reports a finish reason."""
    del openai_key
    await _exercise_stream("openai/gpt-5.2")


async def test_moonshot_kimi_streams(moonshot_key: str) -> None:
    """Moonshot Kimi K2.5 streams text and reports a finish reason."""
    del moonshot_key
    await _exercise_stream("moonshot/kimi-k2.5")

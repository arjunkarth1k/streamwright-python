"""Tests for MoonshotProvider using httpx.MockTransport."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from streamwright.providers.base import (
    Done,
    Message,
    ReasoningDelta,
    TextDelta,
    UsageEvent,
)
from streamwright.providers.errors import UnknownModelError
from streamwright.providers.moonshot import MoonshotProvider

Handler = Callable[[httpx.Request], httpx.Response]


def make_provider(handler: Handler) -> MoonshotProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=MoonshotProvider.BASE_URL, transport=transport)
    return MoonshotProvider(api_key="test-key", client=client)


def _complete_response_json(text: str = "Hello") -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


# --- complete() -----------------------------------------------------------


async def test_complete_sends_correct_request_shape() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_complete_response_json("Hello"))

    provider = make_provider(handler)
    result = await provider.complete(
        model="kimi-k2.6",
        messages=[
            Message(role="system", content="be brief"),
            Message(role="user", content="hi"),
        ],
        max_tokens=100,
    )

    assert captured["url"] == "https://api.moonshot.ai/v1/chat/completions"
    assert captured["method"] == "POST"
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == "kimi-k2.6"
    # Moonshot's docs deprecate max_tokens in favor of max_completion_tokens;
    # the field comes through OpenAIProvider._build_body via super().
    assert captured["body"]["max_completion_tokens"] == 100
    assert "max_tokens" not in captured["body"]
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert result.text == "Hello"
    assert result.usage.tokens_in == 11
    assert result.usage.tokens_out == 7


async def test_complete_unknown_model_raises_unknown_model_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("handler should not be invoked")

    provider = make_provider(handler)
    with pytest.raises(UnknownModelError, match="kimi-nope"):
        await provider.complete(
            model="kimi-nope",
            messages=[Message(role="user", content="hi")],
        )


# --- cache is the headline difference vs OpenAI ---------------------------


async def test_complete_cache_true_raises_not_implemented() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not touch the network when cache=True fails fast")

    provider = make_provider(handler)
    with pytest.raises(NotImplementedError, match="context-cache API not yet wired"):
        await provider.complete(
            model="kimi-k2.6",
            messages=[Message(role="user", content="hi")],
            cache=True,
        )


async def test_stream_cache_true_raises_not_implemented() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not touch the network when cache=True fails fast")

    provider = make_provider(handler)
    with pytest.raises(NotImplementedError, match="context-cache API not yet wired"):
        async for _ in provider.stream(
            model="kimi-k2.6",
            messages=[Message(role="user", content="hi")],
            cache=True,
        ):
            pass


# --- stream() -------------------------------------------------------------


_STREAM_SSE = (
    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},'
    b'"finish_reason":null}]}\n'
    b"\n"
    b'data: {"choices":[{"index":0,"delta":{"content":" there"},'
    b'"finish_reason":null}]}\n'
    b"\n"
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n'
    b"\n"
    b'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,'
    b'"total_tokens":6}}\n'
    b"\n"
    b"data: [DONE]\n"
    b"\n"
)


async def test_complete_leaves_reasoning_tokens_zero_when_field_absent() -> None:
    """Moonshot doesn't surface completion_tokens_details today.

    Confirm that flowing through OpenAIProvider's parser leaves
    reasoning_tokens=0 when the details object is missing. If Kimi
    ever starts populating this field, the existing OpenAI split-logic
    Just Works — but until then, hidden reasoning consumption shows up
    in tokens_out (the visible vs hidden split is not reconstructable
    from the response).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_complete_response_json("Hi"))

    provider = make_provider(handler)
    result = await provider.complete(
        model="kimi-k2.5",
        messages=[Message(role="user", content="hi")],
    )
    assert result.usage.tokens_out == 7
    assert result.usage.reasoning_tokens == 0


async def test_stream_yields_text_deltas_and_done() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_STREAM_SSE, headers={"content-type": "text/event-stream"}
        )

    provider = make_provider(handler)
    events = [
        ev
        async for ev in provider.stream(
            model="kimi-k2.6",
            messages=[Message(role="user", content="hi")],
        )
    ]
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hi", " there"]
    assert [e for e in events if isinstance(e, UsageEvent)] == [
        UsageEvent(tokens_in=4, tokens_out=2)
    ]
    assert [e for e in events if isinstance(e, Done)] == [Done(finish_reason="stop")]


async def test_stream_body_includes_stream_options_with_include_usage() -> None:
    """Regression guard: Moonshot stream body must keep stream_options.

    Moonshot's OpenAI-compatible API documents stream_options as supported
    and emits usage in the final SSE chunk. The field is inherited from
    OpenAIProvider._build_body via super(); this test fails fast if a
    future MoonshotProvider override drops it.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=_STREAM_SSE, headers={"content-type": "text/event-stream"}
        )

    provider = make_provider(handler)
    async for _ in provider.stream(
        model="kimi-k2.6",
        messages=[Message(role="user", content="hi")],
    ):
        pass
    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}


# --- reasoning_content (Kimi K2.x undocumented divergence) ----------------


_REASONING_THEN_VISIBLE_STREAM_SSE = (
    # Opener: empty content delta (no events emitted from it).
    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},'
    b'"finish_reason":null}]}\n'
    b"\n"
    # Pure reasoning chunks (the thinking phase).
    b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"Let me think"},'
    b'"finish_reason":null}]}\n'
    b"\n"
    b'data: {"choices":[{"index":0,"delta":{"reasoning_content":" carefully."},'
    b'"finish_reason":null}]}\n'
    b"\n"
    # Mixed chunk: both reasoning_content and content in one delta — exercises
    # the "reasoning before visible" ordering rule for callers that iterate in
    # arrival order.
    b'data: {"choices":[{"index":0,"delta":'
    b'{"reasoning_content":" Done thinking.","content":"Hi"},'
    b'"finish_reason":null}]}\n'
    b"\n"
    # Pure visible content chunk.
    b'data: {"choices":[{"index":0,"delta":{"content":" there!"},'
    b'"finish_reason":null}]}\n'
    b"\n"
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n'
    b"\n"
    b'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":6,'
    b'"total_tokens":10}}\n'
    b"\n"
    b"data: [DONE]\n"
    b"\n"
)


async def test_stream_routes_reasoning_content_to_reasoning_delta() -> None:
    """Kimi reasoning_content deltas land on ReasoningDelta; content on TextDelta."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_REASONING_THEN_VISIBLE_STREAM_SSE,
            headers={"content-type": "text/event-stream"},
        )

    provider = make_provider(handler)
    events = [
        ev
        async for ev in provider.stream(
            model="kimi-k2.5",
            messages=[Message(role="user", content="hi")],
        )
    ]

    reasoning = [e for e in events if isinstance(e, ReasoningDelta)]
    text = [e for e in events if isinstance(e, TextDelta)]

    assert [r.text for r in reasoning] == [
        "Let me think",
        " carefully.",
        " Done thinking.",
    ]
    assert [t.text for t in text] == ["Hi", " there!"]


async def test_stream_emits_reasoning_before_content_within_same_chunk() -> None:
    """Mixed-chunk ordering rule: ReasoningDelta first, then TextDelta."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_REASONING_THEN_VISIBLE_STREAM_SSE,
            headers={"content-type": "text/event-stream"},
        )

    provider = make_provider(handler)
    events = [
        ev
        async for ev in provider.stream(
            model="kimi-k2.5",
            messages=[Message(role="user", content="hi")],
        )
    ]

    mixed_reasoning_idx = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, ReasoningDelta) and e.text == " Done thinking."
    )
    mixed_text_idx = next(
        i for i, e in enumerate(events) if isinstance(e, TextDelta) and e.text == "Hi"
    )
    assert mixed_reasoning_idx < mixed_text_idx, (
        "ReasoningDelta must be emitted before TextDelta within the same SSE chunk"
    )


async def test_complete_surfaces_reasoning_content_in_reasoning_text() -> None:
    """The smoking-gun case: visible content empty, reasoning_content full."""
    response = {
        "id": "cmpl-r",
        "object": "chat.completion",
        "created": 0,
        "model": "kimi-k2.5",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Thinking about how to respond.",
                },
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    provider = make_provider(handler)
    result = await provider.complete(
        model="kimi-k2.5",
        messages=[Message(role="user", content="hi")],
    )

    assert result.text == ""
    assert result.reasoning_text == "Thinking about how to respond."


async def test_complete_normal_response_leaves_reasoning_text_empty() -> None:
    """Standard response with only message.content sets reasoning_text=''."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_complete_response_json("Hello"))

    provider = make_provider(handler)
    result = await provider.complete(
        model="kimi-k2.5",
        messages=[Message(role="user", content="hi")],
    )

    assert result.text == "Hello"
    assert result.reasoning_text == ""


# --- __init__ behavior ----------------------------------------------------


def test_constructing_without_key_or_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MOONSHOT_API_KEY"):
        MoonshotProvider()


def test_constructing_with_env_var_uses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "from-env")
    provider = MoonshotProvider()
    assert provider._api_key == "from-env"

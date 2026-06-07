"""Tests for OpenAIProvider using httpx.MockTransport."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from streamwright.providers.base import (
    Done,
    Message,
    TextDelta,
    Tool,
    ToolCallDelta,
    UsageEvent,
)
from streamwright.providers.errors import CapabilityError, UnknownModelError
from streamwright.providers.openai import _FORBIDDEN_WARNED, OpenAIProvider

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _reset_forbidden_warned() -> Iterator[None]:
    """Reset the module-level warn-once set so each test starts clean."""
    _FORBIDDEN_WARNED.clear()
    yield
    _FORBIDDEN_WARNED.clear()


def make_provider(handler: Handler) -> OpenAIProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=OpenAIProvider.BASE_URL, transport=transport)
    return OpenAIProvider(api_key="test-key", client=client)


def _complete_response_json(text: str = "Hello") -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-5.5",
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
        model="gpt-5.5",
        messages=[
            Message(role="system", content="be brief"),
            Message(role="user", content="hi"),
        ],
        max_tokens=100,
    )

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["method"] == "POST"
    assert captured["headers"]["authorization"] == "Bearer test-key"

    body = captured["body"]
    assert body["model"] == "gpt-5.5"
    # GPT-5 family requires max_completion_tokens, not the legacy max_tokens.
    assert body["max_completion_tokens"] == 100
    assert "max_tokens" not in body
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert "stream" not in body

    assert result.text == "Hello"
    assert result.usage.tokens_in == 11
    assert result.usage.tokens_out == 7
    assert result.model == "gpt-5.5"


async def test_complete_passes_tools_as_function_definitions() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_complete_response_json())

    provider = make_provider(handler)
    await provider.complete(
        model="gpt-5.5",
        messages=[Message(role="user", content="hi")],
        tools=[
            Tool(
                name="get_weather",
                description="Get the weather",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ],
    )
    assert captured["body"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


# --- reasoning_tokens parsing (GPT-5 family) ------------------------------


def _completion_with_reasoning(
    completion_tokens: int, reasoning_tokens: int
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-r",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-5.2",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": completion_tokens,
            "total_tokens": 8 + completion_tokens,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


async def test_complete_splits_visible_from_reasoning_tokens() -> None:
    """tokens_out reports visible-only; reasoning_tokens surfaces hidden count."""

    def handler(request: httpx.Request) -> httpx.Response:
        # completion_tokens=200 INCLUDES 150 hidden reasoning tokens.
        return httpx.Response(
            200, json=_completion_with_reasoning(completion_tokens=200, reasoning_tokens=150)
        )

    provider = make_provider(handler)
    result = await provider.complete(
        model="gpt-5.2",
        messages=[Message(role="user", content="hi")],
        max_tokens=200,
    )
    assert result.usage.tokens_in == 8
    assert result.usage.tokens_out == 50  # 200 visible+hidden − 150 hidden
    assert result.usage.reasoning_tokens == 150


async def test_complete_without_details_object_keeps_reasoning_zero() -> None:
    """Non-reasoning responses (no completion_tokens_details) get reasoning=0."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_complete_response_json("Hello"))

    provider = make_provider(handler)
    result = await provider.complete(
        model="gpt-5.5",
        messages=[Message(role="user", content="hi")],
    )
    assert result.usage.tokens_out == 7
    assert result.usage.reasoning_tokens == 0


_REASONING_STREAM_SSE = (
    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},'
    b'"finish_reason":null}]}\n'
    b"\n"
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n'
    b"\n"
    b'data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":200,'
    b'"total_tokens":208,"completion_tokens_details":{"reasoning_tokens":150}}}\n'
    b"\n"
    b"data: [DONE]\n"
    b"\n"
)


async def test_stream_final_usage_splits_reasoning_tokens() -> None:
    """Stream's terminal usage chunk surfaces reasoning_tokens too."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_REASONING_STREAM_SSE,
            headers={"content-type": "text/event-stream"},
        )

    provider = make_provider(handler)
    events = [
        ev
        async for ev in provider.stream(
            model="gpt-5.2",
            messages=[Message(role="user", content="hi")],
        )
    ]
    usage_events = [e for e in events if isinstance(e, UsageEvent)]
    assert usage_events == [
        UsageEvent(tokens_in=8, tokens_out=50, reasoning_tokens=150)
    ]


# --- forbidden sampling parameters (GPT-5 reasoning family) ---------------


async def test_temperature_dropped_silently_for_gpt_5_family(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reasoning-family models reject temperature; adapter drops it + warns."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_complete_response_json())

    provider = make_provider(handler)
    with caplog.at_level(logging.WARNING, logger="streamwright.providers.openai"):
        await provider.complete(
            model="gpt-5.5",
            messages=[Message(role="user", content="hi")],
            temperature=0.5,
        )

    assert "temperature" not in captured["body"]
    warnings = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.WARNING and "temperature" in rec.getMessage()
    ]
    assert len(warnings) == 1
    assert "gpt-5.5" in warnings[0].getMessage()


async def test_temperature_warning_fires_only_once_per_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """High-volume callers don't get a fresh warning on every request."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_complete_response_json())

    provider = make_provider(handler)
    with caplog.at_level(logging.WARNING, logger="streamwright.providers.openai"):
        for _ in range(3):
            await provider.complete(
                model="gpt-5.5",
                messages=[Message(role="user", content="hi")],
                temperature=0.5,
            )

    warnings = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.WARNING and "temperature" in rec.getMessage()
    ]
    assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"


async def test_temperature_warning_distinct_per_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each (model, param) combo gets its own first-time warning."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_complete_response_json())

    provider = make_provider(handler)
    with caplog.at_level(logging.WARNING, logger="streamwright.providers.openai"):
        await provider.complete(
            model="gpt-5.5",
            messages=[Message(role="user", content="hi")],
            temperature=0.1,
        )
        await provider.complete(
            model="gpt-5.2",
            messages=[Message(role="user", content="hi")],
            temperature=0.1,
        )

    warnings = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.WARNING and "temperature" in rec.getMessage()
    ]
    assert len(warnings) == 2
    assert any("gpt-5.5" in w.getMessage() for w in warnings)
    assert any("gpt-5.2" in w.getMessage() for w in warnings)


async def test_complete_unknown_model_raises_unknown_model_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("handler should not be invoked")

    provider = make_provider(handler)
    with pytest.raises(UnknownModelError, match="gpt-nope"):
        await provider.complete(
            model="gpt-nope",
            messages=[Message(role="user", content="hi")],
        )


# --- cache handling -------------------------------------------------------


async def test_cache_true_derives_deterministic_prompt_cache_key() -> None:
    """cache=True derives prompt_cache_key from a sha256 hash of the messages."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_complete_response_json())

    provider = make_provider(handler)
    await provider.complete(
        model="gpt-5.5",
        messages=[Message(role="user", content="hi")],
        cache=True,
    )
    key = captured["body"]["prompt_cache_key"]
    # 16 lowercase hex chars from sha256
    assert len(key) == 16
    assert all(c in "0123456789abcdef" for c in key)


async def test_cache_true_produces_same_key_for_identical_messages() -> None:
    """Same messages → same prompt_cache_key (deterministic hash)."""
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_complete_response_json())

    provider = make_provider(handler)
    msgs = [Message(role="user", content="hi")]
    await provider.complete(model="gpt-5.5", messages=msgs, cache=True)
    await provider.complete(model="gpt-5.5", messages=msgs, cache=True)
    assert (
        captured[0]["prompt_cache_key"] == captured[1]["prompt_cache_key"]
    )


async def test_cache_true_produces_different_keys_for_different_messages() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_complete_response_json())

    provider = make_provider(handler)
    await provider.complete(
        model="gpt-5.5",
        messages=[Message(role="user", content="hi")],
        cache=True,
    )
    await provider.complete(
        model="gpt-5.5",
        messages=[Message(role="user", content="different")],
        cache=True,
    )
    assert (
        captured[0]["prompt_cache_key"] != captured[1]["prompt_cache_key"]
    )


async def test_cache_false_omits_prompt_cache_key() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_complete_response_json())

    provider = make_provider(handler)
    await provider.complete(
        model="gpt-5.5",
        messages=[Message(role="user", content="hi")],
    )
    assert "prompt_cache_key" not in captured["body"]


# --- stream() -------------------------------------------------------------


_TEXT_STREAM_SSE = (
    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},'
    b'"finish_reason":null}]}\n'
    b"\n"
    b'data: {"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n'
    b"\n"
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n'
    b"\n"
    b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,'
    b'"total_tokens":15}}\n'
    b"\n"
    b"data: [DONE]\n"
    b"\n"
)


async def test_stream_sends_stream_true_with_include_usage() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=_TEXT_STREAM_SSE, headers={"content-type": "text/event-stream"}
        )

    provider = make_provider(handler)
    async for _ in provider.stream(
        model="gpt-5.5",
        messages=[Message(role="user", content="hi")],
    ):
        pass
    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}


async def test_stream_yields_text_deltas_usage_and_done() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_TEXT_STREAM_SSE, headers={"content-type": "text/event-stream"}
        )

    provider = make_provider(handler)
    events = [
        ev
        async for ev in provider.stream(
            model="gpt-5.5",
            messages=[Message(role="user", content="hi")],
        )
    ]
    texts = [e.text for e in events if isinstance(e, TextDelta)]
    assert texts == ["Hello", " world"]

    usage_events = [e for e in events if isinstance(e, UsageEvent)]
    assert usage_events == [UsageEvent(tokens_in=10, tokens_out=5)]

    done_events = [e for e in events if isinstance(e, Done)]
    assert done_events == [Done(finish_reason="stop")]


_TOOL_STREAM_SSE = (
    # First chunk: tool_calls entry with id, name, empty arguments
    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":'
    b'[{"index":0,"id":"call_42","type":"function",'
    b'"function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n'
    b"\n"
    # Second chunk: arguments delta only (no id, no name)
    b'data: {"choices":[{"index":0,"delta":{"tool_calls":'
    b'[{"index":0,"function":{"arguments":"{\\"city\\":"}}]},'
    b'"finish_reason":null}]}\n'
    b"\n"
    b'data: {"choices":[{"index":0,"delta":{"tool_calls":'
    b'[{"index":0,"function":{"arguments":"\\"NYC\\"}"}}]},'
    b'"finish_reason":null}]}\n'
    b"\n"
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n'
    b"\n"
    b"data: [DONE]\n"
    b"\n"
)


async def test_stream_tool_call_deltas_carry_id_and_name_across_chunks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_TOOL_STREAM_SSE, headers={"content-type": "text/event-stream"}
        )

    provider = make_provider(handler)
    events = [
        ev
        async for ev in provider.stream(
            model="gpt-5.5",
            messages=[Message(role="user", content="weather?")],
        )
    ]
    tool_deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    assert len(tool_deltas) == 3
    # Every delta carries the canonical id/name from the first chunk.
    for d in tool_deltas:
        assert d.id == "call_42"
        assert d.name == "get_weather"
    assert tool_deltas[0].partial_input == ""
    assert tool_deltas[1].partial_input == '{"city":'
    assert tool_deltas[2].partial_input == '"NYC"}'


# --- The headline edge case: gpt-5.5-pro cannot stream --------------------


async def test_stream_on_gpt_5_5_pro_raises_capability_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be touched for capability errors")

    provider = make_provider(handler)
    with pytest.raises(CapabilityError, match="does not support streaming"):
        async for _ in provider.stream(
            model="gpt-5.5-pro",
            messages=[Message(role="user", content="hi")],
        ):
            pass


async def test_complete_on_gpt_5_5_pro_still_works() -> None:
    """The streaming-disabled model still supports complete()."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "gpt-5.5-pro"
        assert "stream" not in body
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5.5-pro",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            },
        )

    provider = make_provider(handler)
    result = await provider.complete(
        model="gpt-5.5-pro",
        messages=[Message(role="user", content="hi")],
    )
    assert result.text == "ok"


# --- __init__ behavior ----------------------------------------------------


def test_constructing_without_key_or_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIProvider()


def test_constructing_with_env_var_uses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    provider = OpenAIProvider()
    assert provider._api_key == "from-env"

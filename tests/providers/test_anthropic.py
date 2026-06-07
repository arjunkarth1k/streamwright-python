"""Tests for AnthropicProvider using httpx.MockTransport."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from streamwright.providers.anthropic import AnthropicProvider
from streamwright.providers.base import (
    Done,
    Message,
    TextDelta,
    Tool,
    ToolCallDelta,
    UsageEvent,
)
from streamwright.providers.errors import UnknownModelError

Handler = Callable[[httpx.Request], httpx.Response]


def make_provider(handler: Handler) -> AnthropicProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(
        base_url=AnthropicProvider.BASE_URL, transport=transport
    )
    return AnthropicProvider(api_key="test-key", client=client)


def _complete_response_json(text: str = "Hello") -> dict[str, Any]:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-haiku-4-5-20251001",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 11, "output_tokens": 7},
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
        model="claude-haiku-4-5",
        messages=[
            Message(role="system", content="be brief"),
            Message(role="user", content="hi"),
        ],
        max_tokens=100,
        temperature=0.5,
    )

    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["method"] == "POST"
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"

    body = captured["body"]
    assert body["model"] == "claude-haiku-4-5"
    assert body["max_tokens"] == 100
    assert body["temperature"] == 0.5
    assert body["system"] == "be brief"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    # Claude doesn't have a separate reasoning_tokens concept here; the
    # field stays at its default 0 regardless of the response shape.
    assert result.usage.reasoning_tokens == 0
    assert "stream" not in body

    assert result.text == "Hello"
    assert result.usage.tokens_in == 11
    assert result.usage.tokens_out == 7
    assert result.model == "claude-haiku-4-5-20251001"
    assert result.raw["id"] == "msg_01"


async def test_complete_defaults_max_tokens_when_unset() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_complete_response_json())

    provider = make_provider(handler)
    await provider.complete(
        model="claude-haiku-4-5",
        messages=[Message(role="user", content="hi")],
    )
    assert captured["body"]["max_tokens"] == AnthropicProvider.DEFAULT_MAX_TOKENS


async def test_complete_passes_tools_in_request_body() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_complete_response_json())

    provider = make_provider(handler)
    await provider.complete(
        model="claude-haiku-4-5",
        messages=[Message(role="user", content="hi")],
        tools=[
            Tool(
                name="get_weather",
                description="Get the weather for a city",
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
            "name": "get_weather",
            "description": "Get the weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


async def test_complete_unknown_model_raises_unknown_model_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("handler should not be invoked")

    provider = make_provider(handler)
    with pytest.raises(UnknownModelError, match="claude-nope"):
        await provider.complete(
            model="claude-nope",
            messages=[Message(role="user", content="hi")],
        )


# --- cache handling -------------------------------------------------------


async def test_cache_true_sets_cache_control_on_last_system_block() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_complete_response_json())

    provider = make_provider(handler)
    await provider.complete(
        model="claude-haiku-4-5",
        messages=[
            Message(role="system", content="be brief"),
            Message(role="user", content="hi"),
        ],
        cache=True,
    )
    assert captured["body"]["system"] == [
        {"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}
    ]


async def test_multiple_system_messages_render_as_list_with_cache_on_last() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_complete_response_json())

    provider = make_provider(handler)
    await provider.complete(
        model="claude-haiku-4-5",
        messages=[
            Message(role="system", content="first"),
            Message(role="system", content="second"),
            Message(role="user", content="hi"),
        ],
        cache=True,
    )
    assert captured["body"]["system"] == [
        {"type": "text", "text": "first"},
        {
            "type": "text",
            "text": "second",
            "cache_control": {"type": "ephemeral"},
        },
    ]


async def test_cache_true_with_no_system_logs_warning_and_omits_system(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """cache=True without a system message is a no-op with a logged warning."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_complete_response_json())

    provider = make_provider(handler)
    with caplog.at_level("WARNING", logger="streamwright.providers.anthropic"):
        await provider.complete(
            model="claude-haiku-4-5",
            messages=[Message(role="user", content="hi")],
            cache=True,
        )
    assert "system" not in captured["body"]
    assert any(
        "cache=True with no system message" in record.message
        for record in caplog.records
    )


async def test_cache_false_with_single_system_uses_string_form() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_complete_response_json())

    provider = make_provider(handler)
    await provider.complete(
        model="claude-haiku-4-5",
        messages=[
            Message(role="system", content="be brief"),
            Message(role="user", content="hi"),
        ],
    )
    assert captured["body"]["system"] == "be brief"


# --- stream() -------------------------------------------------------------


_STREAM_SSE = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":'
    b'{"usage":{"input_tokens":10,"output_tokens":0}}}\n'
    b"\n"
    b"event: content_block_start\n"
    b'data: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"text","text":""}}\n'
    b"\n"
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"Hello"}}\n'
    b"\n"
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":" world"}}\n'
    b"\n"
    b"event: content_block_stop\n"
    b'data: {"type":"content_block_stop","index":0}\n'
    b"\n"
    b"event: message_delta\n"
    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    b'"usage":{"output_tokens":5}}\n'
    b"\n"
    b"event: message_stop\n"
    b'data: {"type":"message_stop"}\n'
    b"\n"
)


async def test_stream_sends_stream_true_in_body() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=_STREAM_SSE, headers={"content-type": "text/event-stream"}
        )

    provider = make_provider(handler)
    async for _ in provider.stream(
        model="claude-haiku-4-5",
        messages=[Message(role="user", content="hi")],
    ):
        pass
    assert captured["body"]["stream"] is True


async def test_stream_yields_text_deltas_usage_and_done() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_STREAM_SSE, headers={"content-type": "text/event-stream"}
        )

    provider = make_provider(handler)
    events = [
        ev
        async for ev in provider.stream(
            model="claude-haiku-4-5",
            messages=[Message(role="user", content="hi")],
        )
    ]

    texts = [e.text for e in events if isinstance(e, TextDelta)]
    assert texts == ["Hello", " world"]

    usage_events = [e for e in events if isinstance(e, UsageEvent)]
    assert usage_events == [UsageEvent(tokens_in=10, tokens_out=5)]

    done_events = [e for e in events if isinstance(e, Done)]
    assert done_events == [Done(finish_reason="end_turn")]


_TOOL_STREAM_SSE = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":'
    b'{"usage":{"input_tokens":10,"output_tokens":0}}}\n'
    b"\n"
    b"event: content_block_start\n"
    b'data: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"tool_use","id":"toolu_1",'
    b'"name":"get_weather","input":{}}}\n'
    b"\n"
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"input_json_delta",'
    b'"partial_json":"{\\"city\\":"}}\n'
    b"\n"
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"input_json_delta",'
    b'"partial_json":"\\"NYC\\"}"}}\n'
    b"\n"
    b"event: content_block_stop\n"
    b'data: {"type":"content_block_stop","index":0}\n'
    b"\n"
    b"event: message_delta\n"
    b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
    b'"usage":{"output_tokens":12}}\n'
    b"\n"
)


async def test_stream_yields_tool_call_deltas_with_id_and_name() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_TOOL_STREAM_SSE,
            headers={"content-type": "text/event-stream"},
        )

    provider = make_provider(handler)
    events = [
        ev
        async for ev in provider.stream(
            model="claude-haiku-4-5",
            messages=[Message(role="user", content="weather in NYC?")],
        )
    ]
    tool_deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    assert len(tool_deltas) == 2
    assert tool_deltas[0].id == "toolu_1"
    assert tool_deltas[0].name == "get_weather"
    assert tool_deltas[0].partial_input == '{"city":'
    assert tool_deltas[1].id == "toolu_1"
    assert tool_deltas[1].name == "get_weather"
    assert tool_deltas[1].partial_input == '"NYC"}'


async def test_stream_unknown_model_raises_unknown_model_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("handler should not be invoked")

    provider = make_provider(handler)
    with pytest.raises(UnknownModelError, match="claude-nope"):
        async for _ in provider.stream(
            model="claude-nope",
            messages=[Message(role="user", content="hi")],
        ):
            pass


# --- __init__ behavior ----------------------------------------------------


def test_constructing_without_key_or_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()


def test_constructing_with_env_var_uses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    provider = AnthropicProvider()
    assert provider._api_key == "from-env"

"""Smoke tests for BaseProvider.stream_json_array."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from streamwright.providers.base import (
    BaseProvider,
    Message,
    StreamEvent,
    TextDelta,
    Tool,
)


class _ScriptedProvider(BaseProvider):
    """Test fixture: yields a hardcoded list of text chunks via stream()."""

    name = "scripted"

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.captured_messages: list[Message] | None = None
        self.captured_kwargs: dict[str, Any] = {}

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        cache: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        self.captured_messages = list(messages)
        self.captured_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": tools,
            "cache": cache,
        }
        for chunk in self._chunks:
            yield TextDelta(text=chunk)


async def test_stream_json_array_yields_objects_as_they_parse() -> None:
    provider = _ScriptedProvider(['[{"a": 1', '}, {"b"', ': 2}]'])
    out: list[dict[str, Any]] = []
    async for obj in provider.stream_json_array(model="x", messages=[]):
        out.append(obj)
    assert out == [{"a": 1}, {"b": 2}]


async def test_stream_json_array_prepends_system_when_none_exists() -> None:
    provider = _ScriptedProvider(["[]"])
    async for _ in provider.stream_json_array(
        model="x",
        messages=[Message(role="user", content="hi")],
    ):
        pass
    assert provider.captured_messages is not None
    assert len(provider.captured_messages) == 2
    assert provider.captured_messages[0].role == "system"
    assert "JSON array" in provider.captured_messages[0].content
    assert provider.captured_messages[1].role == "user"


async def test_stream_json_array_appends_to_existing_system_message() -> None:
    provider = _ScriptedProvider(["[]"])
    async for _ in provider.stream_json_array(
        model="x",
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="hi"),
        ],
    ):
        pass
    assert provider.captured_messages is not None
    assert len(provider.captured_messages) == 2
    system_content = provider.captured_messages[0].content
    assert system_content.startswith("You are a helpful assistant.")
    assert "JSON array" in system_content


async def test_stream_json_array_includes_schema_in_instruction() -> None:
    provider = _ScriptedProvider(["[]"])
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    async for _ in provider.stream_json_array(
        model="x", messages=[], schema=schema
    ):
        pass
    assert provider.captured_messages is not None
    system_content = provider.captured_messages[0].content
    assert '"type": "integer"' in system_content


async def test_stream_json_array_forwards_kwargs_to_stream() -> None:
    provider = _ScriptedProvider(["[]"])
    async for _ in provider.stream_json_array(
        model="x",
        messages=[],
        max_tokens=100,
        temperature=0.5,
    ):
        pass
    assert provider.captured_kwargs["max_tokens"] == 100
    assert provider.captured_kwargs["temperature"] == 0.5


async def test_stream_json_array_ignores_non_text_events() -> None:
    """UsageEvent / ToolCallDelta / Done events are not part of the JSON stream."""
    from streamwright.providers.base import Done, UsageEvent

    class MixedProvider(BaseProvider):
        name = "mixed"

        async def stream(
            self,
            *,
            model: str,
            messages: list[Message],
            max_tokens: int | None = None,
            temperature: float | None = None,
            tools: list[Tool] | None = None,
            cache: bool = False,
        ) -> AsyncIterator[StreamEvent]:
            yield TextDelta(text='[{"a": 1}')
            yield UsageEvent(tokens_in=10, tokens_out=20)
            yield TextDelta(text=", {\"b\": 2}]")
            yield Done(finish_reason="end_turn")

    provider = MixedProvider()
    out: list[dict[str, Any]] = []
    async for obj in provider.stream_json_array(model="x", messages=[]):
        out.append(obj)
    assert out == [{"a": 1}, {"b": 2}]

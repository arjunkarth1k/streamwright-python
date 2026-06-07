"""Tests for the shared SSE line parser."""

from __future__ import annotations

from collections.abc import AsyncIterator

from streamwright.providers._sse import SSEEvent, iter_sse_events


async def _aiter_lines(lines: list[str]) -> AsyncIterator[str]:
    for line in lines:
        yield line


async def _collect(lines: list[str]) -> list[SSEEvent]:
    return [event async for event in iter_sse_events(_aiter_lines(lines))]


async def test_data_only_stream_openai_style() -> None:
    events = await _collect(
        [
            'data: {"choices":[{"delta":{"content":"hi"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":" there"}}]}',
            "",
        ]
    )
    assert events == [
        SSEEvent(event="", data='{"choices":[{"delta":{"content":"hi"}}]}'),
        SSEEvent(event="", data='{"choices":[{"delta":{"content":" there"}}]}'),
    ]


async def test_event_and_data_stream_anthropic_style() -> None:
    events = await _collect(
        [
            "event: message_start",
            'data: {"type":"message_start"}',
            "",
            "event: content_block_delta",
            'data: {"type":"text_delta","text":"hello"}',
            "",
        ]
    )
    assert events == [
        SSEEvent(event="message_start", data='{"type":"message_start"}'),
        SSEEvent(
            event="content_block_delta",
            data='{"type":"text_delta","text":"hello"}',
        ),
    ]


async def test_comment_lines_are_skipped() -> None:
    events = await _collect(
        [
            ": this is a comment",
            'data: {"x":1}',
            "",
        ]
    )
    assert events == [SSEEvent(event="", data='{"x":1}')]


async def test_multiple_data_lines_concatenated_with_newline() -> None:
    events = await _collect(
        [
            "data: line one",
            "data: line two",
            "",
        ]
    )
    assert events == [SSEEvent(event="", data="line one\nline two")]


async def test_done_sentinel_is_yielded_verbatim() -> None:
    """OpenAI's `data: [DONE]` terminator surfaces as a normal event for the adapter to detect."""
    events = await _collect(
        [
            'data: {"x":1}',
            "",
            "data: [DONE]",
            "",
        ]
    )
    assert events == [
        SSEEvent(event="", data='{"x":1}'),
        SSEEvent(event="", data="[DONE]"),
    ]


async def test_trailing_event_without_blank_line_is_flushed() -> None:
    events = await _collect(
        [
            'data: {"x":1}',
        ]
    )
    assert events == [SSEEvent(event="", data='{"x":1}')]


async def test_crlf_line_endings_are_handled() -> None:
    # httpx normally strips terminators, but some servers leave \r.
    events = await _collect(
        [
            "data: hello\r",
            "",
        ]
    )
    assert events == [SSEEvent(event="", data="hello")]

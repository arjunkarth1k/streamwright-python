"""Server-Sent Events line parser shared by HTTP-streaming adapters.

Handles both Anthropic-style framing (``event:`` + ``data:`` lines) and
OpenAI-style framing (``data:`` only). Consumes pre-decoded lines from
``httpx.Response.aiter_lines()``; the caller is responsible for splitting
bytes into lines (httpx does this incrementally and UTF-8 safely).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SSEEvent:
    """A single Server-Sent Event.

    ``event`` is the ``event:`` field value, or the empty string if the
    stream did not specify one (typical for OpenAI-compatible streams).
    ``data`` is the concatenation of all ``data:`` lines in the event,
    joined by ``\\n``.
    """

    event: str
    data: str


async def iter_sse_events(lines: AsyncIterator[str]) -> AsyncIterator[SSEEvent]:
    """Parse SSE-framed lines into discrete events.

    Lines are expected without trailing newline characters (as produced by
    ``httpx.Response.aiter_lines()``). A blank line dispatches the
    accumulated event. Comment lines (starting with ``:``) and unknown
    fields are silently ignored.
    """
    event_type = ""
    data_parts: list[str] = []

    async for raw_line in lines:
        # httpx.aiter_lines may include trailing \r; strip defensively.
        line = raw_line.rstrip("\r")

        if line == "":
            if data_parts:
                yield SSEEvent(event=event_type, data="\n".join(data_parts))
                event_type = ""
                data_parts = []
            continue

        if line.startswith(":"):
            # Comment per SSE spec; skip.
            continue

        if line.startswith("event:"):
            event_type = line[len("event:") :].lstrip()
        elif line.startswith("data:"):
            value = line[len("data:") :]
            # SSE strips exactly one optional leading space after the colon.
            if value.startswith(" "):
                value = value[1:]
            data_parts.append(value)
        # Other fields (id:, retry:) intentionally ignored.

    if data_parts:
        yield SSEEvent(event=event_type, data="\n".join(data_parts))

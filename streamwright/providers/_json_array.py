"""Streaming parser for a JSON array of top-level objects.

The parser is fed text chunks of arbitrary size and yields each complete
top-level JSON object as soon as its closing ``}`` arrives. It is designed
for consuming model-emitted JSON arrays where you want objects available
as soon as they are produced, without waiting for the closing ``]``.

Out of scope: top-level non-object array elements. Strings, numbers, or
arrays as direct array elements are silently skipped — the contract is an
array of objects.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any


class JsonArrayBuffer:
    """Stateful streaming parser for a JSON array of objects.

    Feed text via :py:meth:`feed`, which yields each complete top-level
    object as soon as it closes. State persists across calls, so chunks may
    split anywhere — including mid-string and mid-escape sequence.

    The parser respects JSON string literals: braces inside ``"..."`` and
    escaped quotes (``\\"``) do not affect brace depth. Whitespace and
    commas between objects are skipped. Characters before the opening
    ``[`` (eg model preamble) and after the closing ``]`` (trailing prose)
    are silently discarded.
    """

    def __init__(self) -> None:
        self._buf: list[str] = []
        self._depth: int = 0
        self._in_string: bool = False
        self._escape: bool = False
        self._started: bool = False
        self._closed: bool = False

    @property
    def closed(self) -> bool:
        """``True`` once the closing ``]`` has been consumed."""
        return self._closed

    def feed(self, chunk: str) -> Iterator[dict[str, Any]]:
        """Feed a chunk of text and yield complete JSON objects as they close."""
        for ch in chunk:
            obj = self._consume(ch)
            if obj is not None:
                yield obj

    def _consume(self, ch: str) -> dict[str, Any] | None:
        if not self._started:
            if ch == "[":
                self._started = True
            return None

        if self._closed:
            return None

        if self._depth == 0:
            # Either between elements, before the first element, or
            # inside a top-level string element we're discarding.
            if self._in_string:
                if self._escape:
                    self._escape = False
                elif ch == "\\":
                    self._escape = True
                elif ch == '"':
                    self._in_string = False
                return None
            if ch == "]":
                self._closed = True
            elif ch == "{":
                self._depth = 1
                self._buf = ["{"]
            elif ch == '"':
                self._in_string = True
            # Otherwise: whitespace, commas, numbers, booleans, null — skipped.
            return None

        # depth >= 1: inside an object. Every char is part of the object's text.
        self._buf.append(ch)

        if self._escape:
            self._escape = False
            return None

        if self._in_string:
            if ch == "\\":
                self._escape = True
            elif ch == '"':
                self._in_string = False
            return None

        if ch == '"':
            self._in_string = True
        elif ch == "{":
            self._depth += 1
        elif ch == "}":
            self._depth -= 1
            if self._depth == 0:
                text = "".join(self._buf)
                self._buf = []
                parsed: dict[str, Any] = json.loads(text)
                return parsed
        return None

"""Tests for the streaming JSON-array object parser."""

from __future__ import annotations

from typing import Any

from streamwright.providers._json_array import JsonArrayBuffer


def consume(buf: JsonArrayBuffer, *chunks: str) -> list[dict[str, Any]]:
    """Feed chunks one at a time and collect all yielded objects."""
    objects: list[dict[str, Any]] = []
    for chunk in chunks:
        objects.extend(buf.feed(chunk))
    return objects


# --- Happy paths ---------------------------------------------------------


def test_single_object() -> None:
    assert consume(JsonArrayBuffer(), '[{"a": 1}]') == [{"a": 1}]


def test_multiple_objects() -> None:
    out = consume(JsonArrayBuffer(), '[{"a": 1}, {"b": 2}, {"c": 3}]')
    assert out == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_empty_array() -> None:
    assert consume(JsonArrayBuffer(), "[]") == []


def test_whitespace_between_objects() -> None:
    out = consume(JsonArrayBuffer(), '[\n  {"a": 1},\n  {"b": 2}\n]')
    assert out == [{"a": 1}, {"b": 2}]


def test_whitespace_inside_objects() -> None:
    out = consume(JsonArrayBuffer(), '[{\n  "a": 1,\n  "b": 2\n}]')
    assert out == [{"a": 1, "b": 2}]


# --- String-handling edge cases ------------------------------------------


def test_strings_containing_braces_and_brackets() -> None:
    out = consume(JsonArrayBuffer(), '[{"text": "a{b}c[d]e"}]')
    assert out == [{"text": "a{b}c[d]e"}]


def test_strings_containing_escaped_quotes() -> None:
    out = consume(JsonArrayBuffer(), r'[{"q": "she said \"hi\""}]')
    assert out == [{"q": 'she said "hi"'}]


def test_strings_with_escaped_backslash() -> None:
    # JSON "a\\b" represents the string a\b
    out = consume(JsonArrayBuffer(), r'[{"path": "a\\b"}]')
    assert out == [{"path": "a\\b"}]


def test_strings_with_escape_n() -> None:
    out = consume(JsonArrayBuffer(), r'[{"msg": "line1\nline2"}]')
    assert out == [{"msg": "line1\nline2"}]


def test_string_with_unicode_escape() -> None:
    out = consume(JsonArrayBuffer(), r'[{"x": "café"}]')
    assert out == [{"x": "café"}]


# --- Nesting -------------------------------------------------------------


def test_nested_objects() -> None:
    out = consume(JsonArrayBuffer(), '[{"a": {"b": {"c": 1}}}]')
    assert out == [{"a": {"b": {"c": 1}}}]


def test_nested_arrays_in_object() -> None:
    out = consume(JsonArrayBuffer(), '[{"items": [1, 2, [3, 4]]}]')
    assert out == [{"items": [1, 2, [3, 4]]}]


def test_nested_object_in_array_in_object() -> None:
    out = consume(JsonArrayBuffer(), '[{"xs": [{"a": 1}, {"a": 2}]}]')
    assert out == [{"xs": [{"a": 1}, {"a": 2}]}]


# --- Chunk-splitting -----------------------------------------------------


def test_chunk_split_mid_object() -> None:
    out = consume(JsonArrayBuffer(), '[{"a": 1', ', "b": 2}]')
    assert out == [{"a": 1, "b": 2}]


def test_chunk_split_mid_string() -> None:
    out = consume(JsonArrayBuffer(), '[{"name": "ab', 'cdef"}]')
    assert out == [{"name": "abcdef"}]


def test_chunk_split_right_after_backslash_inside_string() -> None:
    # The first chunk ends with a backslash inside a string; the next char
    # in the following chunk should be consumed as the escape target.
    out = consume(JsonArrayBuffer(), '[{"q": "a\\', '"b"}]')
    assert out == [{"q": 'a"b'}]


def test_chunk_split_between_objects() -> None:
    out = consume(JsonArrayBuffer(), '[{"a": 1}', ', {"b": 2}]')
    assert out == [{"a": 1}, {"b": 2}]


def test_chunk_split_immediately_after_open_bracket() -> None:
    out = consume(JsonArrayBuffer(), "[", '{"a": 1}]')
    assert out == [{"a": 1}]


def test_chunk_split_immediately_before_close_bracket() -> None:
    out = consume(JsonArrayBuffer(), '[{"a": 1}', "]")
    assert out == [{"a": 1}]


def test_chunk_split_per_character() -> None:
    """Stress test: chunk every single character."""
    payload = '[{"a": 1, "b": [2, 3]}, {"c": "x{y}z"}, {"d": "she said \\"hi\\""}]'
    buf = JsonArrayBuffer()
    out: list[dict[str, Any]] = []
    for ch in payload:
        out.extend(buf.feed(ch))
    assert out == [
        {"a": 1, "b": [2, 3]},
        {"c": "x{y}z"},
        {"d": 'she said "hi"'},
    ]


# --- Surrounding noise ---------------------------------------------------


def test_trailing_data_after_close_is_ignored() -> None:
    out = consume(JsonArrayBuffer(), '[{"a": 1}] some trailing prose')
    assert out == [{"a": 1}]


def test_leading_prose_before_array_is_ignored() -> None:
    out = consume(
        JsonArrayBuffer(),
        'Here is the array you asked for:\n[{"a": 1}]',
    )
    assert out == [{"a": 1}]


def test_top_level_string_element_is_skipped() -> None:
    # Contract is array of objects; non-object elements are silently skipped.
    out = consume(JsonArrayBuffer(), '["plain string", {"a": 1}]')
    assert out == [{"a": 1}]


def test_top_level_string_element_with_embedded_braces_is_skipped() -> None:
    # The braces inside the top-level string must not be mistaken for an
    # object opening.
    out = consume(JsonArrayBuffer(), '["a{b}c", {"a": 1}]')
    assert out == [{"a": 1}]


# --- closed property -----------------------------------------------------


def test_buffer_reports_closed_after_terminator() -> None:
    buf = JsonArrayBuffer()
    list(buf.feed('[{"a": 1}]'))
    assert buf.closed is True


def test_buffer_not_closed_mid_stream() -> None:
    buf = JsonArrayBuffer()
    list(buf.feed('[{"a": 1}'))  # no closing ]
    assert buf.closed is False


def test_buffer_not_closed_before_open_bracket() -> None:
    buf = JsonArrayBuffer()
    list(buf.feed("prose with no array"))
    assert buf.closed is False

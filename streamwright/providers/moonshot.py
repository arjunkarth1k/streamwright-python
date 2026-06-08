"""Moonshot provider adapter (OpenAI-compatible).

Inherits the request/response shape from ``OpenAIProvider`` and only
overrides the base URL, env-var name, capabilities table, ``cache``
handling, and the reasoning-content surfacing for Kimi K2.x.

Moonshot's context cache requires a separate ``POST /v1/context_cache``
call to mint a cache id which can then be referenced on subsequent
completion calls. That lifecycle is not yet implemented in this adapter;
passing ``cache=True`` raises ``NotImplementedError`` to surface the
missing feature explicitly.

**Reasoning content (Kimi K2.x)**. Kimi emits its chain-of-thought on a
separate ``reasoning_content`` field (in both streaming deltas and the
non-streaming message) rather than mixing it into ``content``. This is
undocumented in Kimi's public chat-completion reference but observable
in live responses. The adapter routes ``reasoning_content`` to
:py:class:`ReasoningDelta` events (streaming) and
:py:attr:`CompletionResult.reasoning_text` (non-streaming), leaving
visible output on :py:class:`TextDelta` / :py:attr:`CompletionResult.text`.
Kimi does NOT surface a reasoning/visible token-count split in the
``usage`` payload today, so :py:attr:`Usage.reasoning_tokens` stays 0
on Moonshot — see ``docs/ROADMAP.md`` for the upstream feature request.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .base import (
    CompletionResult,
    Message,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    Tool,
    Usage,
)
from .capabilities import MOONSHOT_CAPABILITIES, ModelCapabilities
from .openai import OpenAIProvider


class MoonshotProvider(OpenAIProvider):
    """Provider for Moonshot's OpenAI-compatible Chat Completions API.

    Endpoint: ``POST https://api.moonshot.ai/v1/chat/completions``.
    """

    name: ClassVar[str] = "moonshot"
    BASE_URL: ClassVar[str] = "https://api.moonshot.ai"
    ENV_KEY: ClassVar[str] = "MOONSHOT_API_KEY"
    CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = MOONSHOT_CAPABILITIES

    def _build_body(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None,
        temperature: float | None,
        tools: list[Tool] | None,
        cache: bool,
        stream: bool,
    ) -> dict[str, Any]:
        if cache:
            # TODO: implement Moonshot's two-call context-cache lifecycle:
            #   1. POST /v1/context_cache with the cached prompt segments,
            #      receive a `cache_id`.
            #   2. Send the chat-completion request with the `cache_id`
            #      referenced in the appropriate field (`prompt_cache_id`
            #      or similar — confirm against current Moonshot docs).
            # Until that lands, surface the missing functionality loudly.
            raise NotImplementedError(
                "moonshot: explicit context-cache API not yet wired; "
                "see TODO in moonshot.py. Call without cache=True until "
                "the two-call cache lifecycle is implemented."
            )
        return super()._build_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            cache=False,
            stream=stream,
        )

    @classmethod
    def _events_from_delta(
        cls, delta: dict[str, Any], tool_state: dict[int, dict[str, str]]
    ) -> list[StreamEvent]:
        """Route Kimi's ``reasoning_content`` to ReasoningDelta.

        Both ``reasoning_content`` and ``content`` can appear in the same
        delta in principle; in practice K2.x emits reasoning first then
        visible text. Emit reasoning before visible so callers iterating
        in arrival order see the natural thinking→answering sequence.
        Tool-call handling stays identical to the OpenAI base.
        """
        out: list[StreamEvent] = []
        reasoning_chunk = delta.get("reasoning_content")
        if reasoning_chunk:
            out.append(ReasoningDelta(text=str(reasoning_chunk)))
        content = delta.get("content")
        if content:
            out.append(TextDelta(text=str(content)))
        for emitted in OpenAIProvider._emit_tool_call_deltas(
            delta.get("tool_calls") or [], tool_state
        ):
            out.append(emitted)
        return out

    @staticmethod
    def _parse_completion(data: dict[str, Any], requested_model: str) -> CompletionResult:
        """Surface Kimi's ``message.reasoning_content`` alongside ``content``."""
        text = ""
        reasoning_text = ""
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            text = str(message.get("content") or "")
            reasoning_text = str(message.get("reasoning_content") or "")
        usage = data.get("usage") or {}
        tokens_in, tokens_out, reasoning_tokens = OpenAIProvider._split_usage(usage)
        return CompletionResult(
            text=text,
            usage=Usage(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                reasoning_tokens=reasoning_tokens,
            ),
            model=str(data.get("model", requested_model)),
            raw=data,
            reasoning_text=reasoning_text,
        )

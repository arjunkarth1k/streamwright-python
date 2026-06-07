"""OpenAI Chat Completions provider adapter."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx

from ._sse import iter_sse_events
from .base import (
    BaseProvider,
    CompletionResult,
    Done,
    Message,
    StreamEvent,
    TextDelta,
    Tool,
    ToolCallDelta,
    Usage,
    UsageEvent,
)
from .capabilities import OPENAI_CAPABILITIES, ModelCapabilities
from .errors import CapabilityError, UnknownModelError

logger = logging.getLogger(__name__)

# Per-(provider, model, param) dedup so the "param X dropped on model Y"
# warning fires at most once per process. Tests reset this set via an
# autouse fixture in test_openai.py.
_FORBIDDEN_WARNED: set[tuple[str, str, str]] = set()


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI's Chat Completions API.

    Endpoint: ``POST https://api.openai.com/v1/chat/completions``.
    OpenAI prompt caching is automatic on prefixes regardless of any
    explicit key; ``cache=True`` derives a deterministic
    ``prompt_cache_key`` from a sha256 hash of the messages list so
    cache-hit telemetry groups identical prompts under a stable id.

    Subclasses (eg ``MoonshotProvider``) reuse this implementation by
    overriding the ``BASE_URL``, ``ENV_KEY``, and ``CAPABILITIES`` class
    attributes.

    Reasoning-family notes (o1, GPT-5, GPT-5.x):

    * ``role="system"`` continues to work unchanged. The latest o-series
      and GPT-5 reasoning models accept system messages and treat them
      server-side as developer messages (functionally identical) — see
      Microsoft Learn "Azure OpenAI reasoning models". Do not send both
      ``system`` and ``developer`` roles in the same request.
    * Sampling parameters (``temperature``, ``top_p``,
      ``presence_penalty``, ``frequency_penalty``) are rejected at the
      API level. ``_build_body`` drops them when the model's
      ``ModelCapabilities.forbidden_params`` declares them, logging a
      one-time warning per (provider, model, param).
    """

    name: ClassVar[str] = "openai"
    BASE_URL: ClassVar[str] = "https://api.openai.com"
    CHAT_PATH: ClassVar[str] = "/v1/chat/completions"
    ENV_KEY: ClassVar[str] = "OPENAI_API_KEY"
    CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = OPENAI_CAPABILITIES

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        if api_key is None and client is None:
            api_key = os.environ.get(self.ENV_KEY)
            if not api_key:
                raise RuntimeError(
                    f"{self.ENV_KEY} environment variable is not set "
                    "and no explicit client was provided"
                )
        self._api_key = api_key
        self._client = (
            client
            if client is not None
            else httpx.AsyncClient(base_url=self.BASE_URL, timeout=timeout)
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the underlying client when it was created by this provider."""
        if self._owns_client:
            await self._client.aclose()

    # --- Request shaping --------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _validate_model(self, model: str) -> None:
        if model not in self.CAPABILITIES:
            raise UnknownModelError(f"Unknown {self.name} model {model!r}")

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
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if max_tokens is not None:
            # OpenAI's o1 / GPT-5 / GPT-5.x family rejects the legacy
            # `max_tokens` field with HTTP 400 unsupported_parameter and
            # requires `max_completion_tokens` instead — see the OpenAI
            # changelog when o1 launched. Moonshot's OpenAI-compatible API
            # likewise marks `max_tokens` as deprecated in favor of
            # `max_completion_tokens` (platform.kimi.ai/docs/api/chat),
            # so MoonshotProvider inherits the same wire field via super().
            body["max_completion_tokens"] = max_tokens
        if temperature is not None and not self._is_forbidden_with_warn(
            model, "temperature"
        ):
            body["temperature"] = temperature
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        if cache:
            # OpenAI caches automatically on prefix matches; prompt_cache_key
            # groups cache-hit telemetry for identical message lists. Derive
            # a deterministic short key from the messages so the same prompt
            # always reports under the same telemetry id.
            body["prompt_cache_key"] = self._derive_cache_key(body["messages"])
        if stream:
            body["stream"] = True
            # Moonshot's OpenAI-compatible API documents that
            # stream_options={"include_usage": True} is supported and emits
            # usage in the final SSE chunk (platform.kimi.ai/docs/api/chat,
            # "Use the Streaming Feature of the Kimi API"), so
            # MoonshotProvider inherits this branch unchanged.
            body["stream_options"] = {"include_usage": True}
        return body

    @staticmethod
    def _derive_cache_key(messages: list[dict[str, Any]]) -> str:
        blob = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _is_forbidden_with_warn(self, model: str, param: str) -> bool:
        """Return True if ``param`` must be dropped for ``model``, logging once.

        OpenAI's reasoning family (o1, GPT-5, GPT-5.x) rejects sampling
        parameters at the API level with HTTP 400. Rather than letting
        callers get a 400 from the wire, silently drop the param and log
        a one-time warning per (provider, model, param) so high-volume
        callers don't spam logs. Returns ``False`` for models that
        accept the param so the caller can set it on the body unchanged.
        """
        caps = self.CAPABILITIES.get(model)
        if caps is None or param not in caps.forbidden_params:
            return False
        key = (self.name, model, param)
        if key not in _FORBIDDEN_WARNED:
            logger.warning(
                "%s: dropping unsupported parameter %r for model %r — "
                "this is a reasoning-family model and rejects sampling "
                "parameters at the API level.",
                self.name,
                param,
                model,
            )
            _FORBIDDEN_WARNED.add(key)
        return True

    # --- complete() -------------------------------------------------------

    async def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        cache: bool = False,
    ) -> CompletionResult:
        self._validate_model(model)
        body = self._build_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            cache=cache,
            stream=False,
        )
        response = await self._client.post(
            self.CHAT_PATH, json=body, headers=self._headers()
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return self._parse_completion(data, model)

    @staticmethod
    def _parse_completion(data: dict[str, Any], requested_model: str) -> CompletionResult:
        text = ""
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            text = str(message.get("content") or "")
        usage = data.get("usage") or {}
        tokens_in, tokens_out, reasoning = OpenAIProvider._split_usage(usage)
        return CompletionResult(
            text=text,
            usage=Usage(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                reasoning_tokens=reasoning,
            ),
            model=str(data.get("model", requested_model)),
            raw=data,
        )

    @staticmethod
    def _split_usage(usage: dict[str, Any]) -> tuple[int, int, int]:
        """Return ``(tokens_in, visible_tokens_out, reasoning_tokens)``.

        OpenAI's response shape for reasoning models reports
        ``completion_tokens`` as the SUM of visible output + hidden
        reasoning tokens; the hidden count appears in
        ``completion_tokens_details.reasoning_tokens``. Non-reasoning
        models omit the details object, in which case reasoning_tokens
        defaults to 0 and visible == completion_tokens unchanged.
        Moonshot's OpenAI-compatible API does not currently surface the
        details object at all, so Kimi reasoning models flow through
        with reasoning_tokens=0 and the inflated visible count — see
        the streamwright integration docs.
        """
        completion = int(usage.get("completion_tokens", 0))
        details = usage.get("completion_tokens_details") or {}
        reasoning = int(details.get("reasoning_tokens", 0)) if isinstance(details, dict) else 0
        visible = max(0, completion - reasoning)
        return int(usage.get("prompt_tokens", 0)), visible, reasoning

    # --- stream() ---------------------------------------------------------

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
        self._validate_model(model)
        caps = self.CAPABILITIES[model]
        if not caps.streaming:
            raise CapabilityError(
                f"{self.name} model {model!r} does not support streaming; "
                "use complete() instead"
            )
        body = self._build_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            cache=cache,
            stream=True,
        )
        async with self._client.stream(
            "POST", self.CHAT_PATH, json=body, headers=self._headers()
        ) as response:
            response.raise_for_status()
            async for stream_event in self._parse_stream(response):
                yield stream_event

    @classmethod
    async def _parse_stream(cls, response: httpx.Response) -> AsyncIterator[StreamEvent]:
        # OpenAI streams tool calls as deltas keyed by `index`. The first
        # chunk for a given index carries the id and function name; later
        # chunks repeat the index and stream `arguments` text. Track the
        # canonical id/name per index so every emitted ToolCallDelta is
        # self-describing.
        tool_state: dict[int, dict[str, str]] = {}

        async for sse in iter_sse_events(response.aiter_lines()):
            if not sse.data:
                continue
            if sse.data == "[DONE]":
                break
            try:
                data: dict[str, Any] = json.loads(sse.data)
            except json.JSONDecodeError:
                continue

            for choice in data.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if isinstance(delta, dict):
                    # cls dispatch lets subclasses (MoonshotProvider)
                    # override _events_from_delta to surface provider-
                    # specific delta fields without re-implementing the
                    # surrounding SSE loop.
                    for emitted in cls._events_from_delta(delta, tool_state):
                        yield emitted
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    yield Done(finish_reason=str(finish_reason))

            usage = data.get("usage")
            if usage:
                tokens_in, tokens_out, reasoning = OpenAIProvider._split_usage(usage)
                yield UsageEvent(
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    reasoning_tokens=reasoning,
                )

    @classmethod
    def _events_from_delta(
        cls, delta: dict[str, Any], tool_state: dict[int, dict[str, str]]
    ) -> list[StreamEvent]:
        """Map a single SSE delta dict to zero or more StreamEvents.

        OpenAI's chat-completions delta shape carries ``content`` (visible
        text) and ``tool_calls`` (tool-use chunks). Subclasses override to
        surface provider-specific delta fields — for example,
        :py:meth:`MoonshotProvider._events_from_delta` additionally routes
        Kimi's ``reasoning_content`` field to :py:class:`ReasoningDelta`.
        """
        out: list[StreamEvent] = []
        content = delta.get("content")
        if content:
            out.append(TextDelta(text=str(content)))
        for emitted in OpenAIProvider._emit_tool_call_deltas(
            delta.get("tool_calls") or [], tool_state
        ):
            out.append(emitted)
        return out

    @staticmethod
    def _emit_tool_call_deltas(
        tool_calls: list[Any], tool_state: dict[int, dict[str, str]]
    ) -> list[ToolCallDelta]:
        out: list[ToolCallDelta] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            index = int(tc.get("index", 0))
            fn = tc.get("function") or {}
            state = tool_state.setdefault(index, {"id": "", "name": ""})
            if "id" in tc:
                state["id"] = str(tc.get("id") or "")
            if isinstance(fn, dict) and "name" in fn:
                state["name"] = str(fn.get("name") or "")
            partial = ""
            if isinstance(fn, dict) and "arguments" in fn:
                partial = str(fn.get("arguments") or "")
            out.append(
                ToolCallDelta(
                    id=state["id"],
                    name=state["name"],
                    partial_input=partial,
                )
            )
        return out

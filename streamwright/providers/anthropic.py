"""Anthropic Messages API provider adapter."""

from __future__ import annotations

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
from .capabilities import ANTHROPIC_CAPABILITIES, ModelCapabilities
from .errors import CapabilityError, UnknownModelError

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic's Messages API.

    Endpoint: ``POST https://api.anthropic.com/v1/messages``.
    System messages are extracted from the message list and sent via the
    top-level ``system`` field (Anthropic's API does not accept a system
    role inside ``messages``). When ``cache=True``, the last system block
    is marked with ``cache_control: ephemeral`` so Anthropic treats the
    prefix as cacheable. ``cache=True`` without any system message logs a
    warning and is otherwise a no-op (Anthropic's cache_control only
    attaches to system blocks today).
    """

    name: ClassVar[str] = "anthropic"
    BASE_URL: ClassVar[str] = "https://api.anthropic.com"
    MESSAGES_PATH: ClassVar[str] = "/v1/messages"
    API_VERSION: ClassVar[str] = "2023-06-01"
    DEFAULT_MAX_TOKENS: ClassVar[int] = 4096
    ENV_KEY: ClassVar[str] = "ANTHROPIC_API_KEY"
    CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = ANTHROPIC_CAPABILITIES

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
        headers = {"anthropic-version": self.API_VERSION}
        if self._api_key is not None:
            headers["x-api-key"] = self._api_key
        return headers

    def _validate_model(self, model: str) -> None:
        if model not in ANTHROPIC_CAPABILITIES:
            raise UnknownModelError(f"Unknown Anthropic model {model!r}")

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
        system_messages = [m for m in messages if m.role == "system"]
        non_system_messages = [m for m in messages if m.role != "system"]

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": (
                max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS
            ),
            "messages": [
                {"role": m.role, "content": m.content} for m in non_system_messages
            ],
        }
        if system_messages:
            body["system"] = self._build_system(system_messages, cache)
        elif cache:
            logger.warning(
                "anthropic: cache=True with no system message has no effect; "
                "Anthropic only sets cache_control on system blocks. "
                "Add a system message or call without cache=True."
            )
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]
        if stream:
            body["stream"] = True
        return body

    @staticmethod
    def _build_system(
        system_messages: list[Message], cache: bool
    ) -> str | list[dict[str, Any]]:
        if not cache and len(system_messages) == 1:
            return system_messages[0].content
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": m.content} for m in system_messages
        ]
        if cache:
            blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return blocks

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
            self.MESSAGES_PATH, json=body, headers=self._headers()
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return self._parse_completion(data, model)

    @staticmethod
    def _parse_completion(data: dict[str, Any], requested_model: str) -> CompletionResult:
        text_parts: list[str] = []
        for block in data.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        usage = data.get("usage") or {}
        return CompletionResult(
            text="".join(text_parts),
            usage=Usage(
                tokens_in=int(usage.get("input_tokens", 0)),
                tokens_out=int(usage.get("output_tokens", 0)),
            ),
            model=str(data.get("model", requested_model)),
            raw=data,
        )

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
        caps = ANTHROPIC_CAPABILITIES[model]
        if not caps.streaming:
            raise CapabilityError(
                f"Anthropic model {model!r} does not support streaming; "
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
            "POST", self.MESSAGES_PATH, json=body, headers=self._headers()
        ) as response:
            response.raise_for_status()
            async for stream_event in self._parse_stream(response):
                yield stream_event

    @staticmethod
    async def _parse_stream(response: httpx.Response) -> AsyncIterator[StreamEvent]:
        current_tool_id: str | None = None
        current_tool_name: str | None = None
        input_tokens: int = 0

        async for sse in iter_sse_events(response.aiter_lines()):
            if not sse.data:
                continue
            try:
                data: dict[str, Any] = json.loads(sse.data)
            except json.JSONDecodeError:
                continue

            event_type = sse.event

            if event_type == "message_start":
                usage = data.get("message", {}).get("usage", {})
                input_tokens = int(usage.get("input_tokens", 0))

            elif event_type == "content_block_start":
                block = data.get("content_block", {})
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    current_tool_id = str(block.get("id", ""))
                    current_tool_name = str(block.get("name", ""))

            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                dtype = delta.get("type") if isinstance(delta, dict) else None
                if dtype == "text_delta":
                    yield TextDelta(text=str(delta.get("text", "")))
                elif dtype == "input_json_delta":
                    yield ToolCallDelta(
                        id=current_tool_id or "",
                        name=current_tool_name or "",
                        partial_input=str(delta.get("partial_json", "")),
                    )

            elif event_type == "content_block_stop":
                current_tool_id = None
                current_tool_name = None

            elif event_type == "message_delta":
                usage = data.get("usage", {})
                output_tokens = int(usage.get("output_tokens", 0))
                yield UsageEvent(tokens_in=input_tokens, tokens_out=output_tokens)
                stop_reason = data.get("delta", {}).get("stop_reason")
                if stop_reason:
                    yield Done(finish_reason=str(stop_reason))
            # message_stop / ping: nothing to surface.

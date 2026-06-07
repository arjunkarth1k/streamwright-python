"""Provider abstraction: Protocol, base class, and supporting types."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

from ._json_array import JsonArrayBuffer
from .capabilities import ModelCapabilities

type Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """A single turn in a conversation passed to a provider."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Tool:
    """Tool definition exposed to the model for tool-use / function calling."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage reported by a provider.

    ``tokens_out`` is the **visible** assistant output count. For
    reasoning-family models (OpenAI o1 / GPT-5 family, Moonshot Kimi
    K2.x) the provider also burns hidden ``reasoning_tokens`` that are
    billed but never streamed back to the caller; those are surfaced
    separately so spend tracking and context budgeting are honest.
    Total billed tokens = ``tokens_in + tokens_out + reasoning_tokens``.
    Providers without a reasoning concept leave ``reasoning_tokens=0``.
    """

    tokens_in: int
    tokens_out: int
    reasoning_tokens: int = 0


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Result of a non-streaming completion call.

    ``raw`` preserves the verbatim provider response so callers can read
    provider-specific fields (eg system fingerprint) without us pre-deciding
    what to expose.

    ``reasoning_text`` carries hidden chain-of-thought from reasoning-model
    providers (Moonshot Kimi K2.x today; future Anthropic extended thinking
    and OpenAI Responses-API surfaces) that returned reasoning content
    alongside or instead of visible ``text``. ``text`` is the visible
    assistant reply; ``reasoning_text`` is the model's internal
    deliberation. Both can be populated in the same response. Token counts
    are split via :py:attr:`Usage.reasoning_tokens` when the provider
    surfaces the split (Moonshot does not today — see ROADMAP).
    """

    text: str
    usage: Usage
    model: str
    raw: dict[str, Any]
    reasoning_text: str = ""


@dataclass(frozen=True, slots=True)
class TextDelta:
    """Incremental assistant text emitted during ``stream()``."""

    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    """Incremental hidden chain-of-thought text emitted during ``stream()``.

    Emitted by reasoning-model providers (Moonshot Kimi K2.x today; future
    Anthropic extended-thinking and OpenAI Responses-API surfaces) for
    hidden chain-of-thought tokens. Distinct from :py:class:`TextDelta` to
    let callers display reasoning separately, suppress it entirely, or
    route it to a different UI surface. The tokens carried here are
    billed and — when the provider exposes the split — counted in
    :py:attr:`Usage.reasoning_tokens`.
    """

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """Incremental tool-call event emitted during ``stream()``.

    Tool calls arrive in chunks across multiple events on most providers
    (OpenAI especially). Callers must accumulate ``partial_input`` themselves;
    ``id`` and ``name`` may only be populated on the first chunk for a given
    tool call (downstream chunks repeat the same id with empty name).
    """

    id: str
    name: str
    partial_input: str


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """Token usage reported during or at the end of a stream.

    ``tokens_out`` is the visible-output count; ``reasoning_tokens`` is
    the hidden reasoning-token count for reasoning-family models. See
    :py:class:`Usage` for the rationale.
    """

    tokens_in: int
    tokens_out: int
    reasoning_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Done:
    """Terminal stream event carrying the provider's finish reason."""

    finish_reason: str


type StreamEvent = TextDelta | ReasoningDelta | ToolCallDelta | UsageEvent | Done


class Provider(Protocol):
    """Provider interface used by streamwright pipelines."""

    name: ClassVar[str]

    async def aclose(self) -> None: ...

    async def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        cache: bool = False,
    ) -> CompletionResult: ...

    def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        cache: bool = False,
    ) -> AsyncIterator[StreamEvent]: ...

    def stream_json_array(
        self,
        *,
        model: str,
        messages: list[Message],
        schema: dict[str, Any] | None = None,
        cache: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]: ...


class BaseProvider:
    """Base implementation. Subclasses override ``complete`` and ``stream``.

    The default ``stream_json_array`` streams text via ``self.stream()`` and
    yields complete JSON objects as they parse. Concrete subclasses set
    ``CAPABILITIES`` to their per-model capability table so the registry
    can validate models without hard-coding per-provider knowledge.
    """

    name: ClassVar[str] = ""
    CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    async def aclose(self) -> None:
        """Close any owned HTTP clients. Default no-op; subclasses override."""

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
        raise NotImplementedError

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
        raise NotImplementedError
        yield  # pragma: no cover - unreachable; marks this as an async generator

    async def stream_json_array(
        self,
        *,
        model: str,
        messages: list[Message],
        schema: dict[str, Any] | None = None,
        cache: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a JSON array of objects, yielding each as it parses.

        Injects an instruction asking the model to emit a JSON array of
        objects (optionally matching ``schema``). The instruction is
        appended to the first existing system message; if none is present
        a new system message is prepended. Tool-call and usage events
        from the underlying ``stream()`` are ignored — only ``TextDelta``
        contributes to parsing. ``cache`` is forwarded to the underlying
        ``stream()`` so the same prompt-caching intent applies.
        """
        instruction = (
            "Respond with a JSON array of objects and nothing else. "
            "Emit each object in the array as soon as it is complete so the "
            "response can be streamed."
        )
        if schema is not None:
            instruction += (
                "\nEach object in the array MUST match this JSON Schema:\n"
                f"{json.dumps(schema)}"
            )

        prepared = _inject_system_instruction(messages, instruction)
        buffer = JsonArrayBuffer()

        async for event in self.stream(
            model=model, messages=prepared, cache=cache, **kwargs
        ):
            if isinstance(event, TextDelta):
                for obj in buffer.feed(event.text):
                    yield obj


def _inject_system_instruction(
    messages: list[Message], instruction: str
) -> list[Message]:
    """Append ``instruction`` to the first system message, or prepend a new one."""
    prepared: list[Message] = []
    appended = False
    for msg in messages:
        if msg.role == "system" and not appended:
            prepared.append(
                Message(role="system", content=f"{msg.content}\n\n{instruction}")
            )
            appended = True
        else:
            prepared.append(msg)
    if not appended:
        prepared.insert(0, Message(role="system", content=instruction))
    return prepared

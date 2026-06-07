"""Shared types for declaring and observing streamwright pipelines."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class StepKind(Enum):
    """Kinds of executable steps supported by the pipeline DSL."""

    SINGLE = "single"
    STREAM = "stream"
    MERGE = "merge"


type MergeKeyFn = Callable[..., Hashable]
type StepFn = Callable[..., Any]


type MergeMode = Literal["strict", "lenient"]


@dataclass(slots=True)
class StepSpec:
    """Declarative specification for a pipeline step.

    ``merge_mode`` only applies when ``kind`` is :py:attr:`StepKind.MERGE`.
    ``"strict"`` (the default) emits :py:class:`StepFailed` at end-of-run
    if any merge key never received a value from every source.
    ``"lenient"`` logs a warning and silently drops the partial set.
    """

    name: str
    kind: StepKind
    fn: StepFn
    fan_out_from: str | None = None
    max_concurrency: int | None = None
    retries: int = 3
    merge_sources: list[str] = field(default_factory=list)
    merge_key: MergeKeyFn | None = None
    merge_mode: MergeMode = "strict"


@dataclass(frozen=True, slots=True)
class JobEvent:
    """Base class for all pipeline events."""


@dataclass(frozen=True, slots=True)
class StepStarted(JobEvent):
    """Event emitted when a step starts."""

    step: str


@dataclass(frozen=True, slots=True)
class StepStreaming(JobEvent):
    """Event emitted when a streaming step yields its first item."""

    step: str


@dataclass(frozen=True, slots=True)
class StepOutput(JobEvent):
    """Event emitted when a step produces a value."""

    step: str
    value: Any
    key: Hashable | None = None


@dataclass(frozen=True, slots=True)
class StepDone(JobEvent):
    """Event emitted when a step completes successfully."""

    step: str


@dataclass(frozen=True, slots=True)
class StepFailed(JobEvent):
    """Event emitted when a step fails."""

    step: str
    error: str
    traceback: str


@dataclass(frozen=True, slots=True)
class PipelineDone(JobEvent):
    """Event emitted when the full pipeline completes."""


@dataclass(frozen=True, slots=True)
class Telemetry(JobEvent):
    """Event emitted with provider usage and timing data."""

    step: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float


type JobEventVariant = (
    StepStarted | StepStreaming | StepOutput | StepDone | StepFailed | PipelineDone | Telemetry
)

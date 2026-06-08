"""Tests for the pipeline declaration DSL and end-to-end Pipeline.run."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from streamwright.pipeline import Pipeline
from streamwright.types import PipelineDone, StepKind, StepOutput


def test_register_single_step() -> None:
    """Regular async functions register as SINGLE steps."""
    pipeline = Pipeline("single")

    @pipeline.step()
    async def fetch(ctx: Any, _input: Any) -> str:
        return "value"

    assert pipeline.steps["fetch"].kind is StepKind.SINGLE


def test_register_stream_step() -> None:
    """Async generator functions register as STREAM steps."""
    pipeline = Pipeline("stream")

    @pipeline.step()
    async def stream(ctx: Any, _input: Any) -> AsyncIterator[str]:
        yield "value"

    assert pipeline.steps["stream"].kind is StepKind.STREAM


def test_register_merge_step() -> None:
    """Merge decorators track source names and merge keys."""
    pipeline = Pipeline("merge")

    @pipeline.merge("first", "second", key=lambda value: value)
    async def combine(ctx: Any, key: Any, *values: Any) -> AsyncIterator[Any]:
        yield values

    spec = pipeline.steps["combine"]
    assert spec.kind is StepKind.MERGE
    assert spec.merge_sources == ["first", "second"]
    assert spec.merge_key is not None


def test_dag_validates_fan_out_reference() -> None:
    """fan_out_from must reference a registered step."""
    pipeline = Pipeline("bad-fan-out")

    @pipeline.step(fan_out_from="missing")
    async def child(ctx: Any, _input: Any) -> str:
        return "value"

    with pytest.raises(ValueError, match="unknown fan_out_from"):
        pipeline.run("input")


def test_dag_detects_cycle() -> None:
    """Cycles are rejected during lazy DAG validation."""
    pipeline = Pipeline("cycle")

    @pipeline.step(fan_out_from="second")
    async def first(ctx: Any, _input: Any) -> str:
        return "first"

    @pipeline.step(fan_out_from="first")
    async def second(ctx: Any, _input: Any) -> str:
        return "second"

    with pytest.raises(ValueError, match="dependency cycle"):
        pipeline.run("input")


def test_dag_requires_entry_point() -> None:
    """A pipeline must have at least one root step that consumes the input."""
    pipeline = Pipeline("empty")

    with pytest.raises(ValueError, match="entry point"):
        pipeline.run("input")


async def test_run_executes_pipeline_end_to_end() -> None:
    """Pipeline.run yields a real JobEvent stream ending in PipelineDone."""
    pipeline = Pipeline("end-to-end")

    @pipeline.step()
    async def root(ctx: Any, value: Any) -> str:
        return f"processed:{value}"

    events = [event async for event in pipeline.run("hello")]
    outputs = [e for e in events if isinstance(e, StepOutput)]
    assert len(outputs) == 1
    assert outputs[0].value == "processed:hello"
    assert isinstance(events[-1], PipelineDone)

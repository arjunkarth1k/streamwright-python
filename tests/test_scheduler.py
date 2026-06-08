"""Tests for the DAG scheduler — retry helper, error classification, and execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from streamwright import Pipeline
from streamwright.providers.errors import (
    CapabilityError,
    ProviderError,
    UnknownModelError,
)
from streamwright.scheduler import Scheduler, is_retryable_error, with_retries
from streamwright.types import (
    JobEvent,
    PipelineDone,
    StepDone,
    StepFailed,
    StepOutput,
    StepStarted,
    StepStreaming,
)


def _scheduler_for(pipeline: Pipeline) -> Scheduler:
    return Scheduler(list(pipeline._build_dag()))


async def _collect(pipeline: Pipeline, input_value: Any) -> list[JobEvent]:
    scheduler = _scheduler_for(pipeline)
    return [ev async for ev in scheduler.execute(input_value)]


class _RetryableTestError(Exception):
    """Test-only retryable signal used by the with_retries tests."""


def _is_test_retryable(exc: BaseException) -> bool:
    return isinstance(exc, _RetryableTestError)


# --- with_retries ---------------------------------------------------------


async def test_with_retries_returns_on_first_success() -> None:
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await with_retries(
        fn, max_attempts=3, is_retryable=_is_test_retryable
    )
    assert result == "ok"
    assert calls == 1


async def test_with_retries_retries_then_succeeds(
    fake_clock: list[float],
) -> None:
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise _RetryableTestError()
        return "ok"

    result = await with_retries(
        fn,
        max_attempts=3,
        is_retryable=_is_test_retryable,
        max_jitter=0,
    )
    assert result == "ok"
    assert calls == 2
    assert fake_clock == [1.0]


async def test_with_retries_backoff_doubles_between_attempts(
    fake_clock: list[float],
) -> None:
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _RetryableTestError()
        return "ok"

    await with_retries(
        fn,
        max_attempts=3,
        is_retryable=_is_test_retryable,
        max_jitter=0,
    )
    # 2 retries → 2 sleeps: 1s then 2s with no jitter
    assert fake_clock == [1.0, 2.0]


async def test_with_retries_exhausted_raises_last_exception(
    fake_clock: list[float],
) -> None:
    async def fn() -> str:
        raise _RetryableTestError("retried 3x")

    with pytest.raises(_RetryableTestError, match="retried 3x"):
        await with_retries(
            fn,
            max_attempts=3,
            is_retryable=_is_test_retryable,
            max_jitter=0,
        )
    # 3 attempts → 2 sleeps between them
    assert fake_clock == [1.0, 2.0]


async def test_with_retries_fatal_raises_immediately(
    fake_clock: list[float],
) -> None:
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("fatal — must not retry")

    with pytest.raises(ValueError, match="fatal"):
        await with_retries(
            fn, max_attempts=3, is_retryable=_is_test_retryable
        )
    assert calls == 1
    assert fake_clock == []


# --- is_retryable_error ---------------------------------------------------


def test_is_retryable_asyncio_timeout() -> None:
    # asyncio.TimeoutError is an alias for the builtin TimeoutError on 3.11+.
    assert is_retryable_error(TimeoutError())


def test_is_retryable_httpx_timeout_subclasses() -> None:
    assert is_retryable_error(httpx.ConnectTimeout("x"))
    assert is_retryable_error(httpx.ReadTimeout("x"))


def test_is_retryable_http_500() -> None:
    request = httpx.Request("GET", "http://x")
    response = httpx.Response(500, request=request)
    err = httpx.HTTPStatusError("server", request=request, response=response)
    assert is_retryable_error(err)


def test_is_retryable_http_429() -> None:
    request = httpx.Request("GET", "http://x")
    response = httpx.Response(429, request=request)
    err = httpx.HTTPStatusError("rate", request=request, response=response)
    assert is_retryable_error(err)


def test_is_retryable_http_404_is_fatal() -> None:
    request = httpx.Request("GET", "http://x")
    response = httpx.Response(404, request=request)
    err = httpx.HTTPStatusError("nf", request=request, response=response)
    assert not is_retryable_error(err)


def test_is_retryable_provider_error_with_flag_true() -> None:
    err = ProviderError("transient")
    err.retryable = True
    assert is_retryable_error(err)


def test_is_retryable_provider_error_default_false() -> None:
    assert not is_retryable_error(ProviderError("x"))


def test_is_retryable_unknown_model_is_fatal() -> None:
    assert not is_retryable_error(UnknownModelError("x"))


def test_is_retryable_capability_is_fatal() -> None:
    assert not is_retryable_error(CapabilityError("x"))


def test_is_retryable_generic_exception_is_fatal() -> None:
    assert not is_retryable_error(ValueError("x"))


# --- Linear SINGLE pipeline execution -------------------------------------


async def test_scheduler_executes_linear_single_chain() -> None:
    """A→B→C of SINGLE steps yields per-step Started/Output/Done in order.

    SINGLE steps do NOT emit StepStreaming — that signal is reserved for
    STREAM steps where it marks "first of many".
    """
    pipeline = Pipeline("linear")

    @pipeline.step()
    async def a(ctx: Any, value: Any) -> str:
        return f"A({value})"

    @pipeline.step(fan_out_from="a")
    async def b(ctx: Any, value: Any) -> str:
        return f"B({value})"

    @pipeline.step(fan_out_from="b")
    async def c(ctx: Any, value: Any) -> str:
        return f"C({value})"

    events = await _collect(pipeline, "input")

    assert isinstance(events[-1], PipelineDone)
    # No StepStreaming events for SINGLE steps.
    assert not [e for e in events if isinstance(e, StepStreaming)]
    for step_name, expected in [
        ("a", "A(input)"),
        ("b", "B(A(input))"),
        ("c", "C(B(A(input)))"),
    ]:
        step_events = [e for e in events if getattr(e, "step", None) == step_name]
        kinds = [type(e).__name__ for e in step_events]
        assert kinds == ["StepStarted", "StepOutput", "StepDone"], (
            f"step {step_name} got {kinds}"
        )
        output = step_events[1]
        assert isinstance(output, StepOutput)
        assert output.value == expected
        assert output.key == 0


async def test_scheduler_emits_pipeline_done_last() -> None:
    pipeline = Pipeline("solo")

    @pipeline.step()
    async def root(ctx: Any, value: Any) -> str:
        return str(value)

    events = await _collect(pipeline, "hi")
    assert isinstance(events[-1], PipelineDone)
    # Exactly one PipelineDone, at the end.
    assert sum(1 for e in events if isinstance(e, PipelineDone)) == 1


async def test_scheduler_per_step_event_order() -> None:
    """Within a STREAM step, Started precedes Streaming precedes Output precedes Done."""
    pipeline = Pipeline("ordering")

    @pipeline.step()
    async def root(ctx: Any, value: Any) -> AsyncIterator[str]:
        yield f"processed:{value}"

    events = await _collect(pipeline, "hi")
    root_events = [e for e in events if getattr(e, "step", None) == "root"]
    started_idx = next(i for i, e in enumerate(root_events) if isinstance(e, StepStarted))
    streaming_idx = next(
        i for i, e in enumerate(root_events) if isinstance(e, StepStreaming)
    )
    output_idx = next(i for i, e in enumerate(root_events) if isinstance(e, StepOutput))
    done_idx = next(i for i, e in enumerate(root_events) if isinstance(e, StepDone))
    assert started_idx < streaming_idx < output_idx < done_idx


async def test_scheduler_entry_step_input_is_keyed_zero() -> None:
    pipeline = Pipeline("key-zero")

    @pipeline.step()
    async def root(ctx: Any, value: Any) -> Any:
        return value

    events = await _collect(pipeline, "anything")
    outputs = [e for e in events if isinstance(e, StepOutput)]
    assert len(outputs) == 1
    assert outputs[0].key == 0


async def test_scheduler_keys_propagate_through_chain() -> None:
    """Linear SINGLE chain preserves the entry key on every downstream output."""
    pipeline = Pipeline("key-propagation")

    @pipeline.step()
    async def a(ctx: Any, value: Any) -> Any:
        return value

    @pipeline.step(fan_out_from="a")
    async def b(ctx: Any, value: Any) -> Any:
        return value

    events = await _collect(pipeline, "x")
    for output in (e for e in events if isinstance(e, StepOutput)):
        assert output.key == 0


# --- STREAM sources + fan-out concurrency ---------------------------------


async def test_stream_source_yields_keys_zero_one_two() -> None:
    """A STREAM entry step assigns 0, 1, 2, ... to each yielded item."""
    pipeline = Pipeline("stream-keys")

    @pipeline.step()
    async def src(ctx: Any, value: Any) -> AsyncIterator[int]:
        for i in range(3):
            yield i
            await asyncio.sleep(0)

    events = await _collect(pipeline, "start")
    src_outputs = [
        e for e in events if isinstance(e, StepOutput) and e.step == "src"
    ]
    assert [(o.value, o.key) for o in src_outputs] == [(0, 0), (1, 1), (2, 2)]


async def test_downstream_consumes_during_upstream_streaming() -> None:
    """fan_out downstream begins processing before upstream finishes yielding."""
    pipeline = Pipeline("interleaved")
    timeline: list[str] = []

    @pipeline.step()
    async def source(ctx: Any, value: Any) -> AsyncIterator[int]:
        for i in range(3):
            timeline.append(f"source-yielded-{i}")
            yield i
            await asyncio.sleep(0.005)

    @pipeline.step(fan_out_from="source")
    async def consumer(ctx: Any, value: int) -> str:
        timeline.append(f"consumer-invoked-{value}")
        return f"got-{value}"

    events = await _collect(pipeline, "start")
    assert isinstance(events[-1], PipelineDone)

    # Interleaving evidence: consumer must have processed at least one item
    # before the source emitted its last item.
    last_yield_idx = timeline.index("source-yielded-2")
    first_consumer_idx = next(
        i for i, entry in enumerate(timeline) if entry.startswith("consumer-invoked-")
    )
    assert first_consumer_idx < last_yield_idx, timeline


async def test_fan_out_respects_max_concurrency() -> None:
    """max_concurrency caps in-flight invocations of a fan-out downstream."""
    pipeline = Pipeline("concurrent")
    state = {"in_flight": 0, "max_seen": 0}

    @pipeline.step()
    async def source(ctx: Any, value: Any) -> AsyncIterator[int]:
        for i in range(10):
            yield i

    @pipeline.step(fan_out_from="source", max_concurrency=2)
    async def slow(ctx: Any, value: int) -> int:
        state["in_flight"] += 1
        state["max_seen"] = max(state["max_seen"], state["in_flight"])
        await asyncio.sleep(0.02)
        state["in_flight"] -= 1
        return value * 2

    events = await _collect(pipeline, "start")
    slow_outputs = [
        e for e in events if isinstance(e, StepOutput) and e.step == "slow"
    ]
    assert len(slow_outputs) == 10
    assert {o.value for o in slow_outputs} == {i * 2 for i in range(10)}
    # Concurrency must reach the cap (proving parallelism happens) and never
    # exceed it.
    assert state["max_seen"] == 2, state


async def test_fan_out_unbounded_when_max_concurrency_none() -> None:
    """No max_concurrency → all items dispatch in parallel."""
    pipeline = Pipeline("unbounded")
    state = {"in_flight": 0, "max_seen": 0}

    @pipeline.step()
    async def source(ctx: Any, value: Any) -> AsyncIterator[int]:
        for i in range(5):
            yield i

    @pipeline.step(fan_out_from="source")
    async def parallel(ctx: Any, value: int) -> int:
        state["in_flight"] += 1
        state["max_seen"] = max(state["max_seen"], state["in_flight"])
        await asyncio.sleep(0.02)
        state["in_flight"] -= 1
        return value

    events = await _collect(pipeline, "start")
    outputs = [
        e for e in events if isinstance(e, StepOutput) and e.step == "parallel"
    ]
    assert len(outputs) == 5
    assert state["max_seen"] >= 2, state


# --- MERGE -----------------------------------------------------------------


async def test_merge_joins_two_sources_by_key() -> None:
    """MERGE fires once per key that every source has contributed to."""
    pipeline = Pipeline("merge-happy")

    @pipeline.step()
    async def a(ctx: Any, value: Any) -> AsyncIterator[int]:
        for x in (1, 2, 3):
            yield x

    @pipeline.step()
    async def b(ctx: Any, value: Any) -> AsyncIterator[int]:
        for x in (10, 20, 30):
            yield x

    @pipeline.merge("a", "b", key=lambda v: v if v < 10 else v // 10)
    async def combine(
        ctx: Any, key: Any, a_val: int, b_val: int
    ) -> AsyncIterator[tuple[Any, int, int]]:
        yield (key, a_val, b_val)

    events = await _collect(pipeline, "start")
    merge_outputs = [
        e for e in events if isinstance(e, StepOutput) and e.step == "combine"
    ]
    assert len(merge_outputs) == 3
    triples = sorted(o.value for o in merge_outputs)
    assert triples == [(1, 1, 10), (2, 2, 20), (3, 3, 30)]
    # Each output's key matches the merge bucket key.
    for o in merge_outputs:
        assert o.key == o.value[0]


async def test_merge_lenient_drops_partial_key_set_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """In lenient mode, a missing-source key logs a warning and is dropped."""
    pipeline = Pipeline("merge-lenient")

    @pipeline.step()
    async def a(ctx: Any, value: Any) -> AsyncIterator[int]:
        for x in (1, 2, 3):
            yield x

    @pipeline.step()
    async def b(ctx: Any, value: Any) -> AsyncIterator[int]:
        # Skip 2 — b only contributes for keys 1 and 3.
        for x in (1, 3):
            yield x

    @pipeline.merge("a", "b", key=lambda v: v, mode="lenient")
    async def combine(
        ctx: Any, key: Any, a_val: int, b_val: int
    ) -> AsyncIterator[tuple[Any, int, int]]:
        yield (key, a_val, b_val)

    with caplog.at_level("WARNING", logger="streamwright.scheduler"):
        events = await _collect(pipeline, "start")

    merge_outputs = [
        e for e in events if isinstance(e, StepOutput) and e.step == "combine"
    ]
    assert sorted(o.value for o in merge_outputs) == [(1, 1, 1), (3, 3, 3)]
    # No StepFailed (lenient).
    assert not [e for e in events if isinstance(e, StepFailed)]
    # Key 2 dropped with warning.
    assert any(
        "dropping partial key-set" in record.getMessage()
        and "key=2" in record.getMessage()
        for record in caplog.records
    )


async def test_merge_strict_emits_step_failed_for_incomplete_keys() -> None:
    """In strict mode (default), end-of-run incomplete keys emit StepFailed."""
    pipeline = Pipeline("merge-strict")

    @pipeline.step()
    async def a(ctx: Any, value: Any) -> AsyncIterator[int]:
        for x in (1, 2, 3):
            yield x

    @pipeline.step()
    async def b(ctx: Any, value: Any) -> AsyncIterator[int]:
        # Skip 2 — b only contributes for keys 1 and 3.
        for x in (1, 3):
            yield x

    @pipeline.merge("a", "b", key=lambda v: v)  # mode defaults to "strict"
    async def combine(
        ctx: Any, key: Any, a_val: int, b_val: int
    ) -> AsyncIterator[tuple[Any, int, int]]:
        yield (key, a_val, b_val)

    events = await _collect(pipeline, "start")
    # Successful keys still emit outputs.
    merge_outputs = [
        e for e in events if isinstance(e, StepOutput) and e.step == "combine"
    ]
    assert sorted(o.value for o in merge_outputs) == [(1, 1, 1), (3, 3, 3)]

    # Strict mode emits exactly one StepFailed listing the incomplete keys.
    failures = [
        e for e in events if isinstance(e, StepFailed) and e.step == "combine"
    ]
    assert len(failures) == 1
    err = failures[0].error
    assert "key=2" in err
    assert "'b'" in err


async def test_merge_with_three_sources() -> None:
    """N=3 sources join on the same key — verifies generalization past 2."""
    pipeline = Pipeline("merge-three")

    @pipeline.step()
    async def a(ctx: Any, value: Any) -> AsyncIterator[int]:
        for x in (1, 2):
            yield x

    @pipeline.step()
    async def b(ctx: Any, value: Any) -> AsyncIterator[int]:
        for x in (1, 2):
            yield x

    @pipeline.step()
    async def c(ctx: Any, value: Any) -> AsyncIterator[int]:
        for x in (1, 2):
            yield x

    @pipeline.merge("a", "b", "c", key=lambda v: v)
    async def combine(
        ctx: Any, key: Any, a_val: int, b_val: int, c_val: int
    ) -> AsyncIterator[tuple[Any, int, int, int]]:
        yield (key, a_val, b_val, c_val)

    events = await _collect(pipeline, "start")
    outputs = [
        e for e in events if isinstance(e, StepOutput) and e.step == "combine"
    ]
    assert sorted(o.value for o in outputs) == [(1, 1, 1, 1), (2, 2, 2, 2)]
    # All sources contributed — no failure.
    assert not [e for e in events if isinstance(e, StepFailed)]


# --- Retry integration & fatal errors -------------------------------------


async def test_scheduler_retries_transient_then_succeeds(
    fake_clock: list[float],
) -> None:
    """A retryable failure followed by success emits one StepOutput, no StepFailed."""
    pipeline = Pipeline("retry-transient")
    attempts = {"count": 0}

    @pipeline.step()
    async def flaky(ctx: Any, value: Any) -> str:
        attempts["count"] += 1
        if attempts["count"] < 2:
            err = ProviderError("transient")
            err.retryable = True
            raise err
        return "ok"

    events = await _collect(pipeline, "x")
    outputs = [
        e for e in events if isinstance(e, StepOutput) and e.step == "flaky"
    ]
    assert len(outputs) == 1
    assert outputs[0].value == "ok"
    assert attempts["count"] == 2
    assert not [e for e in events if isinstance(e, StepFailed)]
    assert len(fake_clock) == 1
    # Default jitter is uniform(0, 0.5); backoff[1] is in [1.0, 1.5).
    assert 1.0 <= fake_clock[0] < 1.5


async def test_scheduler_exhausted_retries_emits_step_failed(
    fake_clock: list[float],
) -> None:
    """A retryable error that never recovers emits StepFailed once."""
    pipeline = Pipeline("retry-exhausted")

    @pipeline.step()
    async def always_fails(ctx: Any, value: Any) -> str:
        err = ProviderError("never recovers")
        err.retryable = True
        raise err

    events = await _collect(pipeline, "x")
    failures = [
        e
        for e in events
        if isinstance(e, StepFailed) and e.step == "always_fails"
    ]
    assert len(failures) == 1
    assert "never recovers" in failures[0].error
    # 3 attempts → 2 backoffs.
    assert len(fake_clock) == 2


async def test_scheduler_fatal_error_emits_step_failed_without_retry(
    fake_clock: list[float],
) -> None:
    """A non-retryable exception emits StepFailed immediately with no sleeps."""
    pipeline = Pipeline("retry-fatal")

    @pipeline.step()
    async def fatal(ctx: Any, value: Any) -> str:
        raise ValueError("not retryable")

    events = await _collect(pipeline, "x")
    failures = [e for e in events if isinstance(e, StepFailed)]
    assert len(failures) == 1
    assert "not retryable" in failures[0].error
    assert fake_clock == []


async def test_scheduler_unknown_model_error_is_fatal(
    fake_clock: list[float],
) -> None:
    """UnknownModelError carries retryable=False and is treated as fatal."""
    pipeline = Pipeline("retry-unknown-model")

    @pipeline.step()
    async def step_fn(ctx: Any, value: Any) -> str:
        raise UnknownModelError("nope")

    events = await _collect(pipeline, "x")
    failures = [e for e in events if isinstance(e, StepFailed)]
    assert len(failures) == 1
    assert fake_clock == []


async def test_scheduler_failed_step_does_not_push_downstream_value(
    fake_clock: list[float],
) -> None:
    """When an upstream step fails, downstream receives no value but still completes."""
    pipeline = Pipeline("failure-blocks-output")

    @pipeline.step()
    async def upstream(ctx: Any, value: Any) -> str:
        raise ValueError("upstream broke")

    @pipeline.step(fan_out_from="upstream")
    async def downstream(ctx: Any, value: Any) -> str:
        return f"got-{value}"

    events = await _collect(pipeline, "x")
    # Upstream failed; downstream emitted Started + Done but no Output.
    upstream_failures = [e for e in events if isinstance(e, StepFailed) and e.step == "upstream"]
    assert len(upstream_failures) == 1
    downstream_outputs = [
        e for e in events if isinstance(e, StepOutput) and e.step == "downstream"
    ]
    assert downstream_outputs == []
    # Pipeline still terminates cleanly.
    assert isinstance(events[-1], PipelineDone)


# --- Cancellation & backpressure ------------------------------------------


async def test_scheduler_cancellation_does_not_leak_tasks() -> None:
    """Closing execute()'s generator early must cancel all runners cleanly.

    The test fails if Python's asyncio raises either:
      - "Task was destroyed but it is pending"
      - "coroutine ... was never awaited"
    because those signal that the scheduler's cleanup leaked work.
    """
    import gc
    import warnings

    pipeline = Pipeline("cancellation")

    @pipeline.step()
    async def slow_source(ctx: Any, value: Any) -> AsyncIterator[int]:
        for i in range(100):
            yield i
            await asyncio.sleep(0.001)

    @pipeline.step(fan_out_from="slow_source")
    async def slow_processor(ctx: Any, value: int) -> int:
        await asyncio.sleep(0.005)
        return value * 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        scheduler = _scheduler_for(pipeline)
        gen = scheduler.execute("start")
        # Pull a single event then close early.
        await gen.__anext__()
        await gen.aclose()

        # Force gc + a loop tick so any "destroyed-pending" warnings have a
        # chance to fire before we inspect the captured set.
        gc.collect()
        await asyncio.sleep(0.01)
        gc.collect()

    leaks = [
        str(w.message)
        for w in caught
        if "Task was destroyed but it is pending" in str(w.message)
        or "was never awaited" in str(w.message)
    ]
    assert leaks == [], leaks


async def test_scheduler_queue_backpressure_throttles_producer(
    tiny_queue_maxsize: int,
) -> None:
    """A bounded queue blocks the producer when the consumer is slower.

    With max_concurrency=1 the upper bound on ``produced - consumed`` is
    ``maxsize`` (in-queue) + 1 (runner in-transit, between get() and
    create_task) + 1 (running task) + 1 (producer blocked at put()) =
    ``maxsize + 3``. We also assert ``max_gap >= maxsize`` to prove the
    queue actually filled — otherwise backpressure was never exercised.
    """
    pipeline = Pipeline("backpressure")
    timeline: list[tuple[str, int]] = []

    @pipeline.step()
    async def producer(ctx: Any, value: Any) -> AsyncIterator[int]:
        for i in range(6):
            timeline.append(("produced", i))
            yield i

    @pipeline.step(fan_out_from="producer", max_concurrency=1)
    async def slow_consumer(ctx: Any, value: int) -> int:
        await asyncio.sleep(0.01)
        timeline.append(("consumed", value))
        return value

    await _collect(pipeline, "start")

    produced_count = 0
    consumed_count = 0
    max_gap = 0
    for kind, _ in timeline:
        if kind == "produced":
            produced_count += 1
        else:
            consumed_count += 1
        max_gap = max(max_gap, produced_count - consumed_count)

    upper = tiny_queue_maxsize + 3
    assert max_gap <= upper, (
        f"max gap {max_gap} > upper bound {upper}; timeline={timeline}"
    )
    assert max_gap >= tiny_queue_maxsize, (
        f"max gap {max_gap} < maxsize ({tiny_queue_maxsize}) — "
        "backpressure path not exercised; queue may never have filled"
    )


# --- Documented limitations -----------------------------------------------


async def test_empty_stream_source_emits_started_and_done_only() -> None:
    """A STREAM entry that yields nothing emits Started+Done with no Output.

    Downstream consumers also wind down cleanly with only Started+Done.
    """
    pipeline = Pipeline("empty-stream")

    @pipeline.step()
    async def empty(ctx: Any, value: Any) -> AsyncIterator[int]:
        return  # function body has no yield; still an async generator
        yield  # pragma: no cover — unreachable; marks fn as async generator

    @pipeline.step(fan_out_from="empty")
    async def downstream(ctx: Any, value: int) -> str:
        return f"got-{value}"

    events = await _collect(pipeline, "start")

    empty_events = [e for e in events if getattr(e, "step", None) == "empty"]
    assert [type(e).__name__ for e in empty_events] == ["StepStarted", "StepDone"]

    downstream_events = [
        e for e in events if getattr(e, "step", None) == "downstream"
    ]
    assert [type(e).__name__ for e in downstream_events] == [
        "StepStarted",
        "StepDone",
    ]

    # No outputs anywhere.
    assert not [e for e in events if isinstance(e, StepOutput)]
    # No StepStreaming (empty STREAM never reaches first yield).
    assert not [e for e in events if isinstance(e, StepStreaming)]


async def test_nested_stream_outputs_share_upstream_key() -> None:
    """Documents a known limitation — see docs/ROADMAP.md
    ("Composite keys for nested STREAM steps").

    A downstream STREAM step's sub-yields all carry the upstream item's
    key. This makes STREAM→STREAM→MERGE patterns unexpressible because
    the MERGE sees colliding keys for distinct sub-stream items.
    """
    pipeline = Pipeline("nested-stream-keys")

    @pipeline.step()
    async def source(ctx: Any, value: Any) -> AsyncIterator[int]:
        yield 100
        yield 200

    @pipeline.step(fan_out_from="source")
    async def expand(ctx: Any, value: int) -> AsyncIterator[str]:
        for i in range(3):
            yield f"{value}-{i}"

    events = await _collect(pipeline, "start")
    expand_outputs = [
        e for e in events if isinstance(e, StepOutput) and e.step == "expand"
    ]
    assert len(expand_outputs) == 6  # 2 upstream items × 3 sub-yields each.

    keys_for_100 = {o.key for o in expand_outputs if str(o.value).startswith("100-")}
    keys_for_200 = {o.key for o in expand_outputs if str(o.value).startswith("200-")}
    # All three sub-items from each upstream collapse to one key:
    # the upstream item's position. This is the limitation.
    assert keys_for_100 == {0}
    assert keys_for_200 == {1}

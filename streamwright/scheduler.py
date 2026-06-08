"""DAG scheduler for executing pipeline steps.

This module implements streamwright's executor. The scheduler walks a
:py:class:`Pipeline`'s DAG, spawns one async task per step, and routes
values between steps through bounded :py:class:`asyncio.Queue` channels
so downstream steps consume from upstream as soon as upstream enters
its streaming phase — not when it's fully done.

Key behaviors (described here so the contract lives near the code):

* Each STREAM source step assigns a position-based key (0, 1, 2, …) to
  each item it yields.
* Non-MERGE downstream steps **inherit** their input's key on every
  output they produce. Concretely: if a STREAM A yields three items
  with keys 0/1/2 and a STREAM B fans out from A, every value B yields
  while handling A-item-0 carries key 0 — including sub-stream items
  from the same B invocation. A downstream MERGE will see those
  sub-items as colliding under the same key. This is a known
  limitation; see ``docs/ROADMAP.md`` ("Composite keys for nested
  STREAM steps") for the planned fix.
* :py:func:`with_retries` and :py:func:`is_retryable_error` are exposed
  so callers (and tests) can reuse the same classification the scheduler
  applies internally.

Test seams (rebound via ``monkeypatch.setattr`` in fixtures):

* ``_sleep`` — the function awaited between retry attempts.
* ``DEFAULT_QUEUE_MAXSIZE`` — the bound used for inter-step queues.
"""

from __future__ import annotations

import asyncio
import logging
import random
import traceback
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Hashable, Sequence
from contextlib import suppress
from typing import Any

import httpx

from .context import Context
from .providers.errors import ProviderError
from .types import (
    JobEvent,
    PipelineDone,
    StepDone,
    StepFailed,
    StepKind,
    StepOutput,
    StepSpec,
    StepStarted,
    StepStreaming,
)

logger = logging.getLogger(__name__)

_sleep = asyncio.sleep

DEFAULT_QUEUE_MAXSIZE = 32


class _Sentinel:
    """Internal marker for queue control signals."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Sentinel {self.name}>"


_DONE = _Sentinel("DONE")
_PIPELINE_DONE = _Sentinel("PIPELINE_DONE")


def is_retryable_error(exc: BaseException) -> bool:
    """Classify whether ``exc`` should trigger a scheduler retry.

    Retryable:

    * :py:class:`httpx.TimeoutException` (and its subclasses)
    * :py:class:`asyncio.TimeoutError`
    * :py:class:`httpx.HTTPStatusError` whose response status is 5xx or 429
    * :py:class:`ProviderError` whose ``retryable`` attribute is ``True``

    Everything else (including :py:class:`UnknownModelError`,
    :py:class:`CapabilityError`, generic :py:class:`Exception`) is fatal.
    """
    if isinstance(exc, httpx.TimeoutException | asyncio.TimeoutError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status >= 500 or status == 429
    if isinstance(exc, ProviderError):
        return exc.retryable
    return False


async def with_retries[T](
    factory: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    is_retryable: Callable[[BaseException], bool] = is_retryable_error,
    base_delay: float = 1.0,
    max_jitter: float = 0.5,
) -> T:
    """Call ``factory()`` with exponential backoff on retryable errors.

    Attempts are numbered ``1..max_attempts``. The delay between attempt
    ``N`` and ``N+1`` is ``base_delay * 2**(N-1) + uniform(0, max_jitter)``
    seconds. Fatal (non-retryable) exceptions propagate immediately;
    :py:class:`asyncio.CancelledError` bypasses retry because the
    ``except Exception`` clause does not catch it.

    ``factory`` is invoked once per attempt to produce a fresh awaitable
    — coroutines cannot be awaited twice.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await factory()
        except Exception as exc:
            if not is_retryable(exc):
                raise
            last_exc = exc
            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, max_jitter)
                await _sleep(delay)
    assert last_exc is not None  # loop body always assigns on the path that reaches here
    raise last_exc


# --- Internal runner & topology ------------------------------------------


class _StepRunner:
    """Per-step async runner that consumes input queue items and emits events.

    For MERGE steps this base implementation raises NotImplementedError;
    MERGE arrives in a later commit.
    """

    def __init__(
        self,
        *,
        spec: StepSpec,
        ctx: Context,
        output_queues: list[asyncio.Queue[Any]],
        emit: Callable[[JobEvent], None],
        input_queue: asyncio.Queue[Any] | None = None,
        merge_input_queues: dict[str, asyncio.Queue[Any]] | None = None,
    ) -> None:
        self.spec = spec
        self.ctx = ctx
        self.output_queues = output_queues
        self.emit = emit
        self.input_queue = input_queue
        self.merge_input_queues = merge_input_queues
        self._streaming_emitted = False

    async def run(self) -> None:
        self.emit(StepStarted(step=self.spec.name))
        try:
            if self.spec.kind == StepKind.MERGE:
                assert self.merge_input_queues is not None
                await self._run_merge()
            else:
                assert self.input_queue is not None
                await self._run_non_merge()
        finally:
            for q in self.output_queues:
                await q.put(_DONE)
            self.emit(StepDone(step=self.spec.name))

    async def _run_merge(self) -> None:
        """Consume from every upstream source and fire on complete key-sets.

        Each source has its own queue. Items are stored in a per-merge-key
        state dict. When every source has contributed for a given key, the
        merge function is invoked with ``(ctx, key, *values_in_declared_order)``
        and its yielded outputs are emitted. After every source signals
        DONE, any incomplete key-set is dropped with a warning (lenient
        merge mode — strict mode is a future configurable).
        """
        assert self.merge_input_queues is not None
        source_names = list(self.merge_input_queues.keys())
        state: dict[Hashable, dict[str, Any]] = {}

        async def consume_source(
            source_name: str, queue: asyncio.Queue[Any]
        ) -> None:
            while True:
                item = await queue.get()
                if item is _DONE:
                    return
                value, propagated_key = item
                merge_key: Hashable = (
                    self.spec.merge_key(value)
                    if self.spec.merge_key is not None
                    else propagated_key
                )
                bucket = state.setdefault(merge_key, {})
                bucket[source_name] = value
                if all(name in bucket for name in source_names):
                    values_in_order = [bucket[name] for name in source_names]
                    del state[merge_key]
                    await self._fire_merge(merge_key, values_in_order)

        await asyncio.gather(
            *(
                consume_source(name, queue)
                for name, queue in self.merge_input_queues.items()
            )
        )

        # All sources DONE — any remaining state entries are incomplete
        # key-sets. Dispatch on merge_mode.
        incomplete: list[tuple[Hashable, list[str]]] = []
        for key, partial in state.items():
            missing = [name for name in source_names if name not in partial]
            incomplete.append((key, missing))

        if not incomplete:
            return
        if self.spec.merge_mode == "strict":
            details = "; ".join(
                f"key={k!r} missing {m!r}" for k, m in incomplete
            )
            msg = (
                f"MERGE step {self.spec.name!r} incomplete at end of run: "
                f"{details}"
            )
            try:
                raise RuntimeError(msg)
            except RuntimeError as exc:
                self._emit_failure(exc)
        else:
            for key, missing in incomplete:
                logger.warning(
                    "MERGE %r: dropping partial key-set for key=%r; missing sources %r",
                    self.spec.name,
                    key,
                    missing,
                )

    async def _fire_merge(
        self, merge_key: Hashable, values: list[Any]
    ) -> None:
        try:
            agen = self.spec.fn(self.ctx, merge_key, *values)
            async for result in agen:
                await self._emit_output(result, merge_key)
        except Exception as exc:
            self._emit_failure(exc)

    async def _run_non_merge(self) -> None:
        assert self.input_queue is not None
        in_q = self.input_queue
        sem: asyncio.Semaphore | None = (
            asyncio.Semaphore(self.spec.max_concurrency)
            if self.spec.max_concurrency
            else None
        )
        tasks: set[asyncio.Task[None]] = set()
        try:
            while True:
                item = await in_q.get()
                if item is _DONE:
                    break
                value, key = item
                if sem is not None:
                    await sem.acquire()
                task = asyncio.create_task(self._invoke_with_release(value, key, sem))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        finally:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _invoke_with_release(
        self,
        value: Any,
        key: Hashable,
        sem: asyncio.Semaphore | None,
    ) -> None:
        try:
            await self._invoke(value, key)
        finally:
            if sem is not None:
                sem.release()

    async def _invoke(self, value: Any, key: Hashable) -> None:
        if self.spec.kind == StepKind.SINGLE:
            await self._invoke_single(value, key)
        elif self.spec.kind == StepKind.STREAM:
            await self._invoke_stream(value, key)

    async def _invoke_single(self, value: Any, key: Hashable) -> None:
        try:
            async def factory() -> Any:
                return await self.spec.fn(self.ctx, value)

            result = await with_retries(factory, max_attempts=self.spec.retries)
        except Exception as exc:
            self._emit_failure(exc)
            return
        await self._emit_output(result, key)

    async def _invoke_stream(self, value: Any, key: Hashable) -> None:
        # Entry STREAM steps (no upstream) assign position-based keys to
        # their items: 0, 1, 2, … . Non-entry STREAM invocations inherit
        # the upstream item's key on every yielded sub-item — see the
        # module-level docstring for the limitation this implies.
        is_stream_source = self.spec.fan_out_from is None
        position = 0
        try:
            agen = self.spec.fn(self.ctx, value)
            async for item in agen:
                item_key: Hashable = position if is_stream_source else key
                await self._emit_output(item, item_key)
                position += 1
        except Exception as exc:
            self._emit_failure(exc)

    async def _emit_output(self, value: Any, key: Hashable) -> None:
        # StepStreaming is the "first-of-many" signal — emit it only for
        # STREAM steps. SINGLE produces exactly one output and MERGE is
        # treated like SINGLE for this purpose: callers asked us not to
        # fire a misleading "now streaming" event on either.
        if self.spec.kind == StepKind.STREAM and not self._streaming_emitted:
            self.emit(StepStreaming(step=self.spec.name))
            self._streaming_emitted = True
        self.emit(StepOutput(step=self.spec.name, value=value, key=key))
        for out_q in self.output_queues:
            await out_q.put((value, key))

    def _emit_failure(self, exc: BaseException) -> None:
        # Use exc.__traceback__ when present (real raised exception). For
        # exceptions constructed without a traceback (eg synthesized by
        # the strict-mode merge path), fall back to a single-line render
        # so the StepFailed.traceback field is still useful.
        if exc.__traceback__ is not None:
            tb = "".join(traceback.format_exception(exc))
        else:
            tb = f"{type(exc).__name__}: {exc}\n"
        self.emit(
            StepFailed(
                step=self.spec.name,
                error=f"{type(exc).__name__}: {exc}",
                traceback=tb,
            )
        )


# --- Scheduler -----------------------------------------------------------


class Scheduler:
    """Execute a resolved step DAG, yielding :py:class:`JobEvent`s in real time.

    Construct with a sequence of :py:class:`StepSpec`s in topological order
    (typically obtained via :py:meth:`Pipeline._build_dag`). Call
    :py:meth:`execute` to obtain an async generator that yields events as
    work happens; the final event is always a :py:class:`PipelineDone`.

    The scheduler manages internal queues, spawns one runner task per
    step, and cancels everything cleanly if the consumer of ``execute()``
    stops iterating before completion.
    """

    def __init__(self, steps: Sequence[StepSpec]) -> None:
        self._steps_by_name = {s.name: s for s in steps}
        self._steps_order = [s.name for s in steps]

    async def execute(self, input_value: Any) -> AsyncGenerator[JobEvent, None]:
        """Run the DAG with ``input_value`` fed to the entry step."""
        event_queue: asyncio.Queue[JobEvent | _Sentinel] = asyncio.Queue()

        def emit(event: JobEvent) -> None:
            event_queue.put_nowait(event)

        job_id = uuid.uuid4().hex
        non_merge_inputs, merge_inputs, output_routes = self._build_topology(
            input_value
        )

        runners: list[asyncio.Task[None]] = []
        for name, spec in self._steps_by_name.items():
            ctx = Context(job_id=job_id, step_name=name, emit_fn=emit)
            if spec.kind == StepKind.MERGE:
                runner = _StepRunner(
                    spec=spec,
                    ctx=ctx,
                    output_queues=output_routes[name],
                    emit=emit,
                    merge_input_queues=merge_inputs[name],
                )
            else:
                runner = _StepRunner(
                    spec=spec,
                    ctx=ctx,
                    output_queues=output_routes[name],
                    emit=emit,
                    input_queue=non_merge_inputs[name],
                )
            runners.append(asyncio.create_task(runner.run(), name=f"runner:{name}"))

        async def watcher() -> None:
            await asyncio.gather(*runners, return_exceptions=True)
            event_queue.put_nowait(_PIPELINE_DONE)

        watcher_task = asyncio.create_task(watcher(), name="scheduler-watcher")

        try:
            while True:
                ev = await event_queue.get()
                if isinstance(ev, _Sentinel):
                    break
                yield ev
            yield PipelineDone()
        finally:
            for r in runners:
                r.cancel()
            watcher_task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.gather(*runners, watcher_task, return_exceptions=True)

    def _build_topology(
        self, input_value: Any
    ) -> tuple[
        dict[str, asyncio.Queue[Any]],
        dict[str, dict[str, asyncio.Queue[Any]]],
        dict[str, list[asyncio.Queue[Any]]],
    ]:
        """Create per-step input queues, pre-load entries, and wire outputs.

        Returns ``(non_merge_inputs, merge_inputs, output_routes)``:

        * ``non_merge_inputs[step]`` — single :py:class:`asyncio.Queue` for
          SINGLE / STREAM step ``step``.
        * ``merge_inputs[step]`` — dict of ``{source_name: Queue}`` for the
          MERGE step ``step``; the runner consumes each source in parallel.
        * ``output_routes[step]`` — list of queues the runner pushes to
          when it produces an output.
        """
        non_merge_inputs: dict[str, asyncio.Queue[Any]] = {}
        merge_inputs: dict[str, dict[str, asyncio.Queue[Any]]] = {}

        for name, spec in self._steps_by_name.items():
            if spec.kind == StepKind.MERGE:
                merge_inputs[name] = {
                    src: asyncio.Queue(maxsize=DEFAULT_QUEUE_MAXSIZE)
                    for src in spec.merge_sources
                }
            else:
                non_merge_inputs[name] = asyncio.Queue(maxsize=DEFAULT_QUEUE_MAXSIZE)

        # Pre-load entry-step queues with the pipeline input.
        for name, spec in self._steps_by_name.items():
            if spec.kind != StepKind.MERGE and spec.fan_out_from is None:
                q = non_merge_inputs[name]
                q.put_nowait((input_value, 0))
                q.put_nowait(_DONE)

        output_routes: dict[str, list[asyncio.Queue[Any]]] = {
            name: [] for name in self._steps_by_name
        }
        for downstream_name, downstream_spec in self._steps_by_name.items():
            if downstream_spec.kind == StepKind.MERGE:
                for src in downstream_spec.merge_sources:
                    output_routes[src].append(merge_inputs[downstream_name][src])
            elif downstream_spec.fan_out_from is not None:
                output_routes[downstream_spec.fan_out_from].append(
                    non_merge_inputs[downstream_name]
                )

        return non_merge_inputs, merge_inputs, output_routes

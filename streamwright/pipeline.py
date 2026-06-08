"""Pipeline DSL for composing streaming multi-LLM workflows."""

from __future__ import annotations

import inspect
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from types import MappingProxyType
from typing import Any, TypeVar

from .types import JobEvent, MergeKeyFn, MergeMode, StepFn, StepKind, StepSpec

DecoratedStep = TypeVar("DecoratedStep", bound=Callable[..., Any])


class Pipeline:
    """Collect step declarations and validate their dependency graph."""

    def __init__(self, name: str) -> None:
        """Create an empty pipeline with a stable display name."""
        self.name = name
        self._steps: dict[str, StepSpec] = {}
        self._dag_order: tuple[str, ...] | None = None

    @property
    def steps(self) -> Mapping[str, StepSpec]:
        """Expose registered step specs without allowing registry mutation."""
        return MappingProxyType(self._steps)

    def step(
        self,
        *,
        fan_out_from: str | None = None,
        max_concurrency: int | None = None,
        retries: int = 3,
    ) -> Callable[[DecoratedStep], DecoratedStep]:
        """Register an async single-output or async-generator streaming step."""

        def decorator(fn: DecoratedStep) -> DecoratedStep:
            kind = StepKind.STREAM if inspect.isasyncgenfunction(fn) else StepKind.SINGLE
            self._register(
                StepSpec(
                    name=fn.__name__,
                    kind=kind,
                    fn=fn,
                    fan_out_from=fan_out_from,
                    max_concurrency=max_concurrency,
                    retries=retries,
                )
            )
            return fn

        return decorator

    def merge(
        self,
        *sources: str,
        key: MergeKeyFn,
        mode: MergeMode = "strict",
    ) -> Callable[[DecoratedStep], DecoratedStep]:
        """Register a merge step that joins upstream values by key.

        ``key`` is applied to each upstream value to derive the bucket
        used for joining; whenever every upstream source has contributed
        a value for the same bucket key, the merge function is invoked
        as ``await fn(ctx, key, *values_in_declared_source_order)`` and
        its async-generated outputs are emitted.

        ``mode`` controls behavior when a source signals DONE without
        contributing to a bucket that other sources have already filled:

        * ``"strict"`` (default) — emit a single :py:class:`StepFailed`
          for the merge step at end-of-run that lists every incomplete
          bucket and the sources missing from each.
        * ``"lenient"`` — log a warning per incomplete bucket and drop
          the partial set silently.

        **First-write-wins per source**: if a single source produces
        multiple values mapping to the same bucket key, the *first*
        value fires the merge once all other sources have caught up.
        Subsequent values from that source for the same key create a
        fresh partial bucket; whether that bucket eventually completes
        or triggers the ``mode``-dependent behavior depends on whether
        the other sources produce again under the same key. Callers
        relying on this should ensure their key function is one-to-one
        with the joined values.
        """

        def decorator(fn: DecoratedStep) -> DecoratedStep:
            self._register(
                StepSpec(
                    name=fn.__name__,
                    kind=StepKind.MERGE,
                    fn=fn,
                    merge_sources=list(sources),
                    merge_key=key,
                    merge_mode=mode,
                )
            )
            return fn

        return decorator

    def run(self, input: Any) -> AsyncIterator[JobEvent]:
        """Validate the DAG and return an async iterator of pipeline events.

        DAG validation runs synchronously here, so structural errors
        (cycles, unknown ``fan_out_from`` references, missing entry steps)
        surface immediately at the call site rather than during iteration.
        Execution itself happens lazily inside the returned generator.
        """
        from .scheduler import Scheduler

        steps_in_order = self._build_dag()
        return Scheduler(list(steps_in_order)).execute(input)

    def _register(self, spec: StepSpec) -> None:
        if spec.name in self._steps:
            raise ValueError(f"Step {spec.name!r} is already registered")
        self._steps[spec.name] = spec
        self._dag_order = None

    def _build_dag(self) -> tuple[StepSpec, ...]:
        """Validate step wiring and return specs in topological order."""
        if self._dag_order is not None:
            return tuple(self._steps[name] for name in self._dag_order)

        dependencies_by_step = self._dependencies_by_step()
        dependents_by_step = {name: set[str]() for name in self._steps}
        in_degree: dict[str, int] = {}

        for step_name, dependencies in dependencies_by_step.items():
            in_degree[step_name] = len(dependencies)
            for dependency in dependencies:
                dependents_by_step[dependency].add(step_name)

        ready = deque(name for name, degree in in_degree.items() if degree == 0)
        ordered_names: list[str] = []

        while ready:
            step_name = ready.popleft()
            ordered_names.append(step_name)
            for dependent in sorted(dependents_by_step[step_name]):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    ready.append(dependent)

        if len(ordered_names) != len(self._steps):
            unresolved = sorted(name for name, degree in in_degree.items() if degree > 0)
            raise ValueError(
                f"Pipeline {self.name!r} contains a dependency cycle involving: "
                f"{', '.join(unresolved)}"
            )

        if not any(len(dependencies) == 0 for dependencies in dependencies_by_step.values()):
            raise ValueError(
                f"Pipeline {self.name!r} requires at least one entry point step "
                "with no upstream dependencies"
            )

        self._dag_order = tuple(ordered_names)
        return tuple(self._steps[name] for name in ordered_names)

    def _dependencies_by_step(self) -> dict[str, set[str]]:
        dependencies_by_step: dict[str, set[str]] = {}

        for spec in self._steps.values():
            dependencies: set[str] = set()

            if spec.fan_out_from is not None:
                if spec.fan_out_from not in self._steps:
                    raise ValueError(
                        f"Step {spec.name!r} references unknown fan_out_from step "
                        f"{spec.fan_out_from!r}"
                    )
                dependencies.add(spec.fan_out_from)

            for source in spec.merge_sources:
                if source not in self._steps:
                    raise ValueError(
                        f"Merge step {spec.name!r} references unknown source step {source!r}"
                    )
                dependencies.add(source)

            dependencies_by_step[spec.name] = dependencies

        return dependencies_by_step


def step(fn: StepFn) -> StepFn:
    """Placeholder for a future module-level step decorator."""
    # TODO: Decide whether the module-level decorator should create implicit pipelines.
    raise NotImplementedError


def merge(*sources: str, key: MergeKeyFn) -> Callable[[StepFn], StepFn]:
    """Placeholder for a future module-level merge decorator.

    Mirrors :py:meth:`Pipeline.merge` but at module scope, for use without
    an explicit Pipeline instance.
    """
    # TODO: Decide whether the module-level decorator should create implicit pipelines.
    del sources, key  # unused until the implicit-pipeline design lands
    raise NotImplementedError

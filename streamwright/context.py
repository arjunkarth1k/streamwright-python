"""Execution context passed to pipeline steps."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .providers import get_provider

if TYPE_CHECKING:
    from .providers.base import Provider
    from .types import JobEvent

logger = logging.getLogger(__name__)


class Context:
    """Per-invocation runtime context handed to every step function.

    Steps receive a ``Context`` as their first positional argument. It
    exposes:

    - ``job_id`` — stable id for this pipeline run.
    - ``step_name`` — the running step's registered name.
    - ``emit(event)`` — push a custom :py:class:`JobEvent` into the run's
      event stream (useful for custom telemetry from inside a step).
    - ``log(msg, **kwargs)`` — log with ``job_id`` and ``step_name``
      automatically attached as ``extra`` fields.
    - ``await llm(spec)`` — resolve a ``"provider/model"`` spec to a
      ``(Provider, model_id)`` tuple via the shared provider registry.
    """

    __slots__ = ("job_id", "step_name", "_emit_fn", "_logger")

    def __init__(
        self,
        *,
        job_id: str,
        step_name: str,
        emit_fn: Callable[[JobEvent], None],
        log: logging.Logger | None = None,
    ) -> None:
        self.job_id = job_id
        self.step_name = step_name
        self._emit_fn = emit_fn
        self._logger = log if log is not None else logger

    def emit(self, event: JobEvent) -> None:
        """Push a JobEvent into the active pipeline's event stream."""
        self._emit_fn(event)

    def log(self, msg: str, **kwargs: Any) -> None:
        """Log ``msg`` at INFO level with job_id and step_name attached."""
        extra = {"job_id": self.job_id, "step": self.step_name, **kwargs}
        self._logger.info(msg, extra=extra)

    async def llm(self, spec: str) -> tuple[Provider, str]:
        """Resolve ``"provider/model"`` to a cached provider instance and model id.

        Returns the same tuple shape as :py:func:`streamwright.get_provider`.
        Async-by-design so the registry can later become I/O-bound (eg
        warming up new provider instances over the network) without
        breaking the call site.
        """
        return get_provider(spec)

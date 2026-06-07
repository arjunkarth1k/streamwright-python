"""Shared pytest fixtures for streamwright tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Patch ``streamwright.scheduler._sleep`` with a recorder.

    Returns the list of delay values passed to the fake sleep, in the
    order they were requested. The fake doesn't actually sleep, so retry
    tests run instantly.
    """
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    import streamwright.scheduler as sched

    monkeypatch.setattr(sched, "_sleep", fake_sleep)
    return delays


@pytest.fixture
def tiny_queue_maxsize(monkeypatch: pytest.MonkeyPatch) -> int:
    """Set scheduler.DEFAULT_QUEUE_MAXSIZE to a small value for backpressure tests.

    Returns the new maxsize so tests can assert against it.
    """
    new_size = 2
    import streamwright.scheduler as sched

    monkeypatch.setattr(sched, "DEFAULT_QUEUE_MAXSIZE", new_size)
    return new_size

"""Tests for the per-step Context object."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from streamwright.context import Context
from streamwright.types import JobEvent, StepStarted


def test_emit_forwards_event_to_callback() -> None:
    captured: list[JobEvent] = []
    ctx = Context(job_id="job-1", step_name="alpha", emit_fn=captured.append)
    event = StepStarted(step="alpha")
    ctx.emit(event)
    assert captured == [event]


def test_log_attaches_job_id_and_step_name_to_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctx = Context(job_id="job-42", step_name="beta", emit_fn=lambda _: None)
    with caplog.at_level("INFO", logger="streamwright.context"):
        ctx.log("hello", extra_field="x")
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.message == "hello"
    assert record.job_id == "job-42"  # type: ignore[attr-defined]
    assert record.step == "beta"  # type: ignore[attr-defined]
    assert record.extra_field == "x"  # type: ignore[attr-defined]


@pytest.fixture
def fake_api_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    yield


@pytest.fixture
def isolated_provider_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import streamwright.providers as providers_pkg

    monkeypatch.setattr(providers_pkg, "_provider_instances", {})
    yield


async def test_llm_returns_provider_and_model(
    fake_api_keys: None,
    isolated_provider_cache: None,
) -> None:
    from streamwright.providers import AnthropicProvider

    ctx = Context(job_id="job-1", step_name="gamma", emit_fn=lambda _: None)
    provider, model = await ctx.llm("anthropic/claude-haiku-4-5")
    assert isinstance(provider, AnthropicProvider)
    assert model == "claude-haiku-4-5"

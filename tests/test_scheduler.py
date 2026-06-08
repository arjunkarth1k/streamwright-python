"""Tests for the DAG scheduler — retry helper and error classification.

Execution tests (linear/fan-out/merge/backpressure/cancellation) import the
public ``Pipeline`` API and land in a follow-up PR once Arjun's
``streamwright/__init__.py`` exists; the retry-primitive tests below depend
only on ``streamwright.types`` + ``providers.errors`` and ship with the code.
"""

from __future__ import annotations

import httpx
import pytest

from streamwright.providers.errors import (
    CapabilityError,
    ProviderError,
    UnknownModelError,
)
from streamwright.scheduler import is_retryable_error, with_retries


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

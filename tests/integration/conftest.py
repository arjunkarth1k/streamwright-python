"""Shared scaffolding for the integration test suite.

These tests make real, paid API calls. They are off by default and must
be explicitly opted into either by setting ``STREAMWRIGHT_RUN_INTEGRATION=1``
or by invoking pytest with ``-m integration``. Every test in this
directory is auto-marked ``integration``, so the marker expression is
sufficient to enable the suite.

Per-provider API-key fixtures (``anthropic_key``, ``openai_key``,
``moonshot_key``) skip the requesting test cleanly when the relevant
key is unset, so a partial-key environment still lets the other
providers' tests run.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from dotenv import load_dotenv

# Load .env at the repo root once at import time so per-provider key
# fixtures (and the env-var opt-in gate) see locally-defined credentials
# without each test having to remember to call load_dotenv() itself.
# Idempotent and a no-op if .env is absent — safe for CI.
load_dotenv()


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``integration`` marker so ``-m integration`` works."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that hit real provider APIs (cost money)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-mark every test in this directory and skip unless opted in.

    Opt-in is either ``STREAMWRIGHT_RUN_INTEGRATION=1`` in the environment
    or a ``-m`` expression that mentions ``integration``. Without either,
    these tests are skipped, not collected as failures.
    """
    env_opt_in = os.environ.get("STREAMWRIGHT_RUN_INTEGRATION") == "1"
    marker_expr = config.getoption("-m", default="") or ""
    marker_opt_in = "integration" in marker_expr

    skip_marker = pytest.mark.skip(
        reason=(
            "integration tests are opt-in: set STREAMWRIGHT_RUN_INTEGRATION=1 "
            "or invoke pytest with -m integration"
        )
    )

    for item in items:
        # Only touch items in tests/integration/.
        if "tests/integration" not in item.nodeid.replace("\\", "/"):
            continue
        item.add_marker(pytest.mark.integration)
        if not (env_opt_in or marker_opt_in):
            item.add_marker(skip_marker)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set; skipping integration test that requires it")
    return value


@pytest.fixture
def anthropic_key() -> str:
    """Skip the test cleanly if ``ANTHROPIC_API_KEY`` is not set."""
    return _require_env("ANTHROPIC_API_KEY")


@pytest.fixture
def openai_key() -> str:
    """Skip the test cleanly if ``OPENAI_API_KEY`` is not set."""
    return _require_env("OPENAI_API_KEY")


@pytest.fixture
def moonshot_key() -> str:
    """Skip the test cleanly if ``MOONSHOT_API_KEY`` is not set."""
    return _require_env("MOONSHOT_API_KEY")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Close cached providers at session end to release pooled connections.

    The provider registry caches one HTTP client per provider; without
    this, the session-end teardown leaves them to be garbage-collected
    and triggers a ``ResourceWarning``. Runs in a fresh event loop so it
    doesn't depend on pytest-asyncio's loop scoping.
    """
    del session, exitstatus
    import streamwright

    asyncio.run(streamwright.aclose())

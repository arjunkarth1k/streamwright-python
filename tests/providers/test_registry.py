"""Tests for the get_provider registry."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from streamwright.providers import (
    AnthropicProvider,
    MoonshotProvider,
    OpenAIProvider,
    aclose,
    get_provider,
)
from streamwright.providers.errors import UnknownModelError


@pytest.fixture(autouse=True)
def isolated_provider_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test a clean provider instance cache."""
    import streamwright.providers as providers_pkg

    monkeypatch.setattr(providers_pkg, "_provider_instances", {})
    yield


@pytest.fixture(autouse=True)
def fake_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set fake env vars so provider constructors don't fail on real env state."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-moonshot-key")


def test_get_provider_parses_anthropic_spec() -> None:
    provider, model = get_provider("anthropic/claude-haiku-4-5")
    assert isinstance(provider, AnthropicProvider)
    assert model == "claude-haiku-4-5"


def test_get_provider_parses_openai_spec() -> None:
    provider, model = get_provider("openai/gpt-5.5")
    assert isinstance(provider, OpenAIProvider)
    assert model == "gpt-5.5"


def test_get_provider_parses_moonshot_spec() -> None:
    provider, model = get_provider("moonshot/kimi-k2.6")
    assert isinstance(provider, MoonshotProvider)
    assert model == "kimi-k2.6"


def test_provider_instance_is_cached_across_calls() -> None:
    """Repeated calls for the same provider return the same instance."""
    p1, _ = get_provider("anthropic/claude-haiku-4-5")
    p2, _ = get_provider("anthropic/claude-opus-4-7")
    assert p1 is p2


def test_different_providers_have_distinct_instances() -> None:
    anthropic_provider, _ = get_provider("anthropic/claude-haiku-4-5")
    openai_provider, _ = get_provider("openai/gpt-5.5")
    assert anthropic_provider is not openai_provider


def test_get_provider_dated_snapshot_resolves() -> None:
    provider, model = get_provider("anthropic/claude-haiku-4-5-20251001")
    assert isinstance(provider, AnthropicProvider)
    assert model == "claude-haiku-4-5-20251001"


def test_unknown_provider_raises_unknown_model_error() -> None:
    with pytest.raises(UnknownModelError, match="Unknown provider"):
        get_provider("notreal/anything")


def test_unknown_model_raises_unknown_model_error() -> None:
    with pytest.raises(UnknownModelError, match="Unknown model"):
        get_provider("anthropic/claude-nope")


def test_missing_slash_raises_value_error() -> None:
    with pytest.raises(ValueError, match="provider/model"):
        get_provider("just-a-model")


async def test_aclose_closes_owned_clients_and_clears_cache() -> None:
    """aclose() closes every cached provider's owned HTTP client."""
    import streamwright.providers as providers_pkg

    provider, _ = get_provider("anthropic/claude-haiku-4-5")
    assert isinstance(provider, AnthropicProvider)
    # Capture the client before aclose clears the cache.
    client = provider._client
    assert client.is_closed is False
    assert providers_pkg._provider_instances

    await aclose()

    assert client.is_closed is True
    assert providers_pkg._provider_instances == {}


async def test_aclose_is_idempotent() -> None:
    """Calling aclose() twice doesn't raise."""
    get_provider("openai/gpt-5.5")
    await aclose()
    await aclose()  # second call on empty cache must not error


async def test_aclose_on_empty_cache_is_noop() -> None:
    """Calling aclose() with nothing cached doesn't raise."""
    import streamwright.providers as providers_pkg

    assert providers_pkg._provider_instances == {}
    await aclose()

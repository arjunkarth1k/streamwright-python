"""Tests for capabilities tables and get_capabilities lookup."""

from __future__ import annotations

import pytest

from streamwright.providers.capabilities import (
    ANTHROPIC_CAPABILITIES,
    MOONSHOT_CAPABILITIES,
    OPENAI_CAPABILITIES,
    ModelCapabilities,
    get_capabilities,
)
from streamwright.providers.errors import UnknownModelError


def test_haiku_alias_and_dated_snapshot_share_capabilities() -> None:
    """Anthropic dated snapshots resolve to the same ModelCapabilities object as their alias."""
    alias = get_capabilities("anthropic/claude-haiku-4-5")
    dated = get_capabilities("anthropic/claude-haiku-4-5-20251001")
    assert alias is dated


@pytest.mark.parametrize(
    "model",
    ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"],
)
def test_claude_models_support_full_feature_set(model: str) -> None:
    caps = get_capabilities(f"anthropic/{model}")
    assert caps.streaming is True
    assert caps.tool_use is True
    assert caps.structured_outputs is True
    assert caps.vision is True
    assert caps.prompt_caching is True


@pytest.mark.parametrize("model", ["claude-opus-4-7", "claude-sonnet-4-6"])
def test_claude_4_6_generation_has_1m_context(model: str) -> None:
    """Opus 4.7 and Sonnet 4.6 are 1M-token context-window models."""
    caps = get_capabilities(f"anthropic/{model}")
    assert caps.max_context_tokens == 1_000_000


def test_haiku_4_5_has_200k_context() -> None:
    caps = get_capabilities("anthropic/claude-haiku-4-5")
    assert caps.max_context_tokens == 200_000


def test_gpt_5_5_pro_does_not_support_streaming() -> None:
    """The headline streaming-exception edge case."""
    caps = get_capabilities("openai/gpt-5.5-pro")
    assert caps.streaming is False
    # Other capabilities are still enabled.
    assert caps.tool_use is True
    assert caps.structured_outputs is True
    assert caps.prompt_caching is True


@pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5.2"])
def test_other_openai_models_stream(model: str) -> None:
    caps = get_capabilities(f"openai/{model}")
    assert caps.streaming is True


@pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5.5-pro", "gpt-5.2"])
def test_all_gpt_5_family_models_have_vision(model: str) -> None:
    """All GPT-5 models are natively multimodal."""
    caps = get_capabilities(f"openai/{model}")
    assert caps.vision is True


@pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5.5-pro"])
def test_gpt_5_5_family_has_1m_context(model: str) -> None:
    caps = get_capabilities(f"openai/{model}")
    assert caps.max_context_tokens == 1_000_000


def test_gpt_5_2_has_400k_context() -> None:
    caps = get_capabilities("openai/gpt-5.2")
    assert caps.max_context_tokens == 400_000


def test_kimi_k2_6_has_vision() -> None:
    caps = get_capabilities("moonshot/kimi-k2.6")
    assert caps.vision is True


def test_kimi_k2_5_lacks_vision() -> None:
    caps = get_capabilities("moonshot/kimi-k2.5")
    assert caps.vision is False


@pytest.mark.parametrize("model", ["kimi-k2.6", "kimi-k2.5"])
def test_kimi_context_is_256k(model: str) -> None:
    caps = get_capabilities(f"moonshot/{model}")
    assert caps.max_context_tokens == 256_000


def test_unknown_model_raises_unknown_model_error() -> None:
    with pytest.raises(UnknownModelError, match="Unknown model"):
        get_capabilities("openai/gpt-nope")


def test_unknown_provider_raises_unknown_model_error() -> None:
    with pytest.raises(UnknownModelError, match="Unknown provider"):
        get_capabilities("notreal/model")


def test_missing_slash_raises_value_error() -> None:
    with pytest.raises(ValueError, match="provider/model"):
        get_capabilities("just-a-model")


@pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5.5-pro", "gpt-5.2"])
def test_gpt_5_family_declares_forbidden_samplers(model: str) -> None:
    """All GPT-5 reasoning models reject the standard sampling parameters."""
    caps = get_capabilities(f"openai/{model}")
    assert "temperature" in caps.forbidden_params
    assert "top_p" in caps.forbidden_params
    assert "presence_penalty" in caps.forbidden_params
    assert "frequency_penalty" in caps.forbidden_params


@pytest.mark.parametrize(
    "spec",
    [
        "anthropic/claude-opus-4-7",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-haiku-4-5",
        "moonshot/kimi-k2.6",
        "moonshot/kimi-k2.5",
    ],
)
def test_non_reasoning_models_have_empty_forbidden_params(spec: str) -> None:
    """Models that accept standard sampling params declare an empty set."""
    caps = get_capabilities(spec)
    assert caps.forbidden_params == frozenset()


def test_capabilities_tables_are_well_formed() -> None:
    """All registered entries must be ModelCapabilities with positive context limits."""
    for table in (ANTHROPIC_CAPABILITIES, OPENAI_CAPABILITIES, MOONSHOT_CAPABILITIES):
        assert table, "capabilities table must not be empty"
        for model_id, caps in table.items():
            assert isinstance(caps, ModelCapabilities), model_id
            assert caps.max_context_tokens > 0, model_id

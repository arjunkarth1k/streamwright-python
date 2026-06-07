"""Model capability declarations and lookup by ``provider/model`` spec.

Each provider exposes a ``CAPABILITIES`` dict keyed by model identifier.
Anthropic models register both the alias and the dated snapshot pointing to
the same ``ModelCapabilities`` object, so ``is`` identity holds across
either key.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import UnknownModelError


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-model pricing. Placeholder — all fields default to None."""

    input_per_million_tokens: float | None = None
    output_per_million_tokens: float | None = None
    cache_write_per_million_tokens: float | None = None
    cache_read_per_million_tokens: float | None = None


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Declared capabilities for a single provider model.

    ``forbidden_params`` lists chat-completion request fields that this
    model rejects at the API level. Currently used to silently drop
    sampling parameters (``temperature``, ``top_p``,
    ``presence_penalty``, ``frequency_penalty``) on OpenAI's reasoning
    family — see ``OpenAIProvider._build_body``. Models that accept all
    standard parameters leave it as the default empty frozenset.
    """

    streaming: bool
    tool_use: bool
    structured_outputs: bool
    vision: bool
    prompt_caching: bool
    max_context_tokens: int
    pricing: ModelPricing | None = None
    forbidden_params: frozenset[str] = frozenset()


# --- Anthropic -----------------------------------------------------------
#
# Convention shift starting with the 4.6 generation: dateless model IDs ARE
# the canonical pinned snapshot, not floating aliases. So claude-opus-4-7
# and claude-sonnet-4-6 do NOT have separate dated forms — registering one
# key per model is correct for the 4.6+ generation.
#
# The 4.5-and-earlier generation still uses the alias + dated-snapshot
# pair; claude-haiku-4-5 and claude-haiku-4-5-20251001 both point to the
# same ModelCapabilities instance via object identity.
#
# When adding new Anthropic models, follow the dateless-as-canonical
# convention for 4.6+ unless docs.claude.com explicitly publishes a dated
# alias.
#
# Context windows verified at docs.claude.com → "Context windows": Opus
# 4.7, Opus 4.6, Sonnet 4.6 are 1M tokens; Haiku 4.5 is 200k.

_claude_opus_4_7 = ModelCapabilities(
    streaming=True,
    tool_use=True,
    structured_outputs=True,
    vision=True,
    prompt_caching=True,
    max_context_tokens=1_000_000,
)
_claude_sonnet_4_6 = ModelCapabilities(
    streaming=True,
    tool_use=True,
    structured_outputs=True,
    vision=True,
    prompt_caching=True,
    max_context_tokens=1_000_000,
)
_claude_haiku_4_5 = ModelCapabilities(
    streaming=True,
    tool_use=True,
    structured_outputs=True,
    vision=True,
    prompt_caching=True,
    max_context_tokens=200_000,
)

ANTHROPIC_CAPABILITIES: dict[str, ModelCapabilities] = {
    "claude-opus-4-7": _claude_opus_4_7,
    "claude-sonnet-4-6": _claude_sonnet_4_6,
    "claude-haiku-4-5": _claude_haiku_4_5,
    "claude-haiku-4-5-20251001": _claude_haiku_4_5,
}


# --- OpenAI --------------------------------------------------------------
#
# All GPT-5 family models are natively multimodal (text + image at minimum;
# 5.5 adds audio + video). vision=True across the board.
#
# Context windows verified:
#   gpt-5.5      → openai.com/index/introducing-gpt-5-5 (1M API context)
#   gpt-5.5-pro  → same announcement (1M context class)
#   gpt-5.2      → openai.com/index/introducing-gpt-5-2 (400k)
#
# Sampling-parameter restrictions: every GPT-5 entry below is a reasoning
# model and rejects temperature / top_p / presence_penalty /
# frequency_penalty / logprobs / top_logprobs / logit_bias at the API
# level (HTTP 400 unsupported_parameter). Documented at Microsoft Learn
# "Azure OpenAI reasoning models" and confirmed in the OpenAI Developer
# Community thread "Top_p problem when running gpt-5.2 API". The set
# below is the subset that ``OpenAIProvider._build_body`` actually drops
# today; the others are listed here for reference but aren't accepted
# kwargs on the provider surface yet.
_GPT_5_FORBIDDEN_SAMPLERS = frozenset(
    {"temperature", "top_p", "presence_penalty", "frequency_penalty"}
)

_gpt_5_5 = ModelCapabilities(
    streaming=True,
    tool_use=True,
    structured_outputs=True,
    vision=True,
    prompt_caching=True,
    max_context_tokens=1_000_000,
    forbidden_params=_GPT_5_FORBIDDEN_SAMPLERS,
)
_gpt_5_2 = ModelCapabilities(
    streaming=True,
    tool_use=True,
    structured_outputs=True,
    vision=True,
    prompt_caching=True,
    max_context_tokens=400_000,
    forbidden_params=_GPT_5_FORBIDDEN_SAMPLERS,
)
_gpt_5_5_pro = ModelCapabilities(
    streaming=False,
    tool_use=True,
    structured_outputs=True,
    vision=True,
    prompt_caching=True,
    max_context_tokens=1_000_000,
    forbidden_params=_GPT_5_FORBIDDEN_SAMPLERS,
)

OPENAI_CAPABILITIES: dict[str, ModelCapabilities] = {
    "gpt-5.5": _gpt_5_5,
    "gpt-5.2": _gpt_5_2,
    "gpt-5.5-pro": _gpt_5_5_pro,
}


# --- Moonshot ------------------------------------------------------------

_kimi_k2_6 = ModelCapabilities(
    streaming=True,
    tool_use=True,
    structured_outputs=False,
    vision=True,
    prompt_caching=True,
    max_context_tokens=256_000,
)
_kimi_k2_5 = ModelCapabilities(
    streaming=True,
    tool_use=True,
    structured_outputs=False,
    vision=False,
    prompt_caching=True,
    max_context_tokens=256_000,
)

MOONSHOT_CAPABILITIES: dict[str, ModelCapabilities] = {
    "kimi-k2.6": _kimi_k2_6,
    "kimi-k2.5": _kimi_k2_5,
}


_BY_PROVIDER: dict[str, dict[str, ModelCapabilities]] = {
    "anthropic": ANTHROPIC_CAPABILITIES,
    "openai": OPENAI_CAPABILITIES,
    "moonshot": MOONSHOT_CAPABILITIES,
}


def get_capabilities(provider_model: str) -> ModelCapabilities:
    """Look up capabilities for a ``provider/model`` spec.

    Raises:
        ValueError: If the spec is not in ``provider/model`` form.
        UnknownModelError: If the provider prefix or the model ID is not
            registered.
    """
    if "/" not in provider_model:
        raise ValueError(
            f"Invalid provider/model spec {provider_model!r}; expected 'provider/model'"
        )
    provider, model = provider_model.split("/", 1)
    table = _BY_PROVIDER.get(provider)
    if table is None:
        raise UnknownModelError(f"Unknown provider {provider!r} in spec {provider_model!r}")
    caps = table.get(model)
    if caps is None:
        raise UnknownModelError(f"Unknown model {model!r} for provider {provider!r}")
    return caps

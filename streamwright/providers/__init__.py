"""Provider registry and re-exports.

``get_provider`` parses a ``"provider/model"`` spec and returns a cached
provider instance plus the model identifier. Provider instances are
cached one-per-provider-per-process; repeated calls for the same
provider return the same instance.
"""

from __future__ import annotations

from .anthropic import AnthropicProvider
from .base import (
    BaseProvider,
    CompletionResult,
    Done,
    Message,
    Provider,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    Tool,
    ToolCallDelta,
    Usage,
    UsageEvent,
)
from .capabilities import ModelCapabilities, ModelPricing, get_capabilities
from .errors import CapabilityError, ProviderError, UnknownModelError
from .moonshot import MoonshotProvider
from .openai import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "BaseProvider",
    "CapabilityError",
    "CompletionResult",
    "Done",
    "Message",
    "ModelCapabilities",
    "ModelPricing",
    "MoonshotProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "ReasoningDelta",
    "StreamEvent",
    "TextDelta",
    "Tool",
    "ToolCallDelta",
    "UnknownModelError",
    "Usage",
    "UsageEvent",
    "aclose",
    "get_capabilities",
    "get_provider",
]


_PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    AnthropicProvider.name: AnthropicProvider,
    OpenAIProvider.name: OpenAIProvider,
    MoonshotProvider.name: MoonshotProvider,
}

_provider_instances: dict[str, BaseProvider] = {}


def get_provider(spec: str) -> tuple[Provider, str]:
    """Resolve a ``provider/model`` spec to a cached provider and the model id.

    Examples:
        >>> provider, model = get_provider("anthropic/claude-haiku-4-5")

    Raises:
        ValueError: If ``spec`` is not in ``provider/model`` form.
        UnknownModelError: If the provider prefix or the model identifier
            is not registered.
    """
    if "/" not in spec:
        raise ValueError(
            f"Invalid provider spec {spec!r}; expected 'provider/model'"
        )
    provider_name, model = spec.split("/", 1)
    provider_cls = _PROVIDER_CLASSES.get(provider_name)
    if provider_cls is None:
        raise UnknownModelError(
            f"Unknown provider {provider_name!r} in spec {spec!r}"
        )
    if model not in provider_cls.CAPABILITIES:
        raise UnknownModelError(
            f"Unknown model {model!r} for provider {provider_name!r}"
        )
    instance = _provider_instances.get(provider_name)
    if instance is None:
        instance = provider_cls()
        _provider_instances[provider_name] = instance
    return instance, model


async def aclose() -> None:
    """Close every cached provider's owned HTTP client and clear the cache.

    Idempotent — safe to call multiple times. Call this at process
    shutdown to release pooled connections cleanly instead of letting
    Python's garbage collector tear them down with a ``ResourceWarning``.

    Caution:
        Calling :py:func:`aclose` while a :py:meth:`Pipeline.run`
        iteration is still in progress will close provider HTTP clients
        out from under the in-flight steps. Steps that have already
        awaited their final response are fine; steps mid-request will
        observe ``httpx`` raising a closed-client error and emit
        :py:class:`StepFailed`. Drain (or cancel) all active
        ``Pipeline.run`` iterations *before* calling ``aclose`` to
        avoid this footgun.
    """
    for provider in list(_provider_instances.values()):
        await provider.aclose()
    _provider_instances.clear()

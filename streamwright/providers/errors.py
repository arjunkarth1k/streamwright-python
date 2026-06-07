"""Provider-layer error hierarchy.

``retryable`` is a class-level boolean read by the scheduler's retry
helper: ``True`` means the scheduler should re-attempt the step (after
backoff), ``False`` means raise immediately. Concrete adapters that
detect transient failures (rate limits, 5xx) can either subclass
``ProviderError`` with ``retryable = True`` or set the attribute on a
specific instance before raising.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base exception for streamwright provider-layer errors."""

    retryable: bool = False


class UnknownModelError(ProviderError):
    """Raised when a model identifier is not registered for a provider.

    Either the provider prefix is wrong, or the model ID is not in the
    provider's capabilities table.
    """

    retryable: bool = False


class CapabilityError(ProviderError):
    """Raised when a model does not support a requested operation.

    For example, calling ``stream()`` on a model whose capabilities mark
    ``streaming=False``.
    """

    retryable: bool = False

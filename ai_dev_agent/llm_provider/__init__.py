"""Factory helpers for LLM providers."""
from __future__ import annotations

from .base import (
    LLMClient,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMRetryExhaustedError,
    LLMTimeoutError,
    Message,
    RetryConfig,
)
from .deepseek import DeepSeekClient, DEFAULT_BASE_URL

_PROVIDER_MAP = {
    "deepseek": DeepSeekClient,
}


def create_client(provider: str, api_key: str, model: str, base_url: str | None = None) -> LLMClient:
    try:
        client_cls = _PROVIDER_MAP[provider.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported LLM provider: {provider}") from exc
    if provider.lower() == "deepseek" and base_url:
        return client_cls(api_key=api_key, model=model, base_url=base_url)
    return client_cls(api_key=api_key, model=model)


__all__ = [
    "create_client",
    "LLMClient",
    "LLMError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMConnectionError",
    "LLMResponseError",
    "LLMRetryExhaustedError",
    "Message",
    "RetryConfig",
    "DEFAULT_BASE_URL",
]

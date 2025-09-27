"""Abstractions for LLM providers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Protocol, Sequence


@dataclass(frozen=True)
class Message:
    role: str
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: List[Dict[str, Any]] | None = None

    def to_payload(self) -> Dict[str, Any]:
        payload = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        return payload


@dataclass
class StreamHooks:
    """Hooks for streaming responses."""
    on_start: Callable[[], None] | None = None
    on_chunk: Callable[[str], None] | None = None
    on_complete: Callable[[str], None] | None = None
    on_error: Callable[[Exception], None] | None = None


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 0.5
    max_delay: float = 5.0
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.1
    retryable_status_codes: set[int] = field(default_factory=lambda: {429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """Raised when an LLM provider encounters an error."""


class LLMRateLimitError(LLMError):
    """Raised when the provider reports a rate limit condition."""


class LLMTimeoutError(LLMError):
    """Raised when a request times out before the provider responds."""


class LLMConnectionError(LLMError):
    """Raised when the client is unable to reach the provider."""


class LLMResponseError(LLMError):
    """Raised when the provider returns a malformed or error response."""


class LLMRetryExhaustedError(LLMError):
    """Raised when retry attempts are exhausted without success."""


class LLMClient(Protocol):
    """Protocol for chat-completion capable LLM clients."""

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_headers: Dict[str, str] | None = None,
    ) -> str:
        """Complete a chat conversation."""
        ...

    def stream(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_headers: Dict[str, str] | None = None,
        hooks: StreamHooks | None = None,
    ) -> Iterable[str]:
        """Stream a chat conversation with optional hooks."""
        raise NotImplementedError

    def configure_retry(self, retry_config: RetryConfig) -> None:
        """Configure retry behavior."""
        ...

    def configure_timeout(self, timeout: float) -> None:
        """Configure request timeout."""
        ...

    def invoke_tools(
        self,
        messages: Sequence[Message],
        tools: List[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        tool_choice: str | Dict[str, Any] | None = "auto",
        extra_headers: Dict[str, str] | None = None,
    ) -> "ToolCallResult":
        """Run a chat completion with tool definitions and return parsed tool calls."""
        ...


@dataclass
class ToolCall:
    """Single tool/function invocation requested by the model."""

    name: str
    arguments: Dict[str, Any]
    call_id: str | None = None


@dataclass
class ToolCallResult:
    """Parsed outcome of a tool-enabled completion."""

    calls: List[ToolCall]
    message_content: str | None = None
    raw_tool_calls: List[Dict[str, Any]] | None = None


__all__ = [
    "Message",
    "LLMClient",
    "LLMError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMConnectionError",
    "LLMResponseError",
    "LLMRetryExhaustedError",
    "StreamHooks",
    "RetryConfig",
    "ToolCall",
    "ToolCallResult",
]

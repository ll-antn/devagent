"""Utilities for keeping LLM conversations within configured context budgets."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from ai_dev_agent.providers.llm.base import LLMClient, Message, ToolCallResult
from .constants import (
    DEFAULT_KEEP_LAST_ASSISTANT,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_TOOL_MESSAGES,
    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    DEFAULT_RESPONSE_HEADROOM,
)

LOGGER = logging.getLogger(__name__)

@dataclass
class ContextBudgetConfig:
    """Configuration for context pruning and truncation."""

    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    headroom_tokens: int = DEFAULT_RESPONSE_HEADROOM
    max_tool_messages: int = DEFAULT_MAX_TOOL_MESSAGES
    max_tool_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS
    keep_last_assistant: int = DEFAULT_KEEP_LAST_ASSISTANT


def estimate_tokens(messages: Sequence[Message]) -> int:
    """Roughly estimate token usage for a list of messages.

    Uses a simple heuristic (characters/4) plus a small allowance per message.
    Accurate tokenization is not required for safety margins."""

    total = 0
    for msg in messages:
        content = msg.content or ""
        total += len(content) // 4
        if msg.tool_calls:
            total += 16 * len(msg.tool_calls)
        if msg.tool_call_id:
            total += 4
        total += 8  # base allowance per message
    return total


def summarize_text(text: str, max_chars: int) -> str:
    """Return a truncated text summary that notes omitted content."""

    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rstrip()
    omitted = len(text) - len(truncated)
    return f"{truncated}\n[... {omitted} characters omitted ...]"


def _index_of_last_role(messages: Sequence[Message], role: str) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].role == role:
            return idx
    return -1


def prune_messages(messages: Sequence[Message], config: ContextBudgetConfig) -> List[Message]:
    """Produce a pruned copy of messages that fits within the budget."""

    if not messages:
        return []

    pruned = [Message(role=msg.role, content=msg.content, tool_call_id=msg.tool_call_id, tool_calls=msg.tool_calls) for msg in messages]

    total_allowed = max(config.max_tokens - config.headroom_tokens, 0)

    if estimate_tokens(pruned) <= total_allowed:
        return pruned

    # Always retain the first system message and the most recent user message
    keep_indices = set()
    for idx, msg in enumerate(pruned):
        if msg.role == "system":
            keep_indices.add(idx)
            break

    last_user_index = _index_of_last_role(pruned, "user")
    if last_user_index != -1:
        keep_indices.add(last_user_index)

    # Keep the most recent assistant messages
    assistant_indices = [idx for idx, msg in enumerate(pruned) if msg.role == "assistant"]
    for idx in assistant_indices[-config.keep_last_assistant:]:
        keep_indices.add(idx)

    # Summarize tool messages older than the most recent ones
    tool_indices = [idx for idx, msg in enumerate(pruned) if msg.role == "tool"]
    if len(tool_indices) > config.max_tool_messages:
        protected = set(tool_indices[-config.max_tool_messages:])
    else:
        protected = set(tool_indices)

    for idx in tool_indices:
        msg = pruned[idx]
        if idx not in protected or idx not in keep_indices:
            summary = summarize_text(msg.content or "", config.max_tool_output_chars)
            pruned[idx] = Message(role="tool", content=summary, tool_call_id=msg.tool_call_id)
            keep_indices.add(idx)

    # Recalculate and drop oldest non-protected messages while over budget
    ordered_indices = list(range(len(pruned)))
    for idx in ordered_indices:
        if estimate_tokens(pruned) <= total_allowed:
            break
        if idx in keep_indices:
            continue
        pruned[idx] = Message(role="assistant", content="[context truncated]")
        keep_indices.add(idx)

    # Final pass: if still over budget, aggressively trim earliest tool summaries
    for idx in ordered_indices:
        if estimate_tokens(pruned) <= total_allowed:
            break
        msg = pruned[idx]
        if msg.role == "tool":
            pruned[idx] = Message(role="tool", content="[tool output truncated]")

    return pruned


def ensure_context_budget(messages: Iterable[Message], config: ContextBudgetConfig | None = None) -> List[Message]:
    """Return a context-limited version of messages according to configuration."""

    config = config or ContextBudgetConfig()
    messages_list = list(messages)
    pruned = prune_messages(messages_list, config)
    return pruned


def config_from_settings(settings) -> ContextBudgetConfig:
    """Build a ContextBudgetConfig using settings with safe fallbacks."""

    return ContextBudgetConfig(
        max_tokens=getattr(settings, "max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS),
        headroom_tokens=getattr(settings, "response_headroom_tokens", DEFAULT_RESPONSE_HEADROOM),
        max_tool_messages=getattr(settings, "max_tool_messages_kept", DEFAULT_MAX_TOOL_MESSAGES),
        max_tool_output_chars=getattr(settings, "max_tool_output_chars", DEFAULT_MAX_TOOL_OUTPUT_CHARS),
        keep_last_assistant=getattr(settings, "keep_last_assistant_messages", DEFAULT_KEEP_LAST_ASSISTANT),
    )


class BudgetedLLMClient:
    """LLM client wrapper that enforces context budgets before each call."""

    def __init__(self, inner: LLMClient, config: ContextBudgetConfig | None = None, *, disabled: bool = False):
        self._inner = inner
        self._config = config or ContextBudgetConfig()
        self._disabled = disabled

    @property
    def inner(self) -> LLMClient:
        return self._inner

    def configure_retry(self, retry_config):  # type: ignore[override]
        if hasattr(self._inner, "configure_retry"):
            self._inner.configure_retry(retry_config)

    def configure_timeout(self, timeout: float):  # type: ignore[override]
        if hasattr(self._inner, "configure_timeout"):
            self._inner.configure_timeout(timeout)

    def _prepare_messages(self, messages: Sequence[Message]) -> List[Message]:
        original = list(messages)
        if self._disabled:
            return original
        before_tokens = estimate_tokens(original)
        pruned = ensure_context_budget(original, self._config)
        after_tokens = estimate_tokens(pruned)
        if after_tokens < before_tokens or len(pruned) < len(original):
            LOGGER.debug(
                "Context pruned: %s→%s estimated tokens, %s messages kept",
                before_tokens,
                after_tokens,
                len(pruned),
            )
        return pruned

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_headers: dict | None = None,
    ) -> str:
        prepared = self._prepare_messages(messages)
        kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if extra_headers is not None:
            kwargs["extra_headers"] = extra_headers
        return self._inner.complete(prepared, **kwargs)

    def stream(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_headers: dict | None = None,
        hooks=None,
    ):
        prepared = self._prepare_messages(messages)
        kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if extra_headers is not None:
            kwargs["extra_headers"] = extra_headers
        if hooks is not None:
            kwargs["hooks"] = hooks
        return self._inner.stream(prepared, **kwargs)

    def invoke_tools(
        self,
        messages: Sequence[Message],
        tools: List[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        tool_choice: str | dict | None = "auto",
        extra_headers: dict | None = None,
    ) -> ToolCallResult:
        prepared = self._prepare_messages(messages)
        kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tool_choice": tool_choice,
        }
        if extra_headers is not None:
            kwargs["extra_headers"] = extra_headers
        return self._inner.invoke_tools(prepared, tools, **kwargs)

    def set_config(self, config: ContextBudgetConfig):
        self._config = config

    def disable(self) -> None:
        self._disabled = True

    def enable(self) -> None:
        self._disabled = False

    def __getattr__(self, item):
        return getattr(self._inner, item)


__all__ = [
    "ContextBudgetConfig",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_RESPONSE_HEADROOM",
    "DEFAULT_MAX_TOOL_MESSAGES",
    "DEFAULT_MAX_TOOL_OUTPUT_CHARS",
    "DEFAULT_KEEP_LAST_ASSISTANT",
    "BudgetedLLMClient",
    "config_from_settings",
    "estimate_tokens",
    "summarize_text",
    "prune_messages",
    "ensure_context_budget",
]

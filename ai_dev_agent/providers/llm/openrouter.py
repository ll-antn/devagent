"""OpenRouter API client implementation."""
from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, Iterable, List, Sequence

import requests

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
    StreamHooks,
    ToolCall,
    ToolCallResult,
)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(LLMClient):
    """Chat-completions client for the OpenRouter API with retry and tool support."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        retry_config: RetryConfig | None = None,
        provider_only: Sequence[str] | None = None,
        provider_config: Dict[str, Any] | None = None,
        default_headers: Dict[str, str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self._default_headers = dict(default_headers or {})
        self._provider_config = self._merge_provider_config(provider_only, provider_config)

    @staticmethod
    def _merge_provider_config(
        provider_only: Sequence[str] | None,
        provider_config: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        config: Dict[str, Any] = dict(provider_config or {})
        if provider_only:
            config = {**config, "only": list(provider_only)}
        return config

    def configure_retry(self, retry_config: RetryConfig) -> None:
        """Configure retry behavior."""
        self.retry_config = retry_config

    def configure_timeout(self, timeout: float) -> None:
        """Configure request timeout."""
        self.timeout = timeout

    def _build_headers(self, extra_headers: Dict[str, str] | None = None) -> Dict[str, str]:
        """Construct request headers, merging defaults and per-call values."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self._default_headers:
            headers.update(self._default_headers)
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _calculate_delay(self, attempt: int) -> float:
        """Return an exponential backoff delay with jitter."""
        base_delay = min(
            self.retry_config.max_delay,
            self.retry_config.initial_delay * (self.retry_config.backoff_multiplier ** (attempt - 1)),
        )
        if base_delay <= 0:
            return 0.0
        jitter_ratio = max(0.0, self.retry_config.jitter_ratio)
        if jitter_ratio == 0:
            return base_delay
        jitter_span = base_delay * jitter_ratio
        lower = max(0.0, base_delay - jitter_span)
        upper = base_delay + jitter_span
        return random.uniform(lower, upper)

    def _error_from_status(self, status_code: int, response_text: str) -> LLMError:
        message = f"OpenRouter API error {status_code}: {response_text}"
        if status_code == 429:
            return LLMRateLimitError(message)
        if status_code in {408, 504}:
            return LLMTimeoutError(message)
        if status_code in {502, 503}:
            return LLMConnectionError(message)
        return LLMResponseError(message)

    def _apply_provider_config(self, payload: Dict[str, Any]) -> None:
        """Attach provider selection hints when configured."""
        if self._provider_config:
            payload["provider"] = self._provider_config

    def _post(self, payload: Dict[str, Any], extra_headers: Dict[str, str] | None = None) -> Dict[str, Any]:
        """Execute a POST request with retry logic."""
        url = f"{self.base_url}/chat/completions"
        headers = self._build_headers(extra_headers)
        last_error: LLMError | None = None

        for attempt in range(1, self.retry_config.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )

                if response.status_code in self.retry_config.retryable_status_codes:
                    error = self._error_from_status(response.status_code, response.text)
                    last_error = error
                    if attempt == self.retry_config.max_retries:
                        raise LLMRetryExhaustedError(
                            f"OpenRouter request exhausted retries: {error}"
                        ) from error
                    time.sleep(self._calculate_delay(attempt))
                    continue

                if response.status_code >= 400:
                    raise self._error_from_status(response.status_code, response.text)

                return response.json()

            except (requests.Timeout, requests.ConnectionError) as exc:
                if isinstance(exc, requests.Timeout):
                    last_error = LLMTimeoutError(f"OpenRouter request timed out: {exc}")
                else:
                    last_error = LLMConnectionError(f"OpenRouter connection failed: {exc}")
                if attempt == self.retry_config.max_retries:
                    raise last_error from exc
                time.sleep(self._calculate_delay(attempt))
            except requests.RequestException as exc:
                raise LLMResponseError(f"OpenRouter request failed: {exc}") from exc
            except json.JSONDecodeError as exc:  # pragma: no cover - network failure edge case
                raise LLMResponseError("Invalid JSON response from OpenRouter API") from exc

        if last_error is None:
            message = "OpenRouter request failed for an unknown reason"
            raise LLMRetryExhaustedError(message)
        raise LLMRetryExhaustedError(
            f"OpenRouter request failed after {self.retry_config.max_retries} attempts: {last_error}"
        ) from last_error

    def _prepare_payload(
        self,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int | None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_payload() for message in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        self._apply_provider_config(payload)
        return payload

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_headers: Dict[str, str] | None = None,
    ) -> str:
        payload = self._prepare_payload(messages, temperature, max_tokens)
        data = self._post(payload, extra_headers=extra_headers)
        try:
            content = data["choices"][0]["message"].get("content")
        except (KeyError, IndexError) as exc:
            raise LLMError("Unexpected OpenRouter response structure") from exc
        if content is None:
            return ""
        return content.strip()

    def stream(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_headers: Dict[str, str] | None = None,
        hooks: StreamHooks | None = None,
    ) -> Iterable[str]:  # pragma: no cover - streaming not part of current tests
        payload = self._prepare_payload(messages, temperature, max_tokens)
        payload["stream"] = True
        url = f"{self.base_url}/chat/completions"
        headers = self._build_headers(extra_headers)
        accumulated: List[str] = []
        try:
            if hooks and hooks.on_start:
                hooks.on_start()
            with requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
                stream=True,
            ) as response:
                if response.status_code >= 400:
                    raise self._error_from_status(response.status_code, response.text)
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue
                    chunk = line[len("data: "):]
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                        delta = data["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if not delta:
                        continue
                    accumulated.append(delta)
                    if hooks and hooks.on_chunk:
                        hooks.on_chunk(delta)
                    yield delta
        except Exception as exc:
            if hooks and hooks.on_error and isinstance(exc, Exception):
                hooks.on_error(exc)
            raise
        else:
            if hooks and hooks.on_complete:
                hooks.on_complete("".join(accumulated))

    def invoke_tools(
        self,
        messages: Sequence[Message],
        tools: List[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        tool_choice: str | Dict[str, Any] | None = "auto",
        extra_headers: Dict[str, str] | None = None,
    ) -> ToolCallResult:
        payload = self._prepare_payload(messages, temperature, max_tokens)
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
        data = self._post(payload, extra_headers=extra_headers)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError("Unexpected OpenRouter response structure for tool call") from exc

        tool_calls_raw = message.get("tool_calls") or []
        parsed_calls: List[ToolCall] = []
        for call in tool_calls_raw:
            function_data = call.get("function", {}) or {}
            name = function_data.get("name") or ""
            raw_args = function_data.get("arguments") or "{}"
            if isinstance(raw_args, dict):
                arguments: Dict[str, Any] = raw_args
            else:
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError:
                    arguments = {}
            parsed_calls.append(
                ToolCall(
                    name=name,
                    arguments=arguments,
                    call_id=call.get("id"),
                )
            )

        return ToolCallResult(
            calls=parsed_calls,
            message_content=message.get("content"),
            raw_tool_calls=tool_calls_raw or None,
        )


__all__ = ["OpenRouterClient", "DEFAULT_BASE_URL"]

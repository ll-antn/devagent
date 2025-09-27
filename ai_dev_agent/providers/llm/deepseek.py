"""DeepSeek API client implementation."""
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

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekClient(LLMClient):
    """Enhanced DeepSeek chat-completions client with retry, timeout, and streaming support."""

    def __init__(
        self, 
        api_key: str, 
        model: str, 
        base_url: str = DEFAULT_BASE_URL, 
        timeout: float = 120.0, 
        retry_config: RetryConfig | None = None
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()

    def configure_retry(self, retry_config: RetryConfig) -> None:
        """Configure retry behavior."""
        self.retry_config = retry_config

    def configure_timeout(self, timeout: float) -> None:
        """Configure request timeout."""
        self.timeout = timeout

    def _build_headers(self, extra_headers: Dict[str, str] | None = None) -> Dict[str, str]:
        """Build request headers with optional extras."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _calculate_delay(self, attempt: int) -> float:
        """Return the next backoff delay including jitter."""
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
        message = f"DeepSeek API error {status_code}: {response_text}"
        if status_code == 429:
            return LLMRateLimitError(message)
        return LLMResponseError(message)

    def _post(self, payload: dict, extra_headers: Dict[str, str] | None = None) -> dict:
        """Make a POST request with retry logic."""
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
                
                # Check if status code is retryable
                if response.status_code in self.retry_config.retryable_status_codes:
                    error = self._error_from_status(response.status_code, response.text)
                    last_error = error
                    if attempt == self.retry_config.max_retries:
                        raise LLMRetryExhaustedError(
                            f"DeepSeek request exhausted retries: {error}"
                        ) from error
                    time.sleep(self._calculate_delay(attempt))
                    continue

                if response.status_code >= 400:
                    raise self._error_from_status(response.status_code, response.text)
                return response.json()

            except (requests.Timeout, requests.ConnectionError) as exc:
                if isinstance(exc, requests.Timeout):
                    last_error = LLMTimeoutError(f"DeepSeek request timed out: {exc}")
                else:
                    last_error = LLMConnectionError(f"DeepSeek connection failed: {exc}")
                if attempt == self.retry_config.max_retries:
                    raise last_error from exc
                time.sleep(self._calculate_delay(attempt))
            except requests.RequestException as exc:
                raise LLMResponseError(f"DeepSeek request failed: {exc}") from exc
            except json.JSONDecodeError as exc:  # pragma: no cover - network failure edge case
                raise LLMResponseError("Invalid JSON response from DeepSeek API") from exc

        if last_error is None:
            message = "DeepSeek request failed for an unknown reason"
            raise LLMRetryExhaustedError(message)
        raise LLMRetryExhaustedError(
            f"DeepSeek request failed after {self.retry_config.max_retries} attempts: {last_error}"
        ) from last_error

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [message.to_payload() for message in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        data = self._post(payload)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError("Unexpected DeepSeek response structure") from exc

    def stream(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterable[str]:  # pragma: no cover - streaming not part of PoC tests
        payload = {
            "model": self.model,
            "messages": [message.to_payload() for message in messages],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with requests.post(url, headers=headers, data=json.dumps(payload), stream=True, timeout=60) as response:
            if response.status_code >= 400:
                raise LLMError(f"DeepSeek API error {response.status_code}: {response.text}")
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    chunk = line[len("data: "):]
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                        delta = data["choices"][0]["delta"].get("content")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

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
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_payload() for message in messages],
            "temperature": temperature,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        data = self._post(payload, extra_headers=extra_headers)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError("Unexpected DeepSeek response structure for tool call") from exc

        tool_calls_raw = message.get("tool_calls") or []
        parsed_calls: List[ToolCall] = []
        for call in tool_calls_raw:
            function_data = call.get("function", {}) or {}
            name = function_data.get("name") or ""
            raw_args = function_data.get("arguments") or "{}"
            arguments: Dict[str, Any]
            if isinstance(raw_args, dict):
                arguments = raw_args  # already parsed
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


__all__ = ["DeepSeekClient", "DEFAULT_BASE_URL"]

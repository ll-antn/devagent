import pytest

from ai_dev_agent.llm_provider.base import (
    LLMRateLimitError,
    LLMResponseError,
    Message,
    RetryConfig,
)
from ai_dev_agent.llm_provider.deepseek import DeepSeekClient


def test_deepseek_backoff_jitter_range(monkeypatch):
    config = RetryConfig(initial_delay=1.0, max_delay=10.0, backoff_multiplier=2.0, jitter_ratio=0.25)
    client = DeepSeekClient(api_key="test", model="demo", retry_config=config)

    captured = {}

    def fake_uniform(low: float, high: float) -> float:
        captured["low"] = low
        captured["high"] = high
        return high

    monkeypatch.setattr("ai_dev_agent.llm_provider.deepseek.random.uniform", fake_uniform)

    delay = client._calculate_delay(3)

    assert delay == pytest.approx(5.0)
    assert captured["low"] == pytest.approx(3.0)
    assert captured["high"] == pytest.approx(5.0)


def test_deepseek_error_mapping():
    client = DeepSeekClient(api_key="test", model="demo")

    rate_error = client._error_from_status(429, "Too Many Requests")
    server_error = client._error_from_status(500, "Server error")

    assert isinstance(rate_error, LLMRateLimitError)
    assert isinstance(server_error, LLMResponseError)


def test_deepseek_tool_call_parsing(monkeypatch):
    client = DeepSeekClient(api_key="test", model="demo")

    payload_captured = {}

    def fake_post(payload, extra_headers=None):
        payload_captured["tools"] = payload.get("tools")
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "list_directory",
                                    "arguments": "{\n  \"path\": \".\"\n}",
                                },
                            }
                        ],
                        "content": "Listing current directory",
                    }
                }
            ]
        }

    monkeypatch.setattr(client, "_post", fake_post)

    result = client.invoke_tools(
        [Message(role="user", content="list files")],
        tools=[{"type": "function", "function": {"name": "list_directory", "parameters": {"type": "object"}}}],
        temperature=0.0,
    )

    assert payload_captured["tools"][0]["function"]["name"] == "list_directory"
    assert result.message_content == "Listing current directory"
    assert len(result.calls) == 1
    assert result.calls[0].name == "list_directory"
    assert result.calls[0].arguments == {"path": "."}

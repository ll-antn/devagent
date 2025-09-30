"""Tests for shell conversation history management in the interactive query flow."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable, List, Optional

import click

from ai_dev_agent.cli.react.executor import _execute_react_assistant
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.providers.llm.base import Message, ToolCall, ToolCallResult
from ai_dev_agent.cli.handlers import registry_handlers


class FakeClient:
    """Minimal LLM client stub that records messages passed to invoke_tools."""

    def __init__(self, responses: Iterable[ToolCallResult]) -> None:
        self._responses: List[ToolCallResult] = list(responses)
        self.last_messages: List[Message] | None = None
        self.invocations: List[List[Message]] = []

    def invoke_tools(
        self,
        messages: Iterable[Message],
        *,
        tools: List[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        tool_choice: str | dict | None = "auto",
        extra_headers: dict | None = None,
    ) -> ToolCallResult:
        captured = list(messages)
        self.last_messages = captured
        self.invocations.append(captured)
        if self._responses:
            return self._responses.pop(0)
        return ToolCallResult(calls=[], message_content="", raw_tool_calls=None)


def _make_context(settings: Settings) -> click.Context:
    ctx = click.Context(click.Command("devagent"))
    ctx.obj = {
        "settings": settings,
        "_shell_conversation_history": [],
        "_shell_session_manager": None,
        "_shell_session_id": None,
        "_structure_hints_state": {"symbols": set(), "files": {}, "project_summary": None},
        "_detected_language": "python",
        "_repo_file_count": 5,
        "_project_structure_summary": "Stub structure",
        "devagent_config": SimpleNamespace(react_iteration_global_cap=None),
    }
    return ctx


def test_shell_conversation_history_persists_between_queries(capsys) -> None:
    settings = Settings()
    settings.keep_last_assistant_messages = 4

    client = FakeClient(
        [
            ToolCallResult(calls=[], message_content="There are 1,369 files.", raw_tool_calls=None),
            ToolCallResult(
                calls=[],
                message_content="72,650 is higher than 1,369.",
                raw_tool_calls=None,
            ),
        ]
    )
    ctx = _make_context(settings)

    _execute_react_assistant(ctx, client, settings, "How many files are in this project?", use_planning=False)

    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assert len(history) == 2
    assert history[0].role == "user"
    assert "How many files" in history[0].content
    assert history[1].role == "assistant"
    assert "1,369" in (history[1].content or "")

    _execute_react_assistant(ctx, client, settings, "Which number is higher?", use_planning=False)

    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assert len(history) == 4
    assistant_responses = [msg.content for msg in history if msg.role == "assistant"]
    assert "72,650 is higher" in assistant_responses[-1]

    assert client.last_messages is not None
    user_contents = [msg.content for msg in client.last_messages if msg.role == "user"]
    assert any("How many files" in content for content in user_contents)
    assert any("Which number is higher" in content for content in user_contents)


def test_shell_conversation_history_respects_limit(capsys) -> None:
    settings = Settings()
    settings.keep_last_assistant_messages = 1

    client = FakeClient(
        [
            ToolCallResult(calls=[], message_content="First answer", raw_tool_calls=None),
            ToolCallResult(calls=[], message_content="Second answer", raw_tool_calls=None),
            ToolCallResult(calls=[], message_content="Third answer", raw_tool_calls=None),
        ]
    )
    ctx = _make_context(settings)

    _execute_react_assistant(ctx, client, settings, "Question one?", use_planning=False)
    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assert len(history) == 2

    _execute_react_assistant(ctx, client, settings, "Question two?", use_planning=False)
    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assert len(history) == 2
    assert history[0].content == "Question two?"
    assert history[1].content == "Second answer"

    _execute_react_assistant(ctx, client, settings, "Question three?", use_planning=False)
    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assert len(history) == 2
    assert history[0].content == "Question three?"
    assert history[1].content == "Third answer"


def test_shell_history_ignores_tool_intermediate_assistant(monkeypatch, capsys) -> None:
    settings = Settings()
    settings.keep_last_assistant_messages = 4

    client = FakeClient(
        [
            ToolCallResult(
                calls=[ToolCall(name="fake_tool", arguments={}, call_id="call-1")],
                message_content="Calling tool",
                raw_tool_calls=[{"id": "call-1", "type": "function"}],
            ),
            ToolCallResult(
                calls=[],
                message_content="Tool result summarised.",
                raw_tool_calls=None,
            ),
            ToolCallResult(
                calls=[],
                message_content="Follow-up answer using earlier info.",
                raw_tool_calls=None,
            ),
        ]
    )

    ctx = _make_context(settings)

    def fake_handler(_ctx, _arguments):
        print("tool executed")

    monkeypatch.setitem(registry_handlers.INTENT_HANDLERS, "fake_tool", fake_handler)

    _execute_react_assistant(ctx, client, settings, "Run fake tool", use_planning=False)

    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assistant_messages = [msg.content for msg in history if msg.role == "assistant"]
    assert assistant_messages == ["Tool result summarised."]

    _execute_react_assistant(ctx, client, settings, "Summarise result", use_planning=False)

    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assistant_messages = [msg.content for msg in history if msg.role == "assistant"]
    assert assistant_messages == [
        "Tool result summarised.",
        "Follow-up answer using earlier info.",
    ]

    # Ensure both user prompts are present in payload of final invocation
    user_prompts = [msg.content for msg in client.invocations[-1] if msg.role == "user"]
    assert any("Run fake tool" in prompt for prompt in user_prompts)
    assert any("Summarise result" in prompt for prompt in user_prompts)


def test_shell_history_records_user_without_response(capsys) -> None:
    settings = Settings()
    client = FakeClient([ToolCallResult(calls=[], message_content=None, raw_tool_calls=None)])
    ctx = _make_context(settings)

    _execute_react_assistant(ctx, client, settings, "No answer?", use_planning=False)

    history = ctx.obj.get("_shell_conversation_history")
    assert history is not None
    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].content == "No answer?"

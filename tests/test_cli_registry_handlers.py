"""Tests for CLI registry handlers."""
from __future__ import annotations

import click
import pytest

from ai_dev_agent.cli.handlers.registry_handlers import _build_code_search_payload
from ai_dev_agent.core.utils.config import Settings


def _make_context() -> click.Context:
    ctx = click.Context(click.Command("code.search"))
    ctx.obj = {"settings": Settings()}
    return ctx


@pytest.mark.parametrize(
    "query",
    [
        "Compile.*method",
        r"Compile\w+Method",
        r"[A-Z]+Manager",
        r"(?P<name>Task)",
        r"^TaskManager",
        r"TaskQueue$",
    ],
)
def test_code_search_auto_regex_triggers(query: str) -> None:
    ctx = _make_context()
    payload, _ = _build_code_search_payload(ctx, {"query": query})

    assert payload["regex"] is True


@pytest.mark.parametrize(
    "query",
    [
        "def greet",
        "main.py",
        "TaskManager::CreateTaskQueue",
    ],
)
def test_code_search_auto_regex_not_set_for_plain_text(query: str) -> None:
    ctx = _make_context()
    payload, _ = _build_code_search_payload(ctx, {"query": query})

    assert "regex" not in payload

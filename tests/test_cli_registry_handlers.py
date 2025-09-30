"""Tests for CLI registry handlers."""
from __future__ import annotations

import click

from ai_dev_agent.cli.handlers.registry_handlers import _build_code_search_payload
from ai_dev_agent.core.utils.config import Settings


def _make_context() -> click.Context:
    ctx = click.Context(click.Command("code.search"))
    ctx.obj = {"settings": Settings()}
    return ctx


def test_code_search_auto_regex_with_dot_star() -> None:
    ctx = _make_context()
    payload, _ = _build_code_search_payload(ctx, {"query": "Compile.*method"})

    assert payload["regex"] is True


def test_code_search_auto_regex_not_set_for_plain_text() -> None:
    ctx = _make_context()
    payload, _ = _build_code_search_payload(ctx, {"query": "def greet"})

    assert "regex" not in payload

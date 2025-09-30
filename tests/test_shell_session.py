"""Tests covering shell session persistence and integration."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.tools import ToolContext, registry as tool_registry
from ai_dev_agent.tools.execution.shell_session import ShellSessionManager


def _select_shell() -> str:
    candidates = [os.environ.get("SHELL"), "bash", "zsh", "sh"]
    for candidate in candidates:
        if not candidate:
            continue
        path = shutil.which(candidate) if os.path.sep not in candidate else candidate
        if path and Path(path).exists():
            return path
    pytest.skip("No suitable shell executable available for shell session tests")
    return ""


def _make_context(root: Path, manager: ShellSessionManager, session_id: str) -> ToolContext:
    settings = Settings(workspace_root=root)
    return ToolContext(
        repo_root=root,
        settings=settings,
        sandbox=None,
        devagent_config=None,
        metrics_collector=None,
        extra={
            "shell_session_manager": manager,
            "shell_session_id": session_id,
        },
    )


def test_shell_session_persists_state(tmp_path: Path) -> None:
    shell_path = _select_shell()
    manager = ShellSessionManager(shell=shell_path)
    session_id = manager.create_session(cwd=tmp_path)

    try:
        result = manager.execute(session_id, "pwd")
        assert str(tmp_path) in result.stdout.strip()

        manager.execute(session_id, "mkdir nested_dir")
        manager.execute(session_id, "cd nested_dir")
        nested_result = manager.execute(session_id, "pwd")
        assert nested_result.stdout.strip().endswith("nested_dir")
    finally:
        manager.close_all()


def test_exec_tool_uses_persistent_shell(tmp_path: Path) -> None:
    shell_path = _select_shell()
    manager = ShellSessionManager(shell=shell_path)
    session_id = manager.create_session(cwd=tmp_path)
    context = _make_context(tmp_path, manager, session_id)

    try:
        initial = tool_registry.invoke("exec", {"cmd": "pwd"}, context)
        assert str(tmp_path) in initial["stdout_tail"].strip()

        nested = tmp_path / "persist"
        nested.mkdir()
        tool_registry.invoke("exec", {"cmd": "cd", "args": ["persist"]}, context)
        follow_up = tool_registry.invoke("exec", {"cmd": "pwd"}, context)
        assert str(nested) in follow_up["stdout_tail"].strip()
    finally:
        manager.close_all()

"""Tests for CLI utilities module."""
from __future__ import annotations

import platform
from pathlib import Path

import pytest

from ai_dev_agent.cli.utils import build_system_context


def test_build_system_context():
    """Test build_system_context returns expected structure."""
    ctx = build_system_context()

    # Check required keys exist
    assert "os" in ctx
    assert "os_friendly" in ctx
    assert "os_version" in ctx
    assert "architecture" in ctx
    assert "shell" in ctx
    assert "cwd" in ctx
    assert "home_dir" in ctx
    assert "python_version" in ctx
    assert "shell_type" in ctx
    assert "path_separator" in ctx
    assert "command_separator" in ctx
    assert "null_device" in ctx
    assert "temp_dir" in ctx
    assert "available_tools" in ctx
    assert "command_mappings" in ctx
    assert "platform_examples" in ctx

    # Check types
    assert isinstance(ctx["os"], str)
    assert isinstance(ctx["os_friendly"], str)
    assert isinstance(ctx["cwd"], str)
    assert isinstance(ctx["available_tools"], list)
    assert isinstance(ctx["command_mappings"], dict)

    # Check platform-specific values
    if platform.system() == "Darwin":
        assert ctx["os"] == "Darwin"
        assert ctx["os_friendly"] == "macOS"
        assert ctx["shell_type"] == "unix"
        assert ctx["path_separator"] == "/"
        assert ctx["command_separator"] == "&&"
        assert ctx["null_device"] == "/dev/null"
    elif platform.system() == "Linux":
        assert ctx["os"] == "Linux"
        assert ctx["os_friendly"] == "Linux"
        assert ctx["shell_type"] == "unix"
    elif platform.system() == "Windows":
        assert ctx["os"] == "Windows"
        assert ctx["shell_type"] == "windows"
        assert ctx["path_separator"] == "\\"


def test_build_system_context_command_mappings():
    """Test command mappings are present."""
    ctx = build_system_context()
    mappings = ctx["command_mappings"]

    # Check common commands exist
    assert "list_files" in mappings
    assert "find_files" in mappings
    assert "copy" in mappings
    assert "move" in mappings
    assert "delete" in mappings
    assert "open_file" in mappings

    # Values should be strings
    for key, value in mappings.items():
        assert isinstance(value, str)
        assert len(value) > 0


def test_build_system_context_available_tools():
    """Test available tools detection."""
    ctx = build_system_context()
    tools = ctx["available_tools"]

    # Should be a list (may be empty if no tools installed)
    assert isinstance(tools, list)

    # If git is in the list, it should actually be available
    if "git" in tools:
        import shutil
        assert shutil.which("git") is not None

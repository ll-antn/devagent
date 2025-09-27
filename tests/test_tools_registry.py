"""Tests for registry-backed tools."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ai_dev_agent.tools import ToolContext, registry as tool_registry  # noqa: F401
from ai_dev_agent.core.utils.config import Settings


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)


def _make_context(root: Path, sandbox=None) -> ToolContext:
    settings = Settings(workspace_root=root)
    return ToolContext(
        repo_root=root,
        settings=settings,
        sandbox=sandbox,
        devagent_config=None,
        metrics_collector=None,
        extra={},
    )


def test_fs_read_and_write(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    target = tmp_path / "hello.py"
    target.write_text("print('hello')\n", encoding="utf-8")
    subprocess.run(["git", "add", "hello.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)

    ctx = _make_context(tmp_path)

    read_result = tool_registry.invoke("fs.read", {"paths": ["hello.py"]}, ctx)
    assert read_result["files"][0]["content"].startswith("print"), read_result

    diff = """diff --git a/hello.py b/hello.py
--- a/hello.py
+++ b/hello.py
@@ -1 +1 @@
-print('hello')
+print('world')
"""
    apply_result = tool_registry.invoke("fs.write_patch", {"diff": diff}, ctx)
    assert apply_result["applied"] is True
    assert apply_result["diff_stats"]["lines"] == 2
    assert (tmp_path / "hello.py").read_text(encoding="utf-8").strip() == "print('world')"


def test_code_search(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    module = tmp_path / "module.py"
    module.write_text("def react_loop():\n    return 'ok'\n", encoding="utf-8")
    subprocess.run(["git", "add", "module.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add module"], cwd=tmp_path, check=True)

    ctx = _make_context(tmp_path)
    result = tool_registry.invoke("code.search", {"query": "react_loop"}, ctx)
    assert result["matches"], "expected at least one match"
    assert result["matches"][0]["path"].endswith("module.py")


def _has_universal_ctags() -> bool:
    path = shutil.which("ctags")
    if not path:
        return False
    proc = subprocess.run([path, "--version"], capture_output=True, text=True)
    return "Universal Ctags" in proc.stdout


@pytest.mark.skipif(not _has_universal_ctags(), reason="Universal Ctags not available")
def test_symbols_index_and_find(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    target = tmp_path / "lib.py"
    target.write_text("""def sample():\n    return 1\n""", encoding="utf-8")
    subprocess.run(["git", "add", "lib.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add lib"], cwd=tmp_path, check=True)

    ctx = _make_context(tmp_path)
    idx_result = tool_registry.invoke("symbols.index", {}, ctx)
    assert idx_result["stats"]["symbols"] >= 1

    find_result = tool_registry.invoke("symbols.find", {"name": "sample"}, ctx)
    assert any(match["path"].endswith("lib.py") for match in find_result["defs"])


def test_ast_query(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    path = tmp_path / "sample.py"
    path.write_text("""def alpha():\n    pass\n""", encoding="utf-8")
    subprocess.run(["git", "add", "sample.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add sample"], cwd=tmp_path, check=True)

    ctx = _make_context(tmp_path)
    query = "(function_definition name: (identifier) @id)"
    result = tool_registry.invoke("ast.query", {"path": "sample.py", "query": query}, ctx)
    assert result["nodes"], "Expected AST nodes from query"


def test_exec_tool(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    result = tool_registry.invoke("exec", {"cmd": "echo", "args": ["hello"]}, ctx)
    assert result["exit_code"] == 0
    assert "hello" in result["stdout_tail"]


def test_security_secrets_scan(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    secret_file = tmp_path / "config.txt"
    secret_file.write_text("api_key=AKIA1234567890ABCDEF\n", encoding="utf-8")
    subprocess.run(["git", "add", "config.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add config"], cwd=tmp_path, check=True)

    ctx = _make_context(tmp_path)
    result = tool_registry.invoke("security.secrets_scan", {"paths": ["config.txt"]}, ctx)
    assert result["findings"] >= 1
    report = tmp_path / result["report_path"]
    assert report.is_file()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["findings"], "Report should contain findings"

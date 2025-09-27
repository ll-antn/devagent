from pathlib import Path

import pytest

from ai_dev_agent.tools.code.code_edit.context import ContextGatherer


def test_context_gatherer_includes_structure_summary(tmp_path):
    repo_root = Path(tmp_path)
    source = repo_root / "sample.py"
    source.write_text(
        """
class Demo:
    def method(self, value: int) -> int:
        return value * 2


def helper(value: int) -> int:
    return value + 1
""".strip()
    )

    gatherer = ContextGatherer(repo_root)
    if not getattr(gatherer, "_structure_analyzer", None) or not gatherer._structure_analyzer.available:
        pytest.skip("tree-sitter not available in test environment")

    contexts = gatherer.gather_contexts(["sample.py"])

    summary_context = next((ctx for ctx in contexts if ctx.reason == "project_structure_summary"), None)
    assert summary_context is not None, "Expected tree-sitter project summary context"
    assert "Project Structure (Tree-sitter)" in summary_context.content
    assert "class Demo" in summary_context.content
    assert "function helper" in summary_context.content

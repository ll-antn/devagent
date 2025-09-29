import textwrap

import pytest

from ai_dev_agent.tools.code.code_edit.context import ContextGatherer, ContextGatheringOptions


def test_gather_contexts_discovers_related_test_files(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src/pkg").mkdir(parents=True)
    (repo / "tests").mkdir()

    (repo / "src/pkg/__init__.py").write_text("", encoding="utf-8")
    (repo / "src/pkg/module.py").write_text(
        textwrap.dedent(
            '''
            def foo():
                """Return a sentinel value."""
                return 42
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (repo / "tests/test_module.py").write_text(
        textwrap.dedent(
            '''
            from src.pkg.module import foo


            def test_foo_returns_expected_value():
                assert foo() == 42
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    options = ContextGatheringOptions(
        include_related_files=True,
        include_tests=True,
        include_docs=False,
        include_structure_summary=False,
        max_files=5,
    )
    gatherer = ContextGatherer(repo, options)

    contexts = gatherer.gather_contexts(
        ["src/pkg/module.py"],
        task_description="Add tests for foo in module.py",
        keywords=["foo"],
    )

    rel_paths = {
        ctx.path.relative_to(repo).as_posix(): ctx.reason for ctx in contexts
    }

    assert "src/pkg/module.py" in rel_paths
    assert "tests/test_module.py" in rel_paths
    assert any("test" in reason or reason.startswith("keyword") for reason in rel_paths.values())


def test_structure_summary_covers_c_sources_when_tree_sitter_available(tmp_path):
    pytest.importorskip("tree_sitter_languages")

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)

    (repo / "src/foo.c").write_text(
        """
        int square(int value) {
            return value * value;
        }
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    options = ContextGatheringOptions(
        include_related_files=False,
        include_structure_summary=True,
        max_files=5,
    )
    gatherer = ContextGatherer(repo, options)

    contexts = gatherer.gather_contexts(["src/foo.c"])

    summary = next((ctx for ctx in contexts if ctx.reason == "project_structure_summary"), None)
    assert summary is not None
    assert "function int square(int value)" in summary.content


def test_context_gatherer_includes_structure_outline_and_symbols(tmp_path):
    pytest.importorskip("tree_sitter_languages")

    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)

    (repo / "pkg/sample.py").write_text(
        textwrap.dedent(
            """
            class Alpha:
                def beta(self):
                    return 1


            def gamma():
                return 2
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    options = ContextGatheringOptions(
        include_related_files=False,
        include_structure_summary=False,
        max_files=3,
    )
    gatherer = ContextGatherer(repo, options)

    contexts = gatherer.gather_contexts(["pkg/sample.py"], task_description="Investigate Alpha", keywords=["Alpha"])

    sample = next(ctx for ctx in contexts if ctx.path.relative_to(repo).as_posix() == "pkg/sample.py")
    assert sample.structure_outline
    assert any("class Alpha" in line for line in sample.structure_outline)
    assert {symbol.lower() for symbol in sample.symbols} >= {"alpha", "beta", "gamma"}


def test_symbol_references_discover_consumers(tmp_path):
    pytest.importorskip("tree_sitter_languages")

    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)

    (repo / "pkg/library.py").write_text(
        """
        def helper():
            return 1
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    (repo / "pkg/consumer.py").write_text(
        """
        from .library import helper


        def run():
            return helper()
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    options = ContextGatheringOptions(
        include_related_files=True,
        include_structure_summary=False,
        max_files=5,
    )
    gatherer = ContextGatherer(repo, options)

    contexts = gatherer.gather_contexts(["pkg/library.py"], task_description="Update helper", keywords=None)

    rel_paths = {ctx.path.relative_to(repo).as_posix(): ctx.reason for ctx in contexts}
    assert "pkg/library.py" in rel_paths
    assert "pkg/consumer.py" in rel_paths
    assert any(key in rel_paths["pkg/consumer.py"] for key in ("symbol_reference", "sibling"))

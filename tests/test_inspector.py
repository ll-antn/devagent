from pathlib import Path

from ai_dev_agent.questions.inspector import RepositoryInspector


def test_inspector_finds_root_extension(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("info", encoding="utf-8")

    inspector = RepositoryInspector(tmp_path, max_index_size=50)
    suggestions = inspector.suggest_files(
        "Check the root directory for TOML configuration files.", limit=5
    )

    assert "pyproject.toml" in suggestions


def test_inspector_handles_directory_mentions(tmp_path: Path) -> None:
    adr_dir = tmp_path / "docs" / "adr"
    adr_dir.mkdir(parents=True, exist_ok=True)
    sample = adr_dir / "ADR-0001-sample.md"
    sample.write_text("ADR", encoding="utf-8")

    inspector = RepositoryInspector(tmp_path, max_index_size=50)
    suggestions = inspector.suggest_files(
        "Review docs/adr/ for recent decisions.", limit=5
    )

    assert any(path.startswith("docs/adr/ADR-0001") for path in suggestions)


def test_inspector_keyword_matches(tmp_path: Path) -> None:
    src_dir = tmp_path / "ai_dev_agent"
    src_dir.mkdir()
    calc_file = src_dir / "calc_module.py"
    calc_file.write_text("value = 1\n", encoding="utf-8")

    inspector = RepositoryInspector(tmp_path, max_index_size=50)
    suggestions = inspector.suggest_files(
        "Explain how the calc module works.", limit=5
    )

    assert "ai_dev_agent/calc_module.py" in suggestions

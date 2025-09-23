from pathlib import Path

from ai_dev_agent.adr.adr_manager import ADRManager
from ai_dev_agent.llm_provider.base import LLMError


class StaticClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, messages, temperature=0.4):
        return self.response


class FailingClient:
    def complete(self, messages, temperature=0.4):
        raise LLMError("Service unavailable")


def _write_template(tmp_path: Path) -> Path:
    template = tmp_path / "template.md"
    template.write_text(
        "# {title}\n\n## Status\n{status}\n\n## Context\n{context}\n\n"
        "## Decision\n{decision}\n\n## Consequences\n{consequences}\n",
        encoding="utf-8",
    )
    return template


def test_adr_manager_generate_success(tmp_path: Path) -> None:
    template = _write_template(tmp_path)
    output = tmp_path / "project"
    output.mkdir()

    client = StaticClient("# Decision\n\n## Status\nApproved\n")
    manager = ADRManager(output, client, template)

    result = manager.generate("Improve system", "Adopt tool", "Use new tool")

    assert result.fallback_reason is None
    assert result.path.exists()
    content = result.path.read_text(encoding="utf-8")
    assert "Approved" in content


def test_adr_manager_generate_fallback(tmp_path: Path) -> None:
    template = _write_template(tmp_path)
    output = tmp_path / "repo"
    output.mkdir()

    manager = ADRManager(output, FailingClient(), template)

    result = manager.generate("Improve system", "Adopt tool", "Use new tool")

    assert result.fallback_reason == "Service unavailable"
    assert result.path.exists()
    assert "Decision pending" in result.content

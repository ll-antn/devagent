from pathlib import Path

from ai_dev_agent.tools.code.code_edit.context import ContextGatheringOptions
from ai_dev_agent.providers.llm.base import LLMError
from ai_dev_agent.tools.query.questions import QuestionAnswerer


class DummyClient:
    def __init__(self) -> None:
        self.messages = None

    def complete(self, messages, temperature=0.15, max_tokens=None, extra_headers=None):
        self.messages = messages
        return "Sample answer"

    def configure_timeout(self, timeout: float) -> None:  # pragma: no cover - noop
        return None

    def configure_retry(self, retry_config) -> None:  # pragma: no cover - noop
        return None


def test_question_answerer_gathers_context(tmp_path: Path) -> None:
    module_path = tmp_path / "src" / "module.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("def greet(name):\n    return f'hi {name}'\n", encoding="utf-8")

    client = DummyClient()
    options = ContextGatheringOptions(max_files=3)
    answerer = QuestionAnswerer(tmp_path, client, options)

    result = answerer.answer(
        "How does greet work?",
        files=["src/module.py"],
        keywords=["greet"],
    )

    assert result.answer == "Sample answer"
    assert result.contexts
    context_paths = {str(ctx.path.relative_to(tmp_path)) for ctx in result.contexts}
    assert "src/module.py" in context_paths
    assert client.messages is not None
    assert client.messages[0].content.startswith("You are a pragmatic senior software engineer")
    assert "src/module.py" in client.messages[1].content


class FailingClient:
    def complete(self, messages, temperature=0.15, max_tokens=None, extra_headers=None):
        raise LLMError("Rate limited")

    def configure_timeout(self, timeout: float) -> None:  # pragma: no cover - noop
        return None

    def configure_retry(self, retry_config) -> None:  # pragma: no cover - noop
        return None


def test_question_answerer_fallback(tmp_path: Path) -> None:
    module_path = tmp_path / "src" / "module.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("def greet(name):\n    return f'hi {name}'\n", encoding="utf-8")

    answerer = QuestionAnswerer(tmp_path, FailingClient())
    result = answerer.answer("How does greet work?", files=["src/module.py"], keywords=["greet"])

    assert "LLM service is unavailable" in result.answer
    assert result.fallback_reason == "Rate limited"
    assert result.raw_response == ""
    assert result.contexts

def test_question_answerer_detects_root_toml(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.example]\nname = 'demo'\n", encoding="utf-8")

    client = DummyClient()
    answerer = QuestionAnswerer(tmp_path, client)

    contexts = answerer.gather_context("Which TOML files exist in the root directory?", files=None, keywords=None)
    context_paths = {str(ctx.path.relative_to(tmp_path)) for ctx in contexts}

    assert "pyproject.toml" in context_paths

import json
import os
import subprocess
from pathlib import Path

from click.testing import CliRunner

import ai_dev_agent.cli as cli_module
from ai_dev_agent.cli import _infer_task_files, cli
from ai_dev_agent.intent_router import IntentDecision
from ai_dev_agent.utils.config import Settings
from ai_dev_agent.planning.planner import PlanResult, PlanTask, Planner
from ai_dev_agent.react.types import MetricsSnapshot, Observation


def test_infer_task_files_from_commands(tmp_path: Path) -> None:
    target = tmp_path / "ai_dev_agent" / "core.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('hello')\n", encoding="utf-8")

    task = {
        "commands": [
            "devagent react run --files ai_dev_agent/core.py docs/guide.md",
            "pytest",
        ]
    }

    # The referenced doc file does not exist; only existing files should be returned
    inferred = _infer_task_files(task, tmp_path)
    assert inferred == ["ai_dev_agent/core.py"]


def test_infer_task_files_from_deliverables(tmp_path: Path) -> None:
    doc_path = tmp_path / "docs" / "overview.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("overview", encoding="utf-8")

    task = {
        "commands": [],
        "deliverables": ["docs/overview.md", "docs/missing.md"],
    }

    inferred = _infer_task_files(task, tmp_path)
    assert inferred == ["docs/overview.md"]


def test_infer_task_files_from_keywords(tmp_path: Path) -> None:
    file_path = tmp_path / "ai_dev_agent" / "core.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("calc = 1\n", encoding="utf-8")

    task = {
        "title": "Remove obsolete core module files",
        "description": "Delete unused core module functionality from the project",
    }

    inferred = _infer_task_files(task, tmp_path)
    assert inferred == ["ai_dev_agent/core.py"]


def test_infer_task_files_from_path_hints(tmp_path: Path) -> None:
    doc_path = tmp_path / "docs" / "guide.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("guide", encoding="utf-8")

    task = {
        "title": "Update guides",
        "description": "Ensure docs/guide.md reflects the new workflow",
    }

    inferred = _infer_task_files(task, tmp_path)
    assert inferred == ["docs/guide.md"]


def test_cli_natural_language_directory_listing(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "sample.txt").write_text("hello", encoding="utf-8")

    settings = Settings()
    settings.workspace_root = repo_root
    settings.state_file = repo_root / ".devagent" / "state.json"
    settings.ensure_state_dir()
    settings.api_key = "test"
    monkeypatch.setattr(cli_module, "load_settings", lambda path=None: settings)

    mock_client = object()
    monkeypatch.setattr(cli_module, "_get_llm_client", lambda ctx: mock_client)

    class DummyRouter:
        def __init__(self, client, settings_obj) -> None:  # noqa: D401 - simple stub
            self.client = client
            self.settings = settings_obj

        def route(self, prompt: str) -> IntentDecision:
            assert "дир" in prompt
            return IntentDecision(tool="list_directory", arguments={"path": "."}, rationale="List repo root")

    monkeypatch.setattr(cli_module, "IntentRouter", DummyRouter)
    monkeypatch.chdir(repo_root)

    runner = CliRunner()
    result = runner.invoke(cli, ["покажи", "содержимое", "директории"])

    assert result.exit_code == 0, result.output
    assert "Listing ." in result.output
    assert "sample.txt" in result.output


def test_ask_command_uses_provided_files(monkeypatch, tmp_path: Path) -> None:
    class DummyClient:
        def __init__(self) -> None:
            self.messages = None

        def complete(self, messages, temperature=0.15, max_tokens=None, extra_headers=None):
            self.messages = messages
            return "Dummy answer"

        def configure_timeout(self, timeout: float) -> None:  # pragma: no cover - noop
            return None

        def configure_retry(self, retry_config) -> None:  # pragma: no cover - noop
            return None

    repo_root = tmp_path
    source_file = repo_root / "ai_dev_agent" / "sample.py"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("def feature():\n    return 'ok'\n", encoding="utf-8")

    dummy_client = DummyClient()
    monkeypatch.setenv("DEVAGENT_API_KEY", "test")
    monkeypatch.setattr("ai_dev_agent.cli.create_client", lambda *args, **kwargs: dummy_client)
    monkeypatch.chdir(repo_root)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "ask",
            "--files",
            "ai_dev_agent/sample.py",
            "How does feature work?",
        ],
    )

    assert result.exit_code == 0
    assert "Dummy answer" in result.output
    assert dummy_client.messages is not None
    user_prompt = dummy_client.messages[1].content
    assert "ai_dev_agent/sample.py" in user_prompt
    assert "How does feature work?" in user_prompt


def test_plan_command_exports_markdown(monkeypatch, tmp_path: Path) -> None:
    settings = Settings()
    settings.state_file = tmp_path / ".devagent" / "state.json"
    settings.ensure_state_dir()
    monkeypatch.setattr(cli_module, "load_settings", lambda path=None: settings)
    monkeypatch.setattr(cli_module, "_get_llm_client", lambda ctx: None)

    plan_result = PlanResult(
        goal="Demo goal",
        summary="Demo summary",
        tasks=[
            PlanTask(
                identifier="T1",
                title="Implement feature",
                description="Do the thing",
                category="implementation",
                effort=3,
                reach=2,
                impact=4,
                confidence=0.7,
                deliverables=["src/demo.py"],
                commands=["devagent react run"],
            )
        ],
        raw_response="",
    )

    monkeypatch.setattr(Planner, "generate", lambda self, goal: plan_result)

    markdown_path = tmp_path / "plan.md"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "plan",
            "Demo",
            "goal",
            "--auto-approve",
            "--write-md",
            "--md-path",
            str(markdown_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert markdown_path.is_file()
    content = markdown_path.read_text(encoding="utf-8")
    assert "# Plan:" in content
    assert "```json" in content
    json_start = content.index("```json") + len("```json\n")
    json_end = content.index("```", json_start)
    plan_json = json.loads(content[json_start:json_end])
    assert plan_json["goal"] == "Demo goal"
    assert plan_json["tasks"][0]["id"] == "T1"


def test_run_command_shows_diff_preview(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    env = os.environ | {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    sample_path = repo_root / "sample.py"
    sample_path.write_text("print('hello')\n", encoding="utf-8")
    subprocess.run(["git", "add", "sample.py"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        env=env,
    )

    plan = {
        "goal": "Demo goal",
        "summary": "Demo summary",
        "status": "approved",
        "tasks": [
            {
                "id": "T1",
                "title": "Update sample",
                "description": "Modify sample file",
                "category": "implementation",
                "status": "pending",
                "dependencies": [],
                "deliverables": ["sample.py"],
                "commands": [],
            }
        ],
    }
    created_store: cli_module.StateStore | None = None

    original_build_context = cli_module._build_context

    def fake_build_context(settings: Settings):
        nonlocal created_store
        ctx = original_build_context(settings)
        ctx["state"].update(last_plan=plan)
        created_store = ctx["state"]
        return ctx

    monkeypatch.setattr(cli_module, "_build_context", fake_build_context)

    class StubClient:
        def configure_timeout(self, timeout: float) -> None:
            return None

        def configure_retry(self, retry_config) -> None:
            return None

        def complete(self, messages, temperature=0.2, max_tokens=None):
            return (
                "```diff\n"
                "--- a/sample.py\n"
                "+++ b/sample.py\n"
                "@@ -1 +1 @@\n"
                "-print('hello')\n"
                "+print('hello world')\n"
                "```"
            )

    monkeypatch.setattr("ai_dev_agent.cli.create_client", lambda *args, **kwargs: StubClient())

    settings = Settings(
        api_key="test",
        auto_approve_code=True,
        auto_approve_plan=True,
        auto_approve_shell=True,
    )
    settings.state_file = repo_root / ".devagent" / "state.json"
    monkeypatch.setattr("ai_dev_agent.cli.load_settings", lambda path=None: settings)
    monkeypatch.chdir(repo_root)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "T1", "--files", "sample.py", "--skip-tests", "--hide-thinking"],
    )

    assert result.exit_code == 0, result.output
    assert "Summary:" in result.output
    assert "Validation:" in result.output
    assert "Unified diff:" in result.output

    summary_index = result.output.index("Summary:")
    diff_index = result.output.index("Unified diff:")
    assert summary_index < diff_index

    updated = sample_path.read_text(encoding="utf-8")
    assert "print('hello world')" in updated

    assert created_store is not None
    updated_plan = created_store.load()["last_plan"]
    assert updated_plan["tasks"][0]["status"] == "completed"

def test_react_plan_command_outputs_wbs(monkeypatch, tmp_path: Path) -> None:
    settings = Settings()
    settings.state_file = tmp_path / ".devagent" / "state.json"
    settings.ensure_state_dir()
    monkeypatch.setattr(cli_module, "load_settings", lambda path=None: settings)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["react", "plan"])

    assert result.exit_code == 0
    assert "ReAct automation work breakdown" in result.output
    assert "R1" in result.output


def test_react_run_with_plan_file(monkeypatch, tmp_path: Path) -> None:
    plan_dict = {
        "goal": "Demo goal",
        "summary": "Demo summary",
        "status": "approved",
        "tasks": [
            {
                "id": "T1",
                "title": "Dry run pipeline",
                "description": "Validate pipeline",
                "category": "implementation",
                "status": "pending",
                "dependencies": [],
                "deliverables": ["sample.py"],
                "commands": [],
            }
        ],
    }
    plan_file = tmp_path / "plan.md"
    plan_file.write_text("# Plan\n```json\n" + json.dumps(plan_dict) + "\n```\n", encoding="utf-8")

    sample_path = tmp_path / "sample.py"
    sample_path.write_text("print('hello')\n", encoding="utf-8")

    def fake_pipeline(repo_root, sandbox, collector, commands, run_tests, tokens_cost=None):
        snapshot = MetricsSnapshot(
            tests_passed=True,
            lint_errors=0,
            type_errors=0,
            format_errors=0,
            compile_errors=0,
            diff_lines=0,
            diff_files=0,
            patch_coverage=1.0,
            secrets_found=0,
            sandbox_violations=0,
            flaky_tests=0,
        )
        return Observation(success=True, outcome="mock", metrics=snapshot.model_dump(), tool="qa.pipeline")

    monkeypatch.setattr(cli_module, "run_quality_pipeline", fake_pipeline)

    settings = Settings()
    settings.state_file = tmp_path / ".devagent" / "state.json"
    settings.ensure_state_dir()
    monkeypatch.setattr(cli_module, "load_settings", lambda path=None: settings)

    captured: dict[str, cli_module.StateStore] = {}
    original_build_context = cli_module._build_context

    def fake_build_context(s: Settings):
        ctx = original_build_context(s)
        captured["state"] = ctx["state"]
        return ctx

    monkeypatch.setattr(cli_module, "_build_context", fake_build_context)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "react",
            "run",
            "T1",
            "--plan-file",
            str(plan_file),
            "--skip-tests",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Using plan from" in result.output
    store = captured["state"]
    updated_plan = store.load()["last_plan"]
    assert updated_plan["tasks"][0]["status"] == "completed"


    settings = Settings()
    settings.state_file = tmp_path / ".devagent" / "state.json"
    settings.ensure_state_dir()
    monkeypatch.setattr(cli_module, "load_settings", lambda path=None: settings)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["react", "plan"])

    assert result.exit_code == 0
    assert "ReAct automation work breakdown" in result.output
    assert "R1" in result.output

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ai_dev_agent.git_tools import (
    DiffContext,
    create_feature_branch,
    gather_diff,
    generate_commit_message,
    generate_pr_description,
    guess_default_base_branch,
    slugify_feature_name,
)


class DummyLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, messages, temperature=0.2, max_tokens=None):  # noqa: D401 - test stub signature
        return self.response


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    env = os.environ | {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    (repo_root / "README.md").write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        env=env,
    )
    return repo_root


def test_slugify_feature_name_basic():
    assert slugify_feature_name("Cool Feature: V1") == "cool-feature-v1"



def test_create_feature_branch_creates_unique_branch(git_repo: Path):
    branch_name, base = create_feature_branch(git_repo, "Cool Feature")
    assert branch_name.startswith("feature/cool-feature")
    assert base in {"main", "master", "develop"}
    current = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert current == branch_name



def test_gather_diff_returns_staged_changes(git_repo: Path):
    target = git_repo / "README.md"
    target.write_text("initial\nupdated\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=git_repo, check=True, capture_output=True)

    diff_context = gather_diff(git_repo, include_staged=True, include_unstaged=False)
    assert "Staged changes" in diff_context.diff
    assert "README.md" in diff_context.files



def test_generate_commit_message_sanitizes_output():
    context = DiffContext(
        diff="--- a/file.txt\n+++ b/file.txt\n",
        files=["file.txt"],
        is_truncated=False,
        source="staged",
    )
    client = DummyLLMClient("```\nAdd change\n```\n")
    message = generate_commit_message(client, context)
    assert message == "Add change"



def test_generate_pr_description_includes_sections():
    context = DiffContext(
        diff="--- a/file.txt\n+++ b/file.txt\n",
        files=["file.txt"],
        is_truncated=False,
        source="staged",
    )
    client = DummyLLMClient("## Summary\n- change\n## Testing\n- none\n## Risks\n- low\n## Related Work\n- none")
    description = generate_pr_description(
        client,
        context,
        context="Goal: Example",
        base_branch="main",
        feature_branch="feature/example",
    )
    assert "## Summary" in description
    assert description.strip().startswith("## Summary")



def test_guess_default_base_branch_prefers_known_branches(git_repo: Path):
    base = guess_default_base_branch(git_repo)
    assert base in {"main", "master", "develop"}

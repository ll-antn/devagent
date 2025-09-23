"""Git integration helpers for the DevAgent CLI."""
from __future__ import annotations

import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Iterable, List, Sequence, Tuple

from .llm_provider import LLMClient, LLMError, Message
from .utils.logger import get_logger

LOGGER = get_logger(__name__)
MAX_DIFF_CHARS = 24000


class GitIntegrationError(RuntimeError):
    """Raised when git integration helpers encounter an error."""


@dataclass
class DiffContext:
    """Represents collected diff information for LLM prompts."""

    diff: str
    files: List[str]
    is_truncated: bool
    source: str


def _run_git(args: Sequence[str], repo_root: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command within the repository and return the completed process."""
    process = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if check and process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip() or "Unknown git error"
        raise GitIntegrationError(f"git {' '.join(args)} failed: {message}")
    return process


def ensure_git_repo(repo_root: Path) -> None:
    """Verify that the provided path is inside a git repository."""
    try:
        _run_git(["rev-parse", "--is-inside-work-tree"], repo_root)
    except GitIntegrationError as exc:
        raise GitIntegrationError("Current directory is not a git repository.") from exc


def get_repo_root(start: Path | None = None) -> Path:
    """Resolve the repository root path."""
    start = start or Path.cwd()
    try:
        process = _run_git(["rev-parse", "--show-toplevel"], start)
    except GitIntegrationError as exc:
        raise GitIntegrationError("Unable to determine git repository root.") from exc
    root = process.stdout.strip()
    if not root:
        raise GitIntegrationError("Git did not report a repository root.")
    return Path(root)


def get_current_branch(repo_root: Path) -> str:
    """Return the active branch name."""
    process = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    branch = process.stdout.strip()
    if not branch:
        raise GitIntegrationError("Unable to determine current branch.")
    return branch


def branch_exists(repo_root: Path, branch: str) -> bool:
    """Return True if the branch exists locally."""
    process = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", branch],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    return process.returncode == 0


def guess_default_base_branch(repo_root: Path) -> str | None:
    """Best-effort detection of the repository's default base branch."""
    ensure_git_repo(repo_root)
    symbolic = _run_git(["symbolic-ref", "refs/remotes/origin/HEAD"], repo_root, check=False)
    if symbolic.returncode == 0:
        ref = symbolic.stdout.strip()
        if ref:
            parts = ref.split("/")
            if parts:
                return parts[-1]

    for candidate in ("main", "master", "develop"):
        if branch_exists(repo_root, candidate):
            return candidate
    return None


def slugify_feature_name(name: str) -> str:
    """Convert a human readable feature name into a git-friendly slug."""
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized or "feature"


def create_feature_branch(
    repo_root: Path,
    feature_name: str,
    *,
    prefix: str = "feature",
    base: str | None = None,
    dry_run: bool = False,
) -> Tuple[str, str]:
    """Create and switch to a feature branch derived from the given name."""
    ensure_git_repo(repo_root)
    if not feature_name.strip():
        raise GitIntegrationError("Feature name must not be empty.")

    slug = slugify_feature_name(feature_name)
    base_branch = base or get_current_branch(repo_root)
    candidate = f"{prefix}/{slug}" if prefix else slug
    branch_name = candidate
    counter = 1
    while branch_exists(repo_root, branch_name):
        branch_name = f"{candidate}-{counter}"
        counter += 1

    if branch_name == base_branch:
        raise GitIntegrationError("Feature branch would match the base branch name; choose a different feature name or prefix.")

    if dry_run:
        LOGGER.info("Dry run: would create branch %s from %s", branch_name, base_branch)
        return branch_name, base_branch

    _run_git(["checkout", "-b", branch_name, base_branch], repo_root, check=True)
    return branch_name, base_branch


def gather_diff(
    repo_root: Path,
    *,
    include_staged: bool = True,
    include_unstaged: bool = False,
) -> DiffContext:
    """Collect diff information for the requested change scope."""
    ensure_git_repo(repo_root)
    diff_sections: List[str] = []
    tracked_files: set[str] = set()
    sources: List[str] = []

    if include_staged:
        staged = _run_git(["diff", "--cached"], repo_root, check=False).stdout
        if staged.strip():
            sources.append("staged")
            diff_sections.append("# Staged changes\n" + staged.strip())
            files = _run_git(["diff", "--cached", "--name-only"], repo_root, check=False).stdout
            tracked_files.update(path for path in files.splitlines() if path.strip())

    if include_unstaged:
        unstaged = _run_git(["diff"], repo_root, check=False).stdout
        if unstaged.strip():
            sources.append("unstaged")
            diff_sections.append("# Unstaged changes\n" + unstaged.strip())
            files = _run_git(["diff", "--name-only"], repo_root, check=False).stdout
            tracked_files.update(path for path in files.splitlines() if path.strip())

    if not diff_sections:
        raise GitIntegrationError("No changes detected in the selected diff scope.")

    combined = "\n\n".join(section for section in diff_sections if section)
    is_truncated = False
    if len(combined) > MAX_DIFF_CHARS:
        combined = combined[:MAX_DIFF_CHARS] + "\n\n[diff truncated]"
        is_truncated = True

    source_label = "+".join(sources) if sources else "unknown"
    return DiffContext(
        diff=combined,
        files=sorted(tracked_files),
        is_truncated=is_truncated,
        source=source_label,
    )


def _strip_code_fences(text: str) -> str:
    """Remove Markdown code fences from the provided text."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9]*\n", "", stripped)
    if stripped.endswith("```"):
        stripped = stripped[: -3]
    return stripped.strip()


def _sanitize_message(text: str) -> str:
    """Normalize LLM output for terminal display."""
    text = _strip_code_fences(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def _format_file_list(files: Iterable[str]) -> str:
    entries = [f"- {path}" for path in files]
    return "\n".join(entries) if entries else "(no filenames reported)"


def generate_commit_message(
    client: LLMClient,
    diff_context: DiffContext,
    *,
    context: str | None = None,
) -> str:
    """Ask the LLM to draft a commit message for the collected diff."""
    summary_text = _format_file_list(diff_context.files)
    context_text = context.strip() if context else ""
    prompt = dedent(
        f"""
        Draft a concise git commit message for the following repository changes. Use conventional structure:
        - Subject line: imperative, <= 72 characters, no trailing period.
        - Optional body: up to 4 bullet points summarizing key work. Include test notes if relevant.

        Additional project context (optional):
        {context_text or 'N/A'}

        Files changed:
        {summary_text}

        Diff source: {diff_context.source}
        Diff:
        ```diff
        {diff_context.diff}
        ```
        {"Diff is truncated due to size limits." if diff_context.is_truncated else ""}

        Respond with plain text only. Do not wrap the message in quotes or code fences.
        """
    ).strip()

    messages = (
        Message(
            role="system",
            content=(
                "You are an experienced software engineer who writes excellent git commit messages. "
                "Keep the subject imperative, capitalize only proper nouns, and keep the body concise."
            ),
        ),
        Message(role="user", content=prompt),
    )
    raw = client.complete(messages, temperature=0.2)
    return _sanitize_message(raw)


def generate_pr_description(
    client: LLMClient,
    diff_context: DiffContext,
    *,
    context: str | None = None,
    base_branch: str | None = None,
    feature_branch: str | None = None,
) -> str:
    """Ask the LLM to prepare a pull request description."""
    summary_text = _format_file_list(diff_context.files)
    context_text = context.strip() if context else ""
    branches_text = ""
    if base_branch or feature_branch:
        branches_text = f"Base: {base_branch or 'unknown'}, Feature: {feature_branch or 'unknown'}"

    prompt = dedent(
        f"""
        Prepare a polished pull request description in Markdown summarizing the following changes.
        Structure the response with these sections:
        ## Summary (2-4 short bullets capturing the main work)
        ## Testing (list commands or note if not applicable)
        ## Risks (note regressions or deployment considerations; include "None" if minimal)
        ## Related Work (reference relevant tasks, tickets, or goals; include "None")

        Project context:
        {context_text or 'N/A'}

        Branch information: {branches_text or 'N/A'}

        Files changed:
        {summary_text}

        Diff source: {diff_context.source}
        Diff:
        ```diff
        {diff_context.diff}
        ```
        {"Diff is truncated due to size limits." if diff_context.is_truncated else ""}
        """
    ).strip()

    messages = (
        Message(
            role="system",
            content=(
                "You are assisting with software delivery by drafting clear, actionable pull request descriptions. "
                "Write in concise Markdown, using bullet points where appropriate, and avoid speculative language."
            ),
        ),
        Message(role="user", content=prompt),
    )
    raw = client.complete(messages, temperature=0.2)
    return _sanitize_message(raw)


__all__ = [
    "DiffContext",
    "GitIntegrationError",
    "branch_exists",
    "create_feature_branch",
    "gather_diff",
    "generate_commit_message",
    "generate_pr_description",
    "get_current_branch",
    "get_repo_root",
    "guess_default_base_branch",
    "slugify_feature_name",
]

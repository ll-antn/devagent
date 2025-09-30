"""Helper utilities shared across CLI commands."""
from __future__ import annotations

import os
import re
import shlex
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import click

from ai_dev_agent.core.approval.approvals import ApprovalManager
from ai_dev_agent.core.approval.policy import ApprovalPolicy
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.context_budget import BudgetedLLMClient, config_from_settings
from ai_dev_agent.core.utils.constants import MAX_HISTORY_ENTRIES
from ai_dev_agent.core.utils.devagent_config import load_devagent_yaml
from ai_dev_agent.core.utils.keywords import extract_keywords
from ai_dev_agent.core.utils.logger import get_logger, set_correlation_id
from ai_dev_agent.core.utils.state import InMemoryStateStore, StateStore
from ai_dev_agent.engine.react.tool_strategy import ToolSelectionStrategy
from ai_dev_agent.providers.llm import (
    DEEPSEEK_DEFAULT_BASE_URL,
    RetryConfig,
    create_client,
)
from ai_dev_agent.tools import ToolContext, registry as tool_registry
from ai_dev_agent.tools.code.code_edit.tree_sitter_analysis import (
    TreeSitterProjectAnalyzer,
    extract_symbols_from_outline,
)
from ai_dev_agent.tools.execution.sandbox import SandboxConfig, SandboxExecutor

LOGGER = get_logger(__name__)


def _resolve_repo_path(path_value: str | None) -> Path:
    repo_root = Path.cwd().resolve()
    if not path_value or path_value.strip() in {"", "."}:
        return repo_root
    candidate = (repo_root / path_value).resolve()
    if repo_root not in candidate.parents and candidate != repo_root:
        raise click.ClickException(f"Path '{path_value}' escapes the repository root.")
    return candidate


def _collect_project_structure_outline(
    repo_root: Path,
    *,
    max_files: int = 6,
    max_file_bytes: int = 120_000,
) -> Optional[str]:
    """Return a lightweight project structure summary using tree-sitter when available."""

    try:
        analyzer = TreeSitterProjectAnalyzer(repo_root, max_files=max_files)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.debug("Failed to initialize TreeSitterProjectAnalyzer: %s", exc)
        return None

    if not analyzer.available:
        return None

    skip_dirs = {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "vendor",
        "dist",
        "build",
        "__pycache__",
        ".venv",
        "venv",
    }

    file_entries: List[Tuple[str, str]] = []
    seen_paths: Set[str] = set()

    for suffix in analyzer.SUPPORTED_SUFFIXES:
        if len(file_entries) >= max_files:
            break
        try:
            candidates = sorted(repo_root.rglob(f"*{suffix}"))
        except OSError:  # pragma: no cover - defensive guard
            continue
        for path in candidates:
            if len(file_entries) >= max_files:
                break
            if not path.is_file():
                continue
            if any(part in skip_dirs for part in path.parts):
                continue
            try:
                if path.stat().st_size > max_file_bytes:
                    continue
            except OSError:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            try:
                rel_path = str(path.relative_to(repo_root))
            except ValueError:
                continue
            if rel_path in seen_paths:
                continue
            seen_paths.add(rel_path)
            file_entries.append((rel_path, content))

    if not file_entries:
        return None

    try:
        summary_text = analyzer.build_project_summary(file_entries)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.debug("Failed to build project structure summary: %s", exc)
        return None

    if not summary_text:
        return None

    return _prepare_structure_prompt(summary_text)


def _prepare_structure_prompt(summary_text: str, *, max_lines: int = 80, max_chars: int = 4000) -> str:
    """Compress a markdown structure outline for planner prompts."""

    lines = []
    for raw_line in summary_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        lines.append(stripped)
        if len(lines) >= max_lines:
            break

    compact = "\n".join(lines).strip()
    if len(compact) > max_chars:
        compact = compact[: max_chars - 3].rstrip() + "..."
    return compact


def _get_structure_hints_state(ctx: click.Context) -> Dict[str, Any]:
    state = ctx.obj.setdefault(
        "_structure_hints_state",
        {"symbols": set(), "files": {}, "project_summary": None},
    )

    symbols = state.get("symbols")
    if not isinstance(symbols, set):
        state["symbols"] = set()
    files = state.get("files")
    if not isinstance(files, dict):
        state["files"] = {}
    return state


def _export_structure_hints_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symbols": sorted(state.get("symbols", set())),
        "files": dict(state.get("files", {})),
        "project_summary": state.get("project_summary"),
    }


def _merge_structure_hints_state(state: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> None:
    if not payload:
        return
    symbols = payload.get("symbols")
    if isinstance(symbols, Iterable):
        state.setdefault("symbols", set()).update(str(item) for item in symbols)
    files = payload.get("files") or {}
    if isinstance(files, Mapping):
        for path, info in files.items():
            if not isinstance(info, Mapping):
                continue
            existing = state.setdefault("files", {}).setdefault(path, {})
            outline = info.get("outline")
            if outline and "symbols" not in existing:
                symbols = set(extract_symbols_from_outline(outline))
                existing["symbols"] = sorted(symbols)
            if info.get("symbols") and "symbols" not in existing:
                symbols = set(info.get("symbols"))
                existing["symbols"] = sorted(symbols)
            if info.get("summary"):
                existing.setdefault("summaries", []).append(info["summary"])
    if payload.get("project_summary"):
        state["project_summary"] = payload["project_summary"]


def _update_files_discovered(files_discovered: Set[str], payload: Optional[Dict[str, Any]]) -> None:
    if not payload:
        return
    for file_entry in payload.get("files", []) or []:
        path = file_entry.get("path")
        if path:
            files_discovered.add(str(path))

    for match in payload.get("matches", []) or []:
        path = match.get("path")
        if path:
            files_discovered.add(str(path))

    for summary in payload.get("summaries", []) or []:
        path = summary.get("path")
        if path:
            files_discovered.add(str(path))


def _detect_repository_language(
    strategy: ToolSelectionStrategy,
    repo_root: Path,
    *,
    max_files: int = 400,
) -> Tuple[Optional[str], Optional[int]]:
    file_paths: List[str] = []
    count = 0
    try:
        for path in repo_root.rglob("*"):
            if count >= max_files:
                break
            if path.is_file():
                count += 1
                try:
                    rel = str(path.relative_to(repo_root))
                except ValueError:
                    rel = str(path)
                file_paths.append(rel)
    except OSError:
        pass

    language = strategy.detect_language(file_paths)
    return language, count if count else None


def infer_task_files(task: Mapping[str, Any], repo_root: Path) -> List[str]:
    """Infer target files for a task using commands, deliverables, and textual hints."""
    repo_root = repo_root.resolve()
    results: List[str] = []
    seen: Set[str] = set()

    def _add(candidate: Path) -> None:
        try:
            resolved = candidate.resolve()
        except OSError:
            return
        if not resolved.is_file():
            return
        try:
            relative = resolved.relative_to(repo_root)
        except ValueError:
            return
        rel_str = str(relative)
        if rel_str not in seen:
            seen.add(rel_str)
            results.append(rel_str)

    for command in task.get("commands") or []:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        for idx, token in enumerate(tokens):
            if token == "--files":
                for follow in tokens[idx + 1 :]:
                    if follow.startswith("-"):
                        break
                    _add(repo_root / follow)
            elif "/" in token:
                _add(repo_root / token.strip("\"\'"))

    for deliverable in task.get("deliverables") or []:
        if isinstance(deliverable, str):
            _add(repo_root / deliverable)

    text_blob = " ".join(filter(None, [task.get("title"), task.get("description")]))
    if text_blob:
        for match in re.findall(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+", text_blob):
            cleaned = match.strip(".,;:'\"()[]{}<>")
            _add(repo_root / cleaned)

        keywords = {kw.lower() for kw in extract_keywords(text_blob)}
    else:
        keywords = set()

    if keywords:
        examined = 0
        for path_candidate in repo_root.rglob("*"):
            if examined >= 200:
                break
            examined += 1
            if not path_candidate.is_file():
                continue
            stem = path_candidate.stem.lower()
            parent_names = {
                parent.name.lower() for parent in path_candidate.parents if parent != repo_root
            }
            if stem in keywords or keywords.intersection(parent_names):
                _add(path_candidate)

    return results


def update_task_state(
    store: StateStore,
    plan: Dict[str, Any],
    task: Dict[str, Any],
    updates: Mapping[str, Any],
    *,
    reasoning: Any | None = None,
) -> None:
    """Persist updated plan and task state to the store."""
    updated_task = dict(task)
    updated_task.update(updates)

    if reasoning is not None:
        apply = getattr(reasoning, "apply_to_task", None)
        if callable(apply):
            apply(updated_task)
        merge = getattr(reasoning, "merge_into_plan", None)
        if callable(merge):
            merge(plan)

    task_id = updated_task.get("id")
    tasks = plan.setdefault("tasks", [])
    if task_id and isinstance(tasks, list):
        for existing in tasks:
            if isinstance(existing, dict) and existing.get("id") == task_id:
                existing.clear()
                existing.update(updated_task)
                break
        else:
            tasks.append(updated_task)
    task.clear()
    task.update(updated_task)

    if "status" in updates:
        plan["status"] = updates["status"]

    snapshot = store.load() or {}
    snapshot["last_plan"] = plan
    snapshot["last_updated"] = datetime.utcnow().isoformat()
    store.save(snapshot)


def _create_sandbox(settings: Settings) -> SandboxExecutor:
    config = SandboxConfig(
        allowlist=settings.sandbox_allowlist,
        default_timeout=120.0,
        cpu_time_limit=settings.sandbox_cpu_time_limit,
        memory_limit_mb=settings.sandbox_memory_limit_mb,
    )
    return SandboxExecutor(Path.cwd(), config)


def _make_tool_context(ctx: click.Context, *, with_sandbox: bool = False) -> ToolContext:
    settings: Settings = ctx.obj["settings"]

    sandbox = None
    if with_sandbox:
        sandbox = ctx.obj.get("sandbox_executor")
        if sandbox is None:
            sandbox = _create_sandbox(settings)
            ctx.obj["sandbox_executor"] = sandbox

    devagent_cfg = ctx.obj.get("devagent_config")
    if devagent_cfg is None:
        devagent_cfg = load_devagent_yaml()
        ctx.obj["devagent_config"] = devagent_cfg

    structure_hints_state = _get_structure_hints_state(ctx)

    return ToolContext(
        repo_root=Path.cwd(),
        settings=settings,
        sandbox=sandbox,
        devagent_config=devagent_cfg,
        metrics_collector=None,
        extra={"structure_hints": _export_structure_hints_state(structure_hints_state)},
    )


def _invoke_registry_tool(
    ctx: click.Context,
    tool_name: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    with_sandbox: bool = False,
) -> Dict[str, Any]:
    """Invoke a registry tool with a lazily created ToolContext."""

    context = _make_tool_context(ctx, with_sandbox=with_sandbox)
    return tool_registry.invoke(tool_name, payload or {}, context)


def _normalize_argument_list(
    arguments: Dict[str, Any],
    *,
    plural_key: str,
    singular_key: Optional[str] = None,
) -> List[str]:
    """Return a normalized list of string arguments for registry handlers."""

    values = arguments.get(plural_key)
    if not values and singular_key and arguments.get(singular_key):
        values = [arguments[singular_key]]
    if values is None:
        return []
    if isinstance(values, (str, os.PathLike)):
        values = [values]
    return [str(item) for item in values]


def _build_context(settings: Settings) -> Dict[str, Any]:
    state_store = StateStore(settings.state_file)
    policy = ApprovalPolicy(
        auto_approve_plan=settings.auto_approve_plan,
        auto_approve_code=settings.auto_approve_code,
        auto_approve_shell=settings.auto_approve_shell,
        auto_approve_adr=settings.auto_approve_adr,
        emergency_override=settings.emergency_override,
        audit_file=settings.audit_approvals,
    )
    return {
        "settings": settings,
        "state": state_store,
        "llm_client": None,
        "approval_policy": policy,
        "sandbox_executor": None,
        "devagent_config": None,
    }


def _prompt_yes_no(
    ctx: click.Context,
    purpose: str,
    message: str,
    *,
    default: bool = False,
) -> bool:
    """Consistently request user confirmation using the approval manager."""

    policy: ApprovalPolicy = ctx.obj["approval_policy"]
    audit_file = Path.cwd() / ".devagent" / "audit.log" if policy.audit_file else None
    approvals = ApprovalManager(policy, audit_file)
    return approvals.require(purpose, default=default, prompt=message)


def _record_invocation(ctx: click.Context, overrides: Optional[Dict[str, Any]] = None) -> None:
    """Persist command invocation details for history and replay."""
    if not ctx or not ctx.command_path:
        return
    set_correlation_id(uuid.uuid4().hex[:12])
    state_obj = ctx.obj.get("state") if ctx.obj else None
    if not isinstance(state_obj, InMemoryStateStore):
        return
    root_ctx = ctx.find_root()
    root_name = root_ctx.info_name if root_ctx else None
    parts = ctx.command_path.split()
    if root_name and parts and parts[0] == root_name:
        parts = parts[1:]
    if not parts or parts[0] == "history":
        return
    params = dict(ctx.params)
    if overrides:
        params.update(overrides)
    entry = {
        "command_path": parts,
        "params": params,
        "timestamp": datetime.utcnow().isoformat(),
    }
    state_obj.append_history(entry, limit=MAX_HISTORY_ENTRIES)


def get_llm_client(ctx: click.Context):
    client = ctx.obj.get("llm_client")
    if client:
        return client
    settings: Settings = ctx.obj["settings"]
    if not settings.api_key:
        raise click.ClickException("No API key configured. Set DEVAGENT_API_KEY or update config file.")

    retry_config = RetryConfig(
        max_retries=3,
        initial_delay=0.5,
        max_delay=5.0,
        backoff_multiplier=2.0,
        jitter_ratio=0.3,
    )

    provider_key = settings.provider.lower()
    base_url_override: str | None = settings.base_url or None
    provider_kwargs: Dict[str, Any] = {}

    if provider_key in {"openrouter", "cerebras"}:
        if settings.provider_only:
            provider_kwargs["provider_only"] = tuple(settings.provider_only)
        if settings.provider_config:
            provider_kwargs["provider_config"] = dict(settings.provider_config)
        if settings.request_headers:
            provider_kwargs["default_headers"] = dict(settings.request_headers)
        if not base_url_override or base_url_override == DEEPSEEK_DEFAULT_BASE_URL:
            base_url_override = None

    client = create_client(
        provider=settings.provider,
        api_key=settings.api_key,
        model=settings.model,
        base_url=base_url_override,
        **provider_kwargs,
    )

    if hasattr(client, "configure_timeout"):
        client.configure_timeout(120.0)
    if hasattr(client, "configure_retry"):
        client.configure_retry(retry_config)

    budget_config = config_from_settings(settings)
    disabled = bool(getattr(settings, "disable_context_pruner", False))
    wrapped_client = BudgetedLLMClient(client, budget_config, disabled=disabled)
    ctx.obj["_raw_llm_client"] = client
    ctx.obj["llm_client"] = wrapped_client
    return wrapped_client

# Backwards-compatible aliases for legacy imports
_infer_task_files = infer_task_files
_update_task_state = update_task_state
_get_llm_client = get_llm_client

__all__ = [
    "_resolve_repo_path",
    "_collect_project_structure_outline",
    "_prepare_structure_prompt",
    "_get_structure_hints_state",
    "_export_structure_hints_state",
    "_merge_structure_hints_state",
    "_update_files_discovered",
    "_detect_repository_language",
    "infer_task_files",
    "update_task_state",
    "get_llm_client",
    "_create_sandbox",
    "_make_tool_context",
    "_invoke_registry_tool",
    "_normalize_argument_list",
    "_build_context",
    "_prompt_yes_no",
    "_record_invocation",
    # Backwards-compat exports
    "_infer_task_files",
    "_update_task_state",
    "_get_llm_client",
]

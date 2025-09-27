"""Command line interface for the development agent."""
from __future__ import annotations

import json
import os
import re
import shlex
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import click
from click.shell_completion import get_completion_class

PLAN_JSON_FENCE_PATTERN = re.compile(r"```json\s*(?P<json>{.*?})\s*```", re.DOTALL)

from ..core.approval.approvals import ApprovalManager
from ..core.approval.policy import ApprovalPolicy
from .router import IntentDecision, IntentRouter, IntentRoutingError
from ..providers.llm import (
    LLMConnectionError,
    LLMError,
    LLMRetryExhaustedError,
    LLMTimeoutError,
    RetryConfig,
    create_client,
    DEEPSEEK_DEFAULT_BASE_URL,
)
from ..engine.planning.planner import Planner
from ..engine.react.tool_strategy import ToolSelectionStrategy, TaskType
from ..tools.execution.sandbox import SandboxConfig, SandboxExecutor
from ..core.utils.config import Settings, load_settings
from ..core.utils.context_budget import (
    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    BudgetedLLMClient,
    config_from_settings,
    summarize_text,
)
from ..core.utils.artifacts import write_artifact
from ..core.utils.constants import MAX_HISTORY_ENTRIES, MIN_TOOL_OUTPUT_CHARS
from ..core.utils.devagent_config import load_devagent_yaml
from ..core.utils.keywords import extract_keywords
from ..core.utils.logger import configure_logging, get_correlation_id, get_logger, set_correlation_id
from ..core.utils.state import InMemoryStateStore, StateStore
from ..core.utils.tool_utils import (
    canonical_tool_name,
    display_tool_name,
    tool_category,
    tool_signature,
    FILE_READ_TOOLS,
    SEARCH_TOOLS,
)
from ..tools import ToolContext, registry as tool_registry

LOGGER = get_logger(__name__)


class NaturalLanguageGroup(click.Group):
    """Group that falls back to NL intent routing when no command matches."""

    def resolve_command(self, ctx: click.Context, args: List[str]):  # type: ignore[override]
        planning_flag: Optional[bool] = None
        filtered_args: List[str] = []

        for arg in args:
            if arg == "--plan":
                planning_flag = True
            elif arg == "--direct":
                planning_flag = False
            else:
                filtered_args.append(arg)

        try:
            return super().resolve_command(ctx, filtered_args)
        except click.UsageError:
            if not filtered_args:
                raise
            if any(arg.startswith("-") for arg in filtered_args):
                raise
            query = " ".join(filtered_args).strip()
            if not query:
                raise
            ctx.meta["_pending_nl_prompt"] = query
            ctx.meta["_emit_status_messages"] = True
            if planning_flag is not None:
                ctx.meta["_use_planning"] = planning_flag
            return super().resolve_command(ctx, ["query"])


def _resolve_repo_path(path_value: str | None) -> Path:
    repo_root = Path.cwd().resolve()
    if not path_value or path_value.strip() in {"", "."}:
        return repo_root
    candidate = (repo_root / path_value).resolve()
    if repo_root not in candidate.parents and candidate != repo_root:
        raise click.ClickException(f"Path '{path_value}' escapes the repository root.")
    return candidate


def _infer_task_files(task: Mapping[str, Any], repo_root: Path) -> List[str]:
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


def _update_task_state(
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



def _deprecated_intent_handler(tool_name: str, guidance: str) -> Callable[[click.Context, Dict[str, Any]], None]:
    message = f"❌ {tool_name} is deprecated. {guidance}"

    def _handler(_: click.Context, __: Dict[str, Any]) -> None:
        click.echo(message)

    return _handler



# ------------------------------- Sandbox helpers ---------------------------------------


def _create_sandbox(settings: Settings) -> SandboxExecutor:
    config = SandboxConfig(
        allowlist=settings.sandbox_allowlist,
        default_timeout=120.0,
        cpu_time_limit=settings.sandbox_cpu_time_limit,
        memory_limit_mb=settings.sandbox_memory_limit_mb,
    )
    return SandboxExecutor(Path.cwd(), config)


# ------------------------------- Registry-backed intents -------------------------------

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

    return ToolContext(
        repo_root=Path.cwd(),
        settings=settings,
        sandbox=sandbox,
        devagent_config=devagent_cfg,
        metrics_collector=None,
        extra={},
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
PayloadBuilder = Callable[[click.Context, Dict[str, Any]], Tuple[Dict[str, Any], Dict[str, Any]]]
ResultHandler = Callable[[click.Context, Dict[str, Any], Dict[str, Any], Dict[str, Any]], None]
RecoveryHandler = Callable[[click.Context, Dict[str, Any], Dict[str, Any], Dict[str, Any], Exception], Dict[str, Any]]


@dataclass
class RegistryIntent:
    """Reusable wrapper for registry-backed intent handlers."""

    tool_name: str
    payload_builder: PayloadBuilder
    result_handler: ResultHandler
    with_sandbox: bool = False
    recovery_handler: Optional[RecoveryHandler] = None

    def __call__(self, ctx: click.Context, arguments: Dict[str, Any]) -> None:
        payload, context = self.payload_builder(ctx, arguments)
        extras = context or {}
        try:
            result = _invoke_registry_tool(
                ctx,
                self.tool_name,
                payload,
                with_sandbox=self.with_sandbox,
            )
        except Exception as exc:  # pragma: no cover - defensive
            if not self.recovery_handler:
                raise
            result = self.recovery_handler(ctx, arguments, payload, extras, exc)
        self.result_handler(ctx, arguments, result, extras)


def _build_code_search_payload(
    ctx: click.Context, arguments: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise click.ClickException("code.search requires a 'query' argument.")

    settings: Settings = ctx.obj["settings"]
    default_max_results = max(1, getattr(settings, "search_max_results", 100))
    payload: Dict[str, Any] = {"query": query}

    if "regex" in arguments:
        payload["regex"] = bool(arguments.get("regex"))
    else:
        if any(ch in query for ch in ("|", "^", "$", "[", "]", "(", ")")):
            payload["regex"] = True

    max_results = default_max_results
    if "max_results" in arguments:
        try:
            max_results = int(arguments.get("max_results", default_max_results))
        except (TypeError, ValueError):
            max_results = default_max_results
        payload["max_results"] = max_results
    else:
        payload["max_results"] = max_results

    where = arguments.get("where")
    if isinstance(where, list):
        payload["where"] = [str(x) for x in where]

    return payload, {"max_results": max_results}


def _handle_code_search_result(
    _: click.Context,
    __: Dict[str, Any],
    result: Dict[str, Any],
    context: Dict[str, Any],
) -> None:
    matches = result.get("matches", [])
    if not matches:
        click.echo("No matches found.")
        return

    max_results = context.get("max_results", len(matches))
    limited_matches = matches[:max_results]
    for match in limited_matches:
        path = match.get("path")
        line = match.get("line")
        col = match.get("col")
        preview = match.get("preview", "")
        click.echo(f"{path}:{line}:{col} {preview}")

    remaining = len(matches) - len(limited_matches)
    if remaining > 0:
        click.echo(
            "... (+{remaining} additional matches not shown; refine your query or increase --max-results)".format(
                remaining=remaining
            )
        )


def _build_fs_read_payload(
    ctx: click.Context, arguments: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    paths = _normalize_argument_list(arguments, plural_key="paths", singular_key="path")
    if not paths:
        raise click.ClickException("fs.read requires 'paths' (or a single 'path').")

    payload: Dict[str, Any] = {"paths": paths}
    if "context_lines" in arguments:
        try:
            payload["context_lines"] = int(arguments.get("context_lines"))
        except (TypeError, ValueError):
            pass

    if "byte_range" in arguments:
        byte_range = arguments.get("byte_range")
        if isinstance(byte_range, (list, tuple)) and len(byte_range) == 2:
            payload["byte_range"] = [int(byte_range[0]), int(byte_range[1])]

    return payload, {}


def _handle_fs_read_result(
    ctx: click.Context,
    arguments: Dict[str, Any],
    result: Dict[str, Any],
    _: Dict[str, Any],
) -> None:
    files = result.get("files", [])
    if not files:
        click.echo("No content returned.")
        return

    slicing_requested = any(key in arguments for key in ("start_line", "end_line", "max_lines"))

    try:
        start_line = int(arguments.get("start_line", 1) or 1)
    except (TypeError, ValueError):
        raise click.ClickException("start_line must be an integer.")
    if start_line < 1:
        start_line = 1

    end_line_value = arguments.get("end_line")
    if end_line_value is not None:
        try:
            end_line = int(end_line_value)
        except (TypeError, ValueError):
            raise click.ClickException("end_line must be an integer.")
        if end_line < start_line:
            raise click.ClickException("end_line must be greater than or equal to start_line.")
    else:
        end_line = None

    max_lines = None
    if end_line is None and arguments.get("max_lines") is not None:
        try:
            max_lines = int(arguments.get("max_lines"))
        except (TypeError, ValueError):
            raise click.ClickException("max_lines must be an integer.")
        if max_lines <= 0:
            max_lines = 200

    settings: Settings = ctx.obj["settings"]
    default_window = max(1, getattr(settings, "fs_read_default_max_lines", 200))

    for entry in files:
        rel_path = entry.get("path") or "(unknown)"
        content = entry.get("content", "")
        if not slicing_requested:
            lines = content.splitlines()
            total_lines = len(lines)
            if not lines:
                click.echo(f"== {rel_path} ==")
                click.echo("(empty)")
                continue

            window = min(default_window, total_lines)
            snippet = lines[:window]
            click.echo(f"== {rel_path} ==")
            for line_number, text in enumerate(snippet, start=1):
                click.echo(f"{line_number:5}: {text.rstrip()}")
            if window < total_lines:
                remaining = total_lines - window
                click.echo(
                    f"... ({remaining} more lines not shown; specify --start-line/--end-line to expand)"
                )
            continue

        lines = content.splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            click.echo(f"{rel_path} is empty.")
            continue

        start_index = min(start_line - 1, total_lines)
        if start_index >= total_lines:
            click.echo(
                f"File {rel_path} has only {total_lines} lines; nothing to show from line {start_line}."
            )
            continue

        if end_line is not None:
            end_index = min(end_line, total_lines)
        else:
            window = max_lines or 200
            end_index = min(start_index + window, total_lines)
            if end_index == start_index:
                end_index = min(start_index + 200, total_lines)

        snippet = lines[start_index:end_index]
        if snippet:
            last_line = start_index + len(snippet)
            click.echo(f"Reading {rel_path} (lines {start_index + 1}-{last_line} of {total_lines}):")
            for line_number, text in enumerate(snippet, start=start_index + 1):
                click.echo(f"{line_number:5}: {text.rstrip()}")
            if end_index < total_lines:
                remaining = total_lines - end_index
                click.echo(f"... ({remaining} more lines not shown)")
        else:
            click.echo(f"No content available in the requested range for {rel_path}.")


def _build_symbols_find_payload(
    ctx: click.Context, arguments: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    name = str(arguments.get("name", "")).strip()
    if not name:
        raise click.ClickException("symbols.find requires a 'name' argument.")

    payload: Dict[str, Any] = {"name": name}
    if arguments.get("kind"):
        payload["kind"] = str(arguments.get("kind"))
    if arguments.get("lang"):
        payload["lang"] = str(arguments.get("lang"))
    return payload, {}


def _handle_symbols_find_result(
    _: click.Context,
    __: Dict[str, Any],
    result: Dict[str, Any],
    ___: Dict[str, Any],
) -> None:
    defs = result.get("defs", [])
    if not defs:
        click.echo("No definitions found.")
        return
    for definition in defs:
        click.echo(f"{definition.get('path')}:{definition.get('line')} {definition.get('kind', '')}")


def _recover_symbols_find(
    ctx: click.Context,
    arguments: Dict[str, Any],
    payload: Dict[str, Any],
    context: Dict[str, Any],
    exc: Exception,
) -> Dict[str, Any]:
    if isinstance(exc, FileNotFoundError):
        try:
            _invoke_registry_tool(ctx, "symbols.index", {})
        except Exception as index_exc:
            raise click.ClickException(
                f"Symbol index not found and indexing failed: {index_exc}. "
                "Install Universal Ctags (e.g., brew install universal-ctags) or use code.search."
            ) from index_exc
        return _invoke_registry_tool(ctx, "symbols.find", payload)

    raise click.ClickException(f"symbols.find failed: {exc}") from exc


def _build_symbols_index_payload(
    ctx: click.Context, arguments: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    paths = _normalize_argument_list(arguments, plural_key="paths", singular_key="path")
    payload: Dict[str, Any] = {}
    if paths:
        payload["paths"] = paths
    return payload, {}


def _handle_symbols_index_result(
    _: click.Context,
    __: Dict[str, Any],
    result: Dict[str, Any],
    ___: Dict[str, Any],
) -> None:
    stats = result.get("stats", {})
    files_indexed = stats.get("files_indexed")
    symbols = stats.get("symbols")
    db_path = result.get("db_path")

    click.echo("Symbol index updated.")
    if files_indexed is not None or symbols is not None:
        parts = [
            f"{files_indexed} file(s)" if files_indexed is not None else None,
            f"{symbols} symbol(s)" if symbols is not None else None,
        ]
        click.echo("Indexed " + ", ".join(part for part in parts if part))
    if db_path:
        click.echo(f"Index written to {db_path}")


def _build_ast_query_payload(
    ctx: click.Context, arguments: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = str(arguments.get("path", "")).strip()
    query = str(arguments.get("query", "")).strip()
    if not path or not query:
        raise click.ClickException("ast.query requires 'path' and 'query'.")

    payload: Dict[str, Any] = {"path": path, "query": query}
    if arguments.get("captures"):
        payload["captures"] = [str(c) for c in (arguments.get("captures") or [])]
    return payload, {}


def _handle_ast_query_result(
    _: click.Context,
    __: Dict[str, Any],
    result: Dict[str, Any],
    ___: Dict[str, Any],
) -> None:
    nodes = result.get("nodes", [])
    click.echo(f"Matched {len(nodes)} node(s)")
    for node in nodes[:5]:
        snippet = (node.get("text") or "").strip().replace("\n", " ")
        click.echo(f"- {snippet[:120]}")


def _build_exec_payload(
    ctx: click.Context, arguments: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cmd_value = arguments.get("cmd") or arguments.get("command")
    cmd = str(cmd_value or "").strip()
    if not cmd:
        raise click.ClickException("exec requires 'cmd'.")

    payload: Dict[str, Any] = {"cmd": cmd}
    if arguments.get("args"):
        payload["args"] = [str(a) for a in (arguments.get("args") or [])]
    if arguments.get("cwd"):
        payload["cwd"] = str(arguments.get("cwd"))

    timeout_value = None
    if arguments.get("timeout_sec") is not None:
        timeout_value = arguments.get("timeout_sec")
    elif arguments.get("timeout") is not None:
        timeout_value = arguments.get("timeout")
    if timeout_value is not None:
        try:
            payload["timeout_sec"] = int(timeout_value)
        except (TypeError, ValueError):
            raise click.ClickException("timeout must be an integer.")

    return payload, {}


def _handle_exec_result(
    _: click.Context,
    __: Dict[str, Any],
    result: Dict[str, Any],
    ___: Dict[str, Any],
) -> None:
    exit_code = result.get("exit_code", 0)
    click.echo(f"Exit: {exit_code}")
    if result.get("stdout_tail"):
        click.echo(result["stdout_tail"].rstrip())
    if result.get("stderr_tail"):
        click.echo(result["stderr_tail"].rstrip())


def _build_fs_write_patch_payload(
    ctx: click.Context, arguments: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    diff = arguments.get("diff")
    if not isinstance(diff, str) or not diff.strip():
        raise click.ClickException("fs.write_patch requires 'diff' content.")

    settings: Settings = ctx.obj["settings"]
    if not settings.auto_approve_code:
        click.echo("Warning: auto_approve_code is disabled; applying patches may require manual review.")

    return {"diff": diff}, {}


def _handle_fs_write_patch_result(
    _: click.Context,
    __: Dict[str, Any],
    result: Dict[str, Any],
    ___: Dict[str, Any],
) -> None:
    applied = bool(result.get("applied"))
    changed = result.get("changed_files") or []
    if applied:
        click.echo("Patch applied")
    else:
        click.echo("Patch failed to apply")
        rejected = result.get("rejected_hunks") or []
        if rejected:
            click.echo("Rejected hunks:")
            for hunk in rejected:
                click.echo(f"- {hunk}")
    if changed:
        click.echo("Changed files:")
        for filename in changed:
            click.echo(f"- {filename}")


REGISTRY_INTENTS: Dict[str, RegistryIntent] = {
    "code.search": RegistryIntent(
        tool_name="code.search",
        payload_builder=_build_code_search_payload,
        result_handler=_handle_code_search_result,
    ),
    "fs.read": RegistryIntent(
        tool_name="fs.read",
        payload_builder=_build_fs_read_payload,
        result_handler=_handle_fs_read_result,
    ),
    "symbols.find": RegistryIntent(
        tool_name="symbols.find",
        payload_builder=_build_symbols_find_payload,
        result_handler=_handle_symbols_find_result,
        recovery_handler=_recover_symbols_find,
    ),
    "symbols.index": RegistryIntent(
        tool_name="symbols.index",
        payload_builder=_build_symbols_index_payload,
        result_handler=_handle_symbols_index_result,
    ),
    "ast.query": RegistryIntent(
        tool_name="ast.query",
        payload_builder=_build_ast_query_payload,
        result_handler=_handle_ast_query_result,
    ),
    "exec": RegistryIntent(
        tool_name="exec",
        payload_builder=_build_exec_payload,
        result_handler=_handle_exec_result,
        with_sandbox=False,
    ),
    "fs.write_patch": RegistryIntent(
        tool_name="fs.write_patch",
        payload_builder=_build_fs_write_patch_payload,
        result_handler=_handle_fs_write_patch_result,
    ),
}

INTENT_HANDLERS: Dict[str, Any] = {
    "list_directory": _deprecated_intent_handler(
        "list_directory",
        "Use 'exec' with an 'ls' command instead. Example: exec cmd='ls -la path/to/dir'",
    ),
    "respond_directly": _deprecated_intent_handler(
        "respond_directly",
        "Direct responses are emitted without a tool call; no handler is required.",
    ),
    # Registry-backed intents (synonyms share the same handler instances)
    "code.search": REGISTRY_INTENTS["code.search"],
    "code_search": REGISTRY_INTENTS["code.search"],
    "fs.read": REGISTRY_INTENTS["fs.read"],
    "fs_read": REGISTRY_INTENTS["fs.read"],
    "symbols.find": REGISTRY_INTENTS["symbols.find"],
    "symbols_find": REGISTRY_INTENTS["symbols.find"],
    "symbols.index": REGISTRY_INTENTS["symbols.index"],
    "symbols_index": REGISTRY_INTENTS["symbols.index"],
    "ast.query": REGISTRY_INTENTS["ast.query"],
    "ast_query": REGISTRY_INTENTS["ast.query"],
    "exec": REGISTRY_INTENTS["exec"],
    "execute": REGISTRY_INTENTS["exec"],
    "fs.write_patch": REGISTRY_INTENTS["fs.write_patch"],
    "fs_write_patch": REGISTRY_INTENTS["fs.write_patch"],
}




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




def _get_llm_client(ctx: click.Context):
    client = ctx.obj.get("llm_client")
    if client:
        return client
    settings: Settings = ctx.obj["settings"]
    if not settings.api_key:
        raise click.ClickException("No API key configured. Set DEVAGENT_API_KEY or update config file.")
    
    # Configure retry behavior
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
    
    # Configure timeout and retry behavior after creation
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

    click.echo(result.answer)
    if result.fallback_reason:
        click.echo(
            f"LLM fallback activated: {result.fallback_reason}",
            err=True,
        )

@click.group(cls=NaturalLanguageGroup)
@click.option("--config", "config_path", type=click.Path(path_type=Path), help="Path to config file.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging output.")
@click.option("--plan", is_flag=True, help="Use planning mode for all queries")
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None, verbose: bool, plan: bool) -> None:
    """AI-assisted development agent CLI.

    By default, queries execute directly without planning (fast mode).
    Use --plan for comprehensive planning and multi-step execution.

    Examples:
        devagent "show readme"         # Fast, direct execution
        devagent --plan "show readme"  # Slower, with planning

    Run `devagent shell` for persistent, multi-command sessions; standalone commands keep
    context only for the lifetime of the current process.
    """
    settings = load_settings(config_path)
    if verbose:
        settings.log_level = "DEBUG"
    configure_logging(settings.log_level, structured=settings.structured_logging)
    if not settings.api_key:
        LOGGER.warning("No API key configured. Some commands may fail.")
    ctx.obj = _build_context(settings)
    ctx.obj["default_use_planning"] = plan


@cli.command(name="query")
@click.argument("prompt", nargs=-1)
@click.option("--plan", "force_plan", is_flag=True, help="Force planning for this query")
@click.option("--direct", is_flag=True, help="Force direct execution (no planning)")
@click.pass_context
def query(
    ctx: click.Context,
    prompt: tuple[str, ...],
    force_plan: bool,
    direct: bool,
) -> None:
    """Execute a natural-language query using the ReAct workflow."""
    pending = " ".join(prompt).strip()
    if not pending:
        pending = str(ctx.meta.pop("_pending_nl_prompt", "")).strip()
    if not pending:
        pending = str(ctx.obj.pop("_pending_nl_prompt", "")).strip()
    if not pending:
        raise click.UsageError("Provide a request for the assistant.")

    _record_invocation(ctx, overrides={"prompt": pending, "mode": "query"})

    settings: Settings = ctx.obj["settings"]

    planning_pref = ctx.meta.pop("_use_planning", None)
    if planning_pref is None:
        planning_pref = ctx.obj.get("default_use_planning", False)

    use_planning = bool(planning_pref)
    if getattr(settings, "always_use_planning", False):
        use_planning = True

    if force_plan:
        use_planning = True
    elif direct:
        use_planning = False

    # Require API key; no router fallback without LLM
    if not settings.api_key:
        raise click.ClickException(
            "No API key configured (DEVAGENT_API_KEY). Natural language assistance requires an LLM."
        )

    # API key is present, try to get LLM client
    try:
        client = _get_llm_client(ctx)
    except click.ClickException as exc:
        # Re-raise LLM client creation errors (network issues, invalid key, etc.)
        raise click.ClickException(f"Failed to create LLM client: {exc}") from exc

    # Execute multi-step ReAct reasoning with LLM
    _execute_react_assistant(ctx, client, settings, pending, use_planning=use_planning)


def _execute_react_assistant(
    ctx: click.Context,
    client,
    settings: Settings,
    user_prompt: str,
    use_planning: bool = False,
) -> None:
    """Execute multi-step ReAct reasoning for natural language queries."""
    from ..providers.llm.base import Message

    import time

    start_time = time.time()
    execution_completed = False
    planning_active = bool(use_planning)
    emit_status_requested = bool(ctx.meta.pop("_emit_status_messages", False))
    supports_tool_calls = hasattr(client, "invoke_tools")
    should_emit_status = planning_active or supports_tool_calls or emit_status_requested
    execution_mode = "with planning" if planning_active else "direct"

    truncated_prompt = user_prompt if len(user_prompt) <= 50 else f"{user_prompt[:50]}..."
    direct_mode_announced = not planning_active if should_emit_status else True

    if should_emit_status:
        if planning_active:
            click.echo(f"🗺️ Planning: {truncated_prompt}")
            click.echo("🗺️ Planning mode enabled")
        else:
            click.echo(f"⚡ Executing: {truncated_prompt}")
            click.echo("⚡ Direct execution mode")
            direct_mode_announced = True

    def _finalize() -> None:
        if execution_completed and should_emit_status:
            execution_time = time.time() - start_time
            mode_label = execution_mode
            click.echo(f"\n✅ Completed in {execution_time:.1f}s ({mode_label})")

    # Create intent router to get available tools
    router = IntentRouter(client, settings)
    available_tools = getattr(router, "tools", [])

    if not supports_tool_calls:
        if planning_active and not direct_mode_announced:
            execution_mode = "direct"
            planning_active = False
            click.echo("⚡ Direct execution mode")
            direct_mode_announced = True
        decision: IntentDecision = router.route(user_prompt)
        if not decision.tool:
            text = str(decision.arguments.get("text", "")).strip()
            if text:
                click.echo(text)
            execution_completed = True
            _finalize()
            return

        handler = INTENT_HANDLERS.get(decision.tool)
        if not handler:
            raise click.ClickException(f"Intent tool '{decision.tool}' is not supported yet.")
        handler(ctx, decision.arguments)
        execution_completed = True
        _finalize()
        return

    # Determine task-specific iteration limits based on operation type
    strategy = ToolSelectionStrategy()
    query_lower = user_prompt.lower()
    tool_task_type = strategy.detect_task_type(user_prompt)

    enumerative_markers = [
        "list", "show all", "find all", "enumerate", "which files", "what files",
        "what methods", "which methods", "all methods", "every", "identify all",
        "remove which", "candidates", "unused"
    ]
    computational_markers = ["count", "how many", "number of", "total", "sum"]
    informational_markers = ["what is", "where is", "when", "who", "definition"]
    research_markers = ["explain", "describe", "tell me about", "how does", "why"]

    task_type: str

    if any(marker in query_lower for marker in computational_markers):
        task_type = "computational"
    elif any(marker in query_lower for marker in enumerative_markers):
        task_type = "enumeration"
    elif tool_task_type in {TaskType.RESEARCH, TaskType.CODE_EXPLORATION} or any(
        marker in query_lower for marker in research_markers
    ):
        task_type = "research"
    elif tool_task_type == TaskType.TESTING:
        task_type = "computational"
    elif any(marker in query_lower for marker in informational_markers):
        task_type = "informational"
    else:
        task_type = "general"

    # Determine global iteration cap (fallback 120) and allow overrides via config/env
    devagent_cfg = ctx.obj.get("devagent_config")
    if devagent_cfg is None:
        devagent_cfg = load_devagent_yaml()
        ctx.obj["devagent_config"] = devagent_cfg

    config_global_cap = None
    if devagent_cfg is not None:
        config_global_cap = getattr(devagent_cfg, "react_iteration_global_cap", None)

    env_cap_value: Optional[int] = None
    env_cap_raw = os.getenv("DEVAGENT_MAX_ITERATIONS")
    if env_cap_raw:
        try:
            env_cap_value = int(env_cap_raw)
        except ValueError:
            env_cap_value = None

    default_global_cap = 120
    global_max_iterations = default_global_cap

    if isinstance(config_global_cap, int) and config_global_cap > 0:
        global_max_iterations = config_global_cap

    if isinstance(env_cap_value, int) and env_cap_value > 0:
        global_max_iterations = env_cap_value

    iteration_cap = global_max_iterations

    planner_enabled = planning_active and getattr(settings, "react_enable_planner", True)
    structured_plan = None

    if planning_active and not planner_enabled:
        planning_active = False
        execution_mode = "direct"
        if not direct_mode_announced:
            click.echo("⚡ Direct execution mode")
            direct_mode_announced = True

    if planner_enabled:
        try:
            planner = Planner(client)
            structured_plan = planner.generate(user_prompt)
        except LLMError as exc:
            LOGGER.warning("Planner failed to produce structured plan: %s", exc)
            planning_active = False
            execution_mode = "direct"
            if not direct_mode_announced:
                click.echo("⚡ Direct execution mode")
                direct_mode_announced = True
            structured_plan = None

    if planning_active and structured_plan and getattr(structured_plan, "tasks", None):
        summary_text = structured_plan.summary or structured_plan.goal or "Structured plan"
        total_tasks = len(structured_plan.tasks)
        click.echo(
            f"🗺️ Plan created ({total_tasks} task{'s' if total_tasks != 1 else ''}): {summary_text}"
        )

        for task in structured_plan.tasks:
            click.echo(f"   Step {task.step_number}: {task.title}")
    elif planning_active:
        planning_active = False
        execution_mode = "direct"
        if not direct_mode_announced:
            click.echo("⚡ Direct execution mode")
            direct_mode_announced = True

    # Track tool usage to prevent redundant operations
    consecutive_fails = 0
    used_tools: set[str] = set()
    file_reads: set[str] = set()
    search_queries: set[str] = set()
    tool_repeat_counts: Dict[str, int] = {}

    def _normalize_read_targets(arguments: Dict[str, Any]) -> List[str]:
        targets: List[str] = []
        path_value = arguments.get("path")
        if path_value:
            targets.append(str(path_value))
        paths_value = arguments.get("paths")
        if isinstance(paths_value, list):
            targets.extend(str(item) for item in paths_value if item)
        elif paths_value:
            targets.append(str(paths_value))
        return targets

    # Start conversation with improved system prompt including tool selection guidance
    tool_guidance = ""
    if task_type == "computational":
        tool_guidance = "\nFOR COMPUTATIONAL TASKS (counting, measuring, calculating):\n" \
                       "- Prefer shell commands (sandbox_exec) for bulk file operations\n" \
                       "- Generate and run scripts for complex computations\n" \
                       "- Use 'find', 'wc', 'ls -la', etc. for file system analysis\n" \
                       "- Avoid reading individual files when aggregate data is needed"
    elif task_type == "enumeration":
        tool_guidance = "\nFOR ENUMERATION TASKS (listing, finding all):\n" \
                       "- Use code_search for finding patterns across files\n" \
                       "- Use sandbox_exec for directory traversal and filtering\n" \
                       "- Combine multiple search strategies if needed\n" \
                       "- Aggregate results before presenting"
    
    messages = [
        Message(
            role="system", 
            content="You are a helpful assistant for a software development CLI tool called devagent. "
                   "Use the available tools efficiently to answer the user's question. "
                   f"You have a budget of {iteration_cap} iterations to complete this task. "
                   f"Plan your tool usage accordingly and prioritize synthesis as you approach the limit. "
                   "IMPORTANT GUIDELINES:\n"
                   "1. Choose the most appropriate tool for the task type\n"
                   "2. Avoid calling the same tool with identical arguments repeatedly\n"
                   "3. Don't read individual files when bulk operations are more efficient\n"
                   "4. When you have sufficient information, provide a final response WITHOUT calling more tools\n"
                   "5. If a tool fails, try a different approach rather than repeating\n"
                   "6. For file system operations, prefer shell commands over individual file reads\n"
                   "7. Generate scripts when you need to perform complex computations" + tool_guidance
        ),
        Message(role="user", content=user_prompt)
    ]
    
    iteration = 0

    while iteration < iteration_cap:
        iteration += 1
        
        try:
            # Get LLM response with tools
            result = client.invoke_tools(messages, tools=available_tools, temperature=0.1)
            
            # Add LLM response to conversation
            assistant_message = Message(
                role="assistant",
                content=result.message_content,
                tool_calls=result.raw_tool_calls,
            )
            if assistant_message.content is not None or assistant_message.tool_calls:
                messages.append(assistant_message)
            
            # If no tool calls, we're done
            if not result.calls:
                if result.message_content:
                    click.echo(result.message_content)
                else:
                    click.echo("I was unable to provide a complete answer.")
                execution_completed = True
                _finalize()
                return
            
            # Check for redundant tool usage before executing
            redundant_calls = 0
            last_tool_failed = consecutive_fails > 0
            
            # Count redundant operations first
            for tool_call in result.calls:
                call_signature = tool_signature(tool_call)
                
                # Check for redundant operations, but exclude error recovery
                if tool_call.name in FILE_READ_TOOLS:
                    targets = _normalize_read_targets(tool_call.arguments)
                    if targets and all(target in file_reads for target in targets) and not last_tool_failed:
                        redundant_calls += 1
                elif tool_call.name in SEARCH_TOOLS:
                    query = tool_call.arguments.get("query")
                    if query in search_queries and not last_tool_failed:
                        redundant_calls += 1
                elif call_signature in used_tools and not last_tool_failed:
                    redundant_calls += 1
            
            # Only flag as redundant if most calls are truly redundant (not error recovery)
            if redundant_calls >= len(result.calls) * 0.7 and len(result.calls) > 1:  # 70% or more are redundant
                click.echo("🚫 Detected redundant tool usage. Stopping to avoid loops.")
                execution_completed = True
                _finalize()
                return
            
            # Execute ALL tool calls (DeepSeek API requires responses to all tool calls)
            total_calls = len(result.calls)
            successful_calls = 0
            
            # Initialize file reading context for sequential tracking
            file_reading_context = {"last_file_read": None, "file_line_counts": {}}
            
            for call_index, tool_call in enumerate(result.calls, 1):
                import time
                start_time = time.time()
                
                call_signature = tool_signature(tool_call)

                # Track tool usage
                used_tools.add(call_signature)
                
                # Track specific operations
                if tool_call.name in FILE_READ_TOOLS:
                    for target in _normalize_read_targets(tool_call.arguments):
                        file_reads.add(target)
                elif tool_call.name in SEARCH_TOOLS and tool_call.arguments.get("query"):
                    search_queries.add(tool_call.arguments["query"])
                
                # Find and execute the appropriate intent handler
                handler = INTENT_HANDLERS.get(tool_call.name)
                if not handler:
                    error_msg = f"Tool '{tool_call.name}' is not supported."
                    tool_call_id = getattr(tool_call, 'call_id', None) or getattr(tool_call, 'id', 'unknown')
                    messages.append(Message(
                        role="tool",
                        content=f"Error: {error_msg}",
                        tool_call_id=tool_call_id
                    ))
                    
                    # Single log for unsupported tool
                    log_message = f"❌ {tool_call.name} → Tool not supported"
                    click.echo(log_message)
                    consecutive_fails += 1
                    continue
                
                # Capture tool output
                import io
                import contextlib
                
                # Redirect stdout to capture tool output
                captured_output = io.StringIO()
                # Count repetitions for this tool signature
                call_signature = tool_signature(tool_call)
                repeat_count = tool_repeat_counts.get(call_signature, 0) + 1
                tool_repeat_counts[call_signature] = repeat_count
                
                tool_output_for_message = None

                call_success = False
                try:
                    with contextlib.redirect_stdout(captured_output):
                        handler(ctx, tool_call.arguments)
                    tool_output_raw = captured_output.getvalue()
                    tool_output = tool_output_raw.strip()
                    if not tool_output:
                        tool_output = "Tool executed successfully (no output)"

                    max_chars = max(
                        MIN_TOOL_OUTPUT_CHARS,
                        getattr(settings, "max_tool_output_chars", DEFAULT_MAX_TOOL_OUTPUT_CHARS),
                    )
                    summarized_output = summarize_text(tool_output, max_chars)
                    if summarized_output != tool_output:
                        summary_note = f" (truncated to {max_chars} chars)"
                        artifact_reference = ""
                        try:
                            artifact_path = write_artifact(tool_output)
                            try:
                                display_path = artifact_path.relative_to(Path.cwd())
                            except ValueError:
                                display_path = artifact_path
                            artifact_reference = f"\nFull output saved to {display_path}"
                        except Exception:
                            artifact_reference = ""
                        tool_output_for_message = f"{summarized_output}{summary_note}{artifact_reference}"
                    else:
                        tool_output_for_message = tool_output

                    execution_time = time.time() - start_time

                    # Single comprehensive log message after execution
                    log_message = _format_enhanced_tool_log(
                        tool_call, repeat_count, execution_time, tool_output, 
                        success=True, file_context=file_reading_context
                    )
                    click.echo(log_message)

                    successful_calls += 1
                    consecutive_fails = 0  # Reset fail counter on success
                    call_success = True
                    
                except FileNotFoundError as exc:
                    execution_time = time.time() - start_time
                    tool_output = f"File not found: {exc}"

                    # Single comprehensive error log with suggestions when possible
                    log_message = _format_enhanced_tool_log(
                        tool_call, repeat_count, execution_time, tool_output,
                        success=False, file_context=file_reading_context
                    )
                    click.echo(log_message)

                    consecutive_fails += 1

                    tool_output_for_message = tool_output
                    call_success = False

                except Exception as exc:
                    execution_time = time.time() - start_time
                    tool_output = f"Error executing {tool_call.name}: {exc}"
                    
                    # Single comprehensive error log
                    log_message = _format_enhanced_tool_log(
                        tool_call, repeat_count, execution_time, tool_output,
                        success=False, file_context=file_reading_context
                    )
                    click.echo(log_message)

                    consecutive_fails += 1

                    tool_output_for_message = tool_output
                    call_success = False

                # Add tool result to conversation  
                # For tool messages, we need to include the tool_call_id if the API requires it
                tool_call_id = getattr(tool_call, 'call_id', None) or getattr(tool_call, 'id', None)
                tool_message = Message(
                    role="tool",
                    content=tool_output_for_message or tool_output,
                    tool_call_id=tool_call_id,
                )
                messages.append(tool_message)
            
            if consecutive_fails >= 3:
                click.echo("🚫 Multiple consecutive tool failures. Stopping to avoid loops.")
                execution_completed = True
                _finalize()
                return

            # Only add completion hint for informational tasks after sufficient attempts
            if (task_type == "informational" and iteration >= 6 and successful_calls >= 2) or \
               (task_type in ["computational", "enumeration"] and iteration >= 8 and successful_calls >= 3):
                # Check if we have concrete results before hinting completion
                has_concrete_results = any("completed" in str(msg.content).lower() or 
                                         "found" in str(msg.content).lower() or
                                         "result" in str(msg.content).lower() 
                                         for msg in messages[-3:] if hasattr(msg, 'content') and msg.content)
                
                if has_concrete_results:
                    from ..providers.llm.base import Message
                    hint_message = Message(
                        role="user",
                        content="Based on the information gathered above, please provide a complete answer to my original question."
                    )
                    messages.append(hint_message)
            
        except (LLMConnectionError, LLMTimeoutError, LLMRetryExhaustedError) as exc:
            click.echo(f"⚠️  Unable to reach the LLM: {exc}")
            click.echo("   Falling back to offline analysis using local heuristics.")
            _handle_question_without_llm(user_prompt, reason="LLM unavailable")
            execution_mode = "direct"
            execution_completed = True
            _finalize()
            return
        except LLMError as exc:
            click.echo(f"❌ ReAct execution failed: {exc}")
            break
        except Exception as exc:
            click.echo(f"❌ ReAct execution failed: {exc}")
            break
    
    if iteration >= iteration_cap:
        click.echo(f"⚠️  Reached maximum iteration limit ({iteration_cap}).")
        click.echo("Please refine your request or increase DEVAGENT_MAX_ITERATIONS if you need more steps.")
        execution_completed = True
        _finalize()


@dataclass
class ToolLogDetails:
    """Structured details for a tool log line."""

    metrics: str
    status: str = "success"  # one of: success, info, warning, error
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    suggestions: Tuple[str, ...] = field(default_factory=tuple)


def _get_main_argument(tool_call) -> str:
    """Extract the main argument for concise logging with enhanced context."""
    args = tool_call.arguments or {}
    
    if tool_call.name in FILE_READ_TOOLS:
        primary_path = args.get("path")
        if not primary_path:
            paths_value = args.get("paths")
            if isinstance(paths_value, list) and paths_value:
                primary_path = paths_value[0]
            elif paths_value:
                primary_path = paths_value
        path = str(primary_path or "")
        # Add line range information if specified
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        max_lines = args.get("max_lines")
        
        if start_line is not None:
            if end_line is not None:
                return f"{path}:{start_line}-{end_line}"
            elif max_lines is not None and max_lines != 200:  # 200 is default
                return f"{path}:{start_line}+{max_lines}"
            else:
                return f"{path}:{start_line}+"
        return path
        
    elif tool_call.name in SEARCH_TOOLS:
        return args.get("query", "")
    elif tool_call.name in ("exec", "sandbox.exec", "sandbox_exec", "execute"):
        cmd_value = args.get("cmd") or args.get("command")
        cmd = str(cmd_value or "")
        extra_args = args.get("args")
        if extra_args:
            if isinstance(extra_args, (list, tuple)):
                arg_parts = [str(item) for item in extra_args]
            else:
                arg_parts = [str(extra_args)]
            joined_args = " ".join(part for part in arg_parts if part)
            if cmd:
                return f"{cmd} {joined_args}".strip()
            return joined_args
        return cmd
    else:
        # For unknown tools, try to find a reasonable main argument
        for key in ["path", "paths", "query", "cmd", "command", "goal", "question"]:
            if key in args and args[key]:
                value = args[key]
                if key == "paths" and isinstance(value, list):
                    return value[0] if value else ""
                return str(value)
        return ""

def _format_enhanced_tool_log(
    tool_call, 
    repeat_count: int, 
    execution_time: float, 
    tool_output: str = "",
    success: bool = True,
    file_context: dict = None
) -> str:
    """Format tool execution log with enhanced context and metrics - single comprehensive log."""
    main_arg = _get_main_argument(tool_call)
    
    # Map raw tool names to consistent display names using centralized helpers
    display_name = display_tool_name(tool_call.name)
    canonical_name = canonical_tool_name(tool_call.name)
    category = tool_category(tool_call.name)
    
    # Handle failed operations with enhanced error messages
    if not success:
        return _format_error_message(tool_call, tool_output, main_arg)
    
    # Handle file reading with line range context
    if category == "file_read" and file_context:
        tree_msg = _format_file_reading_tree(tool_call, main_arg, tool_output, file_context)
        if tree_msg:
            return tree_msg
    
    # Build base log message with argument
    if main_arg:
        log_msg = f"{display_name} \"{main_arg}\""
    else:
        log_msg = display_name

    # Add repeat indicator for multiple calls
    if repeat_count > 1:
        log_msg += f" ({repeat_count}x)"
    
    # Extract and add meaningful metrics from tool output
    metrics_info = _extract_tool_metrics(canonical_name, tool_output or "", tool_call=tool_call)
    metrics = metrics_info.metrics.strip()
    if metrics:
        log_msg += f" → {metrics}"

    # Add timing for slow operations (>1 second)
    if execution_time > 1.0:
        log_msg += f" ({execution_time:.1f}s)"

    # Add status indicator emoji
    status_indicator = _resolve_status_indicator(canonical_name, metrics_info.status, success)
    if status_indicator:
        log_msg = f"{status_indicator} {log_msg}"

    return log_msg

def _format_error_message(tool_call, tool_output: str, main_arg: str) -> str:
    """Format enhanced error messages with shared analysis heuristics."""

    tool_name = tool_call.name
    display_name = display_tool_name(tool_name)
    canonical_name = canonical_tool_name(tool_name)
    details = _extract_tool_metrics(canonical_name, tool_output, tool_call=tool_call)
    args = getattr(tool_call, "arguments", {}) or {}

    def build_generic_message(argument: str) -> str:
        message = details.error_message or _fallback_error_message(tool_output)
        argument = argument or main_arg
        return f"❌ {display_name} \"{argument}\" → Error: {message}"

    if tool_name in FILE_READ_TOOLS:
        path = _extract_primary_path(args)
        if details.error_type == "not_found":
            if details.suggestions:
                suggestion_text = ", ".join(details.suggestions[:2])
                return f"❌ {display_name} \"{path}\" → Not found (try: {suggestion_text})"
            return f"❌ {display_name} \"{path}\" → Not found"
        return build_generic_message(path or main_arg)

    if tool_name in SEARCH_TOOLS:
        query = str(args.get("query", ""))
        return build_generic_message(query)

    if tool_name in {"exec", "sandbox.exec", "sandbox_exec", "execute"}:
        raw_cmd = args.get("cmd") or args.get("command", "")
        cmd_display = str(raw_cmd)
        if len(cmd_display) > 50:
            cmd_display = cmd_display[:50] + "..."
        return build_generic_message(cmd_display)

    return build_generic_message(main_arg)


def _get_path_suggestions(failed_path: str) -> List[str]:
    """Get path suggestions when a file read fails."""
    if not failed_path:
        return []
        
    suggestions = []
    repo_root = Path.cwd()
    
    # Extract filename from failed path
    filename = Path(failed_path).name
    if not filename:
        return []
    
    # Search for files with similar names in the project
    for root_dir in [repo_root / "ai_dev_agent", repo_root / "tests", repo_root / "docs", repo_root]:
        if not root_dir.exists():
            continue
            
        # Look for exact filename matches
        for path in root_dir.rglob(f"*{filename}*"):
            if path.is_file():
                rel_path = str(path.relative_to(repo_root))
                if rel_path not in suggestions:
                    suggestions.append(rel_path)
                if len(suggestions) >= 3:
                    break
        if len(suggestions) >= 3:
            break
    
    return suggestions[:3]


def _format_file_reading_tree(tool_call, main_arg: str, tool_output: str, file_context: dict) -> str:
    """Format file reading operations in a tree structure for sequential reads."""
    args = tool_call.arguments or {}
    path = args.get("path", "")
    start_line = args.get("start_line")
    canonical_name = canonical_tool_name(tool_call.name)
    display_name = display_tool_name(tool_call.name)
    
    # Check if this is a continuation of a previous file read
    last_read = file_context.get("last_file_read")
    if last_read and last_read["path"] == path:
        # This is a continuation
        metrics = _extract_tool_metrics(canonical_name, tool_output, tool_call=tool_call).metrics
        
        # Update context for next read
        file_context["last_file_read"] = {
            "path": path,
            "last_end_line": _extract_end_line_from_output(tool_output),
            "is_continuation": True
        }
        
        continuation_symbol = "├─" if not _is_file_complete(tool_output) else "└─"
        return f"📖 {continuation_symbol} continuing from line {start_line} → {metrics}"
    
    else:
        # This is a new file read
        metrics = _extract_tool_metrics(canonical_name, tool_output, tool_call=tool_call).metrics
        file_context["last_file_read"] = {
            "path": path,
            "last_end_line": _extract_end_line_from_output(tool_output),
            "is_continuation": False
        }
        
        return f"📖 {display_name} \"{main_arg}\" → {metrics}"


def _extract_end_line_from_output(tool_output: str) -> int:
    """Extract the ending line number from fs.read output."""
    import re
    match = re.search(r'lines (\d+)-(\d+) of (\d+)', tool_output)
    if match:
        return int(match.group(2))
    return 0


def _is_file_complete(tool_output: str) -> bool:
    """Check if the file reading is complete."""
    return "more lines not shown" not in tool_output


def _extract_tool_metrics(tool_name: str, tool_output: str, *, tool_call=None) -> ToolLogDetails:
    """Extract meaningful metrics, status, and error details from tool output."""

    text = (tool_output or "").strip()
    if not text:
        return ToolLogDetails("no output", status="info")

    args = getattr(tool_call, "arguments", {}) or {}

    if tool_name in ("code.search", "code_search", "search"):
        return _analyze_search_output(text)

    if tool_name in ("fs.read", "fs_read"):
        return _analyze_file_read_output(text, args)

    if tool_name in ("exec", "sandbox.exec", "sandbox_exec", "execute"):
        command_text = str(args.get("cmd") or args.get("command") or "")
        return _analyze_command_output(text, command_text)

    first_line = _first_displayable_line(text)
    if first_line:
        return ToolLogDetails(first_line)
    return ToolLogDetails("executed")


def _analyze_search_output(text: str) -> ToolLogDetails:
    if "No matches found" in text:
        return ToolLogDetails("0 matches found")

    lines = text.splitlines()
    result_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("Listing") or stripped.startswith("Found"):
            continue
        if ":" in line and any(char.isdigit() for char in line):
            result_lines.append(line)
    if result_lines:
        return ToolLogDetails(f"{len(result_lines)} matches found")

    non_header = [line for line in lines if line.strip() and not line.startswith(("Listing", "Found", "No matches"))]
    if non_header:
        return ToolLogDetails(f"{len(non_header)} results")

    first_line = _first_displayable_line(text)
    return ToolLogDetails(first_line or "search completed")


def _analyze_file_read_output(text: str, args: Dict[str, Any]) -> ToolLogDetails:
    lower = text.lower()
    path = _extract_primary_path(args)
    if any(keyword in lower for keyword in ("not found", "no such file", "enoent")):
        suggestions = tuple(_get_path_suggestions(path)) if path else tuple()
        return ToolLogDetails(
            metrics="not found",
            status="error",
            error_message="Not found",
            error_type="not_found",
            suggestions=suggestions,
        )

    patterns = [
        r"Reading .* \(lines (\d+)-(\d+) of (\d+)\)",
        r"lines (\d+)-(\d+) of (\d+)",
        r"\((\d+) lines\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 3:
            start, end, total = (int(groups[0]), int(groups[1]), int(groups[2]))
            read_lines = end - start + 1
            if end >= total:
                return ToolLogDetails(f"{total} lines read")
            return ToolLogDetails(f"{read_lines} lines (of {total})")
        if len(groups) == 1:
            return ToolLogDetails(f"{int(groups[0])} lines read")

    lines = text.splitlines()
    content_lines: List[str] = []
    skip_prefixes = ("== ", "Reading ", "No content", "lines ", "more lines not shown")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(line.startswith(prefix) for prefix in skip_prefixes):
            continue
        if re.match(r"\s*\d+:\s", line):
            content_lines.append(line)
        elif not any(token in line for token in ("==", "Reading", "...")):
            content_lines.append(line)
    if content_lines:
        return ToolLogDetails(f"{len(content_lines)} lines read")

    first_line = _first_displayable_line(text)
    return ToolLogDetails(first_line or "content read")


def _extract_primary_path(args: Dict[str, Any]) -> str:
    path = args.get("path")
    if path:
        return str(path)

    paths = args.get("paths")
    if isinstance(paths, (list, tuple)) and paths:
        return str(paths[0])
    if paths:
        return str(paths)
    return ""


def _analyze_command_output(text: str, command_text: str) -> ToolLogDetails:
    exit_code = _parse_exit_code(text)
    if exit_code is not None:
        if exit_code == 0:
            return ToolLogDetails("✓")
        if exit_code == 1 and _is_pattern_search_command(command_text):
            return ToolLogDetails("no matches", status="info")

        error_line = _extract_first_error_line(text)
        summary, hint = _normalize_error_hint(command_text, error_line)
        detail_parts = [part for part in (summary or error_line, hint) if part]
        detail_text = "; ".join(detail_parts)
        metrics = f"exit {exit_code}"
        metrics_text = f"{metrics} ({detail_text})" if detail_text else metrics
        return ToolLogDetails(
            metrics=metrics_text,
            status="warning",
            error_message=detail_text or metrics,
            error_type="exit_code",
        )

    lowered = text.lower()
    if "completed successfully" in lowered or "done" in lowered:
        return ToolLogDetails("✓")

    first_line = _first_displayable_line(text)
    return ToolLogDetails(first_line or "executed")


def _first_displayable_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("Full output saved"):
            return stripped
    return ""


def _fallback_error_message(tool_output: str) -> str:
    if ":" in tool_output:
        return tool_output.split(":")[-1].strip()
    return tool_output.strip()


def _parse_exit_code(output: str) -> Optional[int]:
    """Return an exit code parsed from tool output, if present."""
    match = re.search(r"Exit:\s*(\d+)", output)
    if match:
        return int(match.group(1))
    match = re.search(r"exit code[:\s]*(\d+)", output.lower())
    if match:
        return int(match.group(1))
    return None


def _extract_first_error_line(output: str) -> str:
    """Extract the first meaningful error line from sandbox output."""
    lines = output.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Exit:") or stripped.lower().startswith("exit code"):
            continue
        if stripped.startswith("Full output saved"):
            continue
        return stripped
    return ""


def _normalize_error_hint(command_text: str, error_line: str) -> tuple[Optional[str], Optional[str]]:
    """Provide cleaner error summaries and optional hints for common failures."""
    if not error_line:
        return None, None

    lower_error = error_line.lower()
    lower_command = command_text.lower()

    if "repetition-operator operand" in lower_error or "invalid regular expression" in lower_error:
        hint = None
        if "grep" in lower_command:
            hint = "use -E"
        elif "rg" in lower_command:
            hint = "use -e"
        return "invalid regex", hint

    if "permission denied" in lower_error:
        return "permission denied", None

    if "not found" in lower_error and _is_pattern_search_command(command_text):
        return "pattern not found", None

    return error_line, None


def _is_pattern_search_command(command_text: str) -> bool:
    """Heuristic to identify commands that treat exit code 1 as 'no matches'."""
    lowered = command_text.lower()
    if not lowered:
        return False
    search_tools = ("grep", "rg", "ripgrep", "ag", "ack", "fd", "find")
    return any(tool in lowered for tool in search_tools)


def _get_tool_status_indicator(tool_name: str, success: bool) -> str:
    """Get a status indicator for the tool based on its type."""
    if not success:
        return "❌"

    category = tool_category(tool_name)
    indicators = {
        "file_read": "📖",
        "search": "🔍",
        "command": "⚡",
        "list": "📁",
        "ast": "🧠",
        "symbols": "🔣",
    }
    return indicators.get(category, "")


def _resolve_status_indicator(tool_name: str, status: str, success: bool) -> str:
    """Pick an appropriate indicator, respecting informational and warning states."""
    if status == "warning":
        return "⚠"
    if status == "error":
        return "❌"
    return _get_tool_status_indicator(tool_name, success)
def _provide_research_summary(messages: List[Any], client, settings: Settings) -> None:
    """Provide a summary when research reaches limits or becomes redundant."""
    from ..providers.llm.base import Message
    from ..core.utils.context_budget import config_from_settings, ensure_context_budget

    # Add synthesis request to the existing conversation for better context
    synthesis_message = Message(
        role="user",
        content=(
            "Based on all the information gathered above, please provide a "
            "comprehensive answer to my original question. Synthesize all findings "
            "into a clear, complete response."
        )
    )
    
    # Use the full conversation context for better synthesis
    messages_with_synthesis = messages + [synthesis_message]
    
    # Reuse existing context budget management
    config = config_from_settings(settings)
    pruned_messages = ensure_context_budget(messages_with_synthesis, config)
    
    try:
        final_result = client.complete(pruned_messages, temperature=0.1)
        if final_result:
            click.echo(final_result)
        else:
            # Fall back to existing summary builder if needed
            fallback_summary = _build_fallback_tool_summary(messages)
            click.echo(fallback_summary or "Unable to provide a complete summary.")
    except Exception:
        # Use existing fallback on any error
        fallback_summary = _build_fallback_tool_summary(messages)
        click.echo(fallback_summary or "Unable to provide a complete summary.")


@dataclass
class CodeLineSnippet:
    number: Optional[int]
    text: str


@dataclass
class BaseFinding:
    """Common base for research summary findings."""

    latest_index: int = field(default=-1, init=False)

    def stamp(self, index: int) -> None:
        if index > self.latest_index:
            self.latest_index = index


@dataclass
class FileFinding(BaseFinding):
    path: str
    snippets: List[CodeLineSnippet] = field(default_factory=list)
    ranges: List[Tuple[Optional[int], Optional[int]]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    total_lines: Optional[int] = None

    def add_snippet(self, snippet: CodeLineSnippet, max_items: int = 12) -> None:
        text = snippet.text.strip()
        if not text:
            return
        for existing in self.snippets:
            if existing.number == snippet.number and existing.text.strip() == text:
                return
        self.snippets.append(CodeLineSnippet(snippet.number, text))
        if len(self.snippets) > max_items:
            self.snippets = self.snippets[-max_items:]

    def add_range(self, start: Optional[int], end: Optional[int]) -> None:
        if start is None and end is None:
            return
        candidate = (start, end)
        if candidate not in self.ranges:
            self.ranges.append(candidate)

    def add_note(self, note: str, max_items: int = 4) -> None:
        note = note.strip()
        if not note:
            return
        if note in self.notes:
            return
        self.notes.append(note)
        if len(self.notes) > max_items:
            self.notes = self.notes[-max_items:]


@dataclass
class SearchMatch:
    path: str
    line: Optional[int]
    column: Optional[int]
    preview: str


@dataclass
class SearchFinding(BaseFinding):
    query: str
    matches: List[SearchMatch] = field(default_factory=list)
    additional_note: Optional[str] = None
    status: str = "success"

    def add_match(self, match: SearchMatch, max_items: int = 12) -> None:
        for existing in self.matches:
            if (
                existing.path == match.path
                and existing.line == match.line
                and existing.column == match.column
                and existing.preview == match.preview
            ):
                return
        self.matches.append(match)
        if len(self.matches) > max_items:
            self.matches = self.matches[-max_items:]


@dataclass
class ASTFinding(BaseFinding):
    path: str
    query: str
    nodes: List[str] = field(default_factory=list)
    match_count: Optional[int] = None

    def add_node(self, node: str, max_items: int = 10) -> None:
        cleaned = _truncate_text(node.strip(), 160)
        if not cleaned:
            return
        if cleaned in self.nodes:
            return
        self.nodes.append(cleaned)
        if len(self.nodes) > max_items:
            self.nodes = self.nodes[-max_items:]


@dataclass
class SymbolMatch:
    path: str
    line: Optional[int]
    kind: str


@dataclass
class SymbolFinding(BaseFinding):
    name: str
    kind: Optional[str] = None
    lang: Optional[str] = None
    matches: List[SymbolMatch] = field(default_factory=list)
    status: Optional[str] = None

    def add_match(self, match: SymbolMatch, max_items: int = 12) -> None:
        for existing in self.matches:
            if (
                existing.path == match.path
                and existing.line == match.line
                and existing.kind == match.kind
            ):
                return
        self.matches.append(match)
        if len(self.matches) > max_items:
            self.matches = self.matches[-max_items:]


@dataclass
class CommandFinding(BaseFinding):
    command: str
    exit_code: Optional[int]
    output_lines: List[str] = field(default_factory=list)


@dataclass
class GenericFinding(BaseFinding):
    tool_name: str
    text: str


@dataclass
class FunctionFinding(BaseFinding):
    signature: str
    path: Optional[str]
    line: Optional[int]
    source: str


class _ResearchSummaryAggregator:
    def __init__(self) -> None:
        self.files: Dict[str, FileFinding] = {}
        self.searches: Dict[str, SearchFinding] = {}
        self.ast: Dict[Tuple[str, str], ASTFinding] = {}
        self.symbols: Dict[Tuple[str, str, str], SymbolFinding] = {}
        self.commands: List[CommandFinding] = []
        self.generics: List[GenericFinding] = []
        self.functions: Dict[Tuple[str, str], FunctionFinding] = {}

    def ingest(
        self,
        tool_name: Optional[str],
        arguments: Dict[str, Any],
        content: str,
        index: int,
    ) -> None:
        if not content.strip():
            return

        canonical = canonical_tool_name(tool_name)
        category = tool_category(tool_name)

        if category == "file_read":
            for finding in _parse_file_read_result(arguments, content):
                self._merge_file_finding(finding, index)
        elif category == "search":
            finding = _parse_search_result(arguments, content)
            if finding:
                self._merge_search_finding(finding, index)
        elif category == "ast":
            finding = _parse_ast_result(arguments, content)
            if finding:
                self._merge_ast_finding(finding, index)
        elif category == "symbols":
            finding = _parse_symbol_result(arguments, content)
            if finding:
                self._merge_symbol_finding(finding, index)
        elif category == "command":
            finding = _parse_command_result(arguments, content)
            if finding:
                finding.stamp(index)
                self.commands.append(finding)
                self.commands = self.commands[-10:]
        else:
            snippet = _truncate_text(content.strip(), 320)
            if snippet:
                generic = GenericFinding(tool_name or "unknown", snippet)
                generic.stamp(index)
                self.generics.append(generic)
                self.generics = self.generics[-10:]

    def build_summary(self, max_chars: int = 3200) -> Optional[str]:
        if not (
            self.files
            or self.searches
            or self.ast
            or self.symbols
            or self.commands
            or self.generics
        ):
            return None

        lines: List[str] = []
        used_chars = 0
        truncated = False
        ellipsis_line = "... (more findings omitted)"

        def append_line(text: str) -> bool:
            nonlocal used_chars, truncated
            if truncated:
                return False
            if text is None or text == "":
                return True
            addition = len(text)
            newline = 1 if lines else 0
            if used_chars + newline + addition <= max_chars:
                if newline:
                    used_chars += 1
                lines.append(text)
                used_chars += addition
                return True
            if ellipsis_line not in lines:
                newline = 1 if lines else 0
                if used_chars + newline + len(ellipsis_line) <= max_chars:
                    if newline:
                        used_chars += 1
                    lines.append(ellipsis_line)
                    used_chars += len(ellipsis_line)
            truncated = True
            return False

        append_line("Research findings (most recent first):")

        if self.files and not truncated:
            append_line("- Files read:")
            for finding in sorted(self.files.values(), key=lambda item: item.latest_index, reverse=True):
                detail_parts: List[str] = []
                range_summary = _format_range_summary(finding)
                if range_summary:
                    detail_parts.append(range_summary)
                if finding.notes:
                    detail_parts.append(
                        "; ".join(_truncate_text(note, 120) for note in finding.notes)
                    )
                detail = "; ".join(part for part in detail_parts if part)
                base_line = f"  - {finding.path}"
                if detail:
                    base_line += f" ({detail})"
                if not append_line(base_line):
                    break
                for snippet in _select_highlight_lines(finding.snippets):
                    line_number = f"{snippet.number}" if snippet.number is not None else "-"
                    snippet_text = _truncate_text(snippet.text)
                    if not append_line(f"    {line_number}: {snippet_text}"):
                        break
                if truncated:
                    break

        if self.searches and not truncated:
            append_line("- Searches:")
            for finding in sorted(self.searches.values(), key=lambda item: item.latest_index, reverse=True):
                query_display = _truncate_text(finding.query or "(no query)", 120)
                base_line = f"  - \"{query_display}\""
                if finding.matches:
                    base_line += f" ({len(finding.matches)} match{'es' if len(finding.matches) != 1 else ''})"
                elif finding.status == "no_matches":
                    base_line += " (no matches)"
                if not append_line(base_line):
                    break
                if finding.additional_note:
                    append_line(f"    {_truncate_text(finding.additional_note, 160)}")
                for match in _select_highlight_matches(finding.matches):
                    location = match.path
                    if match.line is not None:
                        location += f":{match.line}"
                        if match.column is not None:
                            location += f":{match.column}"
                    preview = _truncate_text(match.preview)
                    entry = f"    {location}"
                    if preview:
                        entry += f" {preview}"
                    if not append_line(entry):
                        break
                if truncated:
                    break

        if self.ast and not truncated:
            append_line("- AST queries:")
            for finding in sorted(self.ast.values(), key=lambda item: item.latest_index, reverse=True):
                descriptor_parts: List[str] = []
                if finding.path:
                    descriptor_parts.append(finding.path)
                if finding.query:
                    descriptor_parts.append(f"query \"{_truncate_text(finding.query, 120)}\"")
                header = "  - " + ", ".join(descriptor_parts) if descriptor_parts else "  - (query)"
                if finding.match_count is not None:
                    header += f" ({finding.match_count} node{'s' if finding.match_count != 1 else ''})"
                if not append_line(header):
                    break
                for node in finding.nodes[:3]:
                    if not append_line(f"    {_truncate_text(node, 160)}"):
                        break
                if truncated:
                    break

        if self.symbols and not truncated:
            append_line("- Symbol lookups:")
            for finding in sorted(self.symbols.values(), key=lambda item: item.latest_index, reverse=True):
                descriptor = finding.name or "(no name)"
                qualifiers: List[str] = []
                if finding.kind:
                    qualifiers.append(f"kind={finding.kind}")
                if finding.lang:
                    qualifiers.append(f"lang={finding.lang}")
                if finding.matches:
                    qualifiers.append(
                        f"{len(finding.matches)} match{'es' if len(finding.matches) != 1 else ''}"
                    )
                elif finding.status:
                    qualifiers.append(_truncate_text(finding.status, 120))
                line = f"  - {descriptor}"
                if qualifiers:
                    line += f" ({', '.join(qualifiers)})"
                if not append_line(line):
                    break
                for match in finding.matches[:4]:
                    location = match.path
                    if match.line is not None:
                        location += f":{match.line}"
                    detail = match.kind.strip()
                    entry = f"    {location}"
                    if detail:
                        entry += f" {detail}"
                    if not append_line(entry):
                        break
                if truncated:
                    break

        if self.commands and not truncated:
            append_line("- Commands:")
            for finding in sorted(self.commands, key=lambda item: item.latest_index, reverse=True):
                command_display = _truncate_text(finding.command, 120)
                line = f"  - `{command_display}`"
                if finding.exit_code is not None:
                    line += f" exit {finding.exit_code}"
                if not append_line(line):
                    break
                for output in finding.output_lines[:3]:
                    if not append_line(f"    {_truncate_text(output, 160)}"):
                        break
                if truncated:
                    break

        if self.generics and not truncated:
            append_line("- Other outputs:")
            for finding in sorted(self.generics, key=lambda item: item.latest_index, reverse=True)[:6]:
                line = f"  - {finding.tool_name}: {_truncate_text(finding.text, 160)}"
                if not append_line(line):
                    break
                if truncated:
                    break

        if self.functions and not truncated:
            append_line("- Function highlights:")
            for finding in sorted(self.functions.values(), key=lambda item: item.latest_index, reverse=True)[:12]:
                location = finding.path or "(unknown path)"
                if finding.line is not None:
                    location += f":{finding.line}"
                source_labels = {
                    "file": "read",
                    "search": "search",
                    "ast": "ast",
                    "symbols": "symbols",
                }
                source_label = source_labels.get(finding.source, finding.source)
                line = f"  - {location} {finding.signature}"
                if source_label:
                    line += f" [{source_label}]"
                if not append_line(line):
                    break
                if truncated:
                    break

        if len(lines) == 1:
            return None
        return "\n".join(lines)

    def _merge_file_finding(self, finding: FileFinding, index: int) -> None:
        finding.stamp(index)
        existing = self.files.get(finding.path)
        if existing:
            existing.stamp(index)
            if finding.total_lines is not None:
                if existing.total_lines is None or finding.total_lines > existing.total_lines:
                    existing.total_lines = finding.total_lines
            for rng in finding.ranges:
                existing.add_range(*rng)
            for note in finding.notes:
                existing.add_note(note)
            for snippet in finding.snippets:
                existing.add_snippet(snippet)
            self._collect_functions_from_file(existing, index)
        else:
            self.files[finding.path] = finding
            self._collect_functions_from_file(finding, index)

    def _merge_search_finding(self, finding: SearchFinding, index: int) -> None:
        finding.stamp(index)
        key = finding.query or f"<empty-query-{len(self.searches)}>"
        existing = self.searches.get(key)
        if existing:
            existing.stamp(index)
            if finding.status == "no_matches":
                existing.status = "no_matches"
            if finding.additional_note:
                existing.additional_note = finding.additional_note
            for match in finding.matches:
                existing.add_match(match)
            self._collect_functions_from_search(existing, index)
        else:
            self.searches[key] = finding
            self._collect_functions_from_search(finding, index)

    def _merge_ast_finding(self, finding: ASTFinding, index: int) -> None:
        finding.stamp(index)
        key = (finding.path or "", finding.query)
        existing = self.ast.get(key)
        if existing:
            existing.stamp(index)
            if finding.match_count is not None:
                existing.match_count = finding.match_count
            for node in finding.nodes:
                existing.add_node(node)
            self._collect_functions_from_ast(existing, index)
        else:
            self.ast[key] = finding
            self._collect_functions_from_ast(finding, index)

    def _merge_symbol_finding(self, finding: SymbolFinding, index: int) -> None:
        finding.stamp(index)
        key = (
            finding.name or "",
            finding.kind or "",
            finding.lang or "",
        )
        existing = self.symbols.get(key)
        if existing:
            existing.stamp(index)
            if finding.status:
                existing.status = finding.status
            for match in finding.matches:
                existing.add_match(match)
            self._collect_functions_from_symbols(existing, index)
        else:
            self.symbols[key] = finding
            self._collect_functions_from_symbols(finding, index)

    def _collect_functions_from_file(self, finding: FileFinding, index: int) -> None:
        for snippet in finding.snippets:
            for signature in _extract_function_signatures(snippet.text):
                self._record_function(signature, finding.path, snippet.number, "file", index)

    def _collect_functions_from_search(self, finding: SearchFinding, index: int) -> None:
        for match in finding.matches:
            for signature in _extract_function_signatures(match.preview):
                self._record_function(signature, match.path, match.line, "search", index)

    def _collect_functions_from_ast(self, finding: ASTFinding, index: int) -> None:
        for node in finding.nodes:
            for signature in _extract_function_signatures(node):
                self._record_function(signature, finding.path, None, "ast", index)

    def _collect_functions_from_symbols(self, finding: SymbolFinding, index: int) -> None:
        for match in finding.matches:
            descriptor = match.kind.strip() or finding.name
            self._record_function(descriptor, match.path, match.line, "symbols", index)

    def _record_function(
        self,
        signature: str,
        path: Optional[str],
        line: Optional[int],
        source: str,
        index: int,
    ) -> None:
        signature = _truncate_text(signature, 160)
        if not signature:
            return
        key = (path or "", signature)
        existing = self.functions.get(key)
        if existing:
            existing.stamp(index)
            if existing.line is None and line is not None:
                existing.line = line
            return
        finding = FunctionFinding(signature, path, line, source)
        finding.stamp(index)
        self.functions[key] = finding


def _build_fallback_tool_summary(messages: List[Any]) -> Optional[str]:
    metadata, call_sequence = _collect_tool_call_metadata(messages)
    aggregator = _ResearchSummaryAggregator()
    consumed: set[str] = set()
    sequence_index = 0

    for index, message in enumerate(messages):
        if getattr(message, "role", None) != "tool":
            continue
        content = getattr(message, "content", None)
        if not content:
            continue
        tool_call_id = getattr(message, "tool_call_id", None)
        meta: Dict[str, Any] | None = None
        if tool_call_id and tool_call_id in metadata:
            meta = metadata[tool_call_id]
            consumed.add(tool_call_id)
        else:
            while sequence_index < len(call_sequence):
                candidate = call_sequence[sequence_index]
                sequence_index += 1
                if candidate in consumed:
                    continue
                meta = metadata.get(candidate)
                consumed.add(candidate)
                break
        arguments = meta.get("arguments", {}) if meta else {}
        if not isinstance(arguments, dict):
            arguments = {}
        aggregator.ingest(meta.get("name") if meta else None, arguments, content, index)

    return aggregator.build_summary()


def _collect_tool_call_metadata(messages: List[Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for idx, message in enumerate(messages):
        if getattr(message, "role", None) != "assistant":
            continue
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            call_id = call.get("id") or call.get("tool_call_id") or call.get("call_id")
            if not call_id:
                continue
            function_payload = call.get("function") or {}
            name = function_payload.get("name") or call.get("name")
            raw_args = function_payload.get("arguments") or call.get("arguments") or {}
            if isinstance(raw_args, dict):
                args_dict = raw_args
            else:
                try:
                    args_dict = json.loads(raw_args)
                except (TypeError, ValueError, json.JSONDecodeError):
                    args_dict = {}
            metadata[call_id] = {
                "name": name,
                "arguments": args_dict,
                "assistant_index": idx,
            }
            order.append(call_id)
    return metadata, order
def _parse_file_read_result(arguments: Optional[Dict[str, Any]], content: str) -> List[FileFinding]:
    if not content.strip():
        return []

    default_paths: List[str] = []
    if isinstance(arguments, dict):
        path = arguments.get("path")
        if path:
            default_paths.append(str(path))
        paths = arguments.get("paths")
        if isinstance(paths, (list, tuple)):
            for item in paths:
                if item:
                    default_paths.append(str(item))
    default_paths = list(dict.fromkeys(default_paths))
    default_index = 0

    def take_default_path() -> str:
        nonlocal default_index
        if default_index < len(default_paths):
            value = default_paths[default_index]
            default_index += 1
            return value
        return ""

    file_sections: List[FileFinding] = []
    current: Optional[FileFinding] = None

    file_header_pattern = re.compile(r"^==\s*(.*?)\s*==$")
    range_pattern = re.compile(r"^Reading\s+(.*?)\s+\(lines\s+(\d+)-(\d+)\s+of\s+(\d+)\):$")
    code_pattern = re.compile(r"^\s*(\d+)\s*:\s?(.*)$")

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        header_match = file_header_pattern.match(stripped)
        if header_match:
            if current:
                file_sections.append(current)
            path_value = header_match.group(1) or take_default_path() or "(unknown)"
            current = FileFinding(path=path_value)
            continue

        range_match = range_pattern.match(stripped)
        if range_match:
            path_value = range_match.group(1).strip()
            if current and current.path != path_value:
                file_sections.append(current)
                current = None
            if current is None:
                current = FileFinding(path=path_value or take_default_path() or "(unknown)")
            start = _safe_int(range_match.group(2))
            end = _safe_int(range_match.group(3))
            total = _safe_int(range_match.group(4))
            current.add_range(start, end)
            if total is not None:
                if current.total_lines is None or total > current.total_lines:
                    current.total_lines = total
            continue

        code_match = code_pattern.match(raw_line)
        if code_match:
            if current is None:
                current = FileFinding(path=take_default_path() or "(unknown)")
            line_number = _safe_int(code_match.group(1))
            snippet_text = _truncate_text(code_match.group(2))
            current.add_snippet(CodeLineSnippet(line_number, snippet_text))
            continue

        if stripped.startswith("... ("):
            if current is None:
                current = FileFinding(path=take_default_path() or "(unknown)")
            current.add_note(stripped)
            continue

        if current is None:
            current = FileFinding(path=take_default_path() or "(unknown)")
        current.add_note(stripped)

    if current:
        file_sections.append(current)

    return file_sections


def _parse_search_result(arguments: Optional[Dict[str, Any]], content: str) -> Optional[SearchFinding]:
    if not content.strip():
        return None

    query = ""
    if isinstance(arguments, dict):
        query = str(arguments.get("query", "") or "").strip()

    finding = SearchFinding(query=query)
    match_pattern = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+)\s+(?P<preview>.*)$")

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        match = match_pattern.match(stripped)
        if match:
            finding.add_match(
                SearchMatch(
                    path=match.group("path"),
                    line=_safe_int(match.group("line")),
                    column=_safe_int(match.group("col")),
                    preview=_truncate_text(match.group("preview")),
                )
            )
            continue
        if stripped.lower().startswith("no matches"):
            finding.status = "no_matches"
            continue
        if stripped.startswith("... ("):
            finding.additional_note = stripped
            continue
        if not finding.additional_note and query and stripped.lower().startswith(query.lower()):
            finding.additional_note = stripped

    if not finding.matches and not finding.additional_note and not query and finding.status != "no_matches":
        return None

    return finding


def _parse_ast_result(arguments: Optional[Dict[str, Any]], content: str) -> Optional[ASTFinding]:
    if not content.strip():
        return None

    path = ""
    query = ""
    if isinstance(arguments, dict):
        path = str(arguments.get("path", "") or "").strip()
        query = str(arguments.get("query", "") or "").strip()

    finding = ASTFinding(path=path, query=query)

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("matched ") and "node" in stripped.lower():
            match = re.search(r"matched\s+(\d+)", stripped.lower())
            if match:
                finding.match_count = _safe_int(match.group(1))
            continue
        if stripped.startswith("- "):
            finding.add_node(stripped[2:])
            continue
        if not finding.nodes:
            finding.add_node(stripped)

    if not finding.nodes and finding.match_count is None and not (path or query):
        return None

    return finding


def _parse_symbol_result(arguments: Optional[Dict[str, Any]], content: str) -> Optional[SymbolFinding]:
    if not content.strip():
        return None

    name = ""
    kind: Optional[str] = None
    lang: Optional[str] = None
    if isinstance(arguments, dict):
        name = str(arguments.get("name", "") or "").strip()
        kind_value = arguments.get("kind")
        lang_value = arguments.get("lang")
        if isinstance(kind_value, str):
            kind = kind_value.strip() or None
        if isinstance(lang_value, str):
            lang = lang_value.strip() or None

    finding = SymbolFinding(name=name, kind=kind, lang=lang)
    match_pattern = re.compile(r"^(?P<path>.+?):(?P<line>\d+)\s*(?P<kind>.*)$")

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("no definitions"):
            finding.status = stripped
            continue
        match = match_pattern.match(stripped)
        if match:
            finding.add_match(
                SymbolMatch(
                    path=match.group("path"),
                    line=_safe_int(match.group("line")),
                    kind=match.group("kind").strip(),
                )
            )
            continue
        if not finding.status:
            finding.status = stripped

    if not finding.matches and not finding.status and not name:
        return None

    return finding


def _parse_command_result(arguments: Optional[Dict[str, Any]], content: str) -> Optional[CommandFinding]:
    if not content.strip():
        return None

    lines = [line for line in content.splitlines() if line.strip()]
    if not lines:
        return None

    exit_code: Optional[int] = None
    start_index = 0
    first_line = lines[0].strip()
    exit_match = re.match(r"^Exit:\s*(-?\d+)", first_line, re.IGNORECASE)
    if exit_match:
        exit_code = _safe_int(exit_match.group(1))
        start_index = 1

    command = ""
    if isinstance(arguments, dict):
        cmd_value = arguments.get("cmd") or arguments.get("command")
        if isinstance(cmd_value, (list, tuple)):
            command = " ".join(str(item) for item in cmd_value)
        elif cmd_value:
            command = str(cmd_value)

    command = command.strip() or "(command unknown)"
    output_lines = [
        _truncate_text(line.strip(), 200)
        for line in lines[start_index:]
        if line.strip() and not line.lower().startswith("stdout tail") and not line.lower().startswith("stderr tail")
    ]

    return CommandFinding(command=command, exit_code=exit_code, output_lines=output_lines[:5])


def _format_range_summary(finding: FileFinding) -> str:
    segments: List[str] = []
    for start, end in finding.ranges:
        if start is None and end is None:
            continue
        if start is None:
            segments.append(f"up to line {end}")
        elif end is None or end == start:
            segments.append(f"line {start}")
        else:
            segments.append(f"lines {start}-{end}")
    if finding.total_lines:
        if segments:
            segments[-1] += f" of {finding.total_lines}"
        else:
            segments.append(f"{finding.total_lines} total lines")
    return "; ".join(segments)


def _select_highlight_lines(snippets: List[CodeLineSnippet], limit: int = 3) -> List[CodeLineSnippet]:
    if len(snippets) <= limit:
        return snippets

    prioritized: List[CodeLineSnippet] = []
    seen: set[Tuple[Optional[int], str]] = set()

    def add_candidate(snippet: CodeLineSnippet) -> None:
        key = (snippet.number, snippet.text)
        if key in seen:
            return
        seen.add(key)
        prioritized.append(snippet)

    for snippet in snippets:
        if _extract_function_signatures(snippet.text):
            add_candidate(snippet)

    for snippet in snippets:
        add_candidate(snippet)
        if len(prioritized) >= limit:
            break

    return prioritized[:limit]


def _select_highlight_matches(matches: List[SearchMatch], limit: int = 4) -> List[SearchMatch]:
    if len(matches) <= limit:
        return matches

    prioritized: List[SearchMatch] = []
    seen: set[Tuple[str, Optional[int], Optional[int]]] = set()

    def add_candidate(match: SearchMatch) -> None:
        key = (match.path, match.line, match.column)
        if key in seen:
            return
        seen.add(key)
        prioritized.append(match)

    for match in matches:
        if _extract_function_signatures(match.preview):
            add_candidate(match)

    for match in matches:
        add_candidate(match)
        if len(prioritized) >= limit:
            break

    return prioritized[:limit]


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _truncate_text(text: str, max_length: int = 160) -> str:
    text = text.strip()
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


_FUNCTION_PATTERNS = [
    re.compile(r"\basync\s+def\s+[A-Za-z_][\w]*\s*\([^)]*\)"),
    re.compile(r"\bdef\s+[A-Za-z_][\w]*\s*\([^)]*\)"),
    re.compile(r"\bclass\s+[A-Za-z_][\w]*"),
    re.compile(r"\bfunction\s+[A-Za-z_][\w]*\s*\([^)]*\)"),
    re.compile(r"\bfn\s+[A-Za-z_][\w]*\s*\([^)]*\)?"),
    re.compile(r"\b[A-Za-z_][\w]*\s*=\s*function\s*\([^)]*\)"),
    re.compile(r"\b[A-Za-z_][\w]*\s*=\s*\([^)]*\)\s*=>"),
    re.compile(
        r"\b(?:public|private|protected|static|final|virtual|override|async)\s+[A-Za-z0-9_<>,\[\]\s]*\b[A-Za-z_][\w]*\s*\([^)]*\)"
    ),
]


def _extract_function_signatures(text: str) -> List[str]:
    snippet = text.strip()
    if not snippet:
        return []

    signatures: List[str] = []
    seen: set[str] = set()

    for pattern in _FUNCTION_PATTERNS:
        for match in pattern.finditer(snippet):
            cleaned = _truncate_text(match.group(0).rstrip("{").strip(), 160)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                signatures.append(cleaned)

    return signatures


# --------------------------------------------------------------------------------------
# Locate / Where
# --------------------------------------------------------------------------------------


def _handle_question_without_llm(
    question: str, include_docs: bool = True, max_files: int = 8, *, reason: str = "no API key"
) -> None:
    """Provide multi-step ReAct-style analysis without LLM."""
    click.echo(f"🔍 Analyzing question: '{question}'")
    click.echo(f"⚠️  Operating in fallback mode ({reason}). Using keyword-based analysis.")
    click.echo()
    
    # Step 1: Extract keywords
    click.echo("📋 Step 1: Extracting keywords...")
    keywords = extract_keywords(question)
    if keywords:
        click.echo(f"   Keywords identified: {', '.join(keywords)}")
    else:
        click.echo("   No specific keywords found, using general search.")
    click.echo()
    
    # Step 2: Search for relevant files
    click.echo("🔎 Step 2: Searching for relevant files...")
    repo = Path.cwd()
    relevant_files = []
    
    # Search by keywords in file paths and content
    for keyword in keywords[:3]:  # Limit to top 3 keywords
        matches = _search_paths_and_content(keyword, include_docs=include_docs, max_results=5)
        relevant_files.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file in relevant_files:
        if file not in seen:
            seen.add(file)
            unique_files.append(file)
    
    relevant_files = unique_files[:max_files]
    
    if relevant_files:
        click.echo(f"   Found {len(relevant_files)} relevant files:")
        for file in relevant_files:
            click.echo(f"   - {file}")
    else:
        click.echo("   No specific files found. Showing project structure.")
        # Fallback to showing key directories
        key_dirs = ["ai_dev_agent", "tests", "docs"]
        for dir_name in key_dirs:
            dir_path = repo / dir_name
            if dir_path.exists():
                click.echo(f"   - {dir_name}/ (directory)")
    click.echo()
    
    # Step 3: Analyze file contents
    if relevant_files:
        click.echo("📖 Step 3: Analyzing file contents...")
        for file_path in relevant_files[:3]:  # Show content from top 3 files
            full_path = repo / file_path
            if full_path.exists() and full_path.is_file():
                try:
                    content = full_path.read_text(encoding='utf-8')
                    lines = content.split('\n')
                    
                    # Find relevant lines containing keywords
                    relevant_lines = []
                    for i, line in enumerate(lines):
                        for keyword in keywords:
                            if keyword.lower() in line.lower():
                                # Add context: line before, the line, line after
                                start = max(0, i-1)
                                end = min(len(lines), i+2)
                                context_lines = [f"{j+1:4}: {lines[j]}" for j in range(start, end)]
                                relevant_lines.extend(context_lines)
                                break
                        if len(relevant_lines) >= 10:  # Limit output
                            break
                    
                    if relevant_lines:
                        click.echo(f"   📄 {file_path}:")
                        for line in relevant_lines[:10]:
                            click.echo(f"      {line}")
                        if len(relevant_lines) > 10:
                            click.echo(f"      ... ({len(relevant_lines) - 10} more lines)")
                        click.echo()
                    
                except (UnicodeDecodeError, OSError):
                    click.echo(f"   📄 {file_path}: (binary or inaccessible file)")
                    click.echo()
    
    # Step 4: Provide structured summary
    click.echo("📝 Step 4: Summary")
    
    # Pattern matching for common questions
    question_lower = question.lower()
    if "react" in question_lower and "pattern" in question_lower:
        click.echo("   Based on the repository structure, the ReAct pattern appears to be implemented in:")
        react_files = [f for f in relevant_files if "react" in f.lower()]
        if react_files:
            for file in react_files:
                click.echo(f"   - {file}")
        else:
            click.echo("   - ai_dev_agent/react/ directory contains the ReAct implementation")
            
        click.echo("\n   Key components likely include:")
        click.echo("   - Action providers for decision making")
        click.echo("   - Tool invokers for executing actions")  
        click.echo("   - Evaluation loops for iterative reasoning")
        
    elif "where" in question_lower or "find" in question_lower:
        if relevant_files:
            click.echo(f"   Found {len(relevant_files)} files matching your search:")
            for file in relevant_files:
                click.echo(f"   - {file}")
        else:
            click.echo("   No specific matches found. Try a more specific search term.")
            
    elif "how" in question_lower or "what" in question_lower:
        if relevant_files:
            click.echo("   Based on the files found, you may want to examine:")
            for file in relevant_files[:3]:
                click.echo(f"   - {file}")
            click.echo("\n   For detailed implementation, consider using an API key for full LLM analysis.")
        else:
            click.echo("   This question requires deeper code analysis.")
            click.echo("   Consider setting DEVAGENT_API_KEY for comprehensive answers.")
    else:
        if relevant_files:
            click.echo("   Found potentially relevant files. See the analysis above.")
        else:
            click.echo("   This question might need more specific keywords or an API key for full analysis.")
    
    click.echo("\n💡 Tip: Set DEVAGENT_API_KEY environment variable for detailed LLM-powered analysis.")





@cli.command()
@click.pass_context
def shell(ctx: click.Context) -> None:
    """Start an interactive shell session with persistent context."""
    click.echo("DevAgent Interactive Shell")
    click.echo("Type a question or command, 'help' for guidance, and 'exit' to quit.")
    click.echo("=" * 50)

    while True:
        try:
            user_input = click.prompt("DevAgent> ", prompt_suffix="", show_default=False).strip()
        except (KeyboardInterrupt, EOFError):
            click.echo("\nGoodbye!")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"exit", "quit", "q"}:
            click.echo("Goodbye!")
            break

        if lowered == "help":
            click.echo("Enter any natural-language request to run `devagent query`.")
            click.echo("Use 'exit' to leave the shell.")
            continue

        try:
            ctx.invoke(query, prompt=(user_input,))
        except click.ClickException as exc:
            click.echo(f"Error: {exc}")

def main() -> None:
    cli(prog_name="devagent")


if __name__ == "__main__":  # pragma: no cover
    main()

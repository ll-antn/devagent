"""Command line interface for the development agent."""
from __future__ import annotations

import json
import re
import shlex
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import click
from click.shell_completion import get_completion_class

PLAN_JSON_FENCE_PATTERN = re.compile(r"```json\s*(?P<json>{.*?})\s*```", re.DOTALL)

from .adr.adr_manager import ADRManager
from .approval.approvals import ApprovalManager
from .approval.policy import ApprovalPolicy
from .code_edit.diff_utils import DiffError
from .code_edit.editor import CodeEditor, DiffProposal, IterativeFixConfig
from .code_edit.context import ContextGatherer, ContextGatheringOptions
from .git_tools import (
    GitIntegrationError,
    create_feature_branch,
    gather_diff,
    generate_commit_message,
    generate_pr_description,
    get_current_branch,
    guess_default_base_branch,
)
from .intent_router import IntentDecision, IntentRouter, IntentRoutingError
from .llm_provider import (
    LLMConnectionError,
    LLMError,
    LLMRetryExhaustedError,
    LLMTimeoutError,
    RetryConfig,
    create_client,
)
from .metrics import MetricsCollector
from .planning.planner import PlanResult, Planner
from .planning.frontmatter import (
    parse_frontmatter_plan,
    write_frontmatter_plan,
    validate_frontmatter_plan,
)
from .planning.reasoning import PlanAdjustment, ReasoningStep, TaskReasoning, ToolUse
from .questions import QuestionAnswerer
from .react import ReactiveExecutor, GateConfig, RunResult, TaskSpec
from .react.evaluator import GateEvaluator
from .react.pipeline import PipelineCommands, parse_command, run_quality_pipeline
from .react.types import ActionRequest
from .sandbox import SandboxConfig, SandboxExecutor
from .testing.local_tests import TestRunner
from .testing.qa_gate import passes as qa_passes, summarize as qa_summarize
from .utils.config import Settings, load_settings
from .utils.devagent_config import load_devagent_yaml
from .utils.keywords import extract_keywords
from .utils.logger import configure_logging, get_correlation_id, get_logger, set_correlation_id
from .utils.state import StateStore

LOGGER = get_logger(__name__)
ADR_TEMPLATE_PATH = Path(__file__).resolve().parent / "adr" / "templates" / "default.md"
MAX_HISTORY_ENTRIES = 50


class NaturalLanguageGroup(click.Group):
    """Group that falls back to NL intent routing when no command matches."""

    def resolve_command(self, ctx: click.Context, args: List[str]):  # type: ignore[override]
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as exc:
            if not args:
                raise
            if any(arg.startswith("-") for arg in args):
                raise
            query = " ".join(args).strip()
            if not query:
                raise
            ctx.meta["_pending_nl_prompt"] = query
            return super().resolve_command(ctx, ["assist"])


def _resolve_repo_path(path_value: str | None) -> Path:
    repo_root = Path.cwd().resolve()
    if not path_value or path_value.strip() in {"", "."}:
        return repo_root
    candidate = (repo_root / path_value).resolve()
    if repo_root not in candidate.parents and candidate != repo_root:
        raise click.ClickException(f"Path '{path_value}' escapes the repository root.")
    return candidate


def _intent_list_directory(_: click.Context, arguments: Dict[str, Any]) -> None:
    path_value = arguments.get("path")
    target = _resolve_repo_path(path_value)
    show_hidden = bool(arguments.get("show_hidden", False))
    detailed = bool(arguments.get("detailed", False))

    if not target.exists():
        raise click.ClickException(f"Directory not found: {path_value or '.'}")
    if not target.is_dir():
        raise click.ClickException(f"Not a directory: {path_value or target}")

    entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    visible: List[str] = []
    for entry in entries:
        if not show_hidden and entry.name.startswith("."):
            continue
        if detailed:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            entry_type = "dir" if entry.is_dir() else "file"
            visible.append(f"{entry_type:<4} {size:>8} {entry.name}")
        else:
            suffix = "/" if entry.is_dir() else ""
            visible.append(entry.name + suffix)

    relative_display = Path(path_value or ".")
    click.echo(f"Listing {relative_display.as_posix() if relative_display != Path('.') else '.'}:")
    if not visible:
        click.echo("(empty)")
        return
    for line in visible:
        click.echo(line)


def _intent_search_repository(ctx: click.Context, arguments: Dict[str, Any]) -> None:
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise click.ClickException("Search intent requires a query string.")
    include_docs = bool(arguments.get("include_docs", True))
    matches = _search_paths_and_content(query, include_docs=include_docs)
    if not matches:
        click.echo("No matches found.")
        return
    for match in matches:
        click.echo(match)


def _intent_read_file(ctx: click.Context, arguments: Dict[str, Any]) -> None:
    path_value = str(arguments.get("path", "")).strip()
    if not path_value:
        raise click.ClickException("read_file intent requires a path argument.")

    target = _resolve_repo_path(path_value)
    if not target.exists():
        raise click.ClickException(f"File not found: {path_value}")
    if not target.is_file():
        raise click.ClickException(f"Not a file: {path_value}")

    try:
        start_line = int(arguments.get("start_line", 1) or 1)
    except (TypeError, ValueError):
        raise click.ClickException("start_line must be an integer.")
    if start_line < 1:
        start_line = 1

    end_line: int | None
    end_line_arg = arguments.get("end_line")
    if end_line_arg is not None:
        try:
            end_line = int(end_line_arg)
        except (TypeError, ValueError):
            raise click.ClickException("end_line must be an integer.")
        if end_line < start_line:
            raise click.ClickException("end_line must be greater than or equal to start_line.")
    else:
        end_line = None

    max_lines_arg = arguments.get("max_lines") if end_line is None else None
    if max_lines_arg is not None:
        try:
            max_lines = int(max_lines_arg)
        except (TypeError, ValueError):
            raise click.ClickException("max_lines must be an integer.")
        if max_lines <= 0:
            max_lines = 200
    else:
        max_lines = 200 if end_line is None else None

    try:
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        raise click.ClickException(f"Failed to read file: {exc}") from exc

    total_lines = len(lines)
    if total_lines == 0:
        click.echo(f"{_relative_to_repo(target)} is empty.")
        return

    start_index = min(start_line - 1, total_lines)
    if start_index >= total_lines:
        click.echo(f"File has only {total_lines} lines; nothing to show from line {start_line}.")
        return

    if end_line is not None:
        end_index = min(end_line, total_lines)
    else:
        end_index = min(start_index + (max_lines or 0), total_lines)
        if end_index == start_index:
            end_index = min(start_index + 200, total_lines)

    snippet = lines[start_index:end_index]
    rel_path = _relative_to_repo(target)
    if snippet:
        last_line_number = start_index + len(snippet)
        click.echo(f"Reading {rel_path} (lines {start_index + 1}-{last_line_number} of {total_lines}):")
        for line_number, content in enumerate(snippet, start=start_index + 1):
            click.echo(f"{line_number:5}: {content.rstrip()}")
        if end_index < total_lines:
            remaining = total_lines - end_index
            click.echo(f"... ({remaining} more lines not shown)")
    else:
        click.echo(f"No content available in the requested range for {rel_path}.")


def _intent_generate_plan(ctx: click.Context, arguments: Dict[str, Any]) -> None:
    goal = str(arguments.get("goal", "")).strip()
    if not goal:
        raise click.ClickException("Plan intent requires a goal description.")
    md_path_value = arguments.get("output_path")
    write_markdown_flag = bool(arguments.get("write_markdown", bool(md_path_value)))
    md_path = Path(md_path_value) if md_path_value else None
    ctx.invoke(
        plan,
        goal=(goal,),
        auto_approve=True,
        write_md=write_markdown_flag,
        md_path=md_path,
        show_md_path=False,
    )


def _intent_ask_repository(ctx: click.Context, arguments: Dict[str, Any]) -> None:
    question = str(arguments.get("question", "")).strip()
    if not question:
        raise click.ClickException("Ask intent requires a question.")
    files = tuple(str(value) for value in (arguments.get("files") or []))
    include_docs = bool(arguments.get("include_docs", True))
    try:
        max_files = int(arguments.get("max_files", 8))
    except (TypeError, ValueError):
        max_files = 8
    
    # Check if LLM client is available
    settings: Settings = ctx.obj["settings"]
    if not settings.api_key:
        # Provide fallback multi-step analysis without LLM
        _handle_question_without_llm(question, include_docs, max_files)
        return
        
    ctx.invoke(
        ask,
        question=(question,),
        files=files,
        max_files=max_files,
        include_docs=include_docs,
        show_context=False,
    )


def _intent_execute_plan(ctx: click.Context, arguments: Dict[str, Any]) -> None:
    plan_path_value = arguments.get("plan_path")
    if not plan_path_value:
        raise click.ClickException("Execute plan intent requires a plan_path.")
    checkpoint_value = arguments.get("checkpoint_every")
    checkpoint_every = 0
    if checkpoint_value is not None:
        try:
            checkpoint_every = int(checkpoint_value)
        except (TypeError, ValueError):
            checkpoint_every = 0
    max_diff = arguments.get("max_diff_lines")
    if max_diff is not None:
        try:
            max_diff = int(max_diff)
        except (TypeError, ValueError):
            max_diff = None
    ctx.invoke(
       run_plan_cmd,
       plan_file=Path(plan_path_value),
       apply_changes=bool(arguments.get("apply_changes", False)),
       auto_commit=bool(arguments.get("auto_commit", False)),
       checkpoint_every=checkpoint_every,
       max_diff_lines=max_diff,
    )


def _intent_review_plan(ctx: click.Context, arguments: Dict[str, Any]) -> None:
    plan_path_value = arguments.get("plan_path")
    if not plan_path_value:
        raise click.ClickException("Review intent requires a plan_path.")
    ctx.invoke(
        review_cmd,
        plan_file=Path(plan_path_value),
        fail_on_divergence=bool(arguments.get("fail_on_divergence", False)),
    )


def _intent_direct_response(_: click.Context, arguments: Dict[str, Any]) -> None:
    text = str(arguments.get("text", "")).strip()
    if text:
        click.echo(text)


INTENT_HANDLERS: Dict[str, Any] = {
    "list_directory": _intent_list_directory,
    "search_repository": _intent_search_repository,
    "read_file": _intent_read_file,
    "generate_plan": _intent_generate_plan,
    "ask_repository": _intent_ask_repository,
    "execute_frontmatter_plan": _intent_execute_plan,
    "review_against_plan": _intent_review_plan,
    "respond_directly": _intent_direct_response,
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
    }


def _record_invocation(ctx: click.Context, overrides: Optional[Dict[str, Any]] = None) -> None:
    """Persist command invocation details for history and replay."""
    if not ctx or not ctx.command_path:
        return
    set_correlation_id(uuid.uuid4().hex[:12])
    state_obj = ctx.obj.get("state") if ctx.obj else None
    if not isinstance(state_obj, StateStore):
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


def _resolve_command(path: List[str]) -> click.Command | None:
    """Resolve a command object from a nested command path."""
    current: click.Command = cli  # type: ignore[assignment]
    for part in path:
        if not isinstance(current, click.MultiCommand):
            return None
        ctx = click.Context(current, info_name=current.name)
        next_command = current.get_command(ctx, part)
        if next_command is None:
            return None
        current = next_command
    return current


def _record_task_metric(
    state: StateStore,
    task: Dict[str, Any],
    outcome: str,
    started_at: datetime,
    *,
    dry_run: bool = False,
) -> None:
    """Capture task execution metrics for observability."""
    try:
        duration = max((datetime.utcnow() - started_at).total_seconds(), 0.0)
    except Exception:  # pragma: no cover - defensive
        duration = 0.0
    state.record_metric(
        {
            "task_id": task.get("id"),
            "title": task.get("title"),
            "outcome": outcome,
            "duration_sec": round(duration, 3),
            "timestamp": datetime.utcnow().isoformat(),
            "dry_run": dry_run,
            "correlation_id": get_correlation_id(),
        }
    )


def _build_gate_evaluator(
    settings: Settings,
    pipeline: PipelineCommands | None = None,
    run_tests: bool = True,
) -> GateEvaluator:
    config = GateConfig(
        diff_limit_lines=settings.diff_limit_lines,
        diff_limit_files=settings.diff_limit_files,
        patch_coverage_target=settings.patch_coverage_target,
        stuck_threshold=settings.stuck_threshold,
        steps_budget=settings.steps_budget,
    )
    _apply_gate_requirements(config, pipeline, run_tests)
    return GateEvaluator(config)


def _build_gate_evaluator_for_profile(
    profile: str,
    settings: Settings,
    *,
    pipeline: PipelineCommands | None = None,
    require_tests: bool | None = None,
) -> GateEvaluator:
    """Build a gate evaluator tuned for a specific profile (implement|review|design)."""
    cfg = GateConfig(
        diff_limit_lines=settings.diff_limit_lines,
        diff_limit_files=settings.diff_limit_files,
        patch_coverage_target=settings.patch_coverage_target,
        stuck_threshold=settings.stuck_threshold,
        steps_budget=settings.steps_budget,
    )
    profile = (profile or "implement").lower()
    if profile == "review":
        cfg.require_tests = bool(require_tests) if require_tests is not None else False
        cfg.require_lint = True
        cfg.require_types = True
        cfg.require_format = True
        cfg.require_compile = False
        cfg.require_diff_limits = True
        cfg.require_patch_coverage = True
        cfg.require_secrets = True
        cfg.require_sandbox = True
        cfg.require_flaky = False
        cfg.require_perf = False
    elif profile == "design":
        cfg.require_tests = False
        cfg.require_lint = False
        cfg.require_types = False
        cfg.require_format = False
        cfg.require_compile = False
        cfg.require_diff_limits = False
        cfg.require_patch_coverage = False
        cfg.require_secrets = False
        cfg.require_sandbox = False
        cfg.require_flaky = False
        cfg.require_design_doc = True
    # implement: keep defaults
    _apply_gate_requirements(cfg, pipeline, require_tests if require_tests is not None else True)
    return GateEvaluator(cfg)


def _create_sandbox(settings: Settings) -> SandboxExecutor:
    config = SandboxConfig(
        allowlist=settings.sandbox_allowlist,
        default_timeout=120.0,
        cpu_time_limit=settings.sandbox_cpu_time_limit,
        memory_limit_mb=settings.sandbox_memory_limit_mb,
    )
    return SandboxExecutor(Path.cwd(), config)


def _render_gate_summary(result: RunResult) -> None:
    click.echo("Gate status:")
    notes: Dict[str, str] = {}
    if result.steps:
        notes = result.steps[-1].metrics.gate_notes
    req_map = result.required_gates or {}
    for gate, passed in sorted(result.gates.items()):
        required = req_map.get(gate, False)
        if not required:
            status = "N/A"
        else:
            status = "PASS" if passed else "FAIL"
        note = notes.get(gate)
        if note and (passed or not required):
            # Show note only when informative and not duplicating a failure
            click.echo(f"  {status:<4} {gate}: {note}")
        else:
            click.echo(f"  {status:<4} {gate}")
    metrics = result.metrics or {}
    diff_lines = metrics.get("diff_lines")
    diff_files = metrics.get("diff_files")
    if diff_lines is not None and diff_files is not None:
        click.echo(f"Diff footprint: {diff_lines} lines across {diff_files} file(s)")
    coverage = metrics.get("patch_coverage")
    if coverage is not None:
        click.echo(f"Patch coverage: {coverage * 100:.1f}%")
    secrets = metrics.get("secrets_found")
    if secrets is not None:
        click.echo(f"Secrets detected: {secrets}")
    sandbox_violations = metrics.get("sandbox_violations")
    if sandbox_violations is not None:
        click.echo(f"Sandbox violations: {sandbox_violations}")
    if result.stop_reason:
        click.echo(f"Stop reason: {result.stop_reason}")
    # Next steps hint from the last step if present
    if result.steps:
        last_eval = result.steps[-1].evaluation
        if last_eval and last_eval.next_action_hint:
            click.echo(f"Next: {last_eval.next_action_hint}")


def _default_plan_markdown_path(goal: str) -> Path:
    slug = _slugify(goal) or "plan"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return Path("docs") / "plans" / f"{timestamp}_{slug}.md"


def _export_plan_markdown(plan: Dict[str, Any], target: Path) -> Path:
    target_path = target if target.is_absolute() else Path.cwd() / target
    target_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# Plan: {plan.get('goal', '').strip() or 'Untitled Plan'}")
    lines.append("")
    lines.append(f"Generated: {plan.get('generated_at', datetime.utcnow().isoformat())}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    summary = str(plan.get('summary', '')).strip() or "(no summary provided)"
    lines.append(summary)
    lines.append("")
    lines.append("## Tasks")
    lines.append("")
    lines.append("| ID | Title | Category | Effort | Status | Dependencies | Deliverables | Commands |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for task in plan.get("tasks", []):
        identifier = task.get("id") or task.get("identifier") or "?"
        title = _escape_markdown_cell(task.get("title", ""))
        category = _escape_markdown_cell(task.get("category", ""))
        effort = task.get("effort", "-")
        status = _escape_markdown_cell(task.get("status", "pending"))
        deps = _escape_markdown_cell(", ".join(task.get("dependencies", []) or ["-"]))
        deliverables = _escape_markdown_cell("<br>".join(task.get("deliverables", []) or ["-"]))
        commands = _escape_markdown_cell("<br>".join(task.get("commands", []) or ["-"]))
        lines.append(
            f"| {identifier} | {title} | {category} | {effort} | {status} | {deps} | {deliverables} | {commands} |"
        )

    plan_json = json.dumps(plan, indent=2)
    lines.append("")
    lines.append("```json")
    lines.append(plan_json)
    lines.append("```")

    target_path.write_text("\n".join(lines), encoding="utf-8")
    return target_path


def _escape_markdown_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|")


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return cleaned.strip("-")


def _load_plan_from_path(path: Path) -> Dict[str, Any]:
    actual = path if path.is_absolute() else Path.cwd() / path
    if not actual.is_file():
        raise click.ClickException(f"Plan file not found: {_relative_to_repo(actual)}")
    if actual.suffix.lower() == ".json":
        try:
            return json.loads(actual.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Failed to parse plan JSON: {exc}") from exc
    text = actual.read_text(encoding="utf-8")
    match = PLAN_JSON_FENCE_PATTERN.search(text)
    if not match:
        raise click.ClickException("Plan markdown must contain a JSON code fence.")
    try:
        return json.loads(match.group("json"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Failed to parse plan JSON block: {exc}") from exc


def _normalize_plan_dict(plan: Dict[str, Any]) -> Dict[str, Any]:
    plan.setdefault("status", "approved")
    tasks = plan.setdefault("tasks", [])
    for task in tasks:
        task.setdefault("status", "pending")
        task.setdefault("dependencies", [])
        task.setdefault("deliverables", [])
        task.setdefault("commands", [])
    return plan


def _apply_gate_requirements(
    config: GateConfig,
    pipeline: PipelineCommands | None,
    run_tests: bool,
) -> GateConfig:
    if pipeline is None:
        return config

    if not pipeline.format:
        config.require_format = False
    if not pipeline.lint:
        config.require_lint = False
    if not pipeline.typecheck:
        config.require_types = False
    if not pipeline.compile:
        config.require_compile = False

    if not run_tests or not pipeline.test:
        config.require_tests = False
        config.require_flaky = False
        config.require_patch_coverage = False

    if not pipeline.coverage_xml:
        config.require_patch_coverage = False

    if not pipeline.perf:
        config.require_perf = False

    if pipeline.flake_runs <= 1:
        config.require_flaky = False

    return config

def _check_adr_completeness(path: Path) -> tuple[bool, list[str]]:
    """Heuristic completeness check for ADR: expects several section headers."""
    required = [
        "Context",
        "Decision",
        "Consequences",
        "Alternatives",
        "Risks",
    ]
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return False, required
    missing: list[str] = [name for name in required if name.lower() not in text.lower()]
    # Treat missing sections as advisory notes rather than hard failures.
    return True, missing


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
    
    client = create_client(
        provider=settings.provider,
        api_key=settings.api_key,
        model=settings.model,
        base_url=settings.base_url,
    )
    
    # Configure timeout and retry behavior after creation
    client.configure_timeout(120.0)
    client.configure_retry(retry_config)
    ctx.obj["llm_client"] = client
    return client


def _render_plan(result: PlanResult) -> None:
    click.echo(f"Plan for goal: {result.goal}")
    click.echo(f"Summary: {result.summary}\n")
    ordered = sorted(
        result.tasks,
        key=lambda task: (task.priority_score.rice if task.priority_score else 0),
        reverse=True,
    )
    for task in ordered:
        priority = task.priority_score.rice if task.priority_score else "N/A"
        lines = [
            f"{task.identifier}: {task.title}",
            f"  Category: {task.category} | Priority: {priority}",
            f"  Effort: {task.effort or 'n/a'} | Reach: {task.reach or 'n/a'}",
            f"  Description: {task.description}",
        ]
        if task.deliverables:
            lines.append("  Deliverables:")
            lines.extend(f"    - {item}" for item in task.deliverables)
        if task.commands:
            lines.append("  Suggested commands:")
            lines.extend(f"    - {cmd}" for cmd in task.commands)
        click.echo("\n".join(lines) + "\n")


def _relative_to_repo(path: Path) -> Path | str:
    """Return a path relative to the current working directory when possible."""
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path


def _answer_repository_question(
    ctx: click.Context,
    question_text: str,
    files: Iterable[str] | None = None,
    *,
    max_files: int = 8,
    include_docs: bool = True,
    show_context: bool = False,
) -> None:
    """Resolve a repository question using the LLM."""
    normalized_question = question_text.strip()
    if not normalized_question:
        raise click.UsageError("Please provide a question to answer.")

    llm_client = _get_llm_client(ctx)
    options = ContextGatheringOptions(
        max_files=max_files,
        include_docs=include_docs,
        include_related_files=True,
        include_structure_summary=True,
        include_tests=True,
    )

    answerer = QuestionAnswerer(Path.cwd(), llm_client, options)
    keywords = extract_keywords(normalized_question)
    result = answerer.answer(
        normalized_question,
        files=files or (),
        keywords=keywords,
    )

    if show_context and result.contexts:
        click.echo("Context files:")
        for context in result.contexts:
            rel_path = _relative_to_repo(context.path)
            click.echo(
                f"- {rel_path} (reason={context.reason}, score={context.relevance_score:.2f})"
            )
        click.echo()

    if not result.contexts:
        click.echo(
            "No repository files matched the question; response may rely on general knowledge.",
            err=True,
        )

    state: StateStore = ctx.obj["state"]
    state.update(
        last_question={
            "question": normalized_question,
            "files": list(files or ()),
            "context_files": [str(_relative_to_repo(c.path)) for c in result.contexts],
            "fallback_reason": result.fallback_reason,
            "timestamp": datetime.utcnow().isoformat(),
        },
        last_answer=result.answer,
    )

    click.echo(result.answer)
    if result.fallback_reason:
        click.echo(
            f"LLM fallback activated: {result.fallback_reason}",
            err=True,
        )


def _plan_context_for_git(state: StateStore) -> str | None:
    """Summarize the active plan for git-related prompts."""
    data = state.load()
    plan = data.get("last_plan")
    if not plan:
        return None

    lines: List[str] = []
    goal = str(plan.get("goal", "")).strip()
    summary = str(plan.get("summary", "")).strip()
    if goal:
        lines.append(f"Goal: {goal}")
    if summary and summary != goal:
        lines.append(f"Summary: {summary}")

    tasks = plan.get("tasks") or []
    pending = [
        task for task in tasks
        if task.get("status") not in {"completed", "skipped"}
    ]
    if pending:
        lines.append("Active tasks:")
        for task in pending[:3]:
            identifier = task.get("id") or task.get("identifier") or "?"
            title = str(task.get("title", "")).strip()
            if title:
                lines.append(f"- {identifier}: {title}")
            else:
                lines.append(f"- {identifier}")

    return "\n".join(lines) if lines else None


@click.group(cls=NaturalLanguageGroup)
@click.option("--config", "config_path", type=click.Path(path_type=Path), help="Path to config file.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging output.")
@click.option("--dry-run", is_flag=True, help="Simulate destructive operations without writing changes.")
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None, verbose: bool, dry_run: bool) -> None:
    """AI-assisted development agent CLI.

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
    ctx.obj["dry_run"] = dry_run


# --------------------------------------------------------------------------------------
# Bootstrap / Init
# --------------------------------------------------------------------------------------


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Scaffold devagent.yaml and example plan templates."""
    _record_invocation(ctx)
    repo = Path.cwd()
    config_path = repo / "devagent.yaml"
    if not config_path.exists():
        sample = (
            "version: 1\n"
            "workspace:\n  root: .\n  write_allowlist:\n    - src/**\n    - tests/**\n    - docs/**\n"
            "vcs:\n  auto_commit: true\n  commit_template: \"[devagent] {step_id}: {summary}\"\n"
            "build:\n  cmd: \"make build\"\n"
            "tests:\n  cmd: \"pytest\"\n  pattern_flag: \"-k\"\n"
            "coverage:\n  cmd: \"\"\n  threshold:\n    diff: 0.80\n    project: 0.60\n"
            "lint:\n  cmd: \"ruff .\"\n"
            "types:\n  cmd: \"mypy .\"\n"
            "format:\n  cmd: \"ruff format .\"\n"
            "gates:\n  - name: tests\n    must_be: green\n  - name: lint\n    must_be: zero_errors\n  - name: types\n    must_be: zero_errors\n  - name: coverage.diff\n    gte: 0.80\n  - name: diff.size\n    lte_lines: 400\n    lte_files: 10\n"
        )
        config_path.write_text(sample, encoding="utf-8")
        click.echo(f"Created {config_path}")
    else:
        click.echo("devagent.yaml already exists")

    tmpl_dir = repo / "docs" / "templates"
    tmpl_dir.mkdir(parents=True, exist_ok=True)
    coverage_plan = tmpl_dir / "coverage.plan.md"
    if not coverage_plan.exists():
        steps = [
            {
                "id": "discover_gaps",
                "tool": "coverage.diff",
                "args": {"path": "src/moduleX"},
                "produces": "reports/moduleX_gaps.json",
                "gates": ["files_found>0"],
            },
            {
                "id": "add_tests",
                "tool": "write_patch",
                "args": {"patch_source": "LLM", "targets_from": "reports/moduleX_gaps.json"},
                "gates": ["lint==0", "types==0"],
            },
            {
                "id": "run_tests",
                "tool": "tests.run",
                "args": {"pattern": "moduleX"},
                "gates": ["tests==green", "diff_coverage>=0.85"],
            },
            {
                "id": "refactor_small",
                "tool": "write_patch",
                "args": {"patch_policy": "small", "limit_lines": 60},
                "optional": True,
            },
        ]
        write_frontmatter_plan(
            coverage_plan,
            mode="improve-coverage",
            goal="Поднять diff-coverage модуля src/moduleX до 85%, без регрессий",
            targets=[{"path": "src/moduleX"}],
            metrics={"target_diff_coverage": 0.85, "max_diff_lines": 400},
            budget={"steps": 20, "tokens": 200000},
            strategy={"stuck_after": 3},
            steps=steps,
            body_text="# План повышения покрытия для moduleX\n\n(далее — человеческое описание)",
        )
        click.echo(f"Created {coverage_plan}")
    else:
        click.echo("Template coverage plan already exists")


# --------------------------------------------------------------------------------------
# Natural language assistant
# --------------------------------------------------------------------------------------


@cli.command(hidden=True)
@click.argument("prompt", nargs=-1)
@click.pass_context
def assist(ctx: click.Context, prompt: tuple[str, ...]) -> None:
    """Route natural language prompts to existing CLI behaviours with multi-step ReAct reasoning."""
    pending = " ".join(prompt).strip()
    if not pending:
        pending = str(ctx.meta.pop("_pending_nl_prompt", "")).strip()
    if not pending:
        pending = str(ctx.obj.pop("_pending_nl_prompt", "")).strip()
    if not pending:
        raise click.UsageError("Provide a request for the assistant.")

    _record_invocation(ctx, overrides={"prompt": pending, "mode": "assist"})

    settings: Settings = ctx.obj["settings"]
    try:
        client = _get_llm_client(ctx)
    except click.ClickException as exc:
        # If no API key is configured, provide a helpful message and try basic routing
        if "No API key configured" in str(exc):
            click.echo("⚠️  No API key configured (DEVAGENT_API_KEY). Using basic keyword-based routing.")
            click.echo("   For full natural language capabilities, please set your API key.")
            # Try to provide basic functionality without LLM
            router = IntentRouter(None, settings)  # Pass None client for fallback mode
            try:
                decision: IntentDecision = router.route(pending)
            except IntentRoutingError as routing_exc:
                raise click.ClickException(f"Could not understand the request: {routing_exc}") from routing_exc
            
            # Execute single-shot fallback
            handler = INTENT_HANDLERS.get(decision.tool)
            if not handler:
                raise click.ClickException(f"Intent tool '{decision.tool}' is not supported yet.")
            handler(ctx, decision.arguments)
            return
        else:
            raise click.ClickException(f"Cannot route request without LLM client: {exc}") from exc
    
    # Execute multi-step ReAct reasoning with LLM
    _execute_react_assistant(ctx, client, settings, pending)


def _execute_react_assistant(ctx: click.Context, client, settings: Settings, user_prompt: str) -> None:
    """Execute multi-step ReAct reasoning for natural language queries."""
    from .llm_provider.base import Message
    
    # Create intent router to get available tools
    router = IntentRouter(client, settings)
    available_tools = getattr(router, "tools", [])

    if not hasattr(client, 'invoke_tools'):
        decision: IntentDecision = router.route(user_prompt)
        handler = INTENT_HANDLERS.get(decision.tool)
        if not handler:
            raise click.ClickException(f"Intent tool '{decision.tool}' is not supported yet.")
        handler(ctx, decision.arguments)
        return

    # Start conversation with the user's prompt
    messages = [
        Message(
            role="system", 
            content="You are a helpful assistant for a software development CLI tool called devagent. "
                   "Use the available tools to help answer the user's question thoroughly. "
                   "Continue using tools until you have gathered enough information to provide a complete answer. "
                   "When you have sufficient information, provide a final response without calling any more tools."
        ),
        Message(role="user", content=user_prompt)
    ]
    
    max_iterations = 30  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
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
                return
            
            # Execute the first tool call
            tool_call = result.calls[0]
            click.echo(f"🔄 Step {iteration}: Using {tool_call.name}...")
            
            # Find and execute the appropriate intent handler
            handler = INTENT_HANDLERS.get(tool_call.name)
            if not handler:
                error_msg = f"Tool '{tool_call.name}' is not supported."
                click.echo(f"❌ {error_msg}")
                messages.append(Message(
                    role="tool",
                    content=f"Error: {error_msg}",
                    tool_call_id=getattr(tool_call, 'call_id', None) or getattr(tool_call, 'id', 'unknown')
                ))
                continue
            
            # Capture tool output
            import io
            import contextlib
            
            # Redirect stdout to capture tool output
            captured_output = io.StringIO()
            try:
                with contextlib.redirect_stdout(captured_output):
                    handler(ctx, tool_call.arguments)
                tool_output = captured_output.getvalue().strip()
                if not tool_output:
                    tool_output = "Tool executed successfully (no output)"
                click.echo(f"✅ Completed: {tool_call.name}")
            except Exception as exc:
                tool_output = f"Error executing {tool_call.name}: {exc}"
                click.echo(f"❌ Failed: {tool_output}")
            
            # Add tool result to conversation  
            # For tool messages, we need to include the tool_call_id if the API requires it
            tool_call_id = getattr(tool_call, 'call_id', None) or getattr(tool_call, 'id', None)
            tool_message = Message(role="tool", content=tool_output, tool_call_id=tool_call_id)
            messages.append(tool_message)
            
        except (LLMConnectionError, LLMTimeoutError, LLMRetryExhaustedError) as exc:
            click.echo(f"⚠️  Unable to reach the LLM: {exc}")
            click.echo("   Falling back to offline analysis using local heuristics.")
            _handle_question_without_llm(user_prompt, reason="LLM unavailable")
            return
        except LLMError as exc:
            click.echo(f"❌ ReAct execution failed: {exc}")
            break
        except Exception as exc:
            click.echo(f"❌ ReAct execution failed: {exc}")
            break
    
    if iteration >= max_iterations:
        click.echo("⚠️  Reached maximum iteration limit. Provide a summary of findings...")
        # Try one final call to get a summary
        try:
            summary_messages = messages + [Message(
                role="user", 
                content="Please provide a summary of what you found based on the information gathered."
            )]
            final_result = client.invoke_tools(summary_messages, tools=[], temperature=0.1)
            if final_result.message_content:
                click.echo(final_result.message_content)
        except Exception:
            click.echo("Unable to provide a summary due to technical issues.")


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


def _search_paths_and_content(query: str, include_docs: bool = True, max_results: int = 20) -> List[str]:
    repo = Path.cwd()
    options = ContextGatheringOptions(include_docs=include_docs)
    gatherer = ContextGatherer(repo, options)
    results: List[str] = []
    # Prefer path hits
    for root in [repo / "ai_dev_agent", repo / "tests", repo / "docs", repo]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if len(results) >= max_results:
                break
            if not path.is_file():
                continue
            rel = str(path.relative_to(repo))
            if query.lower() in rel.lower():
                results.append(rel)
        if results:
            break
    if not results:
        results = gatherer.search_files(query, ["py", "md", "txt", "yaml", "yml", "toml"])[:max_results]
    return results


@cli.command(name="where")
@click.argument("query", nargs=-1)
@click.option("--include-docs/--no-include-docs", default=True, show_default=True, help="Search docs as well as code.")
@click.pass_context
def where_cmd(ctx: click.Context, query: tuple[str, ...], include_docs: bool) -> None:
    """Locate files related to a query in the repository."""
    q = " ".join(query).strip()
    if not q:
        raise click.UsageError("Provide a search query.")
    _record_invocation(ctx, overrides={"query": q})
    matches = _search_paths_and_content(q, include_docs)
    if not matches:
        click.echo("No matches found.")
        return
    for m in matches:
        click.echo(m)


@cli.command(name="locate")
@click.argument("query", nargs=-1)
@click.pass_context
def locate_alias(ctx: click.Context, query: tuple[str, ...]) -> None:
    """Alias for 'where'."""
    ctx.invoke(where_cmd, query=query)



@cli.command()
@click.argument("goal", nargs=-1)
@click.option("--auto-approve", is_flag=True, help="Skip manual approval for this plan.")
@click.option("--write-md/--no-write-md", default=False, help="Export the plan to Markdown (defaults to docs/plans/).")
@click.option("--md-path", type=click.Path(path_type=Path), help="Custom Markdown export path.")
@click.option("--show-md-path", is_flag=True, help="Display the Markdown export path without writing.")
@click.pass_context
def plan(
    ctx: click.Context,
    goal: tuple[str, ...],
    auto_approve: bool,
    write_md: bool,
    md_path: Path | None,
    show_md_path: bool,
) -> None:
    """Generate a work breakdown structure for the provided goal (non-persistent)."""
    if not goal:
        raise click.UsageError("Please provide a goal description.")
    description = " ".join(goal)
    _record_invocation(ctx, overrides={"goal": description})
    settings: Settings = ctx.obj["settings"]
    auto = auto_approve or settings.auto_approve_plan
    planner = Planner(_get_llm_client(ctx))
    try:
        result = planner.generate(description)
    except (LLMError, click.ClickException) as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Planning failed")
        raise click.ClickException(f"Planning failed: {exc}") from exc

    _render_plan(result)
    if result.fallback_reason:
        click.echo(
            f"LLM fallback activated: {result.fallback_reason}",
            err=True,
        )

    approval_status = "approved"
    if not auto:
        proceed = click.confirm("Approve this plan?", default=True)
        approval_status = "approved" if proceed else "awaiting_approval"
        if proceed:
            click.echo("Plan approved. Use `devagent run` to execute tasks.")
        else:
            click.echo("Plan saved but awaiting approval.")
    else:
        click.echo("Plan auto-approved by configuration.")

    plan_payload = result.to_dict()
    plan_payload["status"] = approval_status
    plan_payload["auto_approved"] = auto
    plan_payload["generated_at"] = datetime.utcnow().isoformat()

    state: StateStore = ctx.obj["state"]
    state.update(
        last_plan=plan_payload,
        last_updated=datetime.utcnow().isoformat(),
    )

    state_path = settings.state_file if settings.state_file.is_absolute() else _relative_to_repo(settings.state_file)
    click.echo(f"Plan stored in state at {state_path}")

    export_path = md_path
    if md_path and not write_md:
        write_md = True
    if (write_md or show_md_path) and export_path is None:
        export_path = _default_plan_markdown_path(description)

    if show_md_path and export_path is not None:
        click.echo(f"Plan markdown would be saved to {_relative_to_repo(export_path)}")

    if write_md and export_path is not None:
        exported = _export_plan_markdown(plan_payload, export_path)
        click.echo(f"Plan markdown saved to {_relative_to_repo(exported)}")


@cli.command()
@click.argument("question", nargs=-1)
@click.option(
    "--files",
    "files",
    multiple=True,
    help="Specific files to prioritise when gathering context (relative paths).",
)
@click.option(
    "--max-files",
    type=click.IntRange(1, 20),
    default=8,
    show_default=True,
    help="Maximum number of repository files to include in context.",
)
@click.option(
    "--include-docs/--no-include-docs",
    default=True,
    show_default=True,
    help="Allow documentation files to be considered during context gathering.",
)
@click.option(
    "--show-context",
    is_flag=True,
    help="Display the files that were shared with the model before the answer.",
)
@click.pass_context
def ask(
    ctx: click.Context,
    question: tuple[str, ...],
    files: tuple[str, ...],
    max_files: int,
    include_docs: bool,
    show_context: bool,
) -> None:
    """Answer questions about the repository using gathered context."""
    question_text = " ".join(question)
    _record_invocation(
        ctx,
        overrides={
            "question": question_text,
            "files": list(files),
            "max_files": max_files,
            "include_docs": include_docs,
            "show_context": show_context,
        },
    )
    try:
        _answer_repository_question(
            ctx,
            question_text,
            files=files,
            max_files=max_files,
            include_docs=include_docs,
            show_context=show_context,
        )
    except LLMError as exc:
        raise click.ClickException(f"Question answering failed: {exc}") from exc


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show the status of the latest plan and tasks."""
    _record_invocation(ctx)
    state: StateStore = ctx.obj["state"]
    data = state.load()
    if not data.get("last_plan"):
        click.echo("No plan recorded yet. Run `devagent plan` first.")
        return
    plan = data["last_plan"]
    click.echo(f"Goal: {plan.get('goal', 'unknown')} (status: {plan.get('status', 'n/a')})")
    if summary := plan.get("summary"):
        click.echo(f"Summary: {summary}")
    tasks = plan.get("tasks", [])
    if not tasks:
        click.echo("No tasks generated yet.")
        return
    for task in tasks:
        title = task.get("title", "Untitled task")
        identifier = task.get("id", "?")
        status_value = task.get("status", "pending")
        priority = task.get("priority", "n/a")
        click.echo(f"{identifier}: {title} | status={status_value} | priority={priority}")


@cli.command()
@click.option("--limit", type=click.IntRange(1, 100), default=20, show_default=True, help="Number of entries to display.")
@click.option("--replay", type=int, help="Replay the command at the given history index.")
@click.option("--clear", is_flag=True, help="Clear recorded history entries.")
@click.pass_context
def history(ctx: click.Context, limit: int, replay: Optional[int], clear: bool) -> None:
    """Show, clear, or replay recorded command history."""
    state: StateStore = ctx.obj["state"]
    data = state.load()
    history_entries = list(data.get("command_history", []))

    if clear:
        state.update(command_history=[])
        click.echo("Command history cleared.")
        return

    if not history_entries:
        click.echo("No commands recorded yet.")
        return

    if replay is not None:
        if replay < 0 or replay >= len(history_entries):
            raise click.ClickException("Invalid history index.")
        entry = history_entries[replay]
        command_path = entry.get("command_path") or []
        command = _resolve_command(command_path)
        if not command:
            raise click.ClickException("Stored command no longer exists in this CLI version.")
        params = entry.get("params") or {}
        click.echo(f"Replaying command: {' '.join(command_path)}")
        root_ctx = ctx.find_root()
        if not root_ctx:
            raise click.ClickException("Unable to resolve CLI root context for replay.")
        _record_invocation(ctx)
        root_ctx.invoke(command, **params)
        return

    start_index = max(len(history_entries) - limit, 0)
    for idx, entry in enumerate(history_entries[start_index:], start=start_index):
        timestamp = entry.get("timestamp", "")
        command_path = " ".join(entry.get("command_path", []))
        params = entry.get("params", {})
        click.echo(f"[{idx}] {command_path or '<unknown>'} | params={params} | {timestamp}")


@cli.command()
@click.option("--limit", type=click.IntRange(1, 200), default=20, show_default=True, help="Number of metrics entries to display.")
@click.pass_context
def metrics(ctx: click.Context, limit: int) -> None:
    """Show aggregated task metrics tracked during runs."""
    _record_invocation(ctx, overrides={"limit": limit})
    state: StateStore = ctx.obj["state"]
    data = state.load()
    entries = list(data.get("metrics", []))
    if not entries:
        click.echo("No metrics recorded yet.")
        return

    total = len(entries)
    successes = sum(1 for item in entries if item.get("outcome") == "completed")
    dry_runs = sum(1 for item in entries if item.get("dry_run"))
    avg_duration = sum(item.get("duration_sec", 0.0) for item in entries) / total
    success_rate = (successes / total) * 100 if total else 0.0

    click.echo(
        f"Total tasks: {total} | Success rate: {success_rate:.1f}% | Avg duration: {avg_duration:.2f}s | Dry runs: {dry_runs}"
    )
    click.echo("Recent entries:")
    start_index = max(total - limit, 0)
    for entry in entries[start_index:]:
        outcome = entry.get("outcome", "?")
        timestamp = entry.get("timestamp", "?")
        duration = entry.get("duration_sec", 0.0)
        extra = " dry-run" if entry.get("dry_run") else ""
        click.echo(
            f"- [{timestamp}] task={entry.get('task_id','?')} outcome={outcome} duration={duration:.2f}s{extra}"
        )


@cli.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish", "powershell"]),
    required=True,
    help="Target shell for the completion script.",
)
@click.option(
    "--eval",
    "print_eval",
    is_flag=True,
    help="Print an eval helper snippet instead of the raw script.",
)
@click.pass_context
def completion(ctx: click.Context, shell: str, print_eval: bool) -> None:
    """Emit shell completion scripts for the CLI."""
    root_ctx = ctx.find_root()
    prog_name = root_ctx.info_name if root_ctx and root_ctx.info_name else "devagent"
    complete_var = f"_{prog_name.replace('-', '_').upper()}_COMPLETE"
    completion_cls = get_completion_class(shell)
    if completion_cls is None:
        raise click.ClickException(f"Shell completion not supported for {shell}.")
    completer = completion_cls(cli, ctx.obj or {}, prog_name, complete_var)
    if print_eval:
        mapping = {
            "bash": "source_bash",
            "zsh": "source_zsh",
            "fish": "source_fish",
            "powershell": "source_powershell",
        }
        mode = mapping.get(shell, f"source_{shell}")
        click.echo(f'eval "$({complete_var}={mode} {prog_name})"')
    else:
        click.echo(completer.source())
    _record_invocation(ctx, overrides={"shell": shell, "eval": print_eval})


@cli.group()
@click.pass_context
def git(ctx: click.Context) -> None:
    """Git workflow helpers (commit messages, branches, PRs)."""


@cli.group()
@click.pass_context
def react(ctx: click.Context) -> None:
    """ReAct-enabled automation helpers."""

@react.command("plan")
@click.argument("goal", nargs=-1)
@click.pass_context
def react_plan(ctx: click.Context, goal: tuple[str, ...]) -> None:
    """Produce a ReAct work breakdown structure without calling the LLM."""
    description = " ".join(goal).strip()
    planner = Planner(None)  # type: ignore[arg-type]
    result = planner.react_wbs(description or "Deliver PARE/ReAct automation")
    _render_plan(result)
    _record_invocation(ctx, overrides={"goal": description, "react_template": True})



@react.command("run")
@click.argument("task_id", required=False)
@click.option("--diff-base", help="Git ref used for diff and coverage computations.")
@click.option("--test-command", help="Override test command (default from config).")
@click.option("--lint-command", help="Override lint command.")
@click.option("--type-command", help="Override type-check command.")
@click.option("--format-command", help="Override format command.")
@click.option("--compile-command", help="Override compile/build command.")
@click.option("--skip-tests", is_flag=True, help="Skip running automated tests (gates will fail).")
@click.option("--plan-file", type=click.Path(path_type=Path), help="Load plan from a Markdown/JSON file instead of the saved state.")
@click.option("--profile", type=click.Choice(["implementation", "review", "design"]), help="Override task category routing.")
@click.pass_context
def react_run(
    ctx: click.Context,
    task_id: Optional[str],
    diff_base: Optional[str],
    test_command: Optional[str],
    lint_command: Optional[str],
    type_command: Optional[str],
    format_command: Optional[str],
    compile_command: Optional[str],
    plan_file: Path | None,
    skip_tests: bool,
    profile: Optional[str],
) -> None:
    """Execute quality gates for a plan task using the ReAct loop."""
    settings: Settings = ctx.obj["settings"]
    state: StateStore = ctx.obj["state"]

    plan_source = "state store"
    if plan_file:
        plan_path = plan_file if plan_file.is_absolute() else (Path.cwd() / plan_file)
        plan_dict = _load_plan_from_path(plan_path)
        plan = _normalize_plan_dict(plan_dict)
        state_data = state.load()
        state_data["last_plan"] = plan
        state_data["last_updated"] = datetime.utcnow().isoformat()
        state.save(state_data)
        plan_source = _relative_to_repo(plan_path)
    else:
        plan = _normalize_plan_dict(_load_plan(state))

    task = _select_task(plan, task_id)
    profile = (profile or (task.get("category") or "implementation")).strip().lower()
    _record_invocation(
        ctx,
        overrides={
            "task_id": task.get("id"),
            "diff_base": diff_base,
            "test_command": test_command,
            "lint_command": lint_command,
            "type_command": type_command,
            "format_command": format_command,
            "compile_command": compile_command,
            "skip_tests": skip_tests,
            "plan_file": str(plan_file) if plan_file else None,
        },
    )

    click.echo(f"Using plan from {plan_source}")
    click.echo(f"Executing task {task.get('id', '?')}: {task.get('title', '').strip()}")

    if profile == "design":
        click.echo("Routing to design workflow")
        ctx.invoke(react_design, task_id=task.get("id"), dry_run=False)
        return
    if profile == "review":
        click.echo("Routing to review workflow")
        ctx.invoke(react_review, task_id=task.get("id"), require_tests=not skip_tests)
        return

    started_at = datetime.utcnow()
    sandbox = _create_sandbox(settings)
    effective_diff_base = diff_base or "HEAD"
    collector = MetricsCollector(Path.cwd(), diff_base=effective_diff_base)

    pipeline_commands = PipelineCommands(
        format=parse_command(format_command or settings.format_command),
        lint=parse_command(lint_command or settings.lint_command),
        typecheck=parse_command(type_command or settings.typecheck_command),
        compile=parse_command(compile_command or settings.compile_command),
        test=parse_command(test_command or settings.test_command),
        coverage_xml=settings.coverage_xml_path if settings.coverage_xml_path else None,
        perf=parse_command(settings.perf_command),
        flake_runs=settings.flake_check_runs,
    )

    run_tests_flag = not skip_tests

    gate_evaluator = _build_gate_evaluator(settings, pipeline_commands, run_tests_flag)
    executor = ReactiveExecutor(gate_evaluator)

    task_spec = TaskSpec(
        identifier=task.get("id", "?"),
        goal=task.get("title") or plan.get("goal", ""),
        category=task.get("category", "implementation"),
        instructions=task.get("description"),
        files=_infer_task_files(task, Path.cwd()) or None,
    )

    def action_provider(_: TaskSpec, history):
        if history:
            raise StopIteration
        return ActionRequest(
            step_id=f"S{len(history) + 1}",
            thought="Execute automated quality pipeline for the task.",
            tool="qa.pipeline",
            args={"run_tests": run_tests_flag},
            metadata={},
        )

    def tool_invoker(action: ActionRequest):
        return run_quality_pipeline(
            Path.cwd(),
            sandbox,
            collector,
            pipeline_commands,
            run_tests=bool(action.args.get("run_tests", True)),
            tokens_cost=action.metadata.get("tokens_cost") if action.metadata else None,
        )

    try:
        result = executor.run(task_spec, action_provider, tool_invoker)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"ReAct execution failed: {exc}") from exc

    _render_gate_summary(result)

    outcome = "completed" if result.status == "success" else "failed"
    metrics_note = result.stop_reason or "Quality gates incomplete."
    updates: Dict[str, Any] = {
        "status": "completed" if result.status == "success" else "needs_attention",
        "gates": result.gates,
        "last_run_status": result.status,
        "last_run_metrics": result.metrics,
    }
    if result.status != "success":
        updates["note"] = metrics_note
    _update_task_state(state, plan, task, updates)
    _record_task_metric(state, task, outcome, started_at)

    if result.status != "success":
        raise click.ClickException(metrics_note)

    click.echo("All gates satisfied.")


# --------------------------------------------------------------------------------------
# Gate: one-shot gate evaluation (for CI/local parity)
# --------------------------------------------------------------------------------------


@cli.command(name="gate")
@click.option("--skip-tests", is_flag=True, help="Skip running tests during gate evaluation.")
@click.pass_context
def gate_cmd(ctx: click.Context, skip_tests: bool) -> None:
    """Run quality gates using configured commands and thresholds."""
    settings: Settings = ctx.obj["settings"]
    _record_invocation(ctx, overrides={"skip_tests": skip_tests})
    sandbox = _create_sandbox(settings)
    collector = MetricsCollector(Path.cwd(), diff_base="HEAD")

    # Allow devagent.yaml to override commands and limits
    yaml_cfg = load_devagent_yaml()
    fmt_cmd = (yaml_cfg.format_cmd if yaml_cfg and yaml_cfg.format_cmd else settings.format_command)
    lint_cmd = (yaml_cfg.lint_cmd if yaml_cfg and yaml_cfg.lint_cmd else settings.lint_command)
    type_cmd = (yaml_cfg.type_cmd if yaml_cfg and yaml_cfg.type_cmd else settings.typecheck_command)
    test_cmd = (yaml_cfg.test_cmd if yaml_cfg and yaml_cfg.test_cmd else settings.test_command)
    comp_cmd = settings.compile_command

    pipeline_commands = PipelineCommands(
        format=parse_command(fmt_cmd),
        lint=parse_command(lint_cmd),
        typecheck=parse_command(type_cmd),
        compile=parse_command(comp_cmd),
        test=parse_command(test_cmd) if not skip_tests else None,
        coverage_xml=settings.coverage_xml_path if settings.coverage_xml_path else None,
        perf=None,
        flake_runs=settings.flake_check_runs,
    )

    gate_evaluator = _build_gate_evaluator(settings, pipeline_commands, run_tests=not skip_tests)
    executor = ReactiveExecutor(gate_evaluator)

    task_spec = TaskSpec(identifier="GATE", goal="Evaluate project gates", category="review")

    def action_provider(_: TaskSpec, history):
        if history:
            raise StopIteration
        return ActionRequest(step_id="S1", thought="Run gate pipeline", tool="qa.pipeline", args={"run_tests": not skip_tests})

    def tool_invoker(action: ActionRequest):
        return run_quality_pipeline(Path.cwd(), sandbox, collector, pipeline_commands, run_tests=not skip_tests)

    result = executor.run(task_spec, action_provider, tool_invoker)
    _render_gate_summary(result)
    if result.status != "success":
        raise click.ClickException(result.stop_reason or "Gates failed.")
    click.echo("Gates passed.")


# --------------------------------------------------------------------------------------
# Review against plan (traceability)
# --------------------------------------------------------------------------------------


@cli.command(name="review")
@click.option("--plan", "plan_file", type=click.Path(path_type=Path), required=True, help="Path to plan with YAML frontmatter.")
@click.option("--fail-on-divergence", is_flag=True, help="Exit 1 if artifacts/metrics diverge from plan.")
@click.pass_context
def review_cmd(ctx: click.Context, plan_file: Path, fail_on_divergence: bool) -> None:
    """Verify repository artifacts/metrics against the provided plan."""
    _record_invocation(ctx, overrides={"plan": str(plan_file), "fail_on_divergence": fail_on_divergence})
    try:
        plan = parse_frontmatter_plan(plan_file)
    except Exception as exc:
        raise click.ClickException(f"Invalid plan: {exc}") from exc

    # Check produced artifacts exist
    missing: List[str] = []
    for step in plan.steps:
        for artifact in step.produces:
            if artifact and not (Path.cwd() / artifact).exists():
                missing.append(artifact)

    # Optionally run gate pipeline to compute current metrics
    settings: Settings = ctx.obj["settings"]
    sandbox = _create_sandbox(settings)
    collector = MetricsCollector(Path.cwd(), diff_base="HEAD")
    pipeline_commands = PipelineCommands(
        format=parse_command(settings.format_command),
        lint=parse_command(settings.lint_command),
        typecheck=parse_command(settings.typecheck_command),
        compile=parse_command(settings.compile_command),
        test=None,  # fast review by default
        coverage_xml=settings.coverage_xml_path if settings.coverage_xml_path else None,
        perf=None,
        flake_runs=0,
    )
    obs = run_quality_pipeline(Path.cwd(), sandbox, collector, pipeline_commands, run_tests=False)
    gate_eval = _build_gate_evaluator_for_profile("review", settings, pipeline=pipeline_commands, require_tests=False)
    evaluation = gate_eval.evaluate(obs.metrics if hasattr(obs, "metrics") else obs, [])

    if missing:
        click.echo("Missing artifacts:")
        for item in missing:
            click.echo(f"- {item}")

    click.echo("Gate status:")
    for name, required in evaluation.required_gates.items():
        passed = evaluation.gates.get(name, False)
        click.echo(f"  {'PASS' if passed else 'FAIL'} {name}")

    diverged = bool(missing) or any(not v for k, v in evaluation.gates.items() if evaluation.required_gates.get(k))
    if diverged and fail_on_divergence:
        raise click.ClickException("Plan review divergence detected.")
    if diverged:
        click.echo("Divergence detected.")
    else:
        click.echo("Plan artifacts present; gates look healthy.")


# --------------------------------------------------------------------------------------
# Status/Explain/Abort/Resume helpers
# --------------------------------------------------------------------------------------


@cli.command(name="explain")
@click.argument("step_id")
@click.pass_context
def explain_cmd(ctx: click.Context, step_id: str) -> None:
    """Explain why a run chose a given action/step (from last trace)."""
    _record_invocation(ctx, overrides={"step_id": step_id})
    runs_dir = Path.cwd() / ".devagent" / "runs"
    if not runs_dir.is_dir():
        click.echo("No runs recorded yet.")
        return
    # Pick the most recent jsonl
    traces = sorted(runs_dir.glob("*.jsonl"), reverse=True)
    if not traces:
        click.echo("No traces found.")
        return
    latest = traces[0]
    # naive print of lines that reference step_id
    try:
        matched = [line for line in latest.read_text(encoding="utf-8").splitlines() if f'"step_id": "{step_id}"' in line]
        if not matched:
            click.echo(f"Step {step_id} not found in last run.")
            return
        for line in matched:
            click.echo(line)
    except Exception as exc:  # pragma: no cover - best effort
        click.echo(f"Failed to read trace: {exc}")


@cli.command(name="abort")
@click.pass_context
def abort_cmd(ctx: click.Context) -> None:
    """Abort the current loop/session if active."""
    state: StateStore = ctx.obj["state"]
    session = state.get_current_session()
    if not session:
        click.echo("No active session.")
        return
    state.update_session(status="interrupted")
    click.echo(f"Aborted session {session.session_id}.")


@cli.command(name="resume")
@click.pass_context
def resume_cmd(ctx: click.Context) -> None:
    """Resume an interrupted session if available."""
    state: StateStore = ctx.obj["state"]
    if not state.can_resume():
        click.echo("Nothing to resume.")
        return
    session = state.get_current_session()
    if session:
        state.update_session(status="active")
        click.echo(f"Resumed session {session.session_id}.")


# --------------------------------------------------------------------------------------
# Run a frontmatter plan (MVP)
# --------------------------------------------------------------------------------------


@cli.command(name="run-plan")
@click.option("--plan", "plan_file", type=click.Path(path_type=Path), required=True, help="Path to plan with YAML frontmatter.")
@click.option("--apply", "apply_changes", is_flag=True, help="Apply changes; otherwise act as dry-run.")
@click.option("--auto-commit", is_flag=True, help="Auto-commit after passing gates/checkpoints.")
@click.option("--checkpoint", "checkpoint_every", type=int, default=0, show_default=True, help="Checkpoint every N steps (0=off).")
@click.option("--max-diff-lines", type=int, help="Override diff lines limit gate.")
@click.pass_context
def run_plan_cmd(
    ctx: click.Context,
    plan_file: Path,
    apply_changes: bool,
    auto_commit: bool,
    checkpoint_every: int,
    max_diff_lines: int | None,
) -> None:
    """Execute a YAML-frontmatter plan step-by-step (dry-run by default)."""
    _record_invocation(
        ctx,
        overrides={
            "plan": str(plan_file),
            "apply": apply_changes,
            "auto_commit": auto_commit,
            "checkpoint": checkpoint_every,
            "max_diff_lines": max_diff_lines,
        },
    )
    _run_frontmatter_plan(ctx, plan_file, apply_changes, auto_commit, checkpoint_every, max_diff_lines)


def _run_frontmatter_plan(
    ctx: click.Context,
    plan_file: Path,
    apply_changes: bool,
    auto_commit: bool,
    checkpoint_every: int,
    max_diff_lines_override: int | None,
) -> None:
    settings: Settings = ctx.obj["settings"]
    try:
        plan = parse_frontmatter_plan(plan_file)
    except Exception as exc:
        raise click.ClickException(f"Invalid plan: {exc}") from exc

    click.echo(f"Executing plan: {plan.goal}")
    sandbox = _create_sandbox(settings)
    collector = MetricsCollector(Path.cwd(), diff_base="HEAD")

    yaml_cfg = load_devagent_yaml()
    pipeline_commands = PipelineCommands(
        format=parse_command((yaml_cfg.format_cmd if yaml_cfg and yaml_cfg.format_cmd else settings.format_command)),
        lint=parse_command((yaml_cfg.lint_cmd if yaml_cfg and yaml_cfg.lint_cmd else settings.lint_command)),
        typecheck=parse_command((yaml_cfg.type_cmd if yaml_cfg and yaml_cfg.type_cmd else settings.typecheck_command)),
        compile=parse_command(settings.compile_command),
        test=parse_command((yaml_cfg.test_cmd if yaml_cfg and yaml_cfg.test_cmd else settings.test_command)),
        coverage_xml=settings.coverage_xml_path if settings.coverage_xml_path else None,
        perf=None,
        flake_runs=settings.flake_check_runs,
    )

    gate_eval = _build_gate_evaluator(settings, pipeline_commands, run_tests=True)
    if max_diff_lines_override and max_diff_lines_override > 0:
        gate_eval.config.diff_limit_lines = max_diff_lines_override
    executor = ReactiveExecutor(gate_eval)
    task_spec = TaskSpec(identifier="PLAN", goal=plan.goal, category=plan.mode)

    step_count = 0
    for step in plan.steps:
        step_count += 1
        click.echo(f"Step {step.id}: {step.tool}")
        # Dry-run reporting
        if not apply_changes:
            if step.gates:
                click.echo("  Gates: " + ", ".join(step.gates))
            if step.produces:
                click.echo("  Produces: " + ", ".join(step.produces))
            continue

        if step.tool == "tests.run":
            observation = run_quality_pipeline(Path.cwd(), sandbox, collector, pipeline_commands, run_tests=True)
            result = executor.run(
                task_spec,
                action_provider=lambda *_: ActionRequest(step_id=step.id, thought="run tests", tool="qa.pipeline", args={"run_tests": True}),
                tool_invoker=lambda _action: observation,
            )
            _render_gate_summary(result)
            if result.status != "success":
                raise click.ClickException(result.stop_reason or "Gate failure during tests.run")
        elif step.tool == "coverage.diff":
            target = step.args.get("path")
            report_path = step.produces[0] if step.produces else f"reports/{(target or 'target').replace('/', '_')}_gaps.json"
            report_abs = Path.cwd() / report_path
            report_abs.parent.mkdir(parents=True, exist_ok=True)
            files_found = 0
            if target:
                target_path = Path.cwd() / target
                if target_path.is_dir():
                    files_found = sum(1 for _ in target_path.rglob("*.py"))
                elif target_path.is_file():
                    files_found = 1
            content = {"path": target or "", "files_found": files_found}
            report_abs.write_text(json.dumps(content, indent=2), encoding="utf-8")
            click.echo(f"  Wrote {report_path} (files_found={files_found})")
            for gate in step.gates:
                try:
                    if ">=" in gate:
                        key, value = gate.split(">=", 1)
                        ok = content.get(key.strip()) >= float(value)
                    elif "<=" in gate:
                        key, value = gate.split("<=", 1)
                        ok = content.get(key.strip()) <= float(value)
                    elif ">" in gate:
                        key, value = gate.split(">", 1)
                        ok = content.get(key.strip()) > float(value)
                    elif "<" in gate:
                        key, value = gate.split("<", 1)
                        ok = content.get(key.strip()) < float(value)
                    elif "==" in gate:
                        key, value = gate.split("==", 1)
                        ok = str(content.get(key.strip())) == value.strip()
                    else:
                        ok = True
                except Exception:
                    ok = False
                if not ok:
                    raise click.ClickException(f"Gate failed: {gate}")
        elif step.tool == "write_patch":
            click.echo("  (write_patch placeholder — no changes applied)")
        else:
            click.echo(f"  Unsupported tool: {step.tool} (skipped)")

        if auto_commit or (checkpoint_every and (step_count % checkpoint_every == 0)):
            try:
                _git_commit(f"[devagent] {step.id}: apply step")
                click.echo("  Checkpoint committed.")
            except Exception as exc:
                click.echo(f"  Commit skipped: {exc}")

    click.echo("Plan execution complete.")


def _git_commit(message: str) -> None:
    import subprocess
    repo = Path.cwd()
    subprocess.run(["git", "add", "-A"], cwd=repo, check=False)
    proc = subprocess.run(["git", "commit", "-m", message], cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0 and "nothing to commit" not in (proc.stdout + proc.stderr).lower():
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "git commit failed")



@react.command("review")
@click.argument("task_id", required=False)
@click.option("--require-tests", is_flag=True, help="Require tests to pass in review mode.")
@click.pass_context
def react_review(
    ctx: click.Context,
    task_id: Optional[str],
    require_tests: bool,
) -> None:
    """Execute review gates (lint/types/format/diff/coverage/security)."""
    settings: Settings = ctx.obj["settings"]
    state: StateStore = ctx.obj["state"]
    plan = _load_plan(state)
    task = _select_task(plan, task_id)
    _record_invocation(ctx, overrides={"task_id": task.get("id"), "require_tests": require_tests})
    sandbox = _create_sandbox(settings)
    collector = MetricsCollector(Path.cwd(), diff_base="HEAD")
    pipeline_commands = PipelineCommands(
        format=parse_command(settings.format_command),
        lint=parse_command(settings.lint_command),
        typecheck=parse_command(settings.typecheck_command),
        compile=parse_command(settings.compile_command) if require_tests else None,
        test=parse_command(settings.test_command) if require_tests else None,
        coverage_xml=settings.coverage_xml_path if settings.coverage_xml_path else None,
        perf=None,
        flake_runs=0,
    )

    gate_evaluator = _build_gate_evaluator_for_profile(
        "review",
        settings,
        pipeline=pipeline_commands,
        require_tests=require_tests,
    )
    executor = ReactiveExecutor(gate_evaluator)

    task_spec = TaskSpec(
        identifier=task.get("id", "?"),
        goal=task.get("title") or plan.get("goal", ""),
        category="review",
        instructions=task.get("description"),
        files=_infer_task_files(task, Path.cwd()) or None,
    )

    def action_provider(_: TaskSpec, history):
        if history:
            raise StopIteration
        return ActionRequest(
            step_id="S1",
            thought="Run review quality checks.",
            tool="qa.pipeline",
            args={"run_tests": require_tests},
        )

    def tool_invoker(action: ActionRequest):
        return run_quality_pipeline(Path.cwd(), sandbox, collector, pipeline_commands, run_tests=require_tests)

    result = executor.run(task_spec, action_provider, tool_invoker)
    _render_gate_summary(result)
    updates: Dict[str, Any] = {
        "status": "completed" if result.status == "success" else "needs_attention",
        "gates": result.gates,
        "last_run_status": result.status,
        "last_run_metrics": result.metrics,
    }
    _update_task_state(state, plan, task, updates)
    if result.status != "success":
        raise click.ClickException(result.stop_reason or "Review gates failed.")
    click.echo("Review gates satisfied.")


@react.command("design")
@click.argument("task_id", required=False)
@click.option("--dry-run", is_flag=True, help="Preview ADR path without writing.")
@click.pass_context
def react_design(
    ctx: click.Context,
    task_id: Optional[str],
    dry_run: bool,
) -> None:
    """Generate ADR and evaluate document completeness."""
    settings: Settings = ctx.obj["settings"]
    state: StateStore = ctx.obj["state"]
    plan = _load_plan(state)
    task = _select_task(plan, task_id)
    _record_invocation(ctx, overrides={"task_id": task.get("id"), "dry_run": dry_run})
    try:
        llm_client = _get_llm_client(ctx)
    except click.ClickException as exc:  # allow dry-run without key
        if not dry_run:
            raise
        llm_client = None  # type: ignore

    gate_evaluator = _build_gate_evaluator_for_profile("design", settings)
    executor = ReactiveExecutor(gate_evaluator)

    task_spec = TaskSpec(
        identifier=task.get("id", "?"),
        goal=plan.get("goal", ""),
        category="design",
        instructions=task.get("description"),
        files=None,
    )

    def action_provider(_: TaskSpec, history):
        if history:
            raise StopIteration
        return ActionRequest(step_id="S1", thought="Generate ADR document.", tool="design.adr")

    def tool_invoker(action: ActionRequest):
        try:
            adr_manager = ADRManager(Path.cwd(), _get_llm_client(ctx), ADR_TEMPLATE_PATH)
            adr_result = adr_manager.generate(plan.get("goal", ""), task.get("title", "ADR"), task.get("description", ""), dry_run=dry_run)
            doc_ok, missing = _check_adr_completeness(adr_result.path)
            notes = {}
            if missing:
                notes["design_doc"] = "Missing sections: " + ", ".join(missing)
            metrics = {
                "design_doc_ok": doc_ok,
                "doc_path": str(adr_result.path),
                "missing_sections": missing,
                "gate_notes": notes,
            }
            return Observation(success=doc_ok, outcome="ADR generated", metrics=metrics, artifacts=[str(adr_result.path)], tool="design.adr")
        except Exception as exc:
            metrics = {"design_doc_ok": False, "error": str(exc)}
            return Observation(success=False, outcome="ADR generation failed", metrics=metrics, error=str(exc), tool="design.adr")

    result = executor.run(task_spec, action_provider, tool_invoker)
    _render_gate_summary(result)
    updates: Dict[str, Any] = {
        "status": "completed" if result.status == "success" else "needs_attention",
        "gates": result.gates,
        "last_run_status": result.status,
        "last_run_metrics": result.metrics,
    }
    _update_task_state(state, plan, task, updates)
    if result.status != "success":
        raise click.ClickException(result.stop_reason or "Design gates failed.")
    click.echo("Design gate satisfied.")

@git.command("commit-message")
@click.option(
    "--scope",
    type=click.Choice(["staged", "unstaged", "all"]),
    default="staged",
    show_default=True,
    help="Which changes to include when building the diff context.",
)
@click.pass_context
def git_commit_message(ctx: click.Context, scope: str) -> None:
    """Generate a commit message draft from repository changes."""
    _record_invocation(ctx, overrides={"scope": scope})
    include_staged = scope in {"staged", "all"}
    include_unstaged = scope in {"unstaged", "all"}
    repo_root = Path.cwd()
    try:
        diff_context = gather_diff(
            repo_root,
            include_staged=include_staged,
            include_unstaged=include_unstaged,
        )
    except GitIntegrationError as exc:
        raise click.ClickException(str(exc)) from exc

    llm_client = _get_llm_client(ctx)
    state: StateStore = ctx.obj["state"]
    plan_context = _plan_context_for_git(state)

    try:
        message = generate_commit_message(
            llm_client,
            diff_context,
            context=plan_context,
        )
    except LLMError as exc:
        raise click.ClickException(f"Commit message generation failed: {exc}") from exc

    click.echo(message)
    if diff_context.is_truncated:
        click.echo("\nNote: diff was truncated before sending to the model.", err=True)


@git.command("start-feature")
@click.argument("name", required=False)
@click.option("--base", "base_branch", help="Base branch to branch from (defaults to detected main branch).")
@click.option("--prefix", default="feature", show_default=True, help="Prefix to prepend to the branch name.")
@click.option("--dry-run", is_flag=True, help="Preview the branch name without creating it.")
@click.pass_context
def git_start_feature(
    ctx: click.Context,
    name: Optional[str],
    base_branch: Optional[str],
    prefix: str,
    dry_run: bool,
) -> None:
    """Create (or preview) a feature branch with a consistent naming scheme."""
    state: StateStore = ctx.obj["state"]
    if not name:
        plan = state.load().get("last_plan") or {}
        default_name = (plan.get("summary") or plan.get("goal") or "").strip() or None
        if default_name:
            name = click.prompt("Feature name", default=default_name, show_default=True)
        else:
            name = click.prompt("Feature name", default="", show_default=False)
    name = name.strip()
    if not name:
        raise click.ClickException("Feature name is required.")

    repo_root = Path.cwd()
    resolved_base = base_branch or guess_default_base_branch(repo_root)
    _record_invocation(
        ctx,
        overrides={
            "name": name,
            "base": resolved_base,
            "prefix": prefix,
            "dry_run": dry_run,
        },
    )
    try:
        branch_name, base_used = create_feature_branch(
            repo_root,
            name,
            prefix=prefix,
            base=resolved_base,
            dry_run=dry_run,
        )
    except GitIntegrationError as exc:
        raise click.ClickException(str(exc)) from exc

    if dry_run:
        click.echo(f"Would create branch {branch_name} from {base_used}.")
    else:
        click.echo(f"Created and switched to {branch_name} (base {base_used}).")


@git.command("pr-description")
@click.option(
    "--scope",
    type=click.Choice(["staged", "unstaged", "all"]),
    default="staged",
    show_default=True,
    help="Which changes to include when building the diff context.",
)
@click.option("--base", "base_branch", help="Base branch for the PR (defaults to detected main branch).")
@click.option("--feature", "feature_branch", help="Feature branch name (defaults to current branch).")
@click.pass_context
def git_pr_description(
    ctx: click.Context,
    scope: str,
    base_branch: Optional[str],
    feature_branch: Optional[str],
) -> None:
    """Generate a Markdown pull request description from repository changes."""
    include_staged = scope in {"staged", "all"}
    include_unstaged = scope in {"unstaged", "all"}
    repo_root = Path.cwd()
    try:
        diff_context = gather_diff(
            repo_root,
            include_staged=include_staged,
            include_unstaged=include_unstaged,
        )
    except GitIntegrationError as exc:
        raise click.ClickException(str(exc)) from exc

    state: StateStore = ctx.obj["state"]
    plan_context = _plan_context_for_git(state)

    resolved_feature = feature_branch
    if not resolved_feature:
        try:
            resolved_feature = get_current_branch(repo_root)
        except GitIntegrationError:
            resolved_feature = None

    resolved_base = base_branch or guess_default_base_branch(repo_root)

    _record_invocation(
        ctx,
        overrides={
            "scope": scope,
            "base": resolved_base,
            "feature": resolved_feature,
        },
    )

    llm_client = _get_llm_client(ctx)
    try:
        description = generate_pr_description(
            llm_client,
            diff_context,
            context=plan_context,
            base_branch=resolved_base,
            feature_branch=resolved_feature,
        )
    except LLMError as exc:
        raise click.ClickException(f"PR description generation failed: {exc}") from exc

    click.echo(description)
    if diff_context.is_truncated:
        click.echo("\nNote: diff was truncated before sending to the model.", err=True)


def _load_plan(state: StateStore) -> Dict[str, Any]:
    data = state.load()
    plan = data.get("last_plan")
    if not plan:
        raise click.ClickException("No plan available. Generate one with `devagent plan` first.")
    return plan


def _select_task(plan: Dict[str, Any], task_id: Optional[str]) -> Dict[str, Any]:
    tasks = plan.get("tasks", [])
    if not tasks:
        raise click.ClickException("Plan has no tasks to execute.")
    if task_id:
        for task in tasks:
            if task.get("id") == task_id:
                return task
        raise click.ClickException(f"Task {task_id} not found in plan.")
    for task in tasks:
        if task.get("status") not in {"completed", "skipped"}:
            return task
    raise click.ClickException("All tasks are already completed.")


def _collect_files_from_directory(directory: Path, repo_root: Path, seen: set[str], limit: int = 10) -> List[str]:
    files: List[str] = []
    count = 0
    for path in directory.rglob('*'):
        if count >= limit:
            break
        if not path.is_file():
            continue
        rel_path = str(path.relative_to(repo_root))
        if rel_path in seen:
            continue
        seen.add(rel_path)
        files.append(rel_path)
        count += 1
    return files


def _extract_task_text(task: Dict[str, Any]) -> str:
    return " ".join(
        part.strip()
        for part in [task.get("title", ""), task.get("description", "")]
        if part
    )


def _extract_path_hints(text: str, repo_root: Path, seen: set[str]) -> List[str]:
    if not text:
        return []
    hints = set()
    for match in re.findall(r'([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+/?)+', text):
        normalized = match.rstrip('/')
        if not normalized:
            continue
        hints.add(normalized)

    discovered: List[str] = []
    for hint in hints:
        candidate = repo_root / hint
        if candidate.is_file():
            rel = str(Path(hint))
            if rel not in seen:
                seen.add(rel)
                discovered.append(rel)
        elif candidate.is_dir():
            discovered.extend(_collect_files_from_directory(candidate, repo_root, seen))

    return discovered
def _find_files_for_keywords(repo_root: Path, keywords: Iterable[str], seen: set[str]) -> List[str]:
    discovered: List[str] = []
    if not keywords:
        return discovered

    search_roots = [repo_root / name for name in ("ai_dev_agent", "tests", "docs")]
    search_roots = [root for root in search_roots if root.exists()]
    if not search_roots:
        search_roots = [repo_root]

    for keyword in keywords:
        keyword_lower = keyword.lower()
        for root in search_roots:
            for path in root.rglob('*'):
                if len(discovered) >= 10:
                    return discovered
                if not path.is_file():
                    continue
                rel_path = str(path.relative_to(repo_root))
                if rel_path in seen:
                    continue
                if keyword_lower in rel_path.lower():
                    seen.add(rel_path)
                    discovered.append(rel_path)
            if discovered:
                break
        if discovered:
            break

    # If no matches found by path names, fall back to content search via context gatherer
    if not discovered:
        gatherer = ContextGatherer(repo_root)
        for keyword in keywords:
            for match in gatherer.search_files(keyword, ["py", "md", "rst", "txt", "json", "yaml", "yml"]):
                if len(discovered) >= 10:
                    return discovered
                rel_match = match.strip()
                if not rel_match or rel_match in seen:
                    continue
                candidate = repo_root / rel_match
                if candidate.is_file():
                    seen.add(rel_match)
                    discovered.append(rel_match)
            if discovered:
                break

    return discovered


def _infer_paths_from_tokens(tokens: Iterable[str]) -> List[str]:
    """Extract potential file paths from a list of CLI tokens."""
    paths: List[str] = []
    capture_segments = {"--files", "--file", "--path", "--paths"}
    skip_next: Optional[str] = None

    for idx, token in enumerate(tokens):
        if token in capture_segments:
            skip_next = token
            continue

        if skip_next:
            if not token.startswith("--"):
                paths.append(token)
                continue
            skip_next = None

        if token.startswith("--"):
            continue

        if "/" in token or token.endswith(('.py', '.md', '.txt', '.rst', '.json', '.yaml', '.yml')):
            paths.append(token)

    return paths


def _infer_task_files(task: Dict[str, Any], repo_root: Path) -> List[str]:
    """Infer target files for a task using plan metadata and existing files."""
    inferred: List[str] = []
    seen: set[str] = set()

    commands = task.get("commands") or []
    for command in commands:
        try:
            tokens = shlex.split(command)
        except ValueError:
            LOGGER.debug("Failed to parse command for inference: %s", command)
            continue

        for candidate in _infer_paths_from_tokens(tokens):
            normalized = candidate.strip().strip('\"\'')
            if not normalized:
                continue
            # Ignore clearly non-path arguments (e.g. placeholders or flags)
            if normalized.startswith("-"):
                continue
            rel_path = Path(normalized)
            resolved = (repo_root / rel_path).resolve()
            try:
                resolved.relative_to(repo_root.resolve())
            except ValueError:
                continue
            if resolved.is_file() and str(rel_path) not in seen:
                seen.add(str(rel_path))
                inferred.append(str(rel_path))

    for deliverable in task.get("deliverables") or []:
        normalized = str(deliverable).strip().strip('\"\'')
        if not normalized or normalized in seen:
            continue
        candidate_path = (repo_root / normalized)
        if candidate_path.is_file():
            seen.add(normalized)
            inferred.append(normalized)

    if inferred:
        return inferred

    task_text = _extract_task_text(task)
    inferred.extend(_extract_path_hints(task_text, repo_root, seen))
    if inferred:
        return inferred

    keywords = extract_keywords(task_text)
    inferred.extend(_find_files_for_keywords(repo_root, keywords, seen))

    return inferred


def _update_task_state(
    state: StateStore,
    plan: Dict[str, Any],
    task: Dict[str, Any],
    updates: Dict[str, Any],
    reasoning: TaskReasoning | None = None,
) -> None:
    if reasoning:
        reasoning.apply_to_task(task)
    task.update(updates)
    if plan.get("tasks") and all(t.get("status") == "completed" for t in plan["tasks"]):
        plan["status"] = "completed"
    if reasoning:
        reasoning.merge_into_plan(plan)
    state_data = state.load()
    state_data["last_plan"] = plan
    state_data["last_updated"] = datetime.utcnow().isoformat()
    state.save(state_data)


def _emit_thinking_event(show: bool, step: ReasoningStep, event: str, note: Optional[str] = None) -> None:
    if not show:
        return
    suffix = f" – {note}" if note else ""
    click.echo(f"[thinking] {step.identifier} {event}: {step.title}{suffix}")


def _emit_adjustment(show: bool, adjustment: PlanAdjustment) -> None:
    if not show:
        return
    click.echo(f"[thinking] adjustment: {adjustment.summary} – {adjustment.detail}")


@cli.command()
@click.argument("task_id", required=False)
@click.option("--files", multiple=True, help="Target files for implementation tasks (relative paths).")
@click.option("--instructions", help="Additional guidance for the code editor.")
@click.option("--skip-tests", is_flag=True, help="Skip running automated tests after code changes.")
@click.option("--test-command", help="Custom shell command for tests (default: pytest).")
@click.option("--hide-thinking", is_flag=True, help="Hide reasoning step logs during execution.")
@click.option("--dry-run", "dry_run_flag", is_flag=True, help="Preview changes without modifying files or running tests.")
@click.pass_context
def run(
    ctx: click.Context,
    task_id: Optional[str],
    files: tuple[str, ...],
    instructions: Optional[str],
    skip_tests: bool,
    test_command: Optional[str],
    hide_thinking: bool,
    dry_run_flag: bool,
) -> None:
    """Execute a specific task or the next pending task (non-persistent)."""
    state: StateStore = ctx.obj["state"]
    plan = _load_plan(state)
    task = _select_task(plan, task_id)
    dry_run = bool(ctx.obj.get("dry_run")) or dry_run_flag
    if dry_run:
        skip_tests = True
    _record_invocation(
        ctx,
        overrides={
            "task_id": task.get("id"),
            "files": list(files),
            "instructions": instructions,
            "skip_tests": skip_tests,
            "dry_run": dry_run,
        },
    )
    started_at = datetime.utcnow()
    finalized = False

    def finalize(outcome: str) -> None:
        nonlocal finalized
        if finalized:
            return
        finalized = True
        _record_task_metric(
            state,
            task,
            outcome,
            started_at,
            dry_run=dry_run,
        )

    approval_policy: ApprovalPolicy = ctx.obj["approval_policy"]
    cli_overrides = ctx.obj.get("cli_overrides", {})

    # Apply CLI overrides to policy
    if cli_overrides.get("auto_approve"):
        approval_policy.auto_approve_plan = True
        approval_policy.auto_approve_code = True
        approval_policy.auto_approve_shell = True
        approval_policy.auto_approve_adr = True

    if cli_overrides.get("no_approval"):
        approval_policy.emergency_override = True

    audit_file = Path.cwd() / ".devagent" / "audit.log" if approval_policy.audit_file else None
    approvals = ApprovalManager(approval_policy, audit_file)

    # Update session with current task
    state.update_session(
        current_task_id=task.get("id"),
        status="active"
    )

    plan["status"] = "in_progress"
    task["status"] = "in_progress"
    state_data = state.load()
    state_data["last_plan"] = plan
    state_data["last_updated"] = datetime.utcnow().isoformat()
    state.save(state_data)

    category = (task.get("category") or "implementation").lower()
    goal = plan.get("goal", "")
    description = task.get("description", "")

    show_thinking = not hide_thinking

    reasoning = TaskReasoning(
        task_id=task.get("id", "?"),
        goal=goal,
        task_title=task.get("title"),
    )
    dependencies = ", ".join(task.get("dependencies", [])) or "none"
    context_detail = (
        f"Category: {category}; Dependencies: {dependencies}; "
        f"Instructions: {instructions or 'none provided'}"
    )
    context_step = reasoning.start_step("Review task context", context_detail)
    _emit_thinking_event(show_thinking, context_step, "start", context_detail)
    context_step.complete("Context captured for execution.")
    _emit_thinking_event(show_thinking, context_step, "complete", context_step.result)

    if category == "design":
        adr_step = reasoning.start_step(
            "Generate ADR",
            f"Create ADR for goal '{goal}' with task '{task.get('title', 'ADR')}'.",
            tool=ToolUse(name="adr_manager.generate"),
        )
        _emit_thinking_event(show_thinking, adr_step, "start")
        try:
            adr_manager = ADRManager(Path.cwd(), _get_llm_client(ctx), ADR_TEMPLATE_PATH)
            adr_result = adr_manager.generate(
                goal,
                task.get("title", "ADR"),
                description,
                dry_run=dry_run,
            )
        except Exception as exc:
            adr_step.fail(str(exc))
            _emit_thinking_event(show_thinking, adr_step, "fail", adr_step.result)
            adjustment = reasoning.record_adjustment(
                "ADR generation blocked",
                f"Failed to generate ADR: {exc}",
            )
            _emit_adjustment(show_thinking, adjustment)
            _update_task_state(
                state,
                plan,
                task,
                {"status": "blocked", "note": str(exc)},
                reasoning=reasoning,
            )
            finalize("failed")
            raise click.ClickException(f"Failed to generate ADR: {exc}") from exc
        if dry_run:
            adr_step.complete(f"Dry run: ADR would be created at {adr_result.path}")
            _emit_thinking_event(show_thinking, adr_step, "complete", adr_step.result)
            plan["status"] = "planned"
            note = f"Dry run generated ADR draft at {adr_result.path}"
            if adr_result.fallback_reason:
                note += f"; Fallback reason: {adr_result.fallback_reason}"
            updates = {"status": "pending", "note": note}
            if adr_result.fallback_reason:
                updates["fallback_reason"] = adr_result.fallback_reason
            _update_task_state(
                state,
                plan,
                task,
                updates,
                reasoning=reasoning,
            )
            state.update_session(status="paused")
            click.echo(f"Dry run: ADR would be created at {adr_result.path}")
            if adr_result.fallback_reason:
                click.echo(
                    "ADR generated using fallback template due to: "
                    f"{adr_result.fallback_reason}"
                )
                click.echo(
                    f"LLM fallback activated: {adr_result.fallback_reason}",
                    err=True,
                )
            finalize("dry_run")
            return

        adr_step.complete(f"ADR created at {adr_result.path}")
        _emit_thinking_event(show_thinking, adr_step, "complete", adr_step.result)
        relative_artifact = str(adr_result.path.relative_to(Path.cwd()))
        updates = {"status": "completed", "artifact": relative_artifact}
        if adr_result.fallback_reason:
            updates["fallback_reason"] = adr_result.fallback_reason
            updates.setdefault(
                "note",
                "ADR generated using fallback template due to: "
                f"{adr_result.fallback_reason}",
            )
        _update_task_state(
            state,
            plan,
            task,
            updates,
            reasoning=reasoning,
        )
        click.echo(f"ADR created at {adr_result.path}")
        if adr_result.fallback_reason:
            click.echo(
                "ADR generated using fallback template due to: "
                f"{adr_result.fallback_reason}"
            )
            click.echo(
                f"LLM fallback activated: {adr_result.fallback_reason}",
                err=True,
            )
        finalize("completed")
        return

    if category in {"implementation", "documentation"}:
        files_step = reasoning.start_step(
            "Identify target files",
            "Resolve files from CLI --files input or interactive prompt.",
            tool=ToolUse(name="input", description="command-line arguments"),
        )
        _emit_thinking_event(show_thinking, files_step, "start")
        target_files = list(files)
        inferred_files: List[str] = []
        if not target_files:
            inferred_files = _infer_task_files(task, Path.cwd())
            if inferred_files:
                target_files = inferred_files
                click.echo(
                    "Auto-selected files from plan metadata: "
                    + ", ".join(target_files)
                )
        if not target_files:
            files_input = click.prompt(
                "Files to edit (comma separated relative paths)",
                default="",
                show_default=False,
            ).strip()
            target_files = [item.strip() for item in files_input.split(",") if item.strip()]
        if not target_files:
            files_step.fail("No files provided.")
            _emit_thinking_event(show_thinking, files_step, "fail", files_step.result)
            adjustment = reasoning.record_adjustment(
                "Files required",
                "Task returned to pending state until target files are specified.",
            )
            _emit_adjustment(show_thinking, adjustment)
            _update_task_state(
                state,
                plan,
                task,
                {"status": "pending"},
                reasoning=reasoning,
            )
            finalize("pending")
            raise click.ClickException("No files specified for implementation task.")
        selection_note = (
            f"Auto-selected files: {', '.join(target_files)}"
            if inferred_files
            else f"Selected files: {', '.join(target_files)}"
        )
        files_step.complete(selection_note)
        _emit_thinking_event(show_thinking, files_step, "complete", files_step.result)

        run_tests = not skip_tests and not dry_run and category == "implementation"
        fix_config = IterativeFixConfig(
            max_attempts=3,
            run_tests=run_tests,
            require_test_success=run_tests,
            enable_context_expansion=True,
        )
        editor = CodeEditor(Path.cwd(), _get_llm_client(ctx), approvals, fix_config=fix_config)

        diff_step = reasoning.start_step(
            "Generate diff proposal",
            "Use LLM to propose minimal code changes with iterative fixes.",
            tool=ToolUse(name="llm.complete", description="CodeEditor.apply_diff_with_fixes"),
        )
        _emit_thinking_event(show_thinking, diff_step, "start")
        diff_completed = False

        def _render_diff_preview(proposal: DiffProposal, attempt_number: int) -> None:
            nonlocal diff_completed
            touched = ", ".join(str(path) for path in proposal.files) or "no files reported"
            summary_text = f"Attempt {attempt_number}: proposed diff touching {touched}"
            if not diff_completed:
                diff_step.complete(summary_text)
                _emit_thinking_event(show_thinking, diff_step, "complete", diff_step.result)
                diff_completed = True
            else:
                adjustment = reasoning.record_adjustment(
                    "Retry diff proposal",
                    summary_text,
                )
                _emit_adjustment(show_thinking, adjustment)

            click.echo()
            click.echo(summary_text)
            preview = proposal.preview
            if preview:
                click.echo(f"Summary: {preview.summary}")
                validation = preview.validation_result
                click.echo(
                    f"Validation: +{validation.lines_added} / -{validation.lines_removed} across {len(validation.affected_files)} file(s)"
                )
                if validation.warnings:
                    click.echo("Validation warnings:")
                    for warning in validation.warnings:
                        click.echo(f"- {warning}")
            if proposal.validation_errors:
                click.echo("Validation errors:")
                for error in proposal.validation_errors:
                    click.echo(f"- {error}")
            click.echo("\nUnified diff:\n")
            click.echo(proposal.diff)
            click.echo()

        test_args: Optional[List[str]] = shlex.split(test_command) if test_command else None

        try:
            success, attempts = editor.apply_diff_with_fixes(
                description,
                target_files,
                extra_instructions=instructions or "",
                test_command=test_args,
                dry_run=dry_run,
                on_proposal=_render_diff_preview,
            )
        except (LLMError, DiffError) as exc:
            diff_step.fail(str(exc))
            _emit_thinking_event(show_thinking, diff_step, "fail", diff_step.result)
            adjustment = reasoning.record_adjustment(
                "Diff generation failed",
                f"Diff proposal blocked: {exc}",
            )
            _emit_adjustment(show_thinking, adjustment)
            _update_task_state(
                state,
                plan,
                task,
                {"status": "blocked", "note": str(exc)},
                reasoning=reasoning,
            )
            finalize("failed")
            raise click.ClickException(f"Failed to generate diff: {exc}") from exc

        if not diff_completed:
            failure_note = (attempts[-1].error_message if attempts and attempts[-1].error_message else "Diff proposal unavailable.")
            diff_step.fail(failure_note)
            _emit_thinking_event(show_thinking, diff_step, "fail", diff_step.result)

        for attempt_result in attempts:
            if attempt_result.test_result:
                click.echo(qa_summarize(attempt_result.test_result))

        last_attempt = attempts[-1] if attempts else None

        if last_attempt and last_attempt.approved is not None:
            apply_step = reasoning.start_step(
                "Apply diff",
                "Apply validated diff to working tree.",
                tool=ToolUse(name="diff_processor.apply"),
            )
            _emit_thinking_event(show_thinking, apply_step, "start")
            if last_attempt.approved:
                outcome_msg = "Diff validated (dry run)." if dry_run else "Diff applied successfully."
                apply_step.complete(outcome_msg)
                _emit_thinking_event(show_thinking, apply_step, "complete", apply_step.result)
            else:
                apply_step.fail("Diff application declined by user.")
                _emit_thinking_event(show_thinking, apply_step, "fail", apply_step.result)

        if success:
            if dry_run:
                plan["status"] = "planned"
                _update_task_state(
                    state,
                    plan,
                    task,
                    {
                        "status": "pending",
                        "note": "Dry run validated diff; no files were modified.",
                    },
                    reasoning=reasoning,
                )
                state.update_session(status="paused")
                click.echo("Dry run: diff validated successfully. No files were modified.")
                finalize("dry_run")
                return

            if run_tests:
                test_result = last_attempt.test_result if last_attempt else None
                command_display = " ".join(test_args or ["pytest"])
                tool_name = (test_args or ["pytest"])[0]
                test_step = reasoning.start_step(
                    "Run tests",
                    "Execute automated tests to verify changes.",
                    tool=ToolUse(name=tool_name, command=command_display),
                )
                _emit_thinking_event(show_thinking, test_step, "start", command_display)
                test_step.complete("Tests passed.")
                _emit_thinking_event(show_thinking, test_step, "complete", test_step.result)
                _update_task_state(
                    state,
                    plan,
                    task,
                    {"status": "completed"},
                    reasoning=reasoning,
                )
                if test_result is None:
                    LOGGER.debug("Test runner returned no result despite run_tests flag.")
                click.echo("Tests passed. Task marked as completed.")
            else:
                skip_step = reasoning.start_step(
                    "Confirm testing approach",
                    "Skip automated tests per flag or non-code task.",
                )
                _emit_thinking_event(show_thinking, skip_step, "start")
                skip_step.complete("Tests skipped.")
                _emit_thinking_event(show_thinking, skip_step, "complete", skip_step.result)
                _update_task_state(
                    state,
                    plan,
                    task,
                    {"status": "completed"},
                    reasoning=reasoning,
                )
                click.echo("Task marked as completed without running tests.")
            finalize("completed")
            return

        failure_reason = last_attempt.error_message if last_attempt and last_attempt.error_message else "Diff application failed."

        if last_attempt and last_attempt.approved is False:
            _update_task_state(
                state,
                plan,
                task,
                {"status": "pending", "note": "Awaiting manual approval"},
                reasoning=reasoning,
            )
            click.echo("Code change was not approved. Task returned to pending state.")
            finalize("pending")
            return

        if last_attempt and last_attempt.test_result and not last_attempt.test_result.success:
            command_text = " ".join(last_attempt.test_result.command)
            test_step = reasoning.start_step(
                "Run tests",
                "Execute automated tests to verify changes.",
                tool=ToolUse(name=command_text.split()[0], command=command_text),
            )
            _emit_thinking_event(show_thinking, test_step, "start", command_text)
            test_step.fail("Tests failed.")
            _emit_thinking_event(show_thinking, test_step, "fail", test_step.result)
            adjustment = reasoning.record_adjustment(
                "Fix failing tests",
                "Review captured stdout/stderr and update implementation.",
            )
            _emit_adjustment(show_thinking, adjustment)
            _update_task_state(
                state,
                plan,
                task,
                {
                    "status": "needs_attention",
                    "last_test_result": {
                        "command": command_text,
                        "stdout": last_attempt.test_result.stdout,
                        "stderr": last_attempt.test_result.stderr,
                    },
                },
                reasoning=reasoning,
            )
            finalize("failed")
            raise click.ClickException("Tests failed. See output above.")

        _update_task_state(
            state,
            plan,
            task,
            {"status": "blocked", "note": failure_reason},
            reasoning=reasoning,
        )
        finalize("failed")
        raise click.ClickException(failure_reason)

    if category == "testing":
        if dry_run:
            plan["status"] = "planned"
            _update_task_state(
                state,
                plan,
                task,
                {
                    "status": "pending",
                    "note": "Dry run would execute testing command.",
                },
                reasoning=reasoning,
            )
            state.update_session(status="paused")
            click.echo("Dry run: tests not executed.")
            finalize("dry_run")
            return
        runner = TestRunner(Path.cwd())
        command = shlex.split(test_command) if test_command else ["pytest"]
        command_display = " ".join(command)
        test_step = reasoning.start_step(
            "Run tests",
            "Execute requested testing command.",
            tool=ToolUse(name=command[0], command=command_display),
        )
        _emit_thinking_event(show_thinking, test_step, "start", command_display)
        result = runner.run(command)
        click.echo(qa_summarize(result))
        if qa_passes(result):
            test_step.complete("Tests passed.")
            _emit_thinking_event(show_thinking, test_step, "complete", test_step.result)
            status_value = "completed"
        else:
            test_step.fail("Tests failed.")
            _emit_thinking_event(show_thinking, test_step, "fail", test_step.result)
            status_value = "needs_attention"
            adjustment = reasoning.record_adjustment(
                "Address failing tests",
                "Review test output and fix issues before re-running.",
            )
            _emit_adjustment(show_thinking, adjustment)
        _update_task_state(state, plan, task, {"status": status_value}, reasoning=reasoning)
        if status_value != "completed":
            finalize("failed")
            raise click.ClickException("Testing task failed. Please review results.")
        finalize("completed")
        return

    notice_step = reasoning.start_step(
        "Handle unsupported category",
        f"Category '{category}' not supported by run command.",
    )
    _emit_thinking_event(show_thinking, notice_step, "start")
    notice_step.complete("Task marked as skipped.")
    _emit_thinking_event(show_thinking, notice_step, "complete", notice_step.result)
    adjustment = reasoning.record_adjustment(
        "Re-plan unsupported work",
        f"Task category '{category}' requires manual handling or future support.",
    )
    _emit_adjustment(show_thinking, adjustment)
    click.echo(f"Unsupported task category '{category}'. Marking as skipped.")
    _update_task_state(state, plan, task, {"status": "skipped"}, reasoning=reasoning)
    finalize("skipped")

    finalize(task.get("status", "unknown"))


@cli.command(name="config")
@click.pass_context
def show_config(ctx: click.Context) -> None:
    """Display the current configuration."""
    settings: Settings = ctx.obj["settings"]
    redacted = {**settings.__dict__, "api_key": "***" if settings.api_key else None}
    for key, value in redacted.items():
        click.echo(f"{key}: {value}")

@cli.command()
@click.argument("name", required=False, default="Developer")
@click.pass_context
def hello(ctx: click.Context, name: str) -> None:
    """Print a greeting message.

    NAME is optional and defaults to 'Developer'.
    """
    click.echo(f"Hello, {name}!")


@cli.command()
@click.pass_context
def shell(ctx: click.Context) -> None:
    """Start an interactive shell session with persistent context."""
    click.echo("DevAgent Interactive Shell")
    click.echo("Type 'help' for available commands, 'exit' to quit.")
    click.echo("=" * 50)
    
    state: StateStore = ctx.obj["state"]
    
    # Check for resumable session
    if state.can_resume():
        session = state.get_current_session()
        click.echo(f"\nResuming session: {session.goal}")
        resumable_tasks = state.get_resumable_tasks()
        if resumable_tasks:
            click.echo(f"Found {len(resumable_tasks)} resumable tasks.")
            if click.confirm("Resume from where you left off?"):
                next_task = resumable_tasks[0]
                click.echo(f"Next task: {next_task.get('title', 'Unknown')}")
    
    click.echo()
    
    while True:
        try:
            user_input = click.prompt("DevAgent> ", prompt_suffix="", show_default=False)
            user_input = user_input.strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit", "q"]:
                # End current session if active
                if state.get_current_session():
                    state.end_session("interrupted")
                    click.echo("Session ended. Goodbye!")
                else:
                    click.echo("Goodbye!")
                break
                
            if user_input.lower() == "help":
                _show_shell_help()
                continue
                
            if user_input.lower() == "status":
                ctx.invoke(status)
                continue
                
            if user_input.lower() == "config":
                ctx.invoke(show_config)
                continue
                
            if user_input.lower().startswith("ask "):
                question = user_input[4:].strip()
                if question:
                    _invoke_question_command(ctx, question)
                else:
                    click.echo("Please provide a question after 'ask'.")
                continue

            # Handle plan commands
            if user_input.startswith("plan "):
                goal = user_input[5:].strip()
                if goal:
                    try:
                        ctx.invoke(plan, goal=tuple(goal.split()), auto_approve=False)
                    except click.ClickException as exc:
                        click.echo(f"Error: {exc}")
                else:
                    click.echo("Please specify a goal for planning.")
                continue
                
            # Handle run commands
            if user_input.startswith("run"):
                parts = user_input.split()
                task_id = parts[1] if len(parts) > 1 else None
                try:
                    ctx.invoke(
                        run,
                        task_id=task_id,
                        files=(),
                        instructions=None,
                        skip_tests=False,
                        test_command=None,
                        hide_thinking=False,
                    )
                except click.ClickException as exc:
                    click.echo(f"Error: {exc}")
                continue
                
            # Handle natural language commands
            if user_input.startswith(("please ", "can you ", "could you ")):
                _handle_natural_command(ctx, user_input)
                continue
                
            if user_input.endswith("?"):
                if click.confirm("Answer this as a repository question?", default=True):
                    _invoke_question_command(ctx, user_input)
                else:
                    click.echo("Command not recognized. Type 'help' for available commands.")
                continue

            # Default: treat as potential goal for planning
            if click.confirm(f"Create a plan for: '{user_input}'?"):
                try:
                    ctx.invoke(plan, goal=tuple(user_input.split()), auto_approve=False)
                except click.ClickException as exc:
                    click.echo(f"Error: {exc}")
            else:
                click.echo("Command not recognized. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            click.echo("\nUse 'exit' to quit the shell.")
        except EOFError:
            click.echo("\nGoodbye!")
            break
        except Exception as exc:
            click.echo(f"Unexpected error: {exc}")


def _invoke_question_command(
    ctx: click.Context,
    question_text: str,
    *,
    show_context: bool = False,
) -> None:
    """Utility for answering questions inside the interactive shell."""
    if not question_text.strip():
        click.echo("Please provide a question to answer.")
        return

    try:
        ctx.invoke(
            ask,
            question=tuple(question_text.split()),
            files=(),
            max_files=8,
            include_docs=True,
            show_context=show_context,
        )
    except click.ClickException as exc:
        click.echo(f"Error: {exc}")


def _show_shell_help() -> None:
    """Show help for the interactive shell."""
    help_text = """
The interactive shell keeps plan state in memory for the duration of this session.
Standalone commands do not persist context.

Available Commands:
  plan <goal>           - Create a new plan for the specified goal
  run [task_id]         - Execute next task or specific task ID
  ask <question>        - Ask about the repository using gathered context
  status               - Show current plan and task status
  config               - Display current configuration
  help                 - Show this help message
  exit/quit/q          - Exit the shell

Natural Language:
  You can also try natural language commands starting with:
  - "please create a plan for..."
  - "please explain how..."
  - "can you run the next task"
  - "could you show me the status"

Examples:
  DevAgent> plan Add user authentication
  DevAgent> run T1
  DevAgent> ask Where is the plan saved?
  DevAgent> please create a plan for improving test coverage
  DevAgent> status
"""
    click.echo(help_text)


def _handle_natural_command(ctx: click.Context, command: str) -> None:
    """Handle natural language commands in the shell."""
    command_lower = command.lower()
    stripped_lower = command_lower.strip()
    stripped_original = command.strip()

    for prefix in ("please ", "can you ", "could you "):
        if stripped_lower.startswith(prefix):
            stripped_lower = stripped_lower[len(prefix):]
            stripped_original = stripped_original[len(prefix):].lstrip()
            break

    question_patterns = (
        "explain",
        "describe",
        "tell me",
        "what is",
        "what does",
        "how does",
        "how do",
        "where is",
        "why does",
    )

    if any(stripped_lower.startswith(pattern) for pattern in question_patterns) or stripped_original.endswith("?"):
        _invoke_question_command(ctx, stripped_original.rstrip("?"))
        return
    
    if "plan" in command_lower:
        # Extract goal from natural language
        for phrase in ["plan for", "plan to", "create a plan for", "create a plan to"]:
            if phrase in command_lower:
                goal = command[command_lower.find(phrase) + len(phrase):].strip()
                if goal:
                    try:
                        ctx.invoke(plan, goal=tuple(goal.split()), auto_approve=False)
                        return
                    except click.ClickException as exc:
                        click.echo(f"Error: {exc}")
                        return
        click.echo("Could not extract goal from command. Try: 'plan <your goal>'")
        
    elif "run" in command_lower and ("next" in command_lower or "task" in command_lower):
        try:
            ctx.invoke(
                run,
                task_id=None,
                files=(),
                instructions=None,
                skip_tests=False,
                test_command=None,
                hide_thinking=False,
            )
        except click.ClickException as exc:
            click.echo(f"Error: {exc}")
            
    elif "status" in command_lower or "progress" in command_lower:
        ctx.invoke(status)
        
    elif "config" in command_lower:
        ctx.invoke(show_config)
        
    else:
        click.echo("I understand you want to do something, but I'm not sure what.")
        click.echo("Try being more specific or use one of the direct commands.")


def main() -> None:
    cli(prog_name="devagent")


if __name__ == "__main__":  # pragma: no cover
    main()

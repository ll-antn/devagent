"""Execution helpers for the CLI ReAct workflow."""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set
from uuid import uuid4

import click

from ai_dev_agent.cli.handlers import INTENT_HANDLERS
from ai_dev_agent.cli.utils import (
    _collect_project_structure_outline,
    _detect_repository_language,
    _get_structure_hints_state,
    _merge_structure_hints_state,
    _update_files_discovered,
)
from ai_dev_agent.core.utils.artifacts import write_artifact
from ai_dev_agent.core.utils.config import DEFAULT_MAX_ITERATIONS, Settings
from ai_dev_agent.core.utils.context_budget import DEFAULT_MAX_TOOL_OUTPUT_CHARS, summarize_text
from ai_dev_agent.core.utils.constants import MIN_TOOL_OUTPUT_CHARS
from ai_dev_agent.core.utils.devagent_config import load_devagent_yaml
from ai_dev_agent.core.utils.tool_utils import (
    FILE_READ_TOOLS,
    SEARCH_TOOLS,
    canonical_tool_name,
    tool_signature,
)
from ai_dev_agent.engine.planning.planner import Planner, PlanningContext
from ai_dev_agent.engine.react.tool_strategy import (
    ToolContext as StrategyToolContext,
    ToolSelectionStrategy,
)
from ai_dev_agent.providers.llm import (
    LLMConnectionError,
    LLMError,
    LLMRetryExhaustedError,
    LLMTimeoutError,
)
from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.session import SessionManager
from ai_dev_agent.session.context_synthesis import ContextSynthesizer

from ..analysis.formatting import _format_enhanced_tool_log
from ..analysis.research import _handle_question_without_llm
from ..router import IntentDecision, IntentRouter as _DEFAULT_INTENT_ROUTER
from .budget_control import (
    BudgetManager,
    AdaptiveBudgetManager,
    ReflectionContext,
    PHASE_PROMPTS,
    create_text_only_tool,
    auto_generate_summary,
    combine_partial_responses,
    extract_text_content,
    get_tools_for_iteration,
)

# Import ComponentIntegration for centralized feature management
try:
    from ai_dev_agent.core.integration import ComponentIntegration
except ImportError:
    ComponentIntegration = None
from ai_dev_agent.core.utils.budget_integration import (
    create_budget_integration,
    BudgetIntegration,
)
from ai_dev_agent.core.utils.cost_tracker import TokenUsage


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if not normalized:
        return False
    return normalized not in {"0", "false", "off", "no"}


_DEBUG_CONTEXT_ENABLED = _is_truthy(os.getenv("DEVAGENT_DEBUG_REACT_CONTEXT"))


def _format_message_preview(message: Message, *, limit: int | None = None) -> str:
    content = message.content
    if content is None:
        preview = "<no content>"
    else:
        preview = str(content).strip()
        if limit is not None and limit > 0 and len(preview) > limit:
            preview = preview[: limit - 1] + "…"

    suffix_parts: List[str] = []
    if message.tool_call_id:
        suffix_parts.append(f"call_id={message.tool_call_id}")
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        suffix_parts.append(f"tool_calls={len(tool_calls)}")
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    return f"[{message.role}] {preview}{suffix}"


def _emit_context_snapshot(
    messages: Sequence[Message],
    iteration: int,
    *,
    context_metadata: Mapping[str, Any] | None = None,
) -> None:
    if not _DEBUG_CONTEXT_ENABLED:
        return

    iteration_label = "initial context" if iteration == 0 else f"iteration {iteration}"
    click.echo(f"\n🔎 [debug] ReAct context snapshot before {iteration_label}:")

    if isinstance(context_metadata, Mapping) and context_metadata:
        token_estimate = context_metadata.get("token_estimate")
        if isinstance(token_estimate, (int, float)):
            click.echo(f"   [context] token estimate ≈ {int(token_estimate)}")

        events = context_metadata.get("events")
        last_event: Mapping[str, Any] | None = None
        if isinstance(events, Sequence):
            for entry in reversed(events):
                if isinstance(entry, Mapping):
                    last_event = entry
                    break

        if last_event:
            before = last_event.get("token_estimate_before")
            after = last_event.get("token_estimate_after")
            summarized = last_event.get("summarized_messages")
            summary_chars = last_event.get("summary_chars")

            delta_tokens: str | None = None
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                delta_tokens = f"Δtokens≈{int(before - after):+}"

            parts: List[str] = []
            if delta_tokens:
                parts.append(delta_tokens)
            if isinstance(summarized, int):
                parts.append(f"summarized={summarized}")
            if isinstance(summary_chars, int):
                parts.append(f"summary_chars={summary_chars}")

            if parts:
                click.echo(f"   [context] last prune: {'; '.join(parts)}")

        last_summary = context_metadata.get("last_summary")
        if isinstance(last_summary, str) and last_summary.strip():
            preview = last_summary.strip().replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "…"
            click.echo(f"   [context] last summary preview: {preview}")

    for index, msg in enumerate(messages, start=1):
        click.echo(f"   {index:02d}. {_format_message_preview(msg)}")



def _ensure_tool_call_identifiers(
    raw_tool_calls: Optional[Sequence[Any]],
    parsed_calls: Optional[Sequence[Any]],
) -> Optional[List[Dict[str, Any]]]:
    """Guarantee each tool call has a usable identifier for follow-up messages."""

    if not raw_tool_calls:
        return raw_tool_calls

    used_ids: Set[str] = set()
    for entry in raw_tool_calls:
        if isinstance(entry, Mapping):
            existing = entry.get("id") or entry.get("tool_call_id")
            if isinstance(existing, str) and existing:
                used_ids.add(existing)

    normalized: List[Dict[str, Any]] = []
    for index, entry in enumerate(raw_tool_calls):
        if isinstance(entry, Mapping):
            entry_dict: Dict[str, Any] = dict(entry)
        else:
            entry_dict = getattr(entry, "__dict__", {}).copy()
            if not entry_dict:
                entry_dict = {"raw": entry}

        call_id = entry_dict.get("id") or entry_dict.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            while True:
                candidate = f"auto-tool-{index}-{uuid4().hex[:8]}"
                if candidate not in used_ids:
                    call_id = candidate
                    used_ids.add(candidate)
                    break
            entry_dict["id"] = call_id
        else:
            used_ids.add(call_id)

        normalized.append(entry_dict)

        if parsed_calls and index < len(parsed_calls):
            try:
                parsed_calls[index].call_id = call_id
            except AttributeError:
                setattr(parsed_calls[index], "call_id", call_id)

    return normalized


def _summarize_arguments(arguments: Mapping[str, Any]) -> str:
    if not arguments:
        return "{}"
    try:
        raw = json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        raw = str(dict(arguments))
    raw = raw.replace("\n", " ").strip()
    if len(raw) > 120:
        return raw[:117] + "…"
    return raw


def _extract_first_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    match = re.search(r"-?\d+", value)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _resolve_intent_router():
    try:
        cli_module = import_module("ai_dev_agent.cli")
    except ModuleNotFoundError:
        return _DEFAULT_INTENT_ROUTER
    return getattr(cli_module, "IntentRouter", _DEFAULT_INTENT_ROUTER)


def _truncate_shell_history(history: List[Message], max_turns: int) -> List[Message]:
    """Keep only the most recent conversation turns within the configured budget."""

    if max_turns <= 0:
        return []

    turns: List[tuple[Message, Message]] = []
    pending_user: Message | None = None
    for msg in history:
        if msg.role == "user":
            pending_user = msg
        elif msg.role == "assistant" and msg.content is not None:
            if pending_user is not None:
                turns.append((pending_user, msg))
                pending_user = None

    trimmed: List[Message] = []
    for user_msg, assistant_msg in turns[-max_turns:]:
        trimmed.extend([user_msg, assistant_msg])

    if pending_user is not None:
        if not trimmed or trimmed[-1] is not pending_user:
            trimmed.append(pending_user)

    return trimmed

def _execute_react_assistant(
    ctx: click.Context,
    client,
    settings: Settings,
    user_prompt: str,
    use_planning: bool = False,
) -> None:
    """Execute multi-step ReAct reasoning for natural language queries."""

    start_time = time.time()
    execution_completed = False
    planning_active = bool(use_planning)
    emit_status_requested = bool(ctx.meta.pop("_emit_status_messages", False))
    supports_tool_calls = hasattr(client, "invoke_tools")
    should_emit_status = planning_active or supports_tool_calls or emit_status_requested
    execution_mode = "with planning" if planning_active else "direct"

    user_message = Message(role="user", content=user_prompt)

    if not isinstance(getattr(ctx, "obj", None), dict):
        ctx.obj = {}

    ctx_obj: Dict[str, Any] = ctx.obj
    ctx_obj["_low_count_verified"] = False

    history_messages: List[Message] = []
    history_enabled = False
    history_raw = ctx_obj.get("_shell_conversation_history")
    if isinstance(history_raw, list):
        history_messages = [msg for msg in history_raw if isinstance(msg, Message)]
        history_enabled = True
    max_history_turns = max(1, getattr(settings, "keep_last_assistant_messages", 4))
    existing_history = list(history_messages)

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

    # Initialize budget_integration early to avoid closure issues
    budget_integration = None
    budget_manager = None
    session_manager = None
    session_id = None

    def _finalize() -> None:
        # Cost summary display removed - internal tracking still active
        # Save to session metadata if session exists
        if execution_completed and budget_integration and budget_integration.cost_tracker:
            if session_manager and session_id:
                session = session_manager.get_session(session_id)
                session.metadata["cost_summary"] = {
                    "total_cost": budget_integration.cost_tracker.total_cost_usd,
                    "total_tokens": (
                        budget_integration.cost_tracker.total_prompt_tokens +
                        budget_integration.cost_tracker.total_completion_tokens
                    ),
                    "phase_costs": budget_integration.cost_tracker.phase_costs,
                    "model_costs": budget_integration.cost_tracker.model_costs,
                }

            # Cost artifact generation disabled - no output needed

        if execution_completed and should_emit_status:
            execution_time = time.time() - start_time
            mode_label = execution_mode
            click.echo(f"\n✅ Completed in {execution_time:.1f}s ({mode_label})")

    repo_root = Path.cwd()
    strategy = ToolSelectionStrategy()
    structure_state = _get_structure_hints_state(ctx)

    repository_language = ctx_obj.get("_detected_language")
    repository_size_estimate = ctx_obj.get("_repo_file_count")
    if repository_language is None or repository_size_estimate is None:
        detected_language, file_count = _detect_repository_language(strategy, repo_root)
        if repository_language is None:
            repository_language = detected_language
            ctx_obj["_detected_language"] = detected_language
        if repository_size_estimate is None and file_count is not None:
            repository_size_estimate = file_count
            ctx_obj["_repo_file_count"] = file_count

    tool_success_tracker = ctx_obj.setdefault("_tool_success_history", {})

    if isinstance(tool_success_tracker, dict) and tool_success_tracker:
        strategy.seed_aggregated_history(tool_success_tracker)

    project_profile: Dict[str, Any] = {
        "workspace_root": str(settings.workspace_root or repo_root),
        "language": repository_language,
        "repository_size": repository_size_estimate,
        "project_summary": ctx_obj.get("_project_structure_summary"),
        "active_plan_complexity": ctx_obj.get("_active_plan_complexity"),
    }

    discovered_files = structure_state.get("files") if isinstance(structure_state, dict) else {}
    if isinstance(discovered_files, dict) and discovered_files:
        project_profile["recent_files"] = sorted(discovered_files.keys())[:6]

    style_notes = ctx_obj.get("_latest_style_profile")
    if style_notes:
        project_profile["style_notes"] = style_notes

    project_profile = {key: value for key, value in project_profile.items() if value}

    router_cls = _resolve_intent_router()
    router = router_cls(
        client,
        settings,
        project_profile=project_profile,
        tool_success_history=ctx_obj.get("_tool_success_history"),
    )
    ctx_obj.setdefault("_router_state", {})["session_id"] = getattr(router, "session_id", None)
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
            if history_enabled:
                updated = existing_history + [user_message]
                if text:
                    updated.append(Message(role="assistant", content=text))
                ctx.obj["_shell_conversation_history"] = _truncate_shell_history(updated, max_history_turns)
            _finalize()
            return

        handler = INTENT_HANDLERS.get(decision.tool)
        if not handler:
            raise click.ClickException(f"Intent tool '{decision.tool}' is not supported yet.")
        handler(ctx, decision.arguments)
        execution_completed = True
        if history_enabled:
            ctx.obj["_shell_conversation_history"] = _truncate_shell_history(
                existing_history + [user_message],
                max_history_turns,
            )
        _finalize()
        return


    devagent_cfg = ctx.obj.get("devagent_config")
    if devagent_cfg is None:
        devagent_cfg = load_devagent_yaml()
        ctx.obj["devagent_config"] = devagent_cfg

    config_global_cap = getattr(devagent_cfg, "react_iteration_global_cap", None) if devagent_cfg else None

    settings_cap_raw = getattr(settings, "max_iterations", None)
    global_max_iterations = (
        settings_cap_raw
        if isinstance(settings_cap_raw, int) and settings_cap_raw > 0
        else DEFAULT_MAX_ITERATIONS
    )
    if (
        (not isinstance(settings_cap_raw, int) or settings_cap_raw <= 0)
        and isinstance(config_global_cap, int)
        and config_global_cap > 0
    ):
        global_max_iterations = config_global_cap

    iteration_cap = global_max_iterations

    budget_control_settings: Dict[str, Any] = {}
    if devagent_cfg and getattr(devagent_cfg, "budget_control", None):
        if isinstance(devagent_cfg.budget_control, dict):
            budget_control_settings = dict(devagent_cfg.budget_control)

    configured_cap = budget_control_settings.get("max_iterations")
    if isinstance(configured_cap, int) and configured_cap > 0:
        iteration_cap = min(iteration_cap, configured_cap)

    phase_thresholds = budget_control_settings.get("phases")
    warning_settings = budget_control_settings.get("warnings")
    tool_settings = budget_control_settings.get("tools") or {}
    synthesis_settings = budget_control_settings.get("synthesis") or {}
    auto_summary_enabled = bool(synthesis_settings.get("auto_summary_on_failure", True))

    # Initialize adaptive budget manager with model context
    model_context_window = getattr(settings, 'model_context_window', 100000)

    # Initialize ComponentIntegration if available for centralized feature management
    component_integration = None
    if ComponentIntegration:
        try:
            component_integration = ComponentIntegration(
                project_root=Path.cwd(),
                settings=settings,
                enable_all=True  # Enable all features by default
            )
            # Initialize components
            component_integration.initialize_repo_map()
            component_integration.initialize_tool_tracker()
            component_integration.initialize_agent_manager()
        except Exception as e:
            LOGGER.debug(f"ComponentIntegration initialization failed: {e}")
            component_integration = None

    # Always use AdaptiveBudgetManager for best performance (features enabled by default)
    # Only fall back to basic BudgetManager if explicitly disabled
    use_adaptive = getattr(settings, 'enable_reflection', True) or getattr(settings, 'adaptive_budget_scaling', True)

    if use_adaptive:
        # Try to use the enhanced adaptive budget from integration if available
        if component_integration:
            budget_manager = component_integration.initialize_budget_manager(
                max_iterations=iteration_cap,
                model_context_window=model_context_window
            )

        # Fall back to the existing AdaptiveBudgetManager if integration fails
        if not budget_manager:
            budget_manager = AdaptiveBudgetManager(
                iteration_cap,
                phase_thresholds=phase_thresholds if isinstance(phase_thresholds, dict) else None,
                warnings=warning_settings if isinstance(warning_settings, dict) else None,
                model_context_window=model_context_window,
                adaptive_scaling=getattr(settings, 'adaptive_budget_scaling', True),
                enable_reflection=getattr(settings, 'enable_reflection', True),
                max_reflections=getattr(settings, 'max_reflections', 3),
            )
    else:
        # Only use basic BudgetManager if explicitly requested
        budget_manager = BudgetManager(
            iteration_cap,
            phase_thresholds=phase_thresholds if isinstance(phase_thresholds, dict) else None,
            warnings=warning_settings if isinstance(warning_settings, dict) else None,
        )

    # Initialize budget integration for cost tracking and retry (reassign from outer scope)
    # Check if we're in test mode (disable by default for tests)
    is_test_mode = os.environ.get('PYTEST_CURRENT_TEST') is not None

    if not is_test_mode and (settings.enable_cost_tracking or settings.enable_retry or settings.enable_summarization):
        budget_integration = create_budget_integration(settings)
        # Initialize summarizer with the LLM client
        if settings.enable_summarization and hasattr(client, 'complete'):
            budget_integration.initialize_summarizer(client)

    session_manager = SessionManager.get_instance()
    temp_session_id = ctx_obj.get("_session_id")
    if not temp_session_id:
        temp_session_id = f"cli-{uuid4()}"
        ctx_obj["_session_id"] = temp_session_id
    session_id = temp_session_id  # Assign to outer scope variable

    # Start with minimal system messages - will be replaced by unified prompt
    # This avoids tool information being added to confuse the model
    system_messages = [Message(role="system", content="Initializing DEVAGENT assistant...")]

    session = session_manager.ensure_session(
        session_id,
        system_messages=system_messages,
        metadata={
            "iteration_cap": iteration_cap,
            "repository_language": repository_language,
        },
    )

    if history_enabled and existing_history and not session.metadata.get("existing_history_loaded"):
        session_manager.extend_history(session_id, existing_history)
        session.metadata["existing_history_loaded"] = True

    session_manager.extend_history(session_id, [user_message])
    with session.lock:
        session.metadata["history_anchor"] = len(session.history)

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
            project_structure = ctx.obj.get("_project_structure_summary")
            if "_project_structure_summary" not in ctx.obj:
                project_structure = _collect_project_structure_outline(Path.cwd())
                ctx.obj["_project_structure_summary"] = project_structure
            if project_structure:
                structure_state["project_summary"] = project_structure
            tool_history = ctx_obj.get("_tool_success_history") or {}
            historical_success = None
            recent_failures = None
            if isinstance(tool_history, dict) and tool_history:
                metrics: List[tuple[float, float, str]] = []
                for name, stats in tool_history.items():
                    if not isinstance(stats, dict):
                        continue
                    success = float(stats.get("success", 0))
                    failure = float(stats.get("failure", 0))
                    total = success + failure
                    if total <= 0:
                        continue
                    rate = success / total
                    metrics.append((rate, total, name))
                if metrics:
                    metrics.sort(reverse=True)
                    top_samples = [
                        f"{name} ({rate:.0%} of {int(total)} runs)"
                        for rate, total, name in metrics[:3]
                    ]
                    historical_success = ", ".join(top_samples)
                    low_samples = [
                        f"{name} ({rate:.0%} of {int(total)} runs)"
                        for rate, total, name in metrics
                        if rate < 0.45
                    ]
                    if low_samples:
                        recent_failures = ", ".join(low_samples[:3])

            repo_metrics_parts = []
            if repository_size_estimate:
                repo_metrics_parts.append(f"Approximate file count: {repository_size_estimate}")
            repo_metrics_parts.append(f"Workspace root: {repo_root}")
            repository_metrics_text = "; ".join(repo_metrics_parts)

            dependency_overview = None
            if isinstance(discovered_files, dict) and discovered_files:
                dependency_overview = ", ".join(sorted(discovered_files.keys())[:6])

            planning_context = PlanningContext(
                project_structure=project_structure,
                repository_metrics=repository_metrics_text or None,
                dominant_language=repository_language,
                dependency_landscape=dependency_overview,
                code_conventions=style_notes,
                historical_success=historical_success,
                recent_failures=recent_failures,
                related_components=dependency_overview,
            )

            planner = Planner(client)
            planner_session_id = ctx_obj.setdefault("_planner_session_id", f"{session_id}-planner")
            structured_plan = planner.generate(
                user_prompt,
                project_structure=project_structure,
                context=planning_context,
                session_id=planner_session_id,
            )
            if structured_plan.complexity:
                ctx_obj["_active_plan_complexity"] = structured_plan.complexity
            if structured_plan.success_criteria:
                ctx_obj["_latest_plan_success_criteria"] = structured_plan.success_criteria
        except LLMError as exc:
            click.echo(f"Planner unavailable: {exc}. Switching to direct execution.")
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

    consecutive_fails = 0
    used_tools: Set[str] = set()
    file_reads: Set[str] = set()
    search_queries: Set[str] = set()
    tool_repeat_counts: Dict[str, int] = {}
    failure_tracker: Dict[str, Dict[str, Any]] = {}

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

    project_structure = ctx.obj.get("_project_structure_summary")
    if not project_structure:
        project_structure = _collect_project_structure_outline(repo_root)
        if project_structure:
            ctx.obj["_project_structure_summary"] = project_structure
    if project_structure:
        structure_state["project_summary"] = project_structure

    def _commit_shell_history() -> None:
        if not history_enabled or not isinstance(ctx.obj, dict):
            return
        session_local = session_manager.get_session(session_id)
        anchor = session_local.metadata.get("history_anchor", len(session_local.history))
        new_entries: List[Message] = []
        if user_message.content:
            new_entries.append(user_message)
        with session_local.lock:
            for msg in session_local.history[anchor:]:
                if msg.role == "assistant" and msg.content and not msg.tool_calls:
                    new_entries.append(msg)
            session_local.metadata["history_anchor"] = len(session_local.history)
        ctx.obj["_shell_conversation_history"] = _truncate_shell_history(
            existing_history + new_entries,
            max_history_turns,
        )

    def _debug_context_snapshot(iteration_index: int) -> None:
        if not _DEBUG_CONTEXT_ENABLED:
            return
        session_local = session_manager.get_session(session_id)
        with session_local.lock:
            snapshot_messages = session_local.compose()
            context_metadata = session_local.metadata.get("context_service")
        _emit_context_snapshot(
            snapshot_messages,
            iteration_index,
            context_metadata=context_metadata if isinstance(context_metadata, Mapping) else None,
        )

    def _extract_submit_answer(tool_calls: Sequence[Any] | None) -> Optional[str]:
        if not tool_calls:
            return None
        for call in tool_calls:
            name = getattr(call, "name", None)
            if not isinstance(name, str):
                name = call.get("name") if isinstance(call, Mapping) else None
            if not isinstance(name, str):
                name = str(name or "")
            if not name:
                continue
            if canonical_tool_name(name) != "submit_final_answer":
                continue
            arguments = getattr(call, "arguments", None)
            if not isinstance(arguments, Mapping):
                arguments = call.get("arguments") if isinstance(call, Mapping) else None
            if not isinstance(arguments, Mapping):
                continue
            answer = arguments.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
        return None

    def _build_phase_prompt(
        phase: str,
        user_query: str,
        context: str,
        constraints: str,
        workspace: str = None,
        repository_language: str = None,
    ) -> str:
        """Build phase-based system prompt without step tracking."""

        phase_guidance = PHASE_PROMPTS.get(phase)
        if not phase_guidance:
            phase_guidance = PHASE_PROMPTS.get("exploration", "Focus on the task at hand.")

        # Language hints
        lang_hint = ""
        if repository_language:
            lang_hints_map = {
                "python": "Consider Python-specific patterns, check requirements.txt",
                "javascript": "Check package.json, consider JS/TS patterns",
                "java": "Check build files, consider Java patterns",
                "c++": "Check build configuration, consider C++ patterns",
                "go": "Check go.mod, consider Go patterns",
            }
            lang_hint = lang_hints_map.get(repository_language.lower(), "")

        # Build prompt
        prompt = f"""You are a development assistant analyzing a codebase.

TASK: {user_query}

APPROACH:
{phase_guidance}

WORKSPACE: {workspace or 'current directory'}
{f'LANGUAGE: {repository_language} - {lang_hint}' if lang_hint else ''}

{'PREVIOUS DISCOVERIES:' if context else ''}
{context if context else 'Beginning investigation...'}

{'CONSTRAINTS:' if constraints else ''}
{constraints if constraints else ''}"""

        return prompt

    def _build_synthesis_prompt(
        user_query: str,
        context: str,
        workspace: str = None,
    ) -> str:
        """Build final synthesis prompt for last iteration (no tools)."""

        synthesis_guidance = PHASE_PROMPTS.get("synthesis", "Provide your final response.")
        prompt = f"""📋 FINAL SYNTHESIS ONLY

Task: {user_query}

{synthesis_guidance}

Workspace: {workspace or 'current directory'}

Investigation Summary:
{context if context else 'No prior findings recorded.'}

Instructions:
- Respond with a complete, self-contained answer.
- Cite specific files, paths, and relevant context.
- Call out open questions or risks still unresolved.
- DO NOT request more iterations or attempt tool usage.

Begin your final answer now."""

        return prompt

    def _update_system_prompt(phase: str, is_final: bool = False) -> None:
        """Update system prompt based on phase, without step tracking."""

        # Clear all existing system messages
        session_manager.remove_system_messages(
            session_id,
            lambda msg: True  # Remove all system messages
        )

        # Get original user query
        session = session_manager.get_session(session_id)
        user_query = "Complete the user's task"
        for msg in session.history:
            if msg.role == "user" and msg.content:
                user_query = str(msg.content)[:500]
                break

        # Get context synthesis
        synthesizer = ContextSynthesizer()
        context = ""
        constraints = ""

        if len(session.history) > 0:
            context = synthesizer.synthesize_previous_steps(
                session.history,
                current_step=len([m for m in session.history if m.role == "assistant"])
            )
            redundant_ops = synthesizer.get_redundant_operations(session.history)
            constraints = synthesizer.build_constraints_section(redundant_ops)

        # Build appropriate prompt
        if is_final:
            prompt = _build_synthesis_prompt(
                user_query=user_query,
                context=context,
                workspace=str(repo_root),
            )
        else:
            prompt = _build_phase_prompt(
                phase=phase,
                user_query=user_query,
                context=context,
                constraints=constraints,
                workspace=str(repo_root),
                repository_language=repository_language,
            )

        # Set as the only system message
        system_message = Message(role="system", content=prompt)
        with session.lock:
            session.system_messages = [system_message]

    # Initialize with exploration phase
    _update_system_prompt(phase="exploration", is_final=False)
    _debug_context_snapshot(0)

    iteration = 0
    tool_history: List[str] = []
    has_symbol_index = False
    files_discovered_set: Set[str] = set()
    last_tool_success = True
    synthesized = False

    while True:
        context = budget_manager.next_iteration()
        if context is None:
            break

        iteration = context.number

        # Budget progress display removed - only tool logs will be shown

        if context.is_final:
            _update_system_prompt(phase="synthesis", is_final=True)
            is_final_iteration = True
        else:
            _update_system_prompt(phase=context.phase, is_final=False)
            is_final_iteration = False

        if iteration > 5 and isinstance(tool_success_tracker, dict) and tool_success_tracker:
            perf_summary: List[str] = []
            for tool_name, stats in sorted(
                tool_success_tracker.items(),
                key=lambda item: item[1].get("success", 0),
                reverse=True,
            )[:5]:
                count = stats.get("count", 0)
                if count >= 2:
                    success_rate = stats.get("success", 0) / count if count else 0.0
                    perf_summary.append(f"  {tool_name}: {success_rate:.0%} success rate")
            if perf_summary:
                session_manager.remove_system_messages(
                    session_id,
                    lambda msg: isinstance(msg.content, str)
                    and msg.content.startswith("Tool performance this session:")
                )
                session_manager.add_system_message(
                    session_id,
                    "Tool performance this session:\n" + "\n".join(perf_summary),
                    location="system",
                )

        strategy_context = StrategyToolContext(
            language=repository_language,
            has_symbol_index=has_symbol_index,
            files_discovered=set(files_discovered_set),
            tools_used=list(tool_history),
            last_tool_success=last_tool_success,
            iteration_count=iteration - 1,
            repository_size=repository_size_estimate,
            iteration_budget=iteration_cap,
        )

        tools_for_iteration = get_tools_for_iteration(
            context,
            available_tools,
            tool_config=tool_settings if isinstance(tool_settings, dict) else None,
        )
        try:
            sanitized_to_entry: Dict[str, Dict[str, Any]] = {}
            sanitized_to_canonical: Dict[str, str] = {}

            _debug_context_snapshot(iteration)

            # Use filtered tools, not available_tools
            for entry in tools_for_iteration:
                fn = entry.get("function", {})
                sanitized = fn.get("name")
                if not sanitized:
                    continue
                canonical = canonical_tool_name(sanitized)
                sanitized_to_entry[sanitized] = entry
                sanitized_to_canonical[sanitized] = canonical

            canonical_available = [canonical for canonical in sanitized_to_canonical.values()]
            prioritized_canonical = strategy.prioritize_tools(canonical_available, strategy_context)

            prioritized_sanitized: List[str] = []
            used_sanitized: Set[str] = set()
            for canonical in prioritized_canonical:
                for sanitized, mapped in sanitized_to_canonical.items():
                    if mapped == canonical and sanitized not in used_sanitized:
                        prioritized_sanitized.append(sanitized)
                        used_sanitized.add(sanitized)
                        break

            for sanitized in sanitized_to_entry:
                if sanitized not in used_sanitized:
                    prioritized_sanitized.append(sanitized)

            # Rebuild prioritized tools from filtered set
            prioritized_tools = [
                sanitized_to_entry[name]
                for name in prioritized_sanitized
                if name in sanitized_to_entry
            ]
            # Keep the filtered tools if prioritization fails
            if not prioritized_tools:
                prioritized_tools = tools_for_iteration
            tools_for_iteration = prioritized_tools
        except Exception:
            # On error, keep the filtered tools (not all available_tools)
            pass  # tools_for_iteration already set by get_tools_for_iteration

        try:
            conversation = session_manager.compose(session_id)

            # Execute with retry and cost tracking if integration is available
            if budget_integration:
                try:
                    result = budget_integration.execute_with_retry(
                        client.invoke_tools,
                        conversation,
                        tools=tools_for_iteration,
                        temperature=0.1,
                    )

                    # Track cost if response contains usage data
                    if hasattr(result, '_raw_response') and result._raw_response:
                        budget_integration.track_llm_call(
                            model=settings.model,
                            response_data=result._raw_response,
                            operation="tool_invocation",
                            iteration=iteration,
                            phase=context.phase,
                        )

                    # Cost warning disabled - tracking continues silently

                except (LLMError, LLMTimeoutError, LLMConnectionError) as e:
                    # Handle with reflection if available
                    if isinstance(budget_manager, AdaptiveBudgetManager) and \
                       context.reflection_allowed and budget_manager.allow_reflection(str(e)):
                        click.echo(f"💭 Attempting reflection (attempt {budget_manager.reflection.current_reflection}/{budget_manager.reflection.max_reflections})")
                        # Add reflection prompt to conversation
                        reflection_msg = Message(
                            role="system",
                            content=f"Previous attempt failed: {e}. Please adjust your approach and try again."
                        )
                        session_manager.extend_history(session_id, [reflection_msg])
                        continue  # Retry the iteration
                    else:
                        raise
            else:
                # Fallback to direct invocation
                if is_final_iteration:
                    result = client.invoke_tools(
                        conversation,
                        tools=tools_for_iteration,
                        temperature=0.1,
                    )
                else:
                    result = client.invoke_tools(
                        conversation,
                        tools=tools_for_iteration,
                        temperature=0.1,
                    )

            normalized_tool_calls = _ensure_tool_call_identifiers(result.raw_tool_calls, result.calls)
            if normalized_tool_calls is not None:
                result.raw_tool_calls = normalized_tool_calls

            assistant_message = Message(
                role="assistant",
                content=result.message_content,
                tool_calls=normalized_tool_calls,
            )
            if assistant_message.content is not None or assistant_message.tool_calls:
                session_manager.extend_history(session_id, [assistant_message])

            submit_final_answer = None
            if is_final_iteration:
                submit_final_answer = _extract_submit_answer(result.calls)
                if submit_final_answer:
                    for tool_call in result.calls or []:
                        raw_name = getattr(tool_call, "name", "")
                        if not isinstance(raw_name, str):
                            raw_name = str(raw_name or "")
                        if canonical_tool_name(raw_name) != "submit_final_answer":
                            continue
                        tool_call_id = getattr(tool_call, "call_id", None) or getattr(tool_call, "id", None)
                        if tool_call_id:
                            session_manager.add_tool_message(session_id, tool_call_id, submit_final_answer)
                        break
                    click.echo(submit_final_answer)
                    execution_completed = True
                    synthesized = True
                    _commit_shell_history()
                    _finalize()
                    return

            if not result.calls:
                if is_final_iteration:
                    final_text = extract_text_content(result)
                    if not final_text:
                        if auto_summary_enabled:
                            conversation_snapshot = session_manager.compose(session_id)
                            final_text = auto_generate_summary(
                                conversation_snapshot,
                                files_examined=files_discovered_set,
                                searches_performed=search_queries,
                            )
                            click.echo("⚠️ Generated synthesis from investigation history:")
                        else:
                            final_text = "Model did not provide a final synthesis."
                    click.echo(final_text)
                    execution_completed = True
                    synthesized = True
                    _commit_shell_history()
                    _finalize()
                    return

                if context.is_penultimate:
                    early_text = extract_text_content(result)
                    if not early_text:
                        if auto_summary_enabled:
                            conversation_snapshot = session_manager.compose(session_id)
                            generated = auto_generate_summary(
                                conversation_snapshot,
                                files_examined=files_discovered_set,
                                searches_performed=search_queries,
                            )
                            click.echo("⚠️ Early synthesis detected (penultimate iteration).")
                            click.echo(generated)
                            early_text = generated
                        else:
                            early_text = "Model ended without providing synthesis."
                    click.echo(early_text)
                    execution_completed = True
                    synthesized = True
                    _commit_shell_history()
                    _finalize()
                    return

                # Non-final iteration with no tool calls – treat as completion
                if assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        tool_call_id = None
                        if isinstance(tool_call, Mapping):
                            tool_call_id = tool_call.get("call_id") or tool_call.get("id")
                        else:
                            tool_call_id = getattr(tool_call, "call_id", None) or getattr(
                                tool_call, "id", None
                            )
                        if tool_call_id:
                            session_manager.add_tool_message(
                                session_id,
                                tool_call_id,
                                "Tool call was not executed.",
                            )

                fallback_text = extract_text_content(result) or "I was unable to provide a complete answer."
                click.echo(fallback_text)
                execution_completed = True
                synthesized = True
                _commit_shell_history()
                _finalize()
                return

            if is_final_iteration and result.calls:
                click.echo("\n⚠️  Iteration limit reached. Tool usage is blocked during synthesis.")

                for tool_call in result.calls:
                    tool_call_id = getattr(tool_call, "call_id", None) or getattr(tool_call, "id", None)
                    if tool_call_id:
                        session_manager.add_tool_message(
                            session_id,
                            tool_call_id,
                            "Final iteration: tool execution skipped. Provide text synthesis instead.",
                        )

                partial_text = extract_text_content(result)
                conversation_snapshot = session_manager.compose(session_id)
                auto_summary = (
                    auto_generate_summary(
                        conversation_snapshot,
                        files_examined=files_discovered_set,
                        searches_performed=search_queries,
                    )
                    if auto_summary_enabled
                    else ""
                )
                final_output = combine_partial_responses(partial_text, auto_summary)
                if not final_output:
                    final_output = (
                        "Model did not provide a final synthesis. Consider increasing DEVAGENT_MAX_ITERATIONS."
                    )

                click.echo(final_output)
                execution_completed = True
                synthesized = True
                _commit_shell_history()
                _finalize()
                return

            redundant_calls = 0
            last_tool_failed = consecutive_fails > 0

            for tool_call in result.calls:
                tool_call_name = getattr(tool_call, "name", None)
                if tool_call_name and canonical_tool_name(tool_call_name) == "submit_final_answer":
                    # Already handled as final synthesis above
                    continue

                call_signature = tool_signature(tool_call)
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

            if redundant_calls >= len(result.calls) * 0.7 and len(result.calls) > 1:
                click.echo("🚫 Detected redundant tool usage. Stopping to avoid loops.")
                execution_completed = True
                _commit_shell_history()
                _finalize()
                return

            total_calls = len(result.calls)
            successful_calls = 0
            file_reading_context: Dict[str, Any] = {"last_file_read": None, "file_line_counts": {}}

            for call_index, tool_call in enumerate(result.calls, 1):
                call_signature = tool_signature(tool_call)
                canonical_name = canonical_tool_name(tool_call.name)

                used_tools.add(call_signature)

                if tool_call.name in FILE_READ_TOOLS:
                    for target in _normalize_read_targets(tool_call.arguments):
                        file_reads.add(target)
                        files_discovered_set.add(target)
                elif tool_call.name in SEARCH_TOOLS and tool_call.arguments.get("query"):
                    search_queries.add(tool_call.arguments["query"])

                handler = INTENT_HANDLERS.get(tool_call.name)
                if not handler:
                    error_msg = f"Tool '{tool_call.name}' is not supported."
                    tool_call_id = getattr(tool_call, "call_id", None) or getattr(tool_call, "id", "unknown")
                    session_manager.add_tool_message(
                        session_id,
                        tool_call_id,
                        f"Error: {error_msg}",
                    )
                    click.echo(f"❌ {tool_call.name} → Tool not supported")
                    consecutive_fails += 1
                    continue

                repeat_count = tool_repeat_counts.get(call_signature, 0) + 1
                tool_repeat_counts[call_signature] = repeat_count

                tool_output_for_message: Optional[str] = None
                handler_result: Optional[Mapping[str, Any]] = None
                call_success = False
                execution_time = 0.0
                tool_output = ""

                captured_output = io.StringIO()
                start_call = time.time()

                def _register_failure(outcome: str) -> str:
                    failure_meta = failure_tracker.setdefault(
                        call_signature,
                        {"count": 0, "last_error": "", "summary_sent": False},
                    )
                    failure_meta["count"] = failure_meta.get("count", 0) + 1
                    failure_meta["last_error"] = outcome
                    failure_count = failure_meta["count"]
                    summary_text_value = outcome
                    if failure_count > 1:
                        summary_text_value = (
                            f"Repeated failure ({failure_count}x): "
                            f"{summarize_text(outcome, 200)}"
                        )
                    if failure_count >= 2:
                        failure_meta["blocked"] = True
                        if not failure_meta.get("summary_sent"):
                            arg_summary_inner = _summarize_arguments(tool_call.arguments)
                            error_summary = summarize_text(outcome, 160)
                            advisory = (
                                f"Repeated failure detected for {tool_call.name} with arguments {arg_summary_inner}. "
                                f"Last error: {error_summary}. Consider adjusting the approach (modify the command, switch tools, or break the task into smaller steps)."
                            )
                            session_manager.add_system_message(session_id, advisory)
                            failure_meta["summary_sent"] = True
                    else:
                        failure_meta.pop("blocked", None)
                    return summary_text_value

                blocked_entry = failure_tracker.get(call_signature)

                if blocked_entry and blocked_entry.get("blocked"):
                    failure_count = blocked_entry.get("count", 0)
                    last_error = blocked_entry.get("last_error") or "Repeated failure detected"
                    error_summary = summarize_text(str(last_error), 200)
                    arg_summary = _summarize_arguments(tool_call.arguments)
                    tool_output = (
                        f"Skipped after {failure_count} repeated failures: {tool_call.name} {arg_summary}. "
                        f"Last error: {error_summary}. Choose a different approach."
                    )
                    log_message = _format_enhanced_tool_log(
                        tool_call,
                        repeat_count,
                        execution_time,
                        tool_output,
                        success=False,
                        file_context=file_reading_context,
                    )
                    click.echo(log_message)
                    tool_output_for_message = tool_output
                    consecutive_fails += 1
                else:
                    try:
                        with contextlib.redirect_stdout(captured_output):
                            handler_result = handler(ctx, tool_call.arguments)
                        tool_output_raw = captured_output.getvalue()
                        tool_output = tool_output_raw.strip() or "Tool executed successfully (no output)"

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

                        execution_time = time.time() - start_call
                        log_message = _format_enhanced_tool_log(
                            tool_call,
                            repeat_count,
                            execution_time,
                            tool_output,
                            success=True,
                            file_context=file_reading_context,
                        )
                        click.echo(log_message)

                        successful_calls += 1
                        consecutive_fails = 0
                        call_success = True
                        failure_tracker.pop(call_signature, None)

                    except FileNotFoundError as exc:
                        execution_time = time.time() - start_call
                        tool_output = f"File not found: {exc}"
                        log_message = _format_enhanced_tool_log(
                            tool_call,
                            repeat_count,
                            execution_time,
                            tool_output,
                            success=False,
                            file_context=file_reading_context,
                        )
                        click.echo(log_message)
                        consecutive_fails += 1
                        tool_output_for_message = _register_failure(tool_output)

                    except Exception as exc:
                        execution_time = time.time() - start_call
                        tool_output = f"Error executing {tool_call.name}: {exc}"
                        log_message = _format_enhanced_tool_log(
                            tool_call,
                            repeat_count,
                            execution_time,
                            tool_output,
                            success=False,
                            file_context=file_reading_context,
                        )
                        click.echo(log_message)
                        consecutive_fails += 1
                        tool_output_for_message = _register_failure(tool_output)

                strategy.record_tool_result(canonical_name, call_success, execution_time)
                tracker_entry = tool_success_tracker.setdefault(
                    canonical_name,
                    {
                        "success": 0,
                        "failure": 0,
                        "total_duration": 0.0,
                        "count": 0,
                        "avg_duration": 0.0,
                    },
                )
                tracker_entry["count"] = tracker_entry.get("count", 0) + 1
                if call_success:
                    tracker_entry["success"] = tracker_entry.get("success", 0) + 1
                else:
                    tracker_entry["failure"] = tracker_entry.get("failure", 0) + 1
                tracker_entry["total_duration"] = tracker_entry.get("total_duration", 0.0) + execution_time
                if tracker_entry["count"]:
                    tracker_entry["avg_duration"] = tracker_entry["total_duration"] / tracker_entry["count"]

                tool_call_id = getattr(tool_call, "call_id", None) or getattr(tool_call, "id", None)
                session_manager.add_tool_message(
                    session_id,
                    tool_call_id,
                    tool_output_for_message or tool_output,
                )

                if handler_result:
                    _merge_structure_hints_state(structure_state, handler_result)
                    _update_files_discovered(files_discovered_set, handler_result)
                    if structure_state.get("project_summary"):
                        ctx.obj["_project_structure_summary"] = structure_state["project_summary"]

                if canonical_name == "symbols.index" and call_success:
                    has_symbol_index = True

                tool_history.append(canonical_name)
                last_tool_success = call_success

                if call_success and canonical_name == "exec":
                    count_value = _extract_first_int(tool_output.strip())
                    if (
                        count_value is not None
                        and count_value <= 1
                        and not ctx.obj.get("_low_count_verified")
                    ):
                        guard_message = (
                            f"⚠️ LOW COUNT DETECTED ({count_value}):\n"
                            "Verify before finalizing:\n"
                            "1. Check working directory: `pwd`\n"
                            "2. List directory contents: `ls -la`\n"
                            "This prevents false negatives.\n"
                        )
                        session_manager.add_system_message(session_id, guard_message)
                        ctx.obj["_low_count_verified"] = True

            if consecutive_fails >= 3:
                click.echo("🚫 Multiple consecutive tool failures. Stopping to avoid loops.")
                execution_completed = True
                _commit_shell_history()
                _finalize()
                return

        except (LLMConnectionError, LLMTimeoutError, LLMRetryExhaustedError) as exc:
            click.echo(f"⚠️  Unable to reach the LLM: {exc}")
            click.echo("   Falling back to offline analysis using local heuristics.")
            _handle_question_without_llm(user_prompt, reason="LLM unavailable")
            execution_mode = "direct"
            execution_completed = True
            _commit_shell_history()
            _finalize()
            return
        except LLMError as exc:
            click.echo(f"❌ ReAct execution failed: {exc}")
            break
        except Exception as exc:
            click.echo(f"❌ ReAct execution failed: {exc}")
            break

    _commit_shell_history()

    if synthesized:
        _finalize()
        return

    if budget_manager.current >= iteration_cap:
        click.echo(f"⚠️  Reached maximum iteration limit ({iteration_cap}).")
        click.echo("Please refine your request or increase DEVAGENT_MAX_ITERATIONS if you need more steps.")
    else:
        click.echo("⚠️  Execution stopped before providing a final synthesis.")

    execution_completed = True

    session_snapshot: Optional[List[Message]] = None
    try:
        session_snapshot = session_manager.compose(session_id)
    except Exception:
        session_snapshot = None

    final_response_text: Optional[str] = None

    if session_snapshot:
        summary_prompt = (
            "Provide a final synthesis based on the conversation so far. "
            "Summarize findings, actionable conclusions, and open questions."
        )
        session_manager.add_user_message(session_id, summary_prompt)
        final_messages = session_manager.compose(session_id)

        try:
            result = client.invoke_tools(
                final_messages,
                tools=[create_text_only_tool()],
                temperature=0.1,
            )
        except Exception as exc:
            click.echo(f"⚠️  Unable to obtain final response from LLM: {exc}")
            result = None
        else:
            final_response_text = extract_text_content(result)
            if final_response_text:
                session_manager.add_assistant_message(session_id, final_response_text)
                if history_enabled and isinstance(ctx.obj, dict):
                    stored_history_raw = ctx.obj.get("_shell_conversation_history")
                    stored_history = (
                        [msg for msg in stored_history_raw if isinstance(msg, Message)]
                        if isinstance(stored_history_raw, list)
                        else []
                    )
                    stored_history.extend(
                        [
                            Message(role="user", content=summary_prompt),
                            Message(role="assistant", content=final_response_text),
                        ]
                    )
                    ctx.obj["_shell_conversation_history"] = _truncate_shell_history(
                        stored_history,
                        max_history_turns,
                    )

    if not final_response_text and session_snapshot and auto_summary_enabled:
        auto_summary = auto_generate_summary(
            session_snapshot,
            files_examined=files_discovered_set,
            searches_performed=search_queries,
        )
        final_response_text = auto_summary
    elif not final_response_text:
        final_response_text = "Model did not provide a final synthesis."

    if final_response_text:
        click.echo("\n📌 Final response:")
        click.echo(final_response_text)
    else:
        click.echo("❌ Model did not provide a usable final synthesis.")

    _finalize()


__all__ = ["_execute_react_assistant"]

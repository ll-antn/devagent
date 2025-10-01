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
from ai_dev_agent.core.utils.config import Settings
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

from ..analysis.formatting import _format_enhanced_tool_log
from ..analysis.research import _handle_question_without_llm
from ..router import IntentDecision, IntentRouter as _DEFAULT_INTENT_ROUTER


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if not normalized:
        return False
    return normalized not in {"0", "false", "off", "no"}


_DEBUG_CONTEXT_ENABLED = _is_truthy(os.getenv("DEVAGENT_DEBUG_REACT_CONTEXT"))


def _format_message_preview(message: Message, *, limit: int = 200) -> str:
    content = message.content
    if content is None:
        preview = "<no content>"
    else:
        preview = str(content).strip()
        if len(preview) > limit:
            preview = preview[: limit - 1] + "…"

    suffix_parts: List[str] = []
    if message.tool_call_id:
        suffix_parts.append(f"call_id={message.tool_call_id}")
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        suffix_parts.append(f"tool_calls={len(tool_calls)}")
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    return f"[{message.role}] {preview}{suffix}"


def _emit_context_snapshot(messages: Sequence[Message], iteration: int) -> None:
    if not _DEBUG_CONTEXT_ENABLED:
        return

    iteration_label = "initial context" if iteration == 0 else f"iteration {iteration}"
    click.echo(f"\n🔎 [debug] ReAct context snapshot before {iteration_label}:")
    for index, msg in enumerate(messages, start=1):
        click.echo(f"   {index:02d}. {_format_message_preview(msg)}")



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

    def _finalize() -> None:
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
            structured_plan = planner.generate(
                user_prompt,
                project_structure=project_structure,
                context=planning_context,
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

    core_instructions = (
        "You are a helpful assistant for the devagent CLI tool, specialized in efficient software development tasks.\n\n"
        "## MISSION\n"
        "Complete the user's task efficiently using available tools within {iteration_cap} iterations.\n\n"
        "## CORE PRINCIPLES\n"
        "1. EFFICIENCY: Choose the most appropriate tool for each task\n"
        "2. AVOID REDUNDANCY: Never repeat identical tool calls\n"
        "3. BULK OPERATIONS: Prefer batch operations over individual file reads\n"
        "4. EARLY TERMINATION: Stop when you have sufficient information\n"
        "5. ADAPTIVE STRATEGY: Change approach if tools fail\n"
        "6. SCRIPT GENERATION: Create scripts for complex computations\n\n"
        "## ITERATION MANAGEMENT\n"
        f"- Current budget: {iteration_cap} iterations\n"
        f"- At 75% usage ({int(iteration_cap * 0.75)} iterations): Begin consolidating findings\n"
        f"- At 90% usage ({int(iteration_cap * 0.9)} iterations): Finalize answer\n"
    )

    language_hints = {
        "python": "\n- Use ast_query for Python code structure\n- Check requirements.txt/setup.py for dependencies\n- Use import analysis for module relationships",
        "javascript": "\n- Consider package.json for dependencies\n- Use ast_query for JS/TS structure\n- Check for .eslintrc for code standards",
        "typescript": "\n- Check tsconfig.json for compilation settings\n- Use ast_query for TypeScript analysis\n- Consider type definitions in .d.ts files",
        "java": "\n- Check pom.xml or build.gradle for dependencies\n- Use ast_query for class hierarchies\n- Consider package structure for organization",
        "c++": "\n- Check CMakeLists.txt or Makefile for build config\n- Look for .h/.hpp headers separately from .cpp/.cc files\n- Use compile_commands.json if available",
        "c": "\n- Check Makefile or CMakeLists.txt for build setup\n- Analyze header files (.h) for interfaces\n- Use grep for macro definitions",
        "go": "\n- Check go.mod for module dependencies\n- Use go tools for analysis\n- Consider internal vs external packages",
        "rust": "\n- Check Cargo.toml for dependencies\n- Use cargo commands for analysis\n- Consider module structure in lib.rs/main.rs",
        "ruby": "\n- Check Gemfile for dependencies\n- Look for rake tasks\n- Consider Rails structure if applicable",
        "php": "\n- Check composer.json for dependencies\n- Look for autoload configurations\n- Consider framework structure (Laravel, Symfony)",
        "c#": "\n- Check .csproj or .sln files\n- Use NuGet packages info\n- Consider namespace organization",
        "swift": "\n- Check Package.swift for dependencies\n- Look for .xcodeproj/xcworkspace\n- Consider iOS/macOS target differences",
        "kotlin": "\n- Check build.gradle.kts for configuration\n- Consider Android vs JVM targets\n- Use Gradle for dependency info",
        "scala": "\n- Check build.sbt for dependencies\n- Use sbt commands for analysis\n- Consider Play/Akka frameworks if present",
        "dart": "\n- Check pubspec.yaml for dependencies\n- Consider Flutter structure if applicable\n- Use dart analyze for code issues",
    }

    if repository_language:
        language_guidance = language_hints.get(repository_language.lower(), "")
        if language_guidance:
            core_instructions += f"\nLANGUAGE-SPECIFIC ({repository_language}):{language_guidance}\n"

    tool_semantics = (
        "\nTOOL SEMANTICS:\n"
        "- exec: Runs commands in POSIX shell; pipes, globs, redirects work\n"
        "- Prefer machine-parsable output (e.g., find -print0) over formatted listings\n"
        "- Minimize tool calls - stop once you have the answer\n"
    )

    output_discipline = (
        "\nOUTPUT REQUIREMENTS:\n"
        "- State scope explicitly (depth, hidden files, symlinks)\n"
        "- Ensure counts match actual listed items\n"
        "- Stop executing once you have sufficient information\n"
    )

    fs_quick_recipes = (
        "\nCOMMON OPERATIONS:\n"
        "Count files: `find . -maxdepth 1 -type f | wc -l`\n"
        "List safely: `find . -maxdepth 1 -type f -print0 | xargs -0 -n1 basename`\n"
        "Verify location: `pwd` and `ls -la`\n"
    )

    failure_policy = (
        "\nFAILURE HANDLING:\n"
        "- First failure: Adjust parameters\n"
        "- Second failure: Switch tools/approach\n"
        "- System blocks identical calls after 2 failures\n"
        "- 3+ consecutive failures trigger termination\n"
    )

    tool_guidance = (
        "\nUNIVERSAL TOOL STRATEGIES:\n"
        "- For counting/metrics: Use shell commands with find/grep/wc\n"
        "- For pattern search: Start with code_search, then read specific files\n"
        "- For exploration: Use ast_query for structure, symbols for definitions\n"
        "- For bulk operations: Generate and execute scripts\n"
        "- For file operations: Prefer find/xargs over individual reads\n"
        "- Verify unexpected results (especially counts <=1) with pwd and ls -la\n"
    )

    tool_priority_guidance = (
        "\nTOOL SELECTION GUIDE:\n"
        "- Counting/metrics -> exec with scripts\n"
        "- Pattern search -> code_search\n"
        "- Code structure -> ast_query\n"
        "- Specific files -> fs.read\n"
        "- Bulk operations -> exec with find/xargs\n"
        "- Complex analysis -> Generate analysis scripts\n"
    )

    messages = [
        Message(
            role="system",
            content=(
                core_instructions
                + tool_semantics
                + output_discipline
                + fs_quick_recipes
                + failure_policy
                + tool_guidance
                + tool_priority_guidance
            ),
        ),
        user_message,
    ]

    project_structure = ctx.obj.get("_project_structure_summary")
    if not project_structure:
        project_structure = _collect_project_structure_outline(repo_root)
        if project_structure:
            ctx.obj["_project_structure_summary"] = project_structure
    if project_structure:
        structure_state["project_summary"] = project_structure
        messages.insert(
            1,
            Message(
                role="system",
                content=f"Project structure:\n{project_structure}",
            ),
        )

    if history_enabled and existing_history:
        insert_at = len(messages) - 1
        messages[insert_at:insert_at] = existing_history

    history_start_index = len(messages) - 1

    def _commit_shell_history() -> None:
        if not history_enabled or not isinstance(ctx.obj, dict):
            return
        new_entries: List[Message] = []
        if user_message.content:
            new_entries.append(user_message)
        for msg in messages[history_start_index + 1 :]:
            if msg.role == "assistant" and msg.content and not msg.tool_calls:
                new_entries.append(msg)
        ctx.obj["_shell_conversation_history"] = _truncate_shell_history(
            existing_history + new_entries,
            max_history_turns,
        )

    _emit_context_snapshot(messages, 0)

    iteration = 0
    tool_history: List[str] = []
    has_symbol_index = False
    files_discovered_set: Set[str] = set()
    last_tool_success = True

    while iteration < iteration_cap:
        iteration += 1

        if iteration > 0 and iteration % 5 == 0:
            progress_pct = (iteration / iteration_cap) * 100 if iteration_cap else 0.0
            progress_msg = f"\n[Progress: {iteration}/{iteration_cap} iterations ({progress_pct:.0f}%)]"
            if progress_pct >= 75:
                progress_msg += " - Approaching limit, prioritize synthesis"
            messages.append(Message(role="system", content=progress_msg))

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
                messages = [
                    msg
                    for msg in messages
                    if not (
                        msg.role == "system"
                        and isinstance(msg.content, str)
                        and msg.content.startswith("Tool performance this session:")
                    )
                ]
                messages.insert(
                    1,
                    Message(
                        role="system",
                        content="Tool performance this session:\n" + "\n".join(perf_summary),
                    ),
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

        tools_for_iteration = available_tools
        try:
            sanitized_to_entry: Dict[str, Dict[str, Any]] = {}
            sanitized_to_canonical: Dict[str, str] = {}

            _emit_context_snapshot(messages, iteration)

            for entry in available_tools:
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

            tools_for_iteration = [
                sanitized_to_entry[name]
                for name in prioritized_sanitized
                if name in sanitized_to_entry
            ] or available_tools
        except Exception:
            tools_for_iteration = available_tools

        try:
            result = client.invoke_tools(messages, tools=tools_for_iteration, temperature=0.1)

            assistant_message = Message(
                role="assistant",
                content=result.message_content,
                tool_calls=result.raw_tool_calls,
            )
            if assistant_message.content is not None or assistant_message.tool_calls:
                messages.append(assistant_message)

            if not result.calls:
                if result.message_content:
                    click.echo(result.message_content)
                else:
                    click.echo("I was unable to provide a complete answer.")
                execution_completed = True
                _commit_shell_history()
                _finalize()
                return

            redundant_calls = 0
            last_tool_failed = consecutive_fails > 0

            for tool_call in result.calls:
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
                    messages.append(
                        Message(
                            role="tool",
                            content=f"Error: {error_msg}",
                            tool_call_id=tool_call_id,
                        )
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
                            messages.append(Message(role="system", content=advisory))
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
                tool_message = Message(
                    role="tool",
                    content=tool_output_for_message or tool_output,
                    tool_call_id=tool_call_id,
                )
                messages.append(tool_message)

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
                        messages.append(Message(role="system", content=guard_message))
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

    if iteration >= iteration_cap:
        click.echo(f"⚠️  Reached maximum iteration limit ({iteration_cap}).")
        click.echo("Please refine your request or increase DEVAGENT_MAX_ITERATIONS if you need more steps.")
        execution_completed = True
        _finalize()


__all__ = ["_execute_react_assistant"]

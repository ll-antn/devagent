"""Formatting utilities for CLI tool output and logging."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_dev_agent.core.utils.tool_utils import (
    FILE_READ_TOOLS,
    SEARCH_TOOLS,
    canonical_tool_name,
    display_tool_name,
    tool_category,
)

from ai_dev_agent.core.utils.text import first_displayable_line


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
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        max_lines = args.get("max_lines")

        if start_line is not None:
            if end_line is not None:
                return f"{path}:{start_line}-{end_line}"
            if max_lines is not None and max_lines != 200:
                return f"{path}:{start_line}+{max_lines}"
            return f"{path}:{start_line}+"
        return path

    if tool_call.name in SEARCH_TOOLS:
        return args.get("query", "")

    if tool_call.name in ("exec", "sandbox.exec", "sandbox_exec", "execute"):
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

    if tool_call.name in ("fs.write_patch", "fs_write_patch"):
        targets = _extract_patch_targets(args.get("diff"))
        if not targets:
            return ""
        if len(targets) == 1:
            return targets[0]
        return f"{len(targets)} files"

    for key in ["path", "paths", "query", "cmd", "command", "goal", "question", "name"]:
        if key in args and args[key]:
            value = args[key]
            if key == "paths" and isinstance(value, list):
                return value[0] if value else ""
            return str(value)
    return ""


def _extract_patch_targets(diff_content: Any) -> List[str]:
    """Return the set of files targeted by a unified diff."""

    if not isinstance(diff_content, str) or not diff_content.strip():
        return []

    files: List[str] = []
    diff_pattern = re.compile(r"^diff --git a/(.+?) b/(.+)$")
    fallback_pattern = re.compile(r"^\+\+\+ b/(.+)$")

    for line in diff_content.splitlines():
        match = diff_pattern.match(line.strip())
        if not match:
            continue
        old_path = _normalize_patch_path(match.group(1))
        new_path = _normalize_patch_path(match.group(2))
        candidate = new_path if new_path and new_path != "/dev/null" else old_path
        if candidate and candidate != "/dev/null" and candidate not in files:
            files.append(candidate)

    if files:
        return files

    for line in diff_content.splitlines():
        match = fallback_pattern.match(line.strip())
        if not match:
            continue
        candidate = _normalize_patch_path(match.group(1))
        if candidate and candidate != "/dev/null" and candidate not in files:
            files.append(candidate)

    return files


def _normalize_patch_path(path: str) -> str:
    path = (path or "").strip()
    if path.startswith('"') and path.endswith('"') and len(path) >= 2:
        path = path[1:-1]
    return path


def _format_enhanced_tool_log(
    tool_call,
    repeat_count: int,
    execution_time: float,
    tool_output: str = "",
    success: bool = True,
    file_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Format tool execution log with enhanced context and metrics."""
    file_context = file_context or {}
    main_arg = _get_main_argument(tool_call)
    display_name = display_tool_name(tool_call.name)
    canonical_name = canonical_tool_name(tool_call.name)
    category = tool_category(tool_call.name)

    if not success:
        return _format_error_message(tool_call, tool_output, main_arg)

    if category == "file_read" and file_context:
        tree_msg = _format_file_reading_tree(tool_call, main_arg, tool_output, file_context)
        if tree_msg:
            return tree_msg

    log_msg = f"{display_name} \"{main_arg}\"" if main_arg else display_name

    if repeat_count > 1:
        log_msg += f" ({repeat_count}x)"

    metrics_info = _extract_tool_metrics(canonical_name, tool_output or "", tool_call=tool_call)
    metrics = metrics_info.metrics.strip()
    if metrics:
        log_msg += f" → {metrics}"

    if execution_time > 1.0:
        log_msg += f" ({execution_time:.1f}s)"

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

    suggestions: List[str] = []
    repo_root = Path.cwd()
    filename = Path(failed_path).name
    if not filename:
        return []

    for root_dir in [repo_root / "ai_dev_agent", repo_root / "tests", repo_root / "docs", repo_root]:
        if not root_dir.exists():
            continue
        for path in root_dir.rglob(f"*{filename}*"):
            if path.is_file():
                try:
                    rel_path = str(path.relative_to(repo_root))
                except ValueError:
                    rel_path = str(path)
                if rel_path not in suggestions:
                    suggestions.append(rel_path)
                if len(suggestions) >= 3:
                    break
        if len(suggestions) >= 3:
            break

    return suggestions[:3]


def _format_file_reading_tree(tool_call, main_arg: str, tool_output: str, file_context: Dict[str, Any]) -> str:
    """Format file reading operations in a tree structure for sequential reads."""
    args = tool_call.arguments or {}
    path = args.get("path", "")
    start_line = args.get("start_line")
    canonical_name = canonical_tool_name(tool_call.name)
    display_name = display_tool_name(tool_call.name)

    last_read = file_context.get("last_file_read")
    if last_read and last_read.get("path") == path:
        metrics = _extract_tool_metrics(canonical_name, tool_output, tool_call=tool_call).metrics
        file_context["last_file_read"] = {
            "path": path,
            "last_end_line": _extract_end_line_from_output(tool_output),
            "is_continuation": True,
        }
        continuation_symbol = "├─" if not _is_file_complete(tool_output) else "└─"
        return f"📖 {continuation_symbol} continuing from line {start_line} → {metrics}"

    metrics = _extract_tool_metrics(canonical_name, tool_output, tool_call=tool_call).metrics
    file_context["last_file_read"] = {
        "path": path,
        "last_end_line": _extract_end_line_from_output(tool_output),
        "is_continuation": False,
    }
    return f"📖 {display_name} \"{main_arg}\" → {metrics}"


def _extract_end_line_from_output(tool_output: str) -> int:
    match = re.search(r"lines (\d+)-(\d+) of (\d+)", tool_output)
    if match:
        return int(match.group(2))
    return 0


def _is_file_complete(tool_output: str) -> bool:
    return "more lines not shown" not in tool_output


def _extract_tool_metrics(tool_name: str, tool_output: str, *, tool_call=None) -> ToolLogDetails:
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

    first_line = first_displayable_line(text)
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

    first_line = first_displayable_line(text)
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

    first_line = first_displayable_line(text)
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

    first_line = first_displayable_line(text)
    return ToolLogDetails(first_line or "executed")


def _fallback_error_message(tool_output: str) -> str:
    if ":" in tool_output:
        return tool_output.split(":")[-1].strip()
    return tool_output.strip()


def _parse_exit_code(output: str) -> Optional[int]:
    match = re.search(r"Exit:\s*(\d+)", output)
    if match:
        return int(match.group(1))
    match = re.search(r"exit code[:\s]*(\d+)", output.lower())
    if match:
        return int(match.group(1))
    return None


def _extract_first_error_line(output: str) -> str:
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
    lowered = command_text.lower()
    if not lowered:
        return False
    search_tools = ("grep", "rg", "ripgrep", "ag", "ack", "fd", "find")
    return any(tool in lowered for tool in search_tools)


def _get_tool_status_indicator(tool_name: str, success: bool) -> str:
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
    if status == "warning":
        return "⚠"
    if status == "error":
        return "❌"
    return _get_tool_status_indicator(tool_name, success)


__all__ = [
    "ToolLogDetails",
    "_get_main_argument",
    "_format_enhanced_tool_log",
    "_format_error_message",
    "_format_file_reading_tree",
    "_extract_tool_metrics",
    "_analyze_search_output",
    "_analyze_file_read_output",
    "_extract_primary_path",
    "_analyze_command_output",
    "_fallback_error_message",
    "_parse_exit_code",
    "_extract_first_error_line",
    "_normalize_error_hint",
    "_is_pattern_search_command",
    "_get_tool_status_indicator",
    "_resolve_status_indicator",
]

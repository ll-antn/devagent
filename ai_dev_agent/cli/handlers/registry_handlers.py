"""Registry-backed intent handlers used by the CLI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import click

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.tool_utils import expand_tool_aliases

from ..utils import (
    _invoke_registry_tool,
    _normalize_argument_list,
    build_system_context,
)

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

    def __call__(self, ctx: click.Context, arguments: Dict[str, Any]) -> Mapping[str, Any]:
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
        return result


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

        structure_summary = match.get("structure_summary")
        if not structure_summary:
            structure = match.get("structure") or {}
            summary_parts: List[str] = []
            path_components = structure.get("path") or []
            if path_components:
                summary_parts.append(".".join(path_components))
            symbol_kind = structure.get("kind")
            if symbol_kind:
                summary_parts.append(symbol_kind)
            depth = structure.get("depth")
            if depth:
                summary_parts.append(f"depth {depth}")
            symbol_line = structure.get("line")
            if symbol_line:
                summary_parts.append(f"line {symbol_line}")
            if summary_parts:
                structure_summary = "within " + " • ".join(summary_parts)

        if structure_summary:
            click.echo(f"    structure: {structure_summary}")

        import_context = match.get("import_context") or []
        if import_context:
            joined = "; ".join(import_context)
            click.echo(f"    imports: {joined}")

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
    ctx_obj = ctx.obj if isinstance(ctx.obj, dict) else {}
    system_context = ctx_obj.setdefault("_system_context", build_system_context()) if isinstance(ctx_obj, dict) else build_system_context()
    cmd_value = arguments.get("cmd") or arguments.get("command")
    cmd = str(cmd_value or "").strip()
    if not cmd:
        raise click.ClickException(
            "exec requires 'cmd'. Received arguments: "
            f"{arguments}. System: {system_context.get('os')}. "
            "Ensure the LLM provided a valid command for this platform."
        )

    if system_context.get("os") == "Windows" and cmd.startswith("ls"):
        cmd = cmd.replace("ls", "dir", 1)

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

INTENT_HANDLERS: Dict[str, RegistryIntent] = expand_tool_aliases(REGISTRY_INTENTS)

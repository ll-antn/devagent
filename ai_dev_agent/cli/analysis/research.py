"""Research parsing and summarisation utilities for CLI responses."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.constants import DEFAULT_IGNORED_REPO_DIRS
from ai_dev_agent.core.utils.keywords import extract_keywords
from ai_dev_agent.core.utils.tool_utils import canonical_tool_name, tool_category
from ai_dev_agent.core.utils.text import (
    extract_function_signatures,
    safe_int,
    truncate_text,
)

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
        cleaned = truncate_text(node.strip(), 160)
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
            snippet = truncate_text(content.strip(), 320)
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
                        "; ".join(truncate_text(note, 120) for note in finding.notes)
                    )
                detail = "; ".join(part for part in detail_parts if part)
                base_line = f"  - {finding.path}"
                if detail:
                    base_line += f" ({detail})"
                if not append_line(base_line):
                    break
                for snippet in _select_highlight_lines(finding.snippets):
                    line_number = f"{snippet.number}" if snippet.number is not None else "-"
                    snippet_text = truncate_text(snippet.text)
                    if not append_line(f"    {line_number}: {snippet_text}"):
                        break
                if truncated:
                    break

        if self.searches and not truncated:
            append_line("- Searches:")
            for finding in sorted(self.searches.values(), key=lambda item: item.latest_index, reverse=True):
                query_display = truncate_text(finding.query or "(no query)", 120)
                base_line = f"  - \"{query_display}\""
                if finding.matches:
                    base_line += f" ({len(finding.matches)} match{'es' if len(finding.matches) != 1 else ''})"
                elif finding.status == "no_matches":
                    base_line += " (no matches)"
                if not append_line(base_line):
                    break
                if finding.additional_note:
                    append_line(f"    {truncate_text(finding.additional_note, 160)}")
                for match in _select_highlight_matches(finding.matches):
                    location = match.path
                    if match.line is not None:
                        location += f":{match.line}"
                        if match.column is not None:
                            location += f":{match.column}"
                    preview = truncate_text(match.preview)
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
                    descriptor_parts.append(f"query \"{truncate_text(finding.query, 120)}\"")
                header = "  - " + ", ".join(descriptor_parts) if descriptor_parts else "  - (query)"
                if finding.match_count is not None:
                    header += f" ({finding.match_count} node{'s' if finding.match_count != 1 else ''})"
                if not append_line(header):
                    break
                for node in finding.nodes[:3]:
                    if not append_line(f"    {truncate_text(node, 160)}"):
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
                    qualifiers.append(truncate_text(finding.status, 120))
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
                command_display = truncate_text(finding.command, 120)
                line = f"  - `{command_display}`"
                if finding.exit_code is not None:
                    line += f" exit {finding.exit_code}"
                if not append_line(line):
                    break
                for output in finding.output_lines[:3]:
                    if not append_line(f"    {truncate_text(output, 160)}"):
                        break
                if truncated:
                    break

        if self.generics and not truncated:
            append_line("- Other outputs:")
            for finding in sorted(self.generics, key=lambda item: item.latest_index, reverse=True)[:6]:
                line = f"  - {finding.tool_name}: {truncate_text(finding.text, 160)}"
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
            for signature in extract_function_signatures(snippet.text):
                self._record_function(signature, finding.path, snippet.number, "file", index)

    def _collect_functions_from_search(self, finding: SearchFinding, index: int) -> None:
        for match in finding.matches:
            for signature in extract_function_signatures(match.preview):
                self._record_function(signature, match.path, match.line, "search", index)

    def _collect_functions_from_ast(self, finding: ASTFinding, index: int) -> None:
        for node in finding.nodes:
            for signature in extract_function_signatures(node):
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
        signature = truncate_text(signature, 160)
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
            start = safe_int(range_match.group(2))
            end = safe_int(range_match.group(3))
            total = safe_int(range_match.group(4))
            current.add_range(start, end)
            if total is not None:
                if current.total_lines is None or total > current.total_lines:
                    current.total_lines = total
            continue

        code_match = code_pattern.match(raw_line)
        if code_match:
            if current is None:
                current = FileFinding(path=take_default_path() or "(unknown)")
            line_number = safe_int(code_match.group(1))
            snippet_text = truncate_text(code_match.group(2))
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
                    line=safe_int(match.group("line")),
                    column=safe_int(match.group("col")),
                    preview=truncate_text(match.group("preview")),
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
                finding.match_count = safe_int(match.group(1))
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
                    line=safe_int(match.group("line")),
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
        exit_code = safe_int(exit_match.group(1))
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
        truncate_text(line.strip(), 200)
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
        if extract_function_signatures(snippet.text):
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
        if extract_function_signatures(match.preview):
            add_candidate(match)

    for match in matches:
        add_candidate(match)
        if len(prioritized) >= limit:
            break

    return prioritized[:limit]


def _search_paths_and_content(
    keyword: str,
    *,
    include_docs: bool = True,
    max_results: int = 5,
) -> List[str]:
    """Locate files whose path or content mention the given keyword."""
    keyword = keyword.strip()
    if not keyword:
        return []

    repo = Path.cwd()
    keyword_lower = keyword.lower()
    results: List[str] = []
    seen: set[str] = set()
    skip_dirs = DEFAULT_IGNORED_REPO_DIRS

    def maybe_add(path: Path) -> None:
        try:
            rel = path.relative_to(repo)
        except ValueError:
            return
        rel_str = str(rel)
        if not include_docs and rel.parts and rel.parts[0] == "docs":
            return
        if rel_str not in seen:
            seen.add(rel_str)
            results.append(rel_str)

    for path in repo.rglob("*"):
        if len(results) >= max_results:
            break
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.is_dir():
            continue
        maybe_add_path = False
        rel_str = str(path.relative_to(repo)) if path.exists() else str(path)
        if keyword_lower in rel_str.lower():
            maybe_add_path = True
        else:
            try:
                if path.stat().st_size > 200_000:
                    continue
                content = path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue
            if keyword_lower in content.lower():
                maybe_add_path = True
        if maybe_add_path:
            maybe_add(path)

    return results[:max_results]

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

__all__ = [
    "CodeLineSnippet",
    "BaseFinding",
    "FileFinding",
    "SearchMatch",
    "SearchFinding",
    "ASTFinding",
    "SymbolMatch",
    "SymbolFinding",
    "CommandFinding",
    "GenericFinding",
    "FunctionFinding",
    "_ResearchSummaryAggregator",
    "_collect_tool_call_metadata",
    "_parse_file_read_result",
    "_parse_search_result",
    "_parse_ast_result",
    "_parse_symbol_result",
    "_parse_command_result",
    "_format_range_summary",
    "_select_highlight_lines",
    "_select_highlight_matches",
    "_build_fallback_tool_summary",
    "_provide_research_summary",
    "_handle_question_without_llm",
]

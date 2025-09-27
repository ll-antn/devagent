"""Tree-sitter powered project structure summaries."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ai_dev_agent.core.utils.logger import get_logger

LOGGER = get_logger(__name__)

try:  # pragma: no cover - exercised indirectly when dependency available
    from tree_sitter_languages import (  # type: ignore
        get_language as _get_language,
        get_parser as _get_parser,
    )
except ImportError:  # pragma: no cover - graceful degradation when not installed
    _get_language = None
    _get_parser = None

try:  # pragma: no cover - we only reference when tree-sitter is installed
    from tree_sitter import Parser as _TreeSitterParser  # type: ignore
except ImportError:  # pragma: no cover - degrade gracefully
    _TreeSitterParser = None


@dataclass
class ParsedFileSummary:
    """Structured summary extracted from a source file."""

    path: str
    outline: List[str]

    def to_markdown(self) -> str:
        header = f"### {self.path}"
        body = "\n".join(self.outline)
        return f"{header}\n{body}" if body else header


class TreeSitterProjectAnalyzer:
    """Generate structural summaries for source files using tree-sitter."""

    SUPPORTED_SUFFIXES: Dict[str, str] = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
    }

    def __init__(self, repo_root: Path, max_files: int = 8, max_lines_per_file: int = 12) -> None:
        self.repo_root = repo_root
        self._parser_factory = _get_parser
        self._language_factory = _get_language
        self._parser_class = _TreeSitterParser
        self._parsers: Dict[str, object] = {}
        self.max_files = max_files
        self.max_lines_per_file = max_lines_per_file
        self._available = self._has_parser_support()

        if not self._available:
            LOGGER.debug("tree-sitter support not available; project summaries disabled")
        else:
            probe_language = self.SUPPORTED_SUFFIXES.get(".py")
            if probe_language:
                parser = self._create_parser(probe_language)
                if parser is None:
                    self._available = False
                    LOGGER.debug("tree-sitter parser initialization failed; summaries disabled")
                else:
                    self._parsers[probe_language] = parser

    @property
    def available(self) -> bool:
        return self._available

    @property
    def supported_suffixes(self) -> List[str]:
        return list(self.SUPPORTED_SUFFIXES.keys())

    def build_project_summary(self, file_entries: Iterable[Tuple[str, str]]) -> Optional[str]:
        """Return a markdown summary describing the structure of the provided files."""
        if not self.available:
            return None

        summaries: List[ParsedFileSummary] = []

        for rel_path, content in file_entries:
            suffix = Path(rel_path).suffix
            parser = self._get_parser_for_suffix(suffix)
            if parser is None:
                continue

            try:
                source_bytes = content.encode("utf-8", errors="ignore")
            except UnicodeEncodeError:
                LOGGER.debug("Skipping non-UTF8 encodable file for summary: %s", rel_path)
                continue

            try:
                tree = parser.parse(source_bytes)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.debug("tree-sitter failed to parse %s: %s", rel_path, exc)
                continue

            outline = self._summarize_file(Path(rel_path), suffix, tree, source_bytes)
            if outline:
                summaries.append(ParsedFileSummary(rel_path, outline[: self.max_lines_per_file]))

            if len(summaries) >= self.max_files:
                break

        if not summaries:
            return None

        header = [
            "# Project Structure (Tree-sitter)",
            "",
            "Generated outline of relevant files to help the language model navigate the codebase.",
            "",
        ]

        body = "\n\n".join(summary.to_markdown() for summary in summaries)
        return "\n".join(header) + body

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_parser_support(self) -> bool:
        if self._parser_factory is not None:
            return True
        return self._language_factory is not None and self._parser_class is not None

    def _get_parser_for_suffix(self, suffix: str):
        language = self.SUPPORTED_SUFFIXES.get(suffix)
        if not language or not self.available:
            return None

        if language in self._parsers:
            return self._parsers[language]

        parser = self._create_parser(language)
        if parser is None:
            return None

        self._parsers[language] = parser
        return parser

    def _create_parser(self, language: str):
        if self._parser_factory is not None:
            try:
                return self._parser_factory(language)
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.debug("Failed to load tree-sitter parser for %s via factory: %s", language, exc)

        if self._parser_class is not None and self._language_factory is not None:
            try:
                lang_obj = self._language_factory(language)
                if lang_obj is None:
                    return None
                parser = self._parser_class()
                parser.set_language(lang_obj)
                return parser
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.debug("Failed to configure parser for %s via set_language: %s", language, exc)

        return None

    def _summarize_file(self, path: Path, suffix: str, tree, source_bytes: bytes) -> List[str]:
        if suffix == ".py":
            return self._summarize_python(tree, source_bytes)
        # TODO: extend support for other languages as needed
        return []

    # Python -----------------------------------------------------------------

    def _summarize_python(self, tree, source_bytes: bytes) -> List[str]:
        outline: List[str] = []
        root = tree.root_node

        for node in root.children:
            if node.type == "class_definition":
                outline.extend(self._summarize_python_class(node, source_bytes))
            elif node.type == "function_definition":
                signature = self._format_python_function(node, source_bytes)
                outline.append(f"- function {signature}")

            if len(outline) >= self.max_lines_per_file:
                break

        return outline

    def _summarize_python_class(self, node, source_bytes: bytes) -> List[str]:
        lines: List[str] = []
        signature = self._format_python_class(node, source_bytes)
        lines.append(f"- class {signature}")

        method_lines = []
        block = self._first_child_of_type(node, "block")
        if block:
            for child in block.children:
                if child.type == "function_definition":
                    method_signature = self._format_python_function(child, source_bytes)
                    method_lines.append(f"    - method {method_signature}")
                    if len(method_lines) >= 6:  # avoid overly long summaries
                        break
        lines.extend(method_lines)
        return lines

    def _format_python_class(self, node, source_bytes: bytes) -> str:
        name = self._node_text(node.child_by_field_name("name"), source_bytes) or "<anonymous>"
        bases_node = self._first_child_of_type(node, "argument_list")
        bases = self._clean_signature(self._node_text(bases_node, source_bytes)) if bases_node else ""
        line_no = node.start_point[0] + 1
        suffix = f" {bases}" if bases else ""
        return f"{name}{suffix} (line {line_no})"

    def _format_python_function(self, node, source_bytes: bytes) -> str:
        name = self._node_text(node.child_by_field_name("name"), source_bytes) or "<anonymous>"
        params_node = node.child_by_field_name("parameters")
        params = self._node_text(params_node, source_bytes) if params_node else "()"
        params = self._clean_signature(params)
        line_no = node.start_point[0] + 1
        return f"{name}{params} (line {line_no})"

    # Generic helpers --------------------------------------------------------

    def _node_text(self, node, source_bytes: bytes) -> str:
        if node is None:
            return ""
        text = source_bytes[node.start_byte : node.end_byte]
        return text.decode("utf-8", errors="ignore")

    def _clean_signature(self, signature: str) -> str:
        signature = signature.strip()
        signature = re.sub(r"\s+", " ", signature)
        return signature

    def _first_child_of_type(self, node, node_type: str):
        for child in getattr(node, "children", []):
            if child.type == node_type:
                return child
        return None


__all__ = ["TreeSitterProjectAnalyzer"]

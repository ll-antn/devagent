"""Tree-sitter powered project structure summaries."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from ai_dev_agent.core.tree_sitter import (
    EXTENSION_LANGUAGE_MAP,
    MANAGER as TREE_SITTER_MANAGER,
    ensure_parser,
    node_text,
    slice_bytes,
)
from ai_dev_agent.core.utils.logger import get_logger

LOGGER = get_logger(__name__)

_SUPPORTED_LANGUAGES = {"python", "typescript", "tsx", "javascript", "c", "cpp"}
_SUPPORTED_SUFFIXES = tuple(
    suffix for suffix, language in EXTENSION_LANGUAGE_MAP.items() if language in _SUPPORTED_LANGUAGES
)


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

    SUPPORTED_LANGUAGES = _SUPPORTED_LANGUAGES
    SUPPORTED_SUFFIXES = _SUPPORTED_SUFFIXES

    def __init__(self, repo_root: Path, max_files: int = 8, max_lines_per_file: int = 12) -> None:
        self.repo_root = repo_root
        self._manager = TREE_SITTER_MANAGER
        self.max_files = max_files
        self.max_lines_per_file = max_lines_per_file
        self._available = self._manager.available

        if not self._available:
            LOGGER.debug("tree-sitter support not available; project summaries disabled")

    @property
    def available(self) -> bool:
        return self._available

    @property
    def supported_suffixes(self) -> List[str]:
        return list(self.SUPPORTED_SUFFIXES)

    def _get_parser_for_suffix(self, suffix: str):
        """Compatibility shim for existing tests relying on the old API."""

        if not self.available:
            return None

        language = EXTENSION_LANGUAGE_MAP.get(suffix)
        if language not in self.SUPPORTED_LANGUAGES:
            return None

        handle = self._manager.parser_for_language(language)
        return None if handle is None else handle.parser

    def summarize_content(self, rel_path: str, content: str) -> List[str]:
        """Return a structural outline for a single file."""

        if not self.available:
            return []

        path = Path(rel_path)
        abs_path = (self.repo_root / path).resolve()

        handle = ensure_parser(abs_path, content=content)
        if handle is None or handle.language not in self.SUPPORTED_LANGUAGES:
            return []

        parser = handle.parser
        source_bytes = slice_bytes(content)
        try:
            tree = parser.parse(source_bytes)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive guard
            return []

        return self._summarize_file(path, handle.language, tree, source_bytes)

    def build_project_summary(self, file_entries: Iterable[Tuple[str, str]]) -> Optional[str]:
        """Return a markdown summary describing the structure of the provided files."""
        if not self.available:
            return None

        summaries: List[ParsedFileSummary] = []

        for rel_path, content in file_entries:
            path = Path(rel_path)
            abs_path = (self.repo_root / path).resolve()

            handle = ensure_parser(abs_path, content=content)
            if handle is None or handle.language not in self.SUPPORTED_LANGUAGES:
                continue

            parser = handle.parser
            source_bytes = slice_bytes(content)

            try:
                tree = parser.parse(source_bytes)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.debug("tree-sitter failed to parse %s: %s", rel_path, exc)
                continue

            outline = self._summarize_file(path, handle.language, tree, source_bytes)
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

    def _summarize_file(self, path: Path, language_or_suffix: str, tree, source_bytes: bytes) -> List[str]:
        language = language_or_suffix
        if language.startswith('.'):
            language = EXTENSION_LANGUAGE_MAP.get(language, language.lstrip('.'))

        if language == "python":
            return self._summarize_python(tree, source_bytes)
        if language in {"typescript", "tsx"}:
            return self._summarize_typescript(tree, source_bytes)
        if language == "javascript":
            return self._summarize_javascript(tree, source_bytes)
        if language in {"c", "cpp"}:
            return self._summarize_c_cpp(tree, source_bytes, language=language)
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

    # JavaScript / TypeScript -------------------------------------------------

    def _summarize_javascript(self, tree, source_bytes: bytes) -> List[str]:
        return self._summarize_ecmascript(tree, source_bytes, include_typescript=False)

    def _summarize_typescript(self, tree, source_bytes: bytes) -> List[str]:
        return self._summarize_ecmascript(tree, source_bytes, include_typescript=True)

    def _summarize_ecmascript(self, tree, source_bytes: bytes, *, include_typescript: bool) -> List[str]:
        outline: List[str] = []
        root = getattr(tree, "root_node", None)
        if root is None:
            return outline

        for child in getattr(root, "children", []):
            entries = self._summarize_ecmascript_node(
                child,
                source_bytes,
                include_typescript=include_typescript,
            )
            if not entries:
                continue
            outline.extend(entries)
            if len(outline) >= self.max_lines_per_file:
                break

        return outline[: self.max_lines_per_file]

    def _summarize_ecmascript_node(
        self,
        node,
        source_bytes: bytes,
        *,
        include_typescript: bool,
        prefix: str = "",
    ) -> List[str]:
        node_type = getattr(node, "type", "")

        if node_type == "export_statement":
            return self._summarize_export_statement(node, source_bytes, include_typescript)

        if node_type in {"function_declaration", "generator_function_declaration"}:
            return [
                self._format_ecmascript_function(
                    node,
                    source_bytes,
                    prefix,
                    include_typescript=include_typescript,
                )
            ]

        if node_type == "class_declaration":
            return [
                self._format_ecmascript_class(
                    node,
                    source_bytes,
                    prefix,
                )
            ]

        if node_type in {"lexical_declaration", "variable_declaration"}:
            return self._format_ecmascript_variables(
                node,
                source_bytes,
                prefix,
                include_typescript=include_typescript,
            )

        if include_typescript and node_type == "interface_declaration":
            return [
                self._format_typescript_interface(
                    node,
                    source_bytes,
                    prefix,
                )
            ]

        if include_typescript and node_type == "type_alias_declaration":
            return [
                self._format_typescript_type_alias(
                    node,
                    source_bytes,
                    prefix,
                )
            ]

        if include_typescript and node_type == "enum_declaration":
            return [
                self._format_typescript_enum(
                    node,
                    source_bytes,
                    prefix,
                )
            ]

        return []

    def _summarize_export_statement(self, node, source_bytes: bytes, include_typescript: bool) -> List[str]:
        has_default = any(child.type == "default" for child in getattr(node, "children", []))
        prefix = "export default " if has_default else "export "

        declaration = node.child_by_field_name("declaration")
        if declaration is not None:
            return self._summarize_ecmascript_node(
                declaration,
                source_bytes,
                include_typescript=include_typescript,
                prefix=prefix,
            )

        clause = self._first_child_of_type(node, "export_clause")
        if clause is not None:
            clause_text = self._clean_signature(self._node_text(clause, source_bytes))
            line_no = node.start_point[0] + 1
            return [f"- export {clause_text} (line {line_no})"]

        star_target = None
        for child in getattr(node, "children", []):
            child_type = getattr(child, "type", "")
            if child_type == "string":
                star_target = self._clean_signature(self._node_text(child, source_bytes))
            if child_type == "*":
                line_no = node.start_point[0] + 1
                target = star_target or "*"
                return [f"- export * from {target} (line {line_no})"]

        named_child = next(
            (child for child in getattr(node, "children", []) if getattr(child, "is_named", False)),
            None,
        )
        if named_child is not None:
            snippet = self._shorten(self._clean_signature(self._node_text(named_child, source_bytes)))
            line_no = named_child.start_point[0] + 1
            return [f"- {prefix}{snippet} (line {line_no})"]

        text = self._shorten(self._clean_signature(self._node_text(node, source_bytes)))
        line_no = node.start_point[0] + 1
        return [f"- {prefix}{text} (line {line_no})"]

    def _format_ecmascript_function(
        self,
        node,
        source_bytes: bytes,
        prefix: str,
        *,
        include_typescript: bool,
    ) -> str:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            name_node = self._first_child_of_type(node, "identifier")
        name = self._node_text(name_node, source_bytes) or "<anonymous>"

        params_node = node.child_by_field_name("parameters")
        params = self._node_text(params_node, source_bytes) if params_node else "()"
        params = self._clean_signature(params)

        type_params = ""
        if include_typescript:
            type_param_node = self._first_child_of_type(node, "type_parameters")
            if type_param_node is not None:
                type_params = self._clean_signature(self._node_text(type_param_node, source_bytes))

        return_type = ""
        if include_typescript:
            return_node = self._first_child_of_type(node, "type_annotation")
            if return_node is not None:
                return_type = self._clean_signature(self._node_text(return_node, source_bytes))

        kind = "function*" if node.type == "generator_function_declaration" else "function"

        signature = f"{name}{type_params}{params}"
        if return_type:
            signature = f"{signature} {return_type}"

        line_no = node.start_point[0] + 1
        descriptor = f"{prefix}{kind} {signature}".strip()
        descriptor = self._clean_signature(descriptor)
        return f"- {descriptor} (line {line_no})"

    def _format_ecmascript_class(self, node, source_bytes: bytes, prefix: str) -> str:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            name_node = self._first_child_of_type(node, "identifier")
        name = self._node_text(name_node, source_bytes) or "<anonymous>"

        heritage_node = self._first_child_of_type(node, "class_heritage")
        heritage = self._clean_signature(self._node_text(heritage_node, source_bytes)) if heritage_node else ""

        suffix = f" {heritage}" if heritage else ""
        line_no = node.start_point[0] + 1
        descriptor = f"{prefix}class {name}{suffix}".strip()
        descriptor = self._clean_signature(descriptor)
        return f"- {descriptor} (line {line_no})"

    def _format_ecmascript_variables(
        self,
        node,
        source_bytes: bytes,
        prefix: str,
        *,
        include_typescript: bool,
    ) -> List[str]:
        entries: List[str] = []
        keyword = self._lexical_keyword(node, source_bytes)

        for child in getattr(node, "children", []):
            if getattr(child, "type", "") != "variable_declarator":
                continue
            name_node = child.child_by_field_name("name") or self._first_child_of_type(child, "identifier")
            name = self._node_text(name_node, source_bytes) or "<anonymous>"

            type_annotation = ""
            if include_typescript:
                type_node = self._first_child_of_type(child, "type_annotation")
                if type_node is not None:
                    type_annotation = self._clean_signature(self._node_text(type_node, source_bytes))

            value_node = child.child_by_field_name("value")
            value = ""
            if value_node is not None:
                value = self._shorten(
                    self._clean_signature(self._node_text(value_node, source_bytes)),
                    limit=60,
                )

            descriptor = f"{prefix}{keyword} {name}".strip()
            if type_annotation:
                descriptor = f"{descriptor} {type_annotation}".strip()
            if value:
                descriptor = f"{descriptor} = {value}".strip()

            line_no = child.start_point[0] + 1
            entries.append(f"- {descriptor} (line {line_no})")

            if len(entries) >= self.max_lines_per_file:
                break

        return entries

    def _format_typescript_interface(self, node, source_bytes: bytes, prefix: str) -> str:
        name_node = node.child_by_field_name("name")
        name = self._node_text(name_node, source_bytes) or "<anonymous>"

        extends_node = self._first_child_of_type(node, "extends_type_clause")
        extends = self._clean_signature(self._node_text(extends_node, source_bytes)) if extends_node else ""

        suffix = f" {extends}" if extends else ""
        line_no = node.start_point[0] + 1
        descriptor = f"{prefix}interface {name}{suffix}".strip()
        descriptor = self._clean_signature(descriptor)
        return f"- {descriptor} (line {line_no})"

    def _format_typescript_type_alias(self, node, source_bytes: bytes, prefix: str) -> str:
        name_node = node.child_by_field_name("name")
        name = self._node_text(name_node, source_bytes) or "<anonymous>"

        value_node = node.child_by_field_name("value")
        raw_value = self._node_text(value_node, source_bytes) if value_node else ""
        value = self._shorten(self._clean_signature(raw_value), limit=60) if raw_value else ""

        descriptor = f"{prefix}type {name}".strip()
        if value:
            descriptor = f"{descriptor} = {value}".strip()

        line_no = node.start_point[0] + 1
        descriptor = self._clean_signature(descriptor)
        return f"- {descriptor} (line {line_no})"

    def _format_typescript_enum(self, node, source_bytes: bytes, prefix: str) -> str:
        name_node = node.child_by_field_name("name")
        name = self._node_text(name_node, source_bytes) or "<anonymous>"

        members: List[str] = []
        body = node.child_by_field_name("body")
        if body is not None:
            for member in getattr(body, "children", []):
                if not getattr(member, "is_named", False):
                    continue
                member_name_node = member.child_by_field_name("name") or member
                member_name = self._clean_signature(self._node_text(member_name_node, source_bytes))
                value_node = member.child_by_field_name("value")
                if value_node is not None:
                    value_text = self._clean_signature(self._node_text(value_node, source_bytes))
                    member_name = f"{member_name}={value_text}"
                members.append(member_name)

        preview = ", ".join(members[:3])
        if len(members) > 3:
            preview += ", …"

        suffix = f" [{preview}]" if preview else ""
        line_no = node.start_point[0] + 1
        descriptor = f"{prefix}enum {name}{suffix}".strip()
        descriptor = self._clean_signature(descriptor)
        return f"- {descriptor} (line {line_no})"

    # C / C++ ----------------------------------------------------------------

    def _summarize_c_cpp(self, tree, source_bytes: bytes, *, language: str) -> List[str]:
        outline: List[str] = []
        root = getattr(tree, "root_node", None)
        if root is None:
            return outline

        for child in getattr(root, "children", []):
            entries = self._summarize_c_cpp_node(
                child,
                source_bytes,
                language=language,
                indent="",
            )
            if entries:
                outline.extend(entries)
            if len(outline) >= self.max_lines_per_file:
                break

        return outline[: self.max_lines_per_file]

    def _summarize_c_cpp_node(
        self,
        node,
        source_bytes: bytes,
        *,
        language: str,
        indent: str,
    ) -> List[str]:
        if not getattr(node, "is_named", False):
            return []

        node_type = getattr(node, "type", "")
        line_no = node.start_point[0] + 1
        next_indent = indent + "    "
        entries: List[str] = []
        is_cpp = language == "cpp"

        if node_type in {"declaration_list", "field_declaration_list"}:
            for child in getattr(node, "children", []):
                entries.extend(
                    self._summarize_c_cpp_node(
                        child,
                        source_bytes,
                        language=language,
                        indent=indent,
                    )
                )
            return entries

        if node_type in {"compound_statement", "comment", "preproc_include", "preproc_def"}:
            return []

        if node_type == "function_definition":
            header = self._extract_before_field(node, "body", source_bytes)
            header = self._shorten(self._clean_signature(header), limit=80)
            entries.append(f"{indent}- function {header} (line {line_no})")
            return entries

        if node_type == "namespace_definition":
            name_node = node.child_by_field_name("name")
            name = self._node_text(name_node, source_bytes) or "<anonymous>"
            entries.append(f"{indent}- namespace {self._clean_signature(name)} (line {line_no})")
            body = node.child_by_field_name("body")
            if body is not None:
                for child in getattr(body, "children", []):
                    entries.extend(
                        self._summarize_c_cpp_node(
                            child,
                            source_bytes,
                            language=language,
                            indent=next_indent,
                        )
                    )
            return entries

        if node_type == "linkage_specification":
            lang_node = self._first_child_of_type(node, "string_literal")
            lang = self._node_text(lang_node, source_bytes) or '"C"'
            entries.append(f"{indent}- extern {self._clean_signature(lang)} (line {line_no})")
            body = node.child_by_field_name("body")
            if body is not None:
                for child in getattr(body, "children", []):
                    entries.extend(
                        self._summarize_c_cpp_node(
                            child,
                            source_bytes,
                            language=language,
                            indent=next_indent,
                        )
                    )
            return entries

        if node_type == "template_declaration":
            params_node = node.child_by_field_name("parameters")
            params = (
                self._clean_signature(self._node_text(params_node, source_bytes))
                if params_node is not None
                else ""
            )
            descriptor = f"template {params}".strip()
            entries.append(f"{indent}- {descriptor} (line {line_no})")
            for child in getattr(node, "children", []):
                if not getattr(child, "is_named", False) or getattr(child, "type", "") == "template":
                    continue
                entries.extend(
                    self._summarize_c_cpp_node(
                        child,
                        source_bytes,
                        language=language,
                        indent=next_indent,
                    )
                )
            return entries

        if node_type in {"struct_specifier", "union_specifier", "class_specifier"}:
            kind_map = {
                "struct_specifier": "struct",
                "union_specifier": "union",
                "class_specifier": "class",
            }
            kind = kind_map[node_type]
            name_node = node.child_by_field_name("name")
            name = self._node_text(name_node, source_bytes) or "<anonymous>"
            extra = ""
            if node_type == "class_specifier":
                base_clause = self._first_child_of_type(node, "base_class_clause")
                if base_clause is not None:
                    extra = " " + self._clean_signature(self._node_text(base_clause, source_bytes))
            entries.append(
                f"{indent}- {kind} {self._clean_signature(name)}{extra} (line {line_no})"
            )
            body = node.child_by_field_name("body")
            if body is not None:
                for child in getattr(body, "children", []):
                    entries.extend(
                        self._summarize_c_cpp_node(
                            child,
                            source_bytes,
                            language=language,
                            indent=next_indent,
                        )
                    )
            return entries

        if node_type == "enum_specifier":
            name_node = node.child_by_field_name("name")
            name = self._node_text(name_node, source_bytes)
            if not name:
                parent = getattr(node, "parent", None)
                if getattr(parent, "type", "") == "type_definition":
                    alias = self._first_child_of_type(parent, "type_identifier")
                    alias_text = self._node_text(alias, source_bytes)
                    if alias_text:
                        name = alias_text
            name = name or "<anonymous>"
            members: List[str] = []
            body = node.child_by_field_name("body")
            if body is not None:
                for member in getattr(body, "children", []):
                    if getattr(member, "type", "") != "enumerator":
                        continue
                    member_name_node = member.child_by_field_name("name") or member
                    member_name = self._clean_signature(
                        self._node_text(member_name_node, source_bytes)
                    )
                    value_node = member.child_by_field_name("value")
                    if value_node is not None:
                        value = self._clean_signature(self._node_text(value_node, source_bytes))
                        member_name = f"{member_name}={value}"
                    members.append(member_name)
            preview = ", ".join(members[:3])
            if len(members) > 3:
                preview += ", …"
            suffix = f" [{preview}]" if preview else ""
            entries.append(
                f"{indent}- enum {self._clean_signature(name)}{suffix} (line {line_no})"
            )
            return entries

        if node_type == "declaration":
            descriptor = self._format_c_declaration(node, source_bytes)
            if descriptor:
                entries.append(f"{indent}- declaration {descriptor} (line {line_no})")
            return entries

        if node_type == "type_definition":
            descriptor = self._format_c_declaration(node, source_bytes)
            if descriptor:
                entries.append(f"{indent}- definition {descriptor} (line {line_no})")
            for child in getattr(node, "children", []):
                if not getattr(child, "is_named", False):
                    continue
                entries.extend(
                    self._summarize_c_cpp_node(
                        child,
                        source_bytes,
                        language=language,
                        indent=indent,
                    )
                )
            return entries

        if node_type == "field_declaration":
            descriptor = self._format_c_declaration(node, source_bytes)
            if descriptor:
                entries.append(f"{indent}- member {descriptor} (line {line_no})")
            return entries

        if is_cpp and node_type == "alias_declaration":
            descriptor = self._format_c_declaration(node, source_bytes)
            if descriptor:
                entries.append(f"{indent}- alias {descriptor} (line {line_no})")
            return entries

        return []

    # Generic helpers --------------------------------------------------------

    def _segment_text(self, source_bytes: bytes, start: int, end: int) -> str:
        return source_bytes[start:end].decode("utf-8", errors="ignore")

    def _extract_before_field(self, node, field_name: str, source_bytes: bytes) -> str:
        target = node.child_by_field_name(field_name)
        if target is None:
            return self._node_text(node, source_bytes)
        return self._segment_text(source_bytes, node.start_byte, target.start_byte)

    def _format_c_declaration(self, node, source_bytes: bytes) -> str:
        text = self._node_text(node, source_bytes)
        text = text.rstrip(";")
        text = self._clean_signature(text)
        return self._shorten(text, limit=80)

    def _node_text(self, node, source_bytes: bytes) -> str:
        if node is None:
            return ""
        return node_text(node, source_bytes)

    def _clean_signature(self, signature: str) -> str:
        signature = signature.strip()
        signature = re.sub(r"\s+", " ", signature)
        return signature

    def _first_child_of_type(self, node, node_type: str):
        for child in getattr(node, "children", []):
            if child.type == node_type:
                return child
        return None

    def _lexical_keyword(self, node, source_bytes: bytes) -> str:
        for child in getattr(node, "children", []):
            if getattr(child, "is_named", False):
                continue
            token = self._node_text(child, source_bytes).strip()
            if token in {"const", "let", "var"}:
                return token
        return "const"

    def _shorten(self, text: str, *, limit: int = 80) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"


__all__ = ["TreeSitterProjectAnalyzer"]

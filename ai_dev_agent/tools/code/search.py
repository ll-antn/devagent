"""code.search implementation backed by ripgrep."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ai_dev_agent.core.tree_sitter import ensure_parser, node_text, slice_bytes

from ..registry import ToolSpec, ToolContext, registry

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas" / "tools"


def _code_search(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    repo_root = context.repo_root
    query = payload["query"]
    regex = payload.get("regex", False)
    max_results = payload.get("max_results", 100)
    where = payload.get("where") or []

    if not shutil.which("rg"):
        matches = _fallback_search(repo_root, query, where, max_results)
        _enrich_matches_with_structure(repo_root, matches)
        return {"matches": matches}

    command = ["rg", "--json", "--color", "never", "--max-columns", "500"]
    if not regex:
        command.append("--fixed-strings")
    command.append(query)
    if where:
        command.extend(where)

    process = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    if process.returncode not in {0, 1}:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "ripgrep failed")

    matches: list[dict[str, Any]] = []
    for line in process.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data", {})
        path = data.get("path", {}).get("text")
        if not path:
            continue
        subs = data.get("submatches") or []
        if not subs:
            continue
        sub = subs[0]
        preview = sub.get("match", {}).get("text", "")
        start = sub.get("start", 0)
        line_number = data.get("line_number", 1) or 1
        matches.append(
            {
                "path": path,
                "line": int(line_number),
                "col": int(start) + 1,
                "preview": preview,
            }
        )
        if len(matches) >= max_results:
            break

    _enrich_matches_with_structure(repo_root, matches)
    return {"matches": matches}


def _fallback_search(repo_root: Path, query: str, where: List[str], max_results: int) -> List[dict[str, Any]]:
    """Simple Python fallback when ripgrep is not available."""
    query_str = str(query)
    # Determine directories to search
    search_roots: List[Path]
    if where:
        search_roots = []
        for rel in where:
            candidate = (repo_root / rel).resolve()
            if repo_root not in candidate.parents and candidate != repo_root:
                continue
            if candidate.exists():
                search_roots.append(candidate)
        if not search_roots:
            search_roots = [repo_root]
    else:
        search_roots = [repo_root]

    matches: List[dict[str, Any]] = []
    for root in search_roots:
        for path in root.rglob("*"):
            if len(matches) >= max_results:
                break
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            lines = text.splitlines()
            for idx, line in enumerate(lines, start=1):
                col = line.find(query_str)
                if col != -1:
                    try:
                        rel_path = str(path.relative_to(repo_root))
                    except ValueError:
                        rel_path = str(path)
                    matches.append(
                        {
                            "path": rel_path,
                            "line": idx,
                            "col": col + 1,
                            "preview": line.strip(),
                        }
                    )
                    break
    matches = matches[:max_results]
    _enrich_matches_with_structure(repo_root, matches)
    return matches


def _structure_for_position(language: str, tree, source_bytes: bytes, line: int, col_zero_indexed: int) -> Optional[Dict[str, Any]]:
    """Return structural information for the position if available."""

    if line <= 0:
        return None

    # Points are zero indexed in tree-sitter
    point = (line - 1, col_zero_indexed)
    try:
        node = tree.root_node.descendant_for_point_range(point, point)
    except Exception:
        return None

    if node is None:
        return None

    descriptors = list(_iter_structural_descriptors(language, node, source_bytes))
    if not descriptors:
        return None

    # descriptors currently ordered from innermost to outermost
    ordered = list(reversed(descriptors))
    depth = len(ordered)

    path_components: List[str] = []
    for kind, name, _line in ordered:
        if not name:
            continue
        if kind in {"function", "method"} and not name.endswith("()"):
            path_components.append(f"{name}()")
        else:
            path_components.append(name)

    if path_components:
        path_display = ".".join(path_components)
    else:
        # Fall back to the innermost descriptor name/kind
        innermost_kind, innermost_name, _ = descriptors[0]
        path_display = innermost_name or innermost_kind

    innermost_kind, innermost_name, innermost_line = descriptors[0]
    innermost_symbol = innermost_name or innermost_kind

    structure = {
        "kind": innermost_kind,
        "symbol": innermost_symbol,
        "depth": depth,
        "line": innermost_line,
        "path": path_components,
        "ancestors": [
            {"kind": kind, "name": name, "line": node_line}
            for kind, name, node_line in ordered
        ],
    }

    summary = f"{path_display} at line {innermost_line} ({innermost_kind}, depth {depth})"
    structure["summary"] = summary

    return {
        "structure": structure,
        "structure_summary": summary,
    }


def _iter_structural_descriptors(language: str, node, source_bytes: bytes):
    while node is not None:
        descriptor = _describe_structural_node(language, node, source_bytes)
        if descriptor is not None:
            yield descriptor
        node = node.parent


def _describe_structural_node(language: str, node, source_bytes: bytes) -> Optional[tuple[str, str, int]]:
    """Return (kind, name, line) for structural nodes or None."""

    lang = language.lower()
    node_type = node.type
    line = node.start_point[0] + 1

    def field_text(field: str) -> str:
        target = node.child_by_field_name(field)
        return node_text(target, source_bytes).strip() if target is not None else ""

    def child_text(child) -> str:
        return node_text(child, source_bytes).strip() if child is not None else ""

    if lang == "python":
        if node_type == "function_definition":
            return "function", field_text("name"), line
        if node_type == "class_definition":
            return "class", field_text("name"), line
    elif lang in {"javascript", "typescript", "tsx"}:
        if node_type == "function_declaration":
            return "function", field_text("name"), line
        if node_type == "method_definition":
            name_node = node.child_by_field_name("name")
            if name_node is None and node.child_count:
                name_node = node.child_by_field_name("property") or node.children[0]
            name = child_text(name_node)
            return "method", name, line
        if node_type == "arrow_function" and lang != "typescript":
            return "function", "<arrow_function>", line
        if node_type == "class_declaration":
            return "class", field_text("name"), line
        if lang in {"typescript", "tsx"}:
            if node_type == "interface_declaration":
                return "interface", field_text("name"), line
            if node_type == "type_alias_declaration":
                return "type", field_text("name"), line
            if node_type == "enum_declaration":
                return "enum", field_text("name"), line
    elif lang == "java":
        if node_type == "method_declaration":
            return "method", field_text("name"), line
        if node_type == "class_declaration":
            return "class", field_text("name"), line
        if node_type == "interface_declaration":
            return "interface", field_text("name"), line
    elif lang == "go":
        if node_type == "function_declaration":
            return "function", field_text("name"), line
        if node_type == "method_declaration":
            return "method", field_text("name"), line
        if node_type == "type_spec":
            type_node = node.child_by_field_name("type")
            name = field_text("name")
            if type_node and type_node.type in {"struct_type", "interface_type"}:
                kind = "struct" if type_node.type == "struct_type" else "interface"
                return kind, name, line
    elif lang == "rust":
        if node_type == "function_item":
            return "function", field_text("name"), line
        if node_type == "impl_item":
            return "impl", field_text("name"), line
        if node_type == "struct_item":
            return "struct", field_text("name"), line
        if node_type == "enum_item":
            return "enum", field_text("name"), line
        if node_type == "trait_item":
            return "trait", field_text("name"), line
        if node_type == "mod_item":
            return "module", field_text("name"), line
    elif lang == "ruby":
        if node_type == "method":
            return "method", field_text("name"), line
        if node_type == "class":
            constant = node.child_by_field_name("name")
            return "class", child_text(constant), line
        if node_type == "module":
            constant = node.child_by_field_name("name")
            return "module", child_text(constant), line
    elif lang in {"c", "cpp"}:
        if node_type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            return "function", child_text(declarator), line
        if node_type == "function_declarator":
            return "function", node_text(node, source_bytes).strip(), line
        if node_type == "struct_specifier":
            name_node = node.child_by_field_name("name")
            return "struct", child_text(name_node), line
        if node_type == "class_specifier":
            name_node = node.child_by_field_name("name")
            return "class", child_text(name_node), line
        if node_type == "namespace_definition":
            name_node = node.child_by_field_name("name")
            return "namespace", child_text(name_node), line
    elif lang == "c_sharp" or lang == "csharp":
        if node_type == "method_declaration":
            return "method", field_text("name"), line
        if node_type == "class_declaration":
            return "class", field_text("name"), line
        if node_type == "interface_declaration":
            return "interface", field_text("name"), line
        if node_type == "namespace_declaration":
            return "namespace", field_text("name"), line
    elif lang == "php":
        if node_type == "function_definition":
            return "function", field_text("name"), line
        if node_type == "method_declaration":
            return "method", field_text("name"), line
        if node_type == "class_declaration":
            return "class", field_text("name"), line
        if node_type == "interface_declaration":
            return "interface", field_text("name"), line

    return None


def _collect_import_nodes(language: str, tree, source_bytes: bytes) -> List[tuple[int, str]]:
    """Gather import/include statements for later contextual display."""

    lang = language.lower()
    root = tree.root_node
    import_types = {
        "python": {"import_statement", "import_from_statement"},
        "javascript": {"import_statement"},
        "typescript": {"import_statement"},
        "tsx": {"import_statement"},
        "java": {"import_declaration"},
        "go": {"import_declaration"},
        "rust": {"use_declaration"},
        "c": {"preproc_include"},
        "cpp": {"preproc_include"},
        "c_sharp": {"using_directive"},
        "csharp": {"using_directive"},
        "php": {"namespace_use_declaration"},
    }.get(lang)

    if not import_types:
        return []

    snippets: List[tuple[int, str]] = []
    for child in getattr(root, "children", []):
        if child.type in import_types:
            line = child.start_point[0] + 1
            text = node_text(child, source_bytes).strip()
            if text:
                snippets.append((line, text))

    return snippets


def _imports_before_line(import_snippets: List[tuple[int, str]], line: int, limit: int = 3) -> List[str]:
    """Return up to *limit* import statements occurring before the provided line."""

    preceding = [text for import_line, text in import_snippets if import_line < line]
    if not preceding:
        return []
    return preceding[-limit:]


def _enrich_matches_with_structure(repo_root: Path, matches: List[Dict[str, Any]]) -> None:
    """Augment search matches with structural context when tree-sitter is available."""

    if not matches:
        return

    matches_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for match in matches:
        rel_path = match.get("path")
        if not rel_path:
            continue
        matches_by_file.setdefault(rel_path, []).append(match)

    for rel_path, file_matches in matches_by_file.items():
        abs_path = (repo_root / rel_path).resolve()
        if not abs_path.exists():
            continue

        try:
            text = abs_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        handle = ensure_parser(abs_path, content=text)
        if handle is None:
            continue

        parser = handle.parser
        source_bytes = slice_bytes(text)
        try:
            tree = parser.parse(source_bytes)  # type: ignore[attr-defined]
        except Exception:
            continue

        import_snippets = _collect_import_nodes(handle.language, tree, source_bytes)

        for match in file_matches:
            line = int(match.get("line", 0))
            col = max(int(match.get("col", 1)) - 1, 0)
            structure_payload = _structure_for_position(handle.language, tree, source_bytes, line, col)
            if structure_payload:
                match.update(structure_payload)

            if import_snippets:
                imports = _imports_before_line(import_snippets, line)
                if imports:
                    match.setdefault("import_context", imports)



registry.register(
    ToolSpec(
        name="code.search",
        handler=_code_search,
        request_schema_path=SCHEMA_DIR / "code.search.request.json",
        response_schema_path=SCHEMA_DIR / "code.search.response.json",
        description=(
            "Search repository text for the provided query. By DEFAULT uses FIXED STRING matching - "
            "regex patterns like 'def.*test' will be treated as literal strings. Set 'regex': true "
            "to enable regex patterns. Supports optional 'where' (list of directories/files), and "
            "'max_results' (int). Automatically falls back to a Python-based scan when ripgrep is unavailable."
        ),
    )
)


__all__ = ["_code_search"]

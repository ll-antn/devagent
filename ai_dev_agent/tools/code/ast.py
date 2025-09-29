"""AST query tool powered by tree-sitter.

This tool gracefully degrades when tree-sitter is not available by returning
an empty result set instead of failing the entire tool registry import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ai_dev_agent.core.tree_sitter import (
    build_capture_query,
    ensure_parser,
    get_ast_query,
    language_object,
    node_text,
    normalise_language,
    slice_bytes,
)

from ..registry import ToolSpec, ToolContext, registry

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas" / "tools"


def validate_query(query: str, language: str) -> tuple[bool, str, str]:
    """Validate AST query syntax and structure.
    
    Returns:
        (is_valid, corrected_query, error_message)
    """
    
    # Basic validation checks
    if not query or not query.strip():
        return False, "", "Empty query"
    
    query = query.strip()
    
    # Check balanced parentheses
    paren_count = 0
    for char in query:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        if paren_count < 0:
            return False, query, "Unmatched closing parenthesis"
    
    if paren_count != 0:
        # Try to fix common issues
        if paren_count > 0:
            # Missing closing parens
            corrected = query + ')' * paren_count
            return True, corrected, f"Auto-corrected: added {paren_count} closing parenthesis"
        else:
            # Extra closing parens - remove extras
            # Count actual opening parens
            open_count = query.count('(')
            close_count = query.count(')')
            extra_closes = close_count - open_count
            
            # Remove extra closing parens from the end
            corrected = query
            for _ in range(extra_closes):
                corrected = corrected.rstrip(') ')
            
            return True, corrected, f"Auto-corrected: removed {extra_closes} extra closing parenthesis"
    
    # Check for basic query structure
    if not query.startswith('('):
        return False, query, "Query must start with opening parenthesis"
    
    if not query.endswith(')'):
        return False, query, "Query must end with closing parenthesis"
    
    # Check for capture syntax if @ is present
    if '@' in query:
        # Basic check for capture syntax
        import re
        capture_pattern = r'@\w+'
        captures = re.findall(capture_pattern, query)
        if not captures:
            return False, query, "Invalid capture syntax - @ must be followed by identifier"
    
    return True, query, "Valid query"


def get_safe_fallback_query(language: str, original_query: str) -> str:
    """Get a safe fallback query for the language when the original fails."""

    language = normalise_language(language)
    original_lower = original_query.lower()

    heuristic_templates = []

    if language in {"cpp", "c"}:
        if any(word in original_lower for word in ["class", "struct"]):
            heuristic_templates.append("find_classes" if language == "cpp" else "find_structs")
        if any(word in original_lower for word in ["namespace", "module"]):
            heuristic_templates.append("find_namespaces")
        if any(word in original_lower for word in ["method", "member", "function"]):
            heuristic_templates.append("find_functions")
    elif language == "python":
        if "class" in original_lower:
            heuristic_templates.append("find_classes")
        if "import" in original_lower:
            heuristic_templates.append("find_imports")
        if "async" in original_lower:
            heuristic_templates.append("find_async")
        heuristic_templates.append("find_functions")
    elif language in {"javascript", "typescript", "tsx"}:
        if "class" in original_lower:
            heuristic_templates.append("find_classes")
        if "interface" in original_lower:
            heuristic_templates.append("find_interfaces")
        if "export" in original_lower:
            heuristic_templates.append("find_exports")
        if "import" in original_lower:
            heuristic_templates.append("find_imports")
        if "arrow" in original_lower:
            heuristic_templates.append("find_arrow_functions")
        heuristic_templates.append("find_functions")
    else:
        heuristic_templates.append("find_functions")

    for template in heuristic_templates:
        query = get_ast_query(language, template)
        if query:
            return query

    default_templates = {
        "cpp": ("find_functions", "function_definition", "func"),
        "c": ("find_functions", "function_definition", "func"),
        "python": ("find_functions", "function_definition", "func"),
        "javascript": ("find_functions", "function_declaration", "func"),
        "typescript": ("find_functions", "function_declaration", "func"),
        "tsx": ("find_functions", "function_declaration", "func"),
        "java": ("find_methods", "method_declaration", "method"),
        "go": ("find_functions", "function_declaration", "func"),
        "rust": ("find_functions", "function_item", "func"),
        "ruby": ("find_methods", "method", "method"),
        "csharp": ("find_methods", "method_declaration", "method"),
        "php": ("find_functions", "function_definition", "func"),
    }

    template_name, node_type, capture = default_templates.get(language, (None, None, None))
    if template_name:
        query = get_ast_query(language, template_name)
        if query:
            return query
    if node_type and capture:
        return build_capture_query(node_type, capture)

    return "(ERROR) @error"


def _ast_query(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    repo_root = context.repo_root
    rel_path = payload["path"]
    target = (repo_root / rel_path).resolve()
    if not target.exists():
        raise FileNotFoundError(rel_path)

    source = target.read_text(encoding="utf-8", errors="ignore")
    handle = ensure_parser(target, content=source)
    if handle is None:  # pragma: no cover - optional dependency path
        return {"nodes": []}

    language_name = handle.language
    language_obj = language_object(language_name)
    if language_obj is None:  # pragma: no cover - optional dependency path
        return {"nodes": []}

    parser = handle.parser
    source_bytes = slice_bytes(source)
    tree = parser.parse(source_bytes)

    # Validate and potentially fix the query
    original_query = payload["query"]
    is_valid, corrected_query, validation_msg = validate_query(original_query, language_name)

    if not is_valid:
        # Try fallback query
        fallback_query = get_safe_fallback_query(language_name, original_query)
        if fallback_query != "(ERROR) @error":
            corrected_query = fallback_query
            validation_msg = f"Used fallback query '{fallback_query}' due to invalid original: {validation_msg}"
        else:
            raise ValueError(
                f"Invalid query '{original_query}' for language '{language_name}': {validation_msg}. "
                "If tree-sitter is not installed, try code.search instead."
            )

    try:
        query = language_obj.query(corrected_query)
    except Exception as e:
        # If corrected query still fails, try fallback
        fallback_query = get_safe_fallback_query(language_name, original_query)
        if fallback_query != "(ERROR) @error" and fallback_query != corrected_query:
            try:
                query = language_obj.query(fallback_query)
                validation_msg = f"Used fallback query '{fallback_query}' due to parse error: {str(e)}"
                corrected_query = fallback_query
            except Exception as e2:
                raise ValueError(
                    "Both original query "
                    f"'{original_query}' and fallback query '{fallback_query}' failed for language "
                    f"'{language_name}': {str(e2)}. If tree-sitter is not installed, try code.search instead."
                )
        else:
            raise ValueError(
                f"Invalid query '{corrected_query}' for language '{language_name}': {str(e)}. "
                "If tree-sitter is not installed, try code.search instead."
            )

    allowed_captures = set(payload.get("captures") or [])

    nodes = []
    for node, capture_name in query.captures(tree.root_node):
        if allowed_captures and capture_name not in allowed_captures:
            continue
        text = node_text(node, source_bytes)
        start_point = list(node.start_point) if isinstance(node.start_point, tuple) else [node.start_point.row, node.start_point.column]
        end_point = list(node.end_point) if isinstance(node.end_point, tuple) else [node.end_point.row, node.end_point.column]
        nodes.append(
            {
                "start_byte": node.start_byte,
                "end_byte": node.end_byte,
                "start_point": start_point,
                "end_point": end_point,
                "text": text,
                "captures": [capture_name],
            }
        )

    result = {"nodes": nodes}

    # Add diagnostic information if query was corrected
    if corrected_query != original_query or "fallback" in validation_msg.lower():
        result["query_diagnostics"] = {
            "original_query": original_query,
            "corrected_query": corrected_query,
            "message": validation_msg
        }

    return result


registry.register(
    ToolSpec(
        name="ast.query",
        handler=_ast_query,
        request_schema_path=SCHEMA_DIR / "ast.query.request.json",
        response_schema_path=SCHEMA_DIR / "ast.query.response.json",
        description=(
            "Run a tree-sitter query against a source file. Provide 'path' and 'query' (tree-sitter pattern). "
            "Optional 'captures' limits returned capture names. Supported suffixes include: "
            ".py, .js, .ts, .tsx, .c, .h, .cc, .cpp, .cxx, .hh, .hpp, .hxx, .java, .go, .rs, .rb, .cs, .php."
        ),
    )
)


__all__ = ["_ast_query"]

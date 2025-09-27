"""AST query tool powered by tree-sitter.

This tool gracefully degrades when tree-sitter is not available by returning
an empty result set instead of failing the entire tool registry import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:  # Avoid hard dependency at import time
    from tree_sitter import Parser  # type: ignore
    from tree_sitter_languages import get_language  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Parser = None  # type: ignore
    get_language = None  # type: ignore

from ..registry import ToolSpec, ToolContext, registry

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas" / "tools"

LANGUAGE_BY_SUFFIX = {
    # Python / JS / TS
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    # C / C++
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".c++": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    # Other popular languages
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".cs": "c_sharp",
    ".php": "php",
}

# Language detection indicators for ambiguous file extensions
LANGUAGE_INDICATORS = {
    "cpp": ["class ", "namespace ", "template<", "std::", "public:", "private:", "virtual ", "override", "using namespace", "::"],
    "c": ["typedef struct", "#include <stdio.h>", "FILE *", "malloc(", "free(", "printf(", "scanf(", "void main"],
    "python": ["def ", "class ", "import ", "from ", "__init__", "self.", "@property", "async def", "if __name__"],
    "javascript": ["function ", "const ", "let ", "var ", "=>", "async function", "export ", "import ", "document.", "window."],
    "typescript": ["interface ", "type ", "enum ", ": string", ": number", "export interface", "<T>", "as ", ": void"],
    "java": ["public class", "private ", "protected ", "extends ", "implements ", "package ", "@Override", "public static void main"],
    "go": ["func ", "package ", "import (", "type ", "struct {", "interface {", "defer ", "go func", "make("],
    "rust": ["fn ", "impl ", "trait ", "struct ", "enum ", "pub ", "mod ", "use ", "match ", "let mut"],
    "ruby": ["def ", "class ", "module ", "require ", "attr_", "end\n", "elsif", "@", "puts "],
    "csharp": ["namespace ", "public class", "private ", "using ", "override ", "async Task", "public static void Main"],
    "php": ["<?php", "function ", "class ", "$this->", "namespace ", "use ", "trait ", "<?=", "echo "]
}

def detect_language_from_content(file_path: Path, content: str) -> str:
    """Smart language detection based on file content and extension."""
    
    # First check explicit extensions
    ext = file_path.suffix.lower()
    explicit_mappings = {
        ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
        ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
        ".cs": "c_sharp", ".php": "php", ".cpp": "cpp", ".cc": "cpp",
        ".cxx": "cpp", ".c++": "cpp", ".hh": "cpp", ".hpp": "cpp", ".hxx": "cpp",
        ".c": "c"
    }
    
    if ext in explicit_mappings:
        return explicit_mappings[ext]
    
    # For ambiguous extensions like .h, analyze content
    if ext in [".h", ".hpp"]:
        cpp_score = sum(1 for ind in LANGUAGE_INDICATORS["cpp"] if ind in content)
        c_score = sum(1 for ind in LANGUAGE_INDICATORS["c"] if ind in content)
        
        # If we find strong C++ indicators, it's C++
        if cpp_score > c_score or cpp_score > 0:
            return "cpp"
        return "c"
    
    # Fallback: analyze content for any language
    max_score = 0
    detected_lang = None
    content_lower = content.lower()
    
    for lang, indicators in LANGUAGE_INDICATORS.items():
        score = sum(1 for ind in indicators if ind.lower() in content_lower)
        if score > max_score:
            max_score = score
            detected_lang = lang
    
    return detected_lang or "text"


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
    
    # Default safe queries that should work for most languages
    safe_queries = {
        "cpp": "(function_definition) @function",
        "c": "(function_definition) @function", 
        "python": "(function_definition) @function",
        "javascript": "(function_declaration) @function",
        "typescript": "(function_declaration) @function",
        "tsx": "(function_declaration) @function",
        "java": "(method_declaration) @method",
        "go": "(function_declaration) @function",
        "rust": "(function_item) @function",
        "ruby": "(method) @method",
        "c_sharp": "(method_declaration) @method",
        "php": "(function_definition) @function"
    }
    
    # Try to infer intent from original query
    original_lower = original_query.lower()
    
    if language == "cpp" or language == "c":
        if any(word in original_lower for word in ["class", "struct"]):
            return "(class_specifier) @class" if language == "cpp" else "(struct_specifier) @struct"
        elif any(word in original_lower for word in ["method", "member"]):
            return "(function_definition) @method"
        elif "namespace" in original_lower:
            return "(namespace_definition) @namespace"
    elif language == "python":
        if "class" in original_lower:
            return "(class_definition) @class"
        elif "import" in original_lower:
            return "(import_statement) @import"
    elif language in ["javascript", "typescript", "tsx"]:
        if "class" in original_lower:
            return "(class_declaration) @class"
        elif "interface" in original_lower and language in ["typescript", "tsx"]:
            return "(interface_declaration) @interface"
        elif "arrow" in original_lower:
            return "(arrow_function) @arrow_function"
    
    # Default fallback
    return safe_queries.get(language, "(ERROR) @error")


def _ast_query(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    repo_root = context.repo_root
    rel_path = payload["path"]
    target = (repo_root / rel_path).resolve()
    if not target.exists():
        raise FileNotFoundError(rel_path)

    # Defer imports and handle environments without tree-sitter installed
    if Parser is None or get_language is None:  # pragma: no cover - optional dependency path
        return {"nodes": []}

    # Smart language detection using content analysis
    source = target.read_text(encoding="utf-8", errors="ignore")
    lang_name = detect_language_from_content(target, source)
    
    # Map language names to tree-sitter grammar names
    grammar_mapping = {
        "cpp": "cpp",
        "c": "c",
        "python": "python", 
        "javascript": "javascript",
        "typescript": "typescript",
        "tsx": "tsx",
        "java": "java",
        "go": "go",
        "rust": "rust",
        "ruby": "ruby",
        "c_sharp": "c_sharp",
        "csharp": "c_sharp",
        "php": "php"
    }
    
    tree_sitter_lang = grammar_mapping.get(lang_name)
    if not tree_sitter_lang:
        # Fallback to suffix-based detection for unsupported content detection
        suffix = target.suffix
        tree_sitter_lang = LANGUAGE_BY_SUFFIX.get(suffix)
        if not tree_sitter_lang:
            raise ValueError(f"No tree-sitter grammar available for language '{lang_name}' or suffix '{suffix}'")

    try:
        language = get_language(tree_sitter_lang)
    except Exception as e:
        raise ValueError(f"Failed to load tree-sitter grammar for '{tree_sitter_lang}': {e}")

    parser = Parser()
    parser.set_language(language)
    tree = parser.parse(bytes(source, "utf-8"))
    
    # Validate and potentially fix the query
    original_query = payload["query"]
    is_valid, corrected_query, validation_msg = validate_query(original_query, tree_sitter_lang)
    
    if not is_valid:
        # Try fallback query
        fallback_query = get_safe_fallback_query(tree_sitter_lang, original_query)
        if fallback_query != "(ERROR) @error":
            corrected_query = fallback_query
            validation_msg = f"Used fallback query '{fallback_query}' due to invalid original: {validation_msg}"
        else:
            raise ValueError(f"Invalid query '{original_query}' for language '{tree_sitter_lang}': {validation_msg}. If tree-sitter is not installed, try code.search instead.")
    
    try:
        query = language.query(corrected_query)
    except Exception as e:
        # If corrected query still fails, try fallback
        fallback_query = get_safe_fallback_query(tree_sitter_lang, original_query)
        if fallback_query != "(ERROR) @error":
            try:
                query = language.query(fallback_query)
                validation_msg = f"Used fallback query '{fallback_query}' due to parse error: {str(e)}"
            except Exception as e2:
                raise ValueError(f"Both original query '{original_query}' and fallback query '{fallback_query}' failed for language '{tree_sitter_lang}': {str(e2)}. If tree-sitter is not installed, try code.search instead.")
        else:
            raise ValueError(f"Invalid query '{corrected_query}' for language '{tree_sitter_lang}': {str(e)}. If tree-sitter is not installed, try code.search instead.")
    
    allowed_captures = set(payload.get("captures") or [])

    nodes = []
    for node, capture_name in query.captures(tree.root_node):
        if allowed_captures and capture_name not in allowed_captures:
            continue
        text = source[node.start_byte : node.end_byte]
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

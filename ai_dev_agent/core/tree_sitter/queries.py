"""Shared tree-sitter query templates and helpers."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

LANGUAGE_ALIASES = {
    "c++": "cpp",
    "c#": "csharp",
    "c-sharp": "csharp",
    "c_sharp": "csharp",
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
}

# Canonical AST query templates migrated from tool_strategy so tools can
# consistently ask for common structural patterns.
AST_QUERY_TEMPLATES: Dict[str, Dict[str, str]] = {
    "cpp": {
        "find_classes": "(class_declaration name: (type_identifier) @name)",
        "find_functions": "(function_definition declarator: (function_declarator) @func)",
        "find_methods": "(field_declaration declarator: (function_declarator) @method)",
        "find_constructors": "(function_definition declarator: (function_declarator declarator: (destructor_name) @ctor))",
        "find_allocations": "(new_expression type: (_) @type)",
        "find_templates": "(template_declaration) @template",
        "find_namespaces": "(namespace_definition name: (identifier) @namespace)",
    },
    "c": {
        "find_functions": "(function_definition declarator: (function_declarator) @func)",
        "find_structs": "(struct_specifier name: (type_identifier) @name)",
        "find_typedefs": "(type_definition declarator: (type_identifier) @name)",
        "find_macros": "(preproc_def name: (identifier) @macro)",
        "find_enums": "(enum_specifier name: (type_identifier) @name)",
    },
    "python": {
        "find_classes": "(class_definition name: (identifier) @name)",
        "find_functions": "(function_definition name: (identifier) @name)",
        "find_methods": "(function_definition name: (identifier) @method)",
        "find_decorators": "(decorator) @decorator",
        "find_imports": "(import_statement) @import",
        "find_async": "(function_definition (async) @async)",
    },
    "javascript": {
        "find_functions": "(function_declaration name: (identifier) @name)",
        "find_arrow_functions": "(arrow_function) @arrow",
        "find_classes": "(class_declaration name: (identifier) @name)",
        "find_methods": "(method_definition key: (property_identifier) @name)",
        "find_imports": "(import_statement) @import",
        "find_exports": "(export_statement) @export",
        "find_async": "(function_declaration (async) @async)",
    },
    "typescript": {
        "find_interfaces": "(interface_declaration name: (type_identifier) @name)",
        "find_types": "(type_alias_declaration name: (type_identifier) @name)",
        "find_enums": "(enum_declaration name: (identifier) @name)",
        "find_functions": "(function_declaration name: (identifier) @name)",
        "find_classes": "(class_declaration name: (type_identifier) @name)",
        "find_generics": "(type_parameters) @generics",
    },
    "java": {
        "find_classes": "(class_declaration name: (identifier) @name)",
        "find_interfaces": "(interface_declaration name: (identifier) @name)",
        "find_methods": "(method_declaration name: (identifier) @name)",
        "find_fields": "(field_declaration declarator: (variable_declarator name: (identifier) @name))",
        "find_annotations": "(annotation) @annotation",
        "find_enums": "(enum_declaration name: (identifier) @name)",
    },
    "go": {
        "find_functions": "(function_declaration name: (identifier) @name)",
        "find_methods": "(method_declaration name: (field_identifier) @name)",
        "find_structs": "(type_spec name: (type_identifier) @name type: (struct_type))",
        "find_interfaces": "(type_spec name: (type_identifier) @name type: (interface_type))",
        "find_imports": "(import_declaration) @import",
    },
    "rust": {
        "find_functions": "(function_item name: (identifier) @name)",
        "find_structs": "(struct_item name: (type_identifier) @name)",
        "find_enums": "(enum_item name: (type_identifier) @name)",
        "find_traits": "(trait_item name: (type_identifier) @name)",
        "find_impls": "(impl_item) @impl",
        "find_macros": "(macro_invocation macro: (identifier) @name)",
    },
    "ruby": {
        "find_classes": "(class name: (constant) @name)",
        "find_modules": "(module name: (constant) @name)",
        "find_methods": "(method name: (identifier) @name)",
        "find_blocks": "(block) @block",
        "find_requires": "(call method: (identifier) @req (#eq? @req \"require\"))",
    },
    "csharp": {
        "find_classes": "(class_declaration name: (identifier) @name)",
        "find_interfaces": "(interface_declaration name: (identifier) @name)",
        "find_methods": "(method_declaration name: (identifier) @name)",
        "find_properties": "(property_declaration name: (identifier) @name)",
        "find_namespaces": "(namespace_declaration name: (_) @name)",
    },
    "php": {
        "find_classes": "(class_declaration name: (name) @name)",
        "find_functions": "(function_definition name: (name) @name)",
        "find_methods": "(method_declaration name: (name) @name)",
        "find_traits": "(trait_declaration name: (name) @name)",
        "find_namespaces": "(namespace_definition name: (namespace_name) @name)",
    },
}

# Outline-oriented queries used by the project summariser. These provide a
# shared set of building blocks so summary extraction can stay declarative.
SUMMARY_QUERY_TEMPLATES: Dict[str, Dict[str, str]] = {
    "python": {
        "module": "(module) @module",
        "classes": "(class_definition name: (identifier) @name)",
        "functions": "(function_definition name: (identifier) @name)",
        "assignments": "(assignment left: (_) @name)",
    },
    "typescript": {
        "imports": "(import_declaration) @import",
        "exports": "(export_statement) @export",
        "classes": "(class_declaration name: (type_identifier) @name)",
        "functions": "(function_declaration name: (identifier) @name)",
        "interfaces": "(interface_declaration name: (type_identifier) @name)",
    },
    "javascript": {
        "imports": "(import_statement) @import",
        "exports": "(export_statement) @export",
        "classes": "(class_declaration name: (identifier) @name)",
        "functions": "(function_declaration name: (identifier) @name)",
    },
    "tsx": {
        "components": "(jsx_element (jsx_identifier) @component)",
        "imports": "(import_declaration) @import",
        "functions": "(function_declaration name: (identifier) @name)",
    },
    "go": {
        "imports": "(import_declaration) @import",
        "types": "(type_spec name: (type_identifier) @name)",
        "functions": "(function_declaration name: (identifier) @name)",
    },
    "java": {
        "classes": "(class_declaration name: (identifier) @name)",
        "interfaces": "(interface_declaration name: (identifier) @name)",
        "methods": "(method_declaration name: (identifier) @name)",
    },
    "cpp": {
        "classes": "(class_declaration name: (type_identifier) @name)",
        "functions": "(function_definition declarator: (function_declarator) @func)",
        "namespaces": "(namespace_definition name: (identifier) @namespace)",
    },
    "c": {
        "functions": "(function_definition declarator: (function_declarator) @func)",
        "structs": "(struct_specifier name: (type_identifier) @name)",
    },
    "rust": {
        "modules": "(mod_item name: (identifier) @name)",
        "functions": "(function_item name: (identifier) @name)",
        "structs": "(struct_item name: (type_identifier) @name)",
    },
}


def normalise_language(language: str) -> str:
    return LANGUAGE_ALIASES.get(language.lower(), language.lower())


def get_ast_query(language: str, name: str) -> Optional[str]:
    """Return a template by name if defined for the language."""

    lang = normalise_language(language)
    return AST_QUERY_TEMPLATES.get(lang, {}).get(name)


def iter_ast_queries(language: str) -> Iterable[tuple[str, str]]:
    """Yield (name, query) pairs for the given language."""

    lang = normalise_language(language)
    return AST_QUERY_TEMPLATES.get(lang, {}).items()


def get_summary_queries(language: str) -> Dict[str, str]:
    """Return summary query definitions for a language (empty if unsupported)."""

    lang = normalise_language(language)
    return SUMMARY_QUERY_TEMPLATES.get(lang, {})


def build_capture_query(node_type: str, capture: str = "node") -> str:
    """Small helper to create one-line capture queries."""

    return f"({node_type}) @{capture}"


def build_field_capture_query(node_type: str, field: str, capture: str = "node") -> str:
    """Helper for queries targeting a specific named field."""

    return f"({node_type} {field}: (_) @{capture})"


__all__ = [
    "AST_QUERY_TEMPLATES",
    "SUMMARY_QUERY_TEMPLATES",
    "LANGUAGE_ALIASES",
    "build_capture_query",
    "build_field_capture_query",
    "get_ast_query",
    "get_summary_queries",
    "iter_ast_queries",
    "normalise_language",
]

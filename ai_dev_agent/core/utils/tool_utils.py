"""Utilities for working with tool identifiers and signatures."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .constants import (
    TOOL_ALIAS_TO_CANONICAL,
    TOOL_CANONICAL_CATEGORIES,
    TOOL_DISPLAY_NAMES,
)

FILE_READ_TOOLS = {
    alias
    for alias, canonical in TOOL_ALIAS_TO_CANONICAL.items()
    if TOOL_CANONICAL_CATEGORIES.get(canonical) == "file_read"
}

SEARCH_TOOLS = {
    alias
    for alias, canonical in TOOL_ALIAS_TO_CANONICAL.items()
    if TOOL_CANONICAL_CATEGORIES.get(canonical) == "search"
}


def canonical_tool_name(tool_name: Optional[str]) -> str:
    """Return a stable canonical name for a tool alias."""
    if not tool_name:
        return "generic"
    return TOOL_ALIAS_TO_CANONICAL.get(tool_name, tool_name)


def tool_category(tool_name: Optional[str]) -> str:
    """Return the logical category for a tool alias."""
    if not tool_name:
        return "generic"
    canonical = canonical_tool_name(tool_name)
    return TOOL_CANONICAL_CATEGORIES.get(canonical, canonical)


def display_tool_name(tool_name: Optional[str]) -> str:
    """Return the user-facing display name for a tool."""
    if not tool_name:
        return "generic"
    canonical = canonical_tool_name(tool_name)
    return TOOL_DISPLAY_NAMES.get(canonical, canonical)


def build_tool_signature(tool_name: str, arguments: Mapping[str, Any] | None) -> str:
    """Build a deterministic signature for a tool call."""
    import json

    args = arguments or {}

    if tool_name in FILE_READ_TOOLS:
        path_value = args.get("path")
        if path_value:
            return f"{tool_name}:{path_value}"
        paths = args.get("paths") or []
        if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
            try:
                key = ",".join(sorted(str(p) for p in paths if p))
            except Exception:
                key = str(list(paths))
        else:
            key = str(paths)
        return f"{tool_name}:{key}"

    if tool_name in SEARCH_TOOLS:
        return f"{tool_name}:{args.get('query', '')}"

    try:
        args_str = json.dumps(args, sort_keys=True, default=str)
        hashed = hash(args_str)
    except (TypeError, ValueError):
        hashed = hash(str(args))
    return f"{tool_name}:{hashed}"


def tool_signature(tool_call: Any) -> str:
    """Build a signature from an object that looks like a tool call."""
    name = getattr(tool_call, "name", "unknown")
    arguments = getattr(tool_call, "arguments", {}) or {}
    return build_tool_signature(name, arguments)


__all__ = [
    "canonical_tool_name",
    "tool_category",
    "display_tool_name",
    "build_tool_signature",
    "tool_signature",
    "FILE_READ_TOOLS",
    "SEARCH_TOOLS",
]

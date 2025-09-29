"""Text processing utilities."""
from __future__ import annotations

import re
from typing import Any, List, Optional

_FUNCTION_PATTERNS = [
    re.compile(r"\basync\s+def\s+[A-Za-z_][\w]*\s*\([^)]*\)"),
    re.compile(r"\bdef\s+[A-Za-z_][\w]*\s*\([^)]*\)"),
    re.compile(r"\bclass\s+[A-Za-z_][\w]*"),
    re.compile(r"\bfunction\s+[A-Za-z_][\w]*\s*\([^)]*\)"),
    re.compile(r"\bfn\s+[A-Za-z_][\w]*\s*\([^)]*\)?"),
    re.compile(r"\b[A-Za-z_][\w]*\s*=\s*function\s*\([^)]*\)"),
    re.compile(r"\b[A-Za-z_][\w]*\s*=\s*\([^)]*\)\s*=>"),
    re.compile(
        r"\b(?:public|private|protected|static|final|virtual|override|async)\s+"
        r"[A-Za-z0-9_<>,\[\]\s]*\b[A-Za-z_][\w]*\s*\([^)]*\)"
    ),
]


def truncate_text(text: str, max_length: int = 160) -> str:
    """Truncate text to specified length."""
    text = text.strip()
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def safe_int(value: Any) -> Optional[int]:
    """Safely convert value to integer."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def extract_function_signatures(text: str) -> List[str]:
    """Extract function signatures from code text."""
    snippet = text.strip()
    if not snippet:
        return []

    signatures: List[str] = []
    seen: set[str] = set()

    for pattern in _FUNCTION_PATTERNS:
        for match in pattern.finditer(snippet):
            cleaned = truncate_text(match.group(0).rstrip("{").strip(), 160)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                signatures.append(cleaned)

    return signatures


def first_displayable_line(text: str) -> str:
    """Get first meaningful line from text."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("Full output saved"):
            return stripped
    return ""


__all__ = [
    "truncate_text",
    "safe_int",
    "extract_function_signatures",
    "first_displayable_line",
]

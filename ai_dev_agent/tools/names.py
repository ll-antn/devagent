"""Central definitions for canonical tool identifiers."""
from __future__ import annotations

READ = "read"
WRITE = "write"
RUN = "run"
FIND = "find"
GREP = "grep"
SYMBOLS = "symbols"
PARSE_PATCH = "parse_patch"

ALL_TOOLS = (
    READ,
    WRITE,
    RUN,
    FIND,
    GREP,
    SYMBOLS,
    PARSE_PATCH,
)

__all__ = [
    "READ",
    "WRITE",
    "RUN",
    "FIND",
    "GREP",
    "SYMBOLS",
    "PARSE_PATCH",
    "ALL_TOOLS",
]

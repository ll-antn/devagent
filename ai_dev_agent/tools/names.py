"""Central definitions for canonical tool identifiers."""
from __future__ import annotations

READ = "read"
WRITE = "write"
RUN = "run"
FIND = "find"
GREP = "grep"
SYMBOLS = "symbols"
PARSE_PATCH = "parse_patch"
GITCODE_PR = "gitcode_pr"

ALL_TOOLS = (
    READ,
    WRITE,
    RUN,
    FIND,
    GREP,
    SYMBOLS,
    PARSE_PATCH,
    GITCODE_PR,
)

__all__ = [
    "READ",
    "WRITE",
    "RUN",
    "FIND",
    "GREP",
    "SYMBOLS",
    "PARSE_PATCH",
    "GITCODE_PR",
    "ALL_TOOLS",
]

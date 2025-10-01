"""Constants used throughout the application."""

# Repository traversal defaults
DEFAULT_IGNORED_REPO_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "dist",
        "build",
        "vendor",
    }
)

# Tool execution limits
MAX_HISTORY_ENTRIES = 50
MIN_TOOL_OUTPUT_CHARS = 256
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 4_000
MAX_METRICS_ENTRIES = 500

# Conversation context defaults
DEFAULT_MAX_CONTEXT_TOKENS = 100_000
DEFAULT_RESPONSE_HEADROOM = 2_000
DEFAULT_MAX_TOOL_MESSAGES = 10
DEFAULT_KEEP_LAST_ASSISTANT = 4

# Legacy tool identifiers that should no longer surface externally

# Canonical tool definitions keep aliases and display names consistent
TOOL_CANONICAL_SPECS = {
    "fs.read": {
        "display_name": "fs.read",
        "aliases": ("fs_read",),
        "category": "file_read",
    },
    "fs.write_patch": {
        "display_name": "fs.write_patch",
        "aliases": ("fs_write_patch",),
        "category": "command",
    },
    "code.search": {
        "display_name": "code.search",
        "aliases": ("code_search",),
        "category": "search",
    },
    "ast.query": {
        "display_name": "ast.query",
        "aliases": ("ast_query",),
        "category": "ast",
    },
    "symbols.find": {
        "display_name": "symbols.find",
        "aliases": ("symbols_find",),
        "category": "symbols",
    },
    "symbols.index": {
        "display_name": "symbols.index",
        "aliases": ("symbols_index",),
        "category": "symbols",
    },
    "exec": {
        "display_name": "exec",
        "aliases": ("execute",),
        "category": "command",
    },
}

TOOL_ALIAS_TO_CANONICAL = {}
for canonical_name, spec in TOOL_CANONICAL_SPECS.items():
    TOOL_ALIAS_TO_CANONICAL[canonical_name] = canonical_name
    for alias in spec["aliases"]:
        TOOL_ALIAS_TO_CANONICAL[alias] = canonical_name

TOOL_DISPLAY_NAMES = {
    canonical_name: spec["display_name"]
    for canonical_name, spec in TOOL_CANONICAL_SPECS.items()
}

TOOL_CANONICAL_CATEGORIES = {
    canonical_name: spec["category"]
    for canonical_name, spec in TOOL_CANONICAL_SPECS.items()
}


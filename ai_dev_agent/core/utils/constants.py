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
RUN_STDOUT_TAIL_CHARS = 16_000
RUN_STDERR_TAIL_CHARS = 4_000
MAX_METRICS_ENTRIES = 500

# Conversation context defaults
DEFAULT_MAX_CONTEXT_TOKENS = 100_000
DEFAULT_RESPONSE_HEADROOM = 2_000
DEFAULT_MAX_TOOL_MESSAGES = 10
DEFAULT_KEEP_LAST_ASSISTANT = 4

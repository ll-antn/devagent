"""CLI package exposing the DevAgent command entry points."""
from __future__ import annotations

from .commands import (
    NaturalLanguageGroup,
    cli,
    main,
    query,
    shell,
    # Only add other PUBLIC command functions here
)
from .router import IntentDecision, IntentRouter, IntentRoutingError
from .utils import _get_llm_client, _infer_task_files, _update_task_state
from ai_dev_agent.core.utils.config import Settings, load_settings

__all__ = [
    "NaturalLanguageGroup",
    "IntentDecision",
    "IntentRouter",
    "IntentRoutingError",
    "cli",
    "main",
    "query",
    "shell",
    "_get_llm_client",
    "_infer_task_files",
    "_update_task_state",
    "load_settings",
    "Settings",
]

"""Tool registry and validation helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from jsonschema import Draft7Validator, ValidationError

from ai_dev_agent.core.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ToolContext:
    """Runtime context passed to tool handlers."""

    repo_root: Path
    settings: Any
    sandbox: Any
    devagent_config: Any = None
    metrics_collector: Any = None
    extra: Dict[str, Any] | None = None


@dataclass
class ToolSpec:
    """Metadata for a registered tool."""

    name: str
    handler: Callable[[Mapping[str, Any], ToolContext], Mapping[str, Any]]
    request_schema_path: Path
    response_schema_path: Path
    description: str = ""


class ToolRegistry:
    """Registry that manages tool specifications and validation."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            LOGGER.debug("Overwriting existing tool registration for %s", spec.name)
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def available(self) -> Iterable[str]:
        return sorted(self._tools.keys())

    def invoke(self, name: str, payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
        spec = self.get(name)
        validator = _load_validator(spec.request_schema_path)
        errors = sorted(validator.iter_errors(payload), key=lambda exc: exc.path)
        if errors:
            first = errors[0]
            raise ValueError(f"Invalid input for {name}: {first.message}")
        result = spec.handler(payload, context)
        validator_out = _load_validator(spec.response_schema_path)
        errors_out = sorted(validator_out.iter_errors(result), key=lambda exc: exc.path)
        if errors_out:
            first_out = errors_out[0]
            raise ValueError(f"Tool {name} returned invalid response: {first_out.message}")
        return result


@lru_cache(maxsize=64)
def _load_validator(schema_path: Path) -> Draft7Validator:
    with schema_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Draft7Validator(data)


# Global registry instance -------------------------------------------------

registry = ToolRegistry()


__all__ = ["ToolContext", "ToolSpec", "ToolRegistry", "registry"]

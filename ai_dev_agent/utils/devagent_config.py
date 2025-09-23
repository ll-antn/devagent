"""Reader for devagent.yaml (CI/build/test/gates configuration).

This intentionally supports a minimal subset of the example schema described by
the user: build/test/lint/type/format/coverage commands and gate thresholds.
If the YAML file is missing, callers should fall back to Settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


@dataclass
class DevAgentConfig:
    build_cmd: Optional[str] = None
    test_cmd: Optional[str] = None
    lint_cmd: Optional[str] = None
    type_cmd: Optional[str] = None
    format_cmd: Optional[str] = None
    coverage_cmd: Optional[str] = None
    threshold_diff: Optional[float] = None
    threshold_project: Optional[float] = None
    diff_limit_lines: Optional[int] = None
    diff_limit_files: Optional[int] = None


def load_devagent_yaml(path: Path | None = None) -> Optional[DevAgentConfig]:
    """Load devagent.yaml into a DevAgentConfig or return None if unavailable."""
    candidate = path or (Path.cwd() / "devagent.yaml")
    if not candidate.is_file():
        return None
    if yaml is None:
        return None
    try:
        data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except Exception:  # pragma: no cover - best effort
        return None
    if not isinstance(data, dict):
        return None

    cfg = DevAgentConfig()
    build = data.get("build") or {}
    tests = data.get("tests") or {}
    coverage = data.get("coverage") or {}
    lint = data.get("lint") or {}
    types = data.get("types") or {}
    fmt = data.get("format") or {}
    gates = data.get("gates") or []

    def g(name: str, d: Dict[str, Any]) -> Optional[str]:
        v = d.get("cmd")
        return str(v) if v else None

    cfg.build_cmd = g("build", build)
    cfg.test_cmd = g("tests", tests)
    cfg.lint_cmd = g("lint", lint)
    cfg.type_cmd = g("types", types)
    cfg.format_cmd = g("format", fmt)
    cfg.coverage_cmd = g("coverage", coverage)

    # Thresholds
    if isinstance(coverage.get("threshold"), dict):
        cfg.threshold_diff = _as_float(coverage["threshold"].get("diff"))
        cfg.threshold_project = _as_float(coverage["threshold"].get("project"))

    # Gates diff limits (if present)
    for gate in gates if isinstance(gates, list) else []:
        if isinstance(gate, dict) and gate.get("name") == "diff.size":
            cfg.diff_limit_lines = _as_int(gate.get("lte_lines"))
            cfg.diff_limit_files = _as_int(gate.get("lte_files"))
            break

    return cfg


def _as_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


__all__ = ["DevAgentConfig", "load_devagent_yaml"]


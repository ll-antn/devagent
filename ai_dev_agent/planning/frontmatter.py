"""Plan frontmatter (YAML) parsing and minimal validation utilities.

This module supports Markdown plans with a YAML frontmatter followed by
human-readable body text. Frontmatter is delimited by leading and trailing
"---" lines at the top of the file.

Schema (minimal, informal):
  schema: devagent/v1/plan
  mode: <string>
  goal: <string>
  targets: [ { path: <string> }, ... ]  # optional
  metrics: { ... }                      # optional
  budget:  { steps: <int>, tokens: <int> }  # optional
  strategy: { stuck_after: <int> }      # optional
  steps:                                 # required
    - id: <string>
      tool: <string>
      args: { ... }
      produces: <string> | [<string>]
      gates: [ <string>, ... ]
      optional: <bool>

Validation is best-effort and avoids external dependencies (e.g. jsonschema).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


FRONTMATTER_DELIM = "---"


@dataclass
class PlanStep:
    id: str
    tool: str
    args: Dict[str, Any]
    gates: List[str]
    produces: List[str]
    optional: bool = False


@dataclass
class FrontmatterPlan:
    schema: str
    mode: str
    goal: str
    targets: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    budget: Dict[str, Any]
    strategy: Dict[str, Any]
    steps: List[PlanStep]
    body: str


def load_markdown_with_frontmatter(path: Path) -> Tuple[Dict[str, Any], str]:
    """Extract YAML frontmatter and body text from a Markdown file.

    Returns (frontmatter_dict, body_text). Raises ValueError on parse issues.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_DELIM:
        raise ValueError("Frontmatter not found: missing leading '---' delimiter")
    # Find closing delimiter
    end_index = None
    for i in range(1, len(lines)):
        if lines[i].strip() == FRONTMATTER_DELIM:
            end_index = i
            break
    if end_index is None:
        raise ValueError("Frontmatter not closed: missing trailing '---'")

    fm_text = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :]).lstrip("\n")

    if yaml is None:
        raise ValueError("PyYAML is required to parse frontmatter. Install pyyaml.")
    try:
        fm_dict = yaml.safe_load(fm_text) or {}
    except Exception as exc:  # pragma: no cover - delegate parse errors
        raise ValueError(f"Failed to parse YAML frontmatter: {exc}") from exc
    if not isinstance(fm_dict, dict):
        raise ValueError("Frontmatter must be a YAML mapping (object)")
    return fm_dict, body


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def validate_frontmatter_plan(fm: Dict[str, Any]) -> List[str]:
    """Return a list of validation error messages (empty means OK)."""
    errors: List[str] = []
    if fm.get("schema") not in {"devagent/v1/plan"}:
        errors.append("schema must be 'devagent/v1/plan'")
    for key in ("mode", "goal", "steps"):
        if key not in fm:
            errors.append(f"missing required field: {key}")
    steps = fm.get("steps")
    if isinstance(steps, list):
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"steps[{idx}] must be an object")
                continue
            if not step.get("id"):
                errors.append(f"steps[{idx}].id is required")
            if not step.get("tool"):
                errors.append(f"steps[{idx}].tool is required")
            gates = step.get("gates", [])
            if gates is not None and not isinstance(gates, list):
                errors.append(f"steps[{idx}].gates must be a list if provided")
    else:
        errors.append("steps must be a list")
    return errors


def parse_frontmatter_plan(path: Path) -> FrontmatterPlan:
    """Load and minimally validate a frontmatter plan file."""
    fm, body = load_markdown_with_frontmatter(path)
    errors = validate_frontmatter_plan(fm)
    if errors:
        raise ValueError("; ".join(errors))

    steps: List[PlanStep] = []
    for raw in fm.get("steps", []) or []:
        steps.append(
            PlanStep(
                id=str(raw.get("id")),
                tool=str(raw.get("tool")),
                args=dict(raw.get("args") or {}),
                gates=[str(g) for g in (raw.get("gates") or [])],
                produces=_as_list(raw.get("produces")),
                optional=bool(raw.get("optional", False)),
            )
        )

    return FrontmatterPlan(
        schema=str(fm.get("schema")),
        mode=str(fm.get("mode")),
        goal=str(fm.get("goal")),
        targets=list(fm.get("targets") or []),
        metrics=dict(fm.get("metrics") or {}),
        budget=dict(fm.get("budget") or {}),
        strategy=dict(fm.get("strategy") or {}),
        steps=steps,
        body=body,
    )


def write_frontmatter_plan(
    target: Path,
    *,
    mode: str,
    goal: str,
    targets: Optional[List[Dict[str, Any]]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    budget: Optional[Dict[str, Any]] = None,
    strategy: Optional[Dict[str, Any]] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
    body_text: str = "",
) -> Path:
    """Create a Markdown file with YAML frontmatter matching the minimal schema."""
    targets = targets or []
    metrics = metrics or {}
    budget = budget or {}
    strategy = strategy or {}
    steps = steps or []

    target = target if target.is_absolute() else Path.cwd() / target
    target.parent.mkdir(parents=True, exist_ok=True)

    if yaml is None:
        raise RuntimeError("PyYAML required to write frontmatter plan. Install pyyaml.")

    fm = {
        "schema": "devagent/v1/plan",
        "mode": mode,
        "goal": goal,
        "targets": targets,
        "metrics": metrics,
        "budget": budget,
        "strategy": strategy,
        "steps": steps,
    }
    fm_text = yaml.safe_dump(fm, sort_keys=False).strip()
    body = body_text.strip()
    contents = f"{FRONTMATTER_DELIM}\n{fm_text}\n{FRONTMATTER_DELIM}\n\n{body}\n"
    target.write_text(contents, encoding="utf-8")
    return target


__all__ = [
    "FrontmatterPlan",
    "PlanStep",
    "parse_frontmatter_plan",
    "write_frontmatter_plan",
    "load_markdown_with_frontmatter",
    "validate_frontmatter_plan",
]


"""Task prioritization utilities (RICE scoring)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class PriorityScore:
    task_id: str
    rice: float
    reach: float
    impact: float
    confidence: float
    effort: float


CATEGORY_DEFAULTS = {
    "design": {"impact": 4.0, "effort": 2.0},
    "implementation": {"impact": 5.0, "effort": 4.0},
    "testing": {"impact": 4.5, "effort": 3.0},
    "documentation": {"impact": 3.0, "effort": 1.5},
}


def _normalize_category(category: str | None) -> str:
    if not category:
        return "implementation"
    return category.strip().lower()


def compute_priority(task: "PlanTask") -> PriorityScore:
    """Compute a RICE score for a task using heuristics."""
    category = _normalize_category(getattr(task, "category", None))
    defaults = CATEGORY_DEFAULTS.get(category, CATEGORY_DEFAULTS["implementation"])
    reach = getattr(task, "reach", None) or 1.0
    impact = getattr(task, "impact", None) or defaults["impact"]
    effort = getattr(task, "effort", None) or defaults["effort"]
    confidence = getattr(task, "confidence", None) or 0.7
    rice = (reach * impact * confidence) / max(effort, 0.5)
    return PriorityScore(
        task_id=task.identifier,
        rice=round(rice, 2),
        reach=reach,
        impact=impact,
        confidence=confidence,
        effort=effort,
    )


def rank_tasks(tasks: Iterable["PlanTask"]) -> List[PriorityScore]:
    scores = [compute_priority(task) for task in tasks]
    scores.sort(key=lambda score: score.rice, reverse=True)
    return scores


__all__ = ["PriorityScore", "compute_priority", "rank_tasks"]

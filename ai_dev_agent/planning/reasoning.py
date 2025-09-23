"""Reasoning helpers for multi-step task execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _now_ts() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.utcnow().strftime(ISO_FORMAT)


@dataclass
class ToolUse:
    """Describe a tool invocation that supports a reasoning step."""

    name: str
    command: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        payload: Dict[str, str] = {"name": self.name}
        if self.command:
            payload["command"] = self.command
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass
class ReasoningStep:
    """Represent a single step in a multi-step reasoning trace."""

    identifier: str
    title: str
    detail: str
    status: str = "pending"
    result: Optional[str] = None
    started_at: str = field(default_factory=_now_ts)
    completed_at: Optional[str] = None
    tool: Optional[ToolUse] = None

    def mark_in_progress(self) -> None:
        self.status = "in_progress"
        self.started_at = self.started_at or _now_ts()

    def complete(self, result: Optional[str] = None) -> None:
        self.status = "completed"
        self.result = result
        self.completed_at = _now_ts()

    def fail(self, result: Optional[str] = None) -> None:
        self.status = "failed"
        self.result = result
        self.completed_at = _now_ts()

    def to_dict(self) -> Dict[str, Optional[str]]:
        payload: Dict[str, Optional[str]] = {
            "id": self.identifier,
            "title": self.title,
            "detail": self.detail,
            "status": self.status,
            "result": self.result,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if self.tool:
            payload["tool"] = self.tool.to_dict()
        return payload


@dataclass
class PlanAdjustment:
    """Describe an on-the-fly adjustment proposed during execution."""

    summary: str
    detail: str
    created_at: str = field(default_factory=_now_ts)

    def to_dict(self) -> Dict[str, str]:
        return {
            "summary": self.summary,
            "detail": self.detail,
            "created_at": self.created_at,
        }


@dataclass
class TaskReasoning:
    """Aggregate reasoning steps and plan adjustments for a task."""

    task_id: str
    goal: Optional[str] = None
    task_title: Optional[str] = None
    steps: List[ReasoningStep] = field(default_factory=list)
    adjustments: List[PlanAdjustment] = field(default_factory=list)

    def start_step(
        self,
        title: str,
        detail: str,
        tool: Optional[ToolUse] = None,
    ) -> ReasoningStep:
        identifier = f"S{len(self.steps) + 1}"
        step = ReasoningStep(
            identifier=identifier,
            title=title,
            detail=detail,
            status="in_progress",
            tool=tool,
        )
        self.steps.append(step)
        return step

    def record_adjustment(self, summary: str, detail: str) -> PlanAdjustment:
        adjustment = PlanAdjustment(summary=summary, detail=detail)
        self.adjustments.append(adjustment)
        return adjustment

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "task_title": self.task_title,
            "steps": [step.to_dict() for step in self.steps],
            "adjustments": [adj.to_dict() for adj in self.adjustments],
        }

    def apply_to_task(self, task: Dict[str, object]) -> None:
        task["reasoning_log"] = [step.to_dict() for step in self.steps]
        if self.adjustments:
            task["plan_adjustments"] = [adj.to_dict() for adj in self.adjustments]

    def merge_into_plan(self, plan: Dict[str, object]) -> None:
        if self.adjustments:
            adjustments = plan.setdefault("adjustments", [])
            seen = {
                (entry.get("summary"), entry.get("detail"))
                for entry in adjustments
                if isinstance(entry, dict)
            }
            for adj in self.adjustments:
                data = adj.to_dict()
                key = (data["summary"], data["detail"])
                if key not in seen:
                    adjustments.append(data)
                    seen.add(key)


__all__ = [
    "TaskReasoning",
    "ReasoningStep",
    "ToolUse",
    "PlanAdjustment",
]

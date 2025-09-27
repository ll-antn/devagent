"""Track iterative efficiency across ReAct steps."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StepDelta:
    """Represents delta metrics captured after a step."""

    duration: float
    tokens: float
    improved: bool


@dataclass
class IterationTracker:
    """Tracks wall time, token usage, and improvement streaks."""

    started_at: float = field(default_factory=time.perf_counter)
    steps: List[StepDelta] = field(default_factory=list)

    def record_step(
        self,
        *,
        duration: float,
        tokens: float | None = None,
        improved: bool | None = None,
    ) -> None:
        self.steps.append(
            StepDelta(
                duration=max(duration, 0.0),
                tokens=float(tokens or 0.0),
                improved=bool(improved),
            )
        )

    @property
    def wall_time(self) -> float:
        if not self.steps:
            return 0.0
        return sum(step.duration for step in self.steps)

    @property
    def tokens_cost(self) -> float:
        return sum(step.tokens for step in self.steps)

    @property
    def steps_to_green(self) -> int:
        return len(self.steps)

    def improvement_streak(self) -> int:
        streak = 0
        for step in reversed(self.steps):
            if step.improved:
                streak += 1
            else:
                break
        return streak


__all__ = ["IterationTracker", "StepDelta"]

"""Planning module for generating work breakdown structures."""
from __future__ import annotations

import json
import re
import time
from dataclasses import InitVar, dataclass, field
from typing import List, Optional, Sequence

from ai_dev_agent.providers.llm import (
    LLMClient,
    LLMConnectionError,
    LLMError,
    LLMTimeoutError,
    Message,
)
from ai_dev_agent.core.utils.logger import get_logger

LOGGER = get_logger(__name__)

SYSTEM_PROMPT = """You are a planning assistant. Generate appropriate plans based on task complexity.
Be practical and thorough. Simple tasks need 2-5 steps, complex tasks may need 6-15 steps.
Always respond with a JSON object containing the plan and nothing else."""

USER_TEMPLATE = """Create a plan for: {goal}

Respond with JSON:
{{
  "summary": "One-line summary",
  "tasks": [
    {{
      "step_number": 1,
      "title": "Short title",
      "description": "What to do (1-2 sentences max)",
      "dependencies": []
    }}
  ]
}}

Guidelines:
- Match plan complexity to task complexity:
  * Simple tasks (basic queries, single file changes): 2-5 steps
  * Medium tasks (feature additions, refactoring): 6-10 steps  
  * Complex tasks (architecture changes, multi-component work): 8-15 steps
- Keep descriptions clear and practical
- Include all necessary steps for thorough completion
- Break down complex work into logical, manageable chunks
- Ensure dependencies are properly sequenced
"""

JSON_PATTERN = re.compile(r"```json\s*(?P<json>{.*?})\s*```", re.DOTALL)


@dataclass
class PlanTask:
    step_number: int | None = None
    title: str = "Untitled"
    description: str = ""
    status: str = "pending"
    dependencies: List[int] = field(default_factory=list)
    category: str = "implementation"
    effort: int | None = None
    reach: int | None = None
    impact: int | None = None
    confidence: float | None = None
    deliverables: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    identifier: InitVar[str | None] = None

    _identifier: str = field(init=False)

    def __post_init__(self, identifier: str | None) -> None:
        candidate = (identifier or "").strip()
        if self.step_number is None:
            if candidate.startswith("T") and candidate[1:].isdigit():
                self.step_number = int(candidate[1:])
            else:
                self.step_number = 1
        if not candidate:
            candidate = f"T{self.step_number}"
        self._identifier = candidate
        self.identifier = candidate

        normalized_deps: List[int] = []
        for dep in self.dependencies:
            try:
                normalized_deps.append(int(dep))
            except (TypeError, ValueError):
                continue
        self.dependencies = normalized_deps

        self.deliverables = [str(item) for item in (self.deliverables or [])]
        self.commands = [str(item) for item in (self.commands or [])]

    def to_dict(self) -> dict:
        data = {
            "id": self._identifier,
            "step_number": self.step_number,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "dependencies": self.dependencies,
        }
        optional_fields = {
            "category": self.category,
            "effort": self.effort,
            "reach": self.reach,
            "impact": self.impact,
            "confidence": self.confidence,
            "deliverables": self.deliverables,
            "commands": self.commands,
        }
        for key, value in optional_fields.items():
            if value not in (None, [], {}):
                data[key] = value
        return data

@dataclass
class PlanResult:
    goal: str
    summary: str
    tasks: List[PlanTask]
    raw_response: str
    fallback_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "summary": self.summary,
            "tasks": [task.to_dict() for task in self.tasks],
            "raw_response": self.raw_response,
            "status": "planned",
            "fallback_reason": self.fallback_reason,
        }


class Planner:
    """Generates structured plans using an LLM provider."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def generate(self, goal: str) -> PlanResult:
        LOGGER.info("Requesting plan from LLM for goal: %s", goal)
        messages: Sequence[Message] = (
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=USER_TEMPLATE.format(goal=goal.strip())),
        )
        start_time = time.time()
        next_heartbeat = start_time + 10.0
        try:
            while True:
                try:
                    response_text = self.client.complete(messages, temperature=0.1)
                    break
                except LLMTimeoutError:
                    now = time.time()
                    LOGGER.warning("Planner request timed out, retrying for goal: %s", goal)
                    if now >= next_heartbeat:
                        elapsed = now - start_time
                        LOGGER.info("Still waiting for planner response (%.1fs elapsed)", elapsed)
                        next_heartbeat = now + 10.0
                except LLMConnectionError as exc:
                    now = time.time()
                    LOGGER.warning("Planner connection issue (%s). Retrying goal: %s", exc, goal)
                    if now >= next_heartbeat:
                        elapsed = now - start_time
                        LOGGER.info("Still waiting for planner response (%.1fs elapsed)", elapsed)
                        next_heartbeat = now + 10.0
        except LLMError as exc:
            LOGGER.warning("LLM planning failed, generating fallback plan: %s", exc)
            tasks = self._create_generic_fallback(goal)
            summary = f"Fallback plan for: {goal}"[:80]
            return PlanResult(
                goal=goal,
                summary=summary,
                tasks=tasks,
                raw_response="",
                fallback_reason=str(exc),
            )
        try:
            payload = self._extract_json(response_text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Planner response was not valid JSON: {exc}") from exc
        tasks = [self._task_from_dict(entry, idx) for idx, entry in enumerate(payload.get("tasks", []), 1)]
        summary = payload.get("summary", goal)
        return PlanResult(goal=goal, summary=summary, tasks=tasks, raw_response=response_text)

    def _extract_json(self, text: str) -> dict:
        match = JSON_PATTERN.search(text)
        if match:
            candidate = match.group("json")
        else:
            # Fallback: attempt to locate first JSON object in text
            start = text.find("{")
            end = text.rfind("}")
            candidate = text[start : end + 1] if start != -1 and end != -1 else "{}"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            LOGGER.error("Failed to parse planner response as JSON: %s", exc)
            raise

    def _task_from_dict(self, data: dict, index: int) -> PlanTask:
        step_number = data.get("step_number", index)
        deliverables = data.get("deliverables") or []
        if isinstance(deliverables, (str, bytes)):
            deliverables = [str(deliverables)]
        commands = data.get("commands") or []
        if isinstance(commands, (str, bytes)):
            commands = [str(commands)]
        return PlanTask(
            step_number=step_number,
            title=data.get("title", "Untitled"),
            description=data.get("description", ""),
            status=data.get("status", "pending"),
            dependencies=_normalize_int_list(data.get("dependencies")),
            category=data.get("category", "implementation"),
            effort=data.get("effort"),
            reach=data.get("reach"),
            impact=data.get("impact"),
            confidence=data.get("confidence"),
            deliverables=[str(item) for item in deliverables],
            commands=[str(item) for item in commands],
            identifier=data.get("id") or data.get("identifier"),
        )

    def _create_generic_fallback(self, goal: str) -> List[PlanTask]:
        """Create a simple fallback plan when the LLM is unavailable."""
        return [
            PlanTask(
                step_number=1,
                title="Understand Requirements",
                description=f"Analyze and understand the requirements for: {goal}. "
                           "Identify what needs to be accomplished and gather necessary context.",
            ),
            PlanTask(
                step_number=2,
                title="Execute Task",
                description="Implement the requested functionality based on the requirements. "
                           "This may involve code changes, analysis, or other operations.",
                dependencies=[1],
            ),
            PlanTask(
                step_number=3,
                title="Verify Results",
                description="Test and validate that the implementation meets the requirements. "
                           "Ensure the solution works correctly and document any findings.",
                dependencies=[2],
            ),
        ]

    def react_wbs(self, goal: str | None = None) -> PlanResult:
        """Produce a ReAct integration work breakdown without calling the LLM.
        
        This method is kept for backward compatibility but simplified.
        """
        goal_text = (goal or "Deliver PARE/ReAct automation").strip()
        tasks = [
            PlanTask(
                step_number=1,
                title="Design ReAct Architecture",
                description="Define and document the PARE (Plan-Act-Observe-Evaluate) loop architecture, "
                           "quality gates, and success metrics. Create documentation in docs/react_loop.md",
                identifier="R1",
            ),
            PlanTask(
                step_number=2,
                title="Implement Core ReAct Components",
                description="Build the reactive executor in ai_dev_agent/react/loop.py, gate evaluator "
                           "in evaluator.py, and type definitions in types.py for auditable execution.",
                dependencies=[1],
                identifier="R2",
            ),
            PlanTask(
                step_number=3,
                title="Add Metrics and Analytics",
                description="Implement metrics collection in ai_dev_agent/metrics/, including diff metrics, "
                           "coverage analysis, and iteration tracking to support gate evaluation.",
                dependencies=[2],
                identifier="R3",
            ),
            PlanTask(
                step_number=4,
                title="Security and Sandbox Hardening",
                description="Implement secure command execution in ai_dev_agent/sandbox/exec.py and "
                           "secret scanning in ai_dev_agent/security/secrets.py for safe automation.",
                dependencies=[2],
                identifier="R4",
            ),
            PlanTask(
                step_number=5,
                title="Build Quality Pipeline",
                description="Create unified quality pipeline in ai_dev_agent/react/pipeline.py combining "
                           "formatting, linting, typing, testing, and coverage checks.",
                dependencies=[3, 4],
                identifier="R5",
            ),
            PlanTask(
                step_number=6,
                title="CLI Integration",
                description="Wire up ReAct commands in ai_dev_agent/cli.py and update configuration "
                           "to make ReAct the default execution path.",
                dependencies=[5],
                identifier="R6",
            ),
            PlanTask(
                step_number=7,
                title="Documentation and Testing",
                description="Update all documentation, add comprehensive tests for the ReAct system, "
                           "and ensure test coverage for all new functionality.",
                dependencies=[6],
                identifier="R7",
            ),
        ]
        summary = "ReAct automation implementation plan"
        return PlanResult(goal=goal_text, summary=summary, tasks=tasks, raw_response="", fallback_reason="react_template")


def _normalize_int_list(value: object) -> List[int]:
    """Convert a value to a list of integers for dependencies."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        # Try to parse as integer
        try:
            return [int(value)]
        except ValueError:
            return []
    try:
        result = []
        for item in value:
            if isinstance(item, int):
                result.append(item)
            elif isinstance(item, str):
                try:
                    result.append(int(item))
                except ValueError:
                    # Skip non-integer strings
                    pass
        return result
    except (TypeError, ValueError):
        return []


__all__ = ["Planner", "PlanResult", "PlanTask"]

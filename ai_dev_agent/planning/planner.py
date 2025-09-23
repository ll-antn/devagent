"""Planning module for generating work breakdown structures."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Sequence

from ..llm_provider import LLMClient, LLMError, Message
from ..utils.logger import get_logger
from .prioritize import PriorityScore, rank_tasks

LOGGER = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an experienced software delivery planner. "
    "Produce concise, implementation-ready work breakdown structures for engineers. "
    "Always respond with a JSON object matching the provided schema and nothing else."
)

USER_TEMPLATE = """
Plan the following goal for a software development agent that will execute tasks sequentially.
Goal: {goal}

You are planning for a DevAgent codebase with these core modules:
- cli: Command-line interface and user interaction
- llm_provider: LLM abstraction and API clients  
- planning: Work breakdown structures and task management
- approval: Human-in-the-loop checkpoints and policies
- adr: Architecture Decision Record management
- code_edit: Code generation, diff application, and context gathering
- testing: Test execution, CI integration, and QA gates
- utils: Configuration, logging, and state management
- react: ReAct execution loop and gate evaluation
- metrics: Diff/coverage analytics and iteration tracking
- sandbox: Sandboxed command execution and safety controls
- security: Secret detection and static analysis helpers

You must output JSON with these fields:
- summary: short overview of the plan (<= 40 words)
- tasks: array of 8-15 task objects. Each task must include:
  - id (T1, T2, …)
  - title (imperative verb + specific deliverable)
  - description (2-3 sentences explaining scope and referencing specific modules like cli, llm_provider, planning, approval, adr, code_edit, testing, utils, security)
  - category (design|implementation|testing|documentation)
  - dependencies (list of task ids that must be completed first)
  - effort (1-5 numeric, where 1=trivial, 5=complex)
  - reach (1-5 numeric, users/systems affected)
  - impact (1-5 numeric, importance to goal)
  - confidence (0-1 float, certainty of estimate)
  - deliverables (specific file paths like `ai_dev_agent/module/file.py`, `docs/file.md`, `tests/test_file.py`)
  - commands (suggested CLI commands like `devagent react run --files ai_dev_agent/planning/planner.py` or `pytest tests/test_module.py`)

Requirements:
- Generate exactly 8-15 tasks with clear progression from design → implementation → testing → documentation
- Reference concrete modules and file paths in descriptions and deliverables
- Include dependencies that form a logical execution order
- Cover the full software development lifecycle: planning, coding, testing, approval, documentation
- Ensure deliverables specify actual repository paths starting with `ai_dev_agent/`, `docs/`, `tests/`, etc.
- Include both module enhancements and new feature development
- Add ADR tasks for significant architectural decisions
- Include approval checkpoints and testing validation
"""

JSON_PATTERN = re.compile(r"```json\s*(?P<json>{.*?})\s*```", re.DOTALL)


@dataclass
class PlanTask:
    identifier: str
    title: str
    description: str
    category: str
    status: str = "pending"
    dependencies: List[str] = field(default_factory=list)
    effort: float | None = None
    reach: float | None = None
    impact: float | None = None
    confidence: float | None = None
    deliverables: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    priority_score: PriorityScore | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.identifier,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "status": self.status,
            "dependencies": self.dependencies,
            "effort": self.effort,
            "reach": self.reach,
            "impact": self.impact,
            "confidence": self.confidence,
            "deliverables": self.deliverables,
            "commands": self.commands,
            "priority": self.priority_score.rice if self.priority_score else None,
        }


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
        try:
            response_text = self.client.complete(messages, temperature=0.1)
        except LLMError as exc:
            LOGGER.warning("LLM planning failed, generating fallback plan: %s", exc)
            tasks = self._fallback_tasks(goal)
            summary = f"Fallback plan generated without LLM for: {goal}"[:80]
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
        tasks = [self._task_from_dict(entry) for entry in payload.get("tasks", [])]
        self._apply_priorities(tasks)
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

    def _task_from_dict(self, data: dict) -> PlanTask:
        identifier = data.get("id") or data.get("identifier")
        if not identifier:
            identifier = f"T{len(data)}"
        return PlanTask(
            identifier=identifier,
            title=data.get("title", "Untitled"),
            description=data.get("description", ""),
            category=data.get("category", "implementation"),
            dependencies=data.get("dependencies", []) or [],
            effort=_safe_float(data.get("effort")),
            reach=_safe_float(data.get("reach")),
            impact=_safe_float(data.get("impact")),
            confidence=_safe_float(data.get("confidence")),
            deliverables=_normalize_list(data.get("deliverables")),
            commands=_normalize_list(data.get("commands")),
        )

    def _apply_priorities(self, tasks: List[PlanTask]) -> None:
        scores = rank_tasks(tasks)
        score_map = {score.task_id: score for score in scores}
        for task in tasks:
            task.priority_score = score_map.get(task.identifier)

    def _fallback_tasks(self, goal: str) -> List[PlanTask]:
        """Create a deterministic fallback plan when the LLM is unavailable."""
        goal_text = goal.strip() or "Deliver PARE/ReAct automation"
        tasks = self._react_wbs_tasks(goal_text)
        self._apply_priorities(tasks)
        return tasks

    def react_wbs(self, goal: str | None = None) -> PlanResult:
        """Produce a ReAct integration work breakdown without calling the LLM."""
        goal_text = (goal or "Deliver PARE/ReAct automation").strip()
        tasks = self._react_wbs_tasks(goal_text)
        self._apply_priorities(tasks)
        summary = "ReAct automation work breakdown covering phases 1-4"
        return PlanResult(goal=goal_text, summary=summary, tasks=tasks, raw_response="", fallback_reason="react_template")

    def _react_wbs_tasks(self, goal: str) -> List[PlanTask]:
        base_goal = goal if goal else "Deliver PARE/ReAct automation"
        tasks = [
            PlanTask(
                identifier="R1",
                title="Define ReAct architecture baseline",
                description=(
                    "Document the PARE (Plan-Act-Observe-Evaluate) loop, quality gates, and profile modes "
                    "(implement/review/design) to align the team on success metrics. (Goal: {goal})"
                ).format(goal=base_goal),
                category="design",
                effort=2,
                reach=3,
                impact=5,
                confidence=0.7,
                deliverables=[
                    "docs/react_loop.md",
                    "docs/metrics.md",
                    "README.md",
                ],
                commands=["devagent react plan"],
            ),
            PlanTask(
                identifier="R2",
                title="Implement ReAct execution core and reporting",
                description=(
                    "Create the reactive executor, gate evaluator, and typed payloads so each "
                    "action-observation pair is auditable."
                ),
                category="implementation",
                dependencies=["R1"],
                effort=4,
                reach=3,
                impact=5,
                confidence=0.6,
                deliverables=[
                    "ai_dev_agent/react/loop.py",
                    "ai_dev_agent/react/evaluator.py",
                    "ai_dev_agent/react/types.py",
                ],
                commands=["devagent react run"],
            ),
            PlanTask(
                identifier="R3",
                title="Build metrics, coverage, and iteration analytics",
                description=(
                    "Instrument diff metrics, patch coverage, secret detection, and iteration tracking to feed gate "
                    "evaluation with normalized snapshots."
                ),
                category="implementation",
                dependencies=["R2"],
                effort=3,
                reach=3,
                impact=5,
                confidence=0.6,
                deliverables=[
                    "ai_dev_agent/metrics/collectors.py",
                    "ai_dev_agent/metrics/diff.py",
                    "ai_dev_agent/metrics/coverage.py",
                    "ai_dev_agent/security/secrets.py",
                ],
                commands=["devagent react run"],
            ),
            PlanTask(
                identifier="R4",
                title="Harden sandbox and security tooling",
                description=(
                    "Enforce allowlisted command execution, resource limits, and secret scanning to keep automation "
                    "safe by default."
                ),
                category="implementation",
                dependencies=["R3"],
                effort=3,
                reach=3,
                impact=4,
                confidence=0.6,
                deliverables=[
                    "ai_dev_agent/sandbox/exec.py",
                    "ai_dev_agent/security/secrets.py",
                ],
                commands=["devagent react run"],
            ),
            PlanTask(
                identifier="R5",
                title="Compose quality pipeline and gate evaluation",
                description=(
                    "Combine formatting, linting, typing, compile, test, coverage, and perf checks into a single "
                    "qa.pipeline tool with configurable profiles."
                ),
                category="implementation",
                dependencies=["R4"],
                effort=4,
                reach=4,
                impact=5,
                confidence=0.7,
                deliverables=[
                    "ai_dev_agent/react/pipeline.py",
                    "tests/test_react_gate.py",
                    "tests/test_patch_coverage.py",
                ],
                commands=["devagent react run"],
            ),
            PlanTask(
                identifier="R6",
                title="Wire CLI and planner defaults for ReAct insights",
                description=(
                    "Expose `devagent react` commands, make React the default execution path in plans, and surface "
                    "gate status within CLI status/history views."
                ),
                category="implementation",
                dependencies=["R5"],
                effort=3,
                reach=4,
                impact=5,
                confidence=0.7,
                deliverables=[
                    "ai_dev_agent/cli.py",
                    "ai_dev_agent/utils/config.py",
                ],
                commands=["devagent react run"],
            ),
            PlanTask(
                identifier="R7",
                title="Update planner, documentation, and templates",
                description=(
                    "Ensure fallback planning provides a ReAct work breakdown, add `react plan`, and refresh docs" 
                    "so the team has a canonical WBS for the automation stack."
                ),
                category="documentation",
                dependencies=["R6"],
                effort=2,
                reach=3,
                impact=4,
                confidence=0.8,
                deliverables=[
                    "ai_dev_agent/planning/planner.py",
                    "docs/react_loop.md",
                    "docs/metrics.md",
                ],
                commands=["devagent react plan"],
            ),
            PlanTask(
                identifier="R8",
                title="Complete regression testing and rollout",
                description=(
                    "Add targeted tests for metrics, sandbox, secret scanning, and finalize release notes covering "
                    "Phase 1-4 capabilities."
                ),
                category="testing",
                dependencies=["R6"],
                effort=2,
                reach=4,
                impact=4,
                confidence=0.7,
                deliverables=[
                    "tests/test_diff_metrics.py",
                    "tests/test_secret_scan.py",
                    "README.md",
                ],
                commands=["pytest"],
            ),
        ]
        return tasks


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


__all__ = ["Planner", "PlanResult", "PlanTask"]

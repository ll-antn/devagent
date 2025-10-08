from __future__ import annotations

from typing import Any, List, Sequence

from ai_dev_agent.cli.react.executor import BudgetAwareExecutor, BudgetManager, _record_search_query
from ai_dev_agent.engine.react.types import ActionRequest, Observation, TaskSpec


class _DummyActionProvider:
    def __init__(self, actions: Sequence[ActionRequest]) -> None:
        self._actions = list(actions)
        self._invocations = 0

    def update_phase(self, phase: str, *, is_final: bool = False) -> None:  # pragma: no cover - no-op stub
        return None

    def __call__(self, task: TaskSpec, history: Sequence[Any]) -> ActionRequest:
        if self._invocations >= len(self._actions):
            raise StopIteration()
        action = self._actions[self._invocations]
        self._invocations += 1
        return action


class _DummyToolInvoker:
    def __init__(self, observations: Sequence[Observation]) -> None:
        self._observations = list(observations)
        self._invocations = 0

    def __call__(self, action: ActionRequest) -> Observation:
        observation = self._observations[self._invocations]
        self._invocations += 1
        return observation


def _make_action(tool: str = "run") -> ActionRequest:
    return ActionRequest(
        step_id="S1",
        thought="test",
        tool=tool,
        args={},
    )


def _make_observation(success: bool, outcome: str) -> Observation:
    return Observation(
        success=success,
        outcome=outcome,
        tool="run",
        metrics={"exit_code": 0 if success else 1},
    )


def test_budget_executor_marks_failure_when_last_observation_fails() -> None:
    manager = BudgetManager(1)
    executor = BudgetAwareExecutor(manager)
    action_provider = _DummyActionProvider([_make_action()])
    tool_invoker = _DummyToolInvoker([_make_observation(False, "Command exited with 1")])
    task = TaskSpec(identifier="T1", goal="Run command", category="assist")

    result = executor.run(task, action_provider, tool_invoker)

    assert result.status == "failed"
    assert "Command exited with 1" in (result.stop_reason or "")


def test_budget_executor_reports_success_after_final_success() -> None:
    manager = BudgetManager(1)
    executor = BudgetAwareExecutor(manager)
    action_provider = _DummyActionProvider([_make_action()])
    tool_invoker = _DummyToolInvoker([_make_observation(True, "Command exited with 0")])
    task = TaskSpec(identifier="T2", goal="Run command", category="assist")

    result = executor.run(task, action_provider, tool_invoker)

    assert result.status == "success"
    assert result.stop_reason == "Completed"


def test_record_search_query_prefers_query_and_pattern() -> None:
    recorded: set[str] = set()
    action = ActionRequest(
        step_id="S1",
        thought="search",
        tool="grep",
        args={"pattern": "TODO"},
    )

    _record_search_query(action, recorded)

    assert "TODO" in recorded

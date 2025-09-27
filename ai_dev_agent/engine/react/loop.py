"""Reactive execution loop orchestrating action → observation → evaluation."""
from __future__ import annotations

import time
from typing import Callable, Iterable, List, Optional, Sequence, Union

from pydantic import ValidationError

from .evaluator import GateEvaluator
from .types import (
    ActionRequest,
    EvaluationResult,
    MetricsSnapshot,
    Observation,
    RunResult,
    StepRecord,
    TaskSpec,
)

ActionProvider = Callable[[TaskSpec, Sequence[StepRecord]], Union[ActionRequest, dict]]
ToolInvoker = Callable[[ActionRequest], Union[Observation, dict]]


class ReactiveExecutor:
    """Coordinates ReAct steps until gates succeed or budgets expire."""

    def __init__(
        self,
        evaluator: GateEvaluator,
    ) -> None:
        self.evaluator = evaluator

    def run(
        self,
        task: TaskSpec,
        action_provider: ActionProvider,
        tool_invoker: ToolInvoker,
        *,
        prior_steps: Optional[Iterable[StepRecord]] = None,
    ) -> RunResult:
        start_time = time.perf_counter()
        history: List[StepRecord] = list(prior_steps or [])
        steps: List[StepRecord] = []
        last_eval: Optional[EvaluationResult] = None
        exc_info: Optional[Exception] = None
        try:
            step_index = len(history) + 1
            while step_index <= self.evaluator.config.steps_budget:
                try:
                    action_payload = action_provider(task, history + steps)
                except StopIteration:
                    break
                try:
                    action = self._ensure_action(action_payload, step_index)
                except ValidationError as exc:  # noqa: TRY003 - translate to runtime error
                    raise ValueError(f"Action provider returned invalid payload: {exc}") from exc

                observation = self._invoke_tool(tool_invoker, action)
                metrics = self._metrics_from_observation(observation)
                evaluation = self.evaluator.evaluate(metrics, history + steps)
                record = StepRecord(
                    action=action,
                    observation=observation,
                    metrics=metrics,
                    evaluation=evaluation,
                    step_index=step_index,
                )
                steps.append(record)
                last_eval = evaluation

                if evaluation.should_stop:
                    break
                step_index += 1

        except Exception as exc:  # noqa: BLE001 - propagate after trace finalization
            exc_info = exc
            if not last_eval:
                last_eval = EvaluationResult(
                    gates={},
                    should_stop=True,
                    stop_reason=f"Exception: {exc}",
                    next_action_hint=None,
                    improved_metrics={},
                    status="failed",
                )
        finally:
            runtime = time.perf_counter() - start_time
            gates = last_eval.gates if last_eval else {}
            status = last_eval.status if last_eval else "failed"
            stop_reason = last_eval.stop_reason if last_eval else None
            if last_eval and not last_eval.should_stop and status == "in_progress":
                status = "failed"
                stop_reason = stop_reason or "Execution stopped before gates were satisfied."
            metrics_dict = steps[-1].metrics.model_dump() if steps else {}

            result = RunResult(
                task_id=task.identifier,
                status=status,
                steps=history + steps,
                gates=gates,
                required_gates=(last_eval.required_gates if last_eval else {}),
                stop_reason=stop_reason,
                runtime_seconds=round(runtime, 3),
                metrics=metrics_dict,
            )

        if exc_info:
            raise exc_info
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_action(self, payload: Union[ActionRequest, dict], step_index: int) -> ActionRequest:
        if isinstance(payload, ActionRequest):
            return payload
        if not isinstance(payload, dict):
            raise TypeError(f"Action provider returned unsupported type: {type(payload)!r}")
        payload.setdefault("step_id", f"S{step_index}")
        return ActionRequest.model_validate(payload)

    def _invoke_tool(self, invoker: ToolInvoker, action: ActionRequest) -> Observation:
        try:
            raw = invoker(action)
        except Exception as exc:  # noqa: BLE001 - capture tool failure as observation
            return Observation(
                success=False,
                outcome="Tool invocation raised exception.",
                tool=action.tool,
                error=str(exc),
            )
        if isinstance(raw, Observation):
            if raw.tool is None:
                raw.tool = action.tool
            return raw
        if not isinstance(raw, dict):
            raise TypeError(f"Tool invoker returned unsupported type: {type(raw)!r}")
        raw.setdefault("tool", action.tool)
        return Observation.model_validate(raw)

    def _metrics_from_observation(self, observation: Observation) -> MetricsSnapshot:
        metrics = observation.metrics or {}
        if isinstance(metrics, MetricsSnapshot):
            return metrics
        if not isinstance(metrics, dict):
            raise TypeError(f"Observation metrics must be dict-like, received {type(metrics)!r}")
        return MetricsSnapshot.model_validate(metrics)


__all__ = ["ReactiveExecutor", "ActionProvider", "ToolInvoker"]

from ai_dev_agent.react.evaluator import GateEvaluator
from ai_dev_agent.react.loop import ReactiveExecutor
from ai_dev_agent.react.types import (
    ActionRequest,
    GateConfig,
    MetricsSnapshot,
    Observation,
    TaskSpec,
)


def test_reactive_executor_passes_once_all_gates_satisfied():
    config = GateConfig(
        diff_limit_lines=200,
        diff_limit_files=4,
        patch_coverage_target=0.75,
        steps_budget=3,
    )
    evaluator = GateEvaluator(config)
    executor = ReactiveExecutor(evaluator)

    metrics = MetricsSnapshot(
        tests_passed=True,
        lint_errors=0,
        type_errors=0,
        format_errors=0,
        compile_errors=0,
        diff_lines=12,
        diff_files=1,
        patch_coverage=0.9,
        secrets_found=0,
        sandbox_violations=0,
        flaky_tests=0,
    )

    def action_provider(task, history):
        assert not history
        return ActionRequest(
            step_id="S1",
            thought="Run quality pipeline",
            tool="qa.pipeline",
            args={},
        )

    def tool_invoker(action):
        return Observation(
            success=True,
            outcome="quality gates satisfied",
            tool=action.tool,
            metrics=metrics.model_dump(),
        )

    task = TaskSpec(identifier="T1", goal="Validate gating", category="review")
    result = executor.run(task, action_provider, tool_invoker)

    assert result.status == "success"
    assert result.gates["tests"] is True
    assert result.gates["diff_limits"] is True
    assert result.gates["patch_coverage"] is True

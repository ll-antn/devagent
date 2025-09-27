from ai_dev_agent.engine.react.tool_strategy import (
    ToolContext,
    ToolSelectionStrategy,
    TaskType,
)


def test_code_search_hints_scale_with_iteration_budget() -> None:
    strategy = ToolSelectionStrategy()

    early_context = ToolContext(
        task_type=TaskType.RESEARCH,
        iteration_count=5,
        iteration_budget=40,
    )
    early_hints = strategy.get_tool_hints("code.search", early_context)
    assert early_hints["max_results"] == 30

    late_context = ToolContext(
        task_type=TaskType.RESEARCH,
        iteration_count=30,
        iteration_budget=40,
    )
    late_hints = strategy.get_tool_hints("code.search", late_context)
    assert late_hints["max_results"] == 50


def test_apply_context_rules_uses_relative_threshold() -> None:
    strategy = ToolSelectionStrategy()
    tools = ["code.search", "fs.read", "symbols.find", "ast.query"]

    context = ToolContext(
        task_type=TaskType.CODE_EXPLORATION,
        iteration_count=18,
        iteration_budget=20,
    )

    reordered = strategy._apply_context_rules(tools, context)
    assert set(reordered[:3]) == {"symbols.find", "ast.query", "fs.read"}
    assert reordered[-1] == "code.search"

from ai_dev_agent.engine.planning.planner import Planner
from ai_dev_agent.providers.llm.base import LLMError, Message


class DummyClient:
    def complete(self, messages, temperature=0.2, max_tokens=None):
        assert isinstance(messages[0], Message)
        return """```json
{
  "summary": "Demo plan",
  "tasks": [
    {"step_number": 1, "title": "Design", "description": "Design the feature architecture and define interfaces"},
    {"step_number": 2, "title": "Implement", "description": "Write the code to implement the feature", "dependencies": [1]}
  ]
}
```"""


def test_planner_generates_tasks():
    planner = Planner(DummyClient())
    result = planner.generate("Build feature")
    assert result.summary == "Demo plan"
    assert len(result.tasks) == 2
    assert result.tasks[0].step_number == 1
    assert result.tasks[0].title == "Design"
    assert result.tasks[1].dependencies == [1]


class SimpleClient:
    def complete(self, messages, temperature=0.2, max_tokens=None):
        return """```json
{
  "summary": "Research plan",
  "tasks": [
    {"step_number": 1, "title": "Investigate usage", "description": "Search repository for method usage"},
    {"step_number": 2, "title": "Analyze findings", "description": "Review and count occurrences", "dependencies": [1]},
    {"step_number": 3, "title": "Summarize results", "description": "Write report with findings", "dependencies": [2]}
  ]
}
```"""


def test_planner_simple_format():
    planner = Planner(SimpleClient())
    result = planner.generate("What methods can be removed from ETSChecker API?")
    assert result.summary == "Research plan"
    assert len(result.tasks) == 3
    assert [task.step_number for task in result.tasks] == [1, 2, 3]
    assert result.tasks[2].dependencies == [2]


class FailingClient:
    def complete(self, messages, temperature=0.2, max_tokens=None):
        raise LLMError("Temporary outage")


def test_planner_fallback_when_llm_unavailable():
    planner = Planner(FailingClient())
    result = planner.generate("Improve resilience")

    assert result.fallback_reason == "Temporary outage"
    assert len(result.tasks) == 3
    assert result.tasks[0].title == "Understand Requirements"
    assert result.tasks[1].title == "Execute Task"
    assert result.tasks[2].title == "Verify Results"
    assert result.raw_response == ""
    # Check dependencies
    assert result.tasks[0].dependencies == []
    assert result.tasks[1].dependencies == [1]
    assert result.tasks[2].dependencies == [2]

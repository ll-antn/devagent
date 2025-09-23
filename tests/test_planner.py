from ai_dev_agent.planning.planner import Planner
from ai_dev_agent.llm_provider.base import LLMError, Message


class DummyClient:
    def complete(self, messages, temperature=0.2, max_tokens=None):
        assert isinstance(messages[0], Message)
        return """```json\n{\n  \"summary\": \"Demo plan\",\n  \"tasks\": [\n    {\"id\": \"T1\", \"title\": \"Design\", \"description\": \"Do design\", \"category\": \"design\", \"effort\": 2, \"reach\": 1, \"impact\": 4, \"confidence\": 0.8},\n    {\"id\": \"T2\", \"title\": \"Implement\", \"description\": \"Do code\", \"category\": \"implementation\", \"effort\": 3, \"reach\": 1, \"impact\": 5, \"confidence\": 0.7}\n  ]\n}\n```"""


def test_planner_generates_tasks():
    planner = Planner(DummyClient())
    result = planner.generate("Build feature")
    assert result.summary == "Demo plan"
    assert len(result.tasks) == 2
    assert result.tasks[0].priority_score is not None


class FailingClient:
    def complete(self, messages, temperature=0.2, max_tokens=None):
        raise LLMError("Temporary outage")


def test_planner_fallback_when_llm_unavailable():
    planner = Planner(FailingClient())
    result = planner.generate("Improve resilience")

    assert result.fallback_reason == "Temporary outage"
    assert [task.identifier for task in result.tasks] == [
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R8",
    ]
    assert result.raw_response == ""
    assert all(task.priority_score is not None for task in result.tasks)

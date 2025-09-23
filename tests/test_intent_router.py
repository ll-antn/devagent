from ai_dev_agent.intent_router import IntentDecision, IntentRouter, IntentRoutingError
from ai_dev_agent.llm_provider.base import ToolCall, ToolCallResult
from ai_dev_agent.utils.config import Settings


class DummyClient:
    def __init__(self, result: ToolCallResult) -> None:
        self.result = result
        self.captured_messages = None
        self.captured_tools = None

    def invoke_tools(self, messages, tools, temperature=0.2, max_tokens=None, tool_choice="auto", extra_headers=None):
        self.captured_messages = messages
        self.captured_tools = tools
        return self.result


def test_intent_router_returns_tool_decision(tmp_path):
    settings = Settings()
    settings.workspace_root = tmp_path
    result = ToolCallResult(calls=[ToolCall(name="list_directory", arguments={"path": "."})], message_content="Listing workspace")
    client = DummyClient(result)

    router = IntentRouter(client, settings)
    decision = router.route("покажи содержимое директории")

    assert isinstance(decision, IntentDecision)
    assert decision.tool == "list_directory"
    assert decision.arguments["path"] == "."
    assert decision.rationale == "Listing workspace"
    assert client.captured_messages is not None
    assert client.captured_tools is not None


def test_intent_router_direct_response(tmp_path):
    settings = Settings()
    settings.workspace_root = tmp_path
    result = ToolCallResult(calls=[], message_content="Просто ответ")
    client = DummyClient(result)

    router = IntentRouter(client, settings)
    decision = router.route("скажи привет")

    assert decision.tool == "respond_directly"
    assert decision.arguments["text"] == "Просто ответ"


def test_intent_router_empty_prompt_raises(tmp_path):
    settings = Settings()
    settings.workspace_root = tmp_path
    client = DummyClient(ToolCallResult(calls=[], message_content=""))

    router = IntentRouter(client, settings)
    try:
        router.route("   ")
    except IntentRoutingError:
        pass
    else:  # pragma: no cover - ensure exception raised
        raise AssertionError("Expected routing error for empty prompt")

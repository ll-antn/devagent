"""Natural-language intent routing leveraging LLM tool calling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json

from ai_dev_agent.providers.llm.base import (
    LLMClient,
    LLMError,
    Message,
    ToolCallResult,
)
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.constants import LEGACY_TOOL_NAMES
from ai_dev_agent.core.utils.tool_utils import sanitize_tool_name
from ai_dev_agent.tools import registry as tool_registry

DEFAULT_TOOLS: List[Dict[str, Any]] = []


@dataclass
class IntentDecision:
    """Result returned by the intent router."""

    tool: Optional[str]
    arguments: Dict[str, Any]
    rationale: Optional[str] = None


class IntentRoutingError(RuntimeError):
    """Raised when the router cannot derive a suitable intent."""


class IntentRouter:
    """Use LLM tool-calling to map natural language prompts onto CLI intents."""

    def __init__(
        self,
        client: Optional[LLMClient],
        settings: Settings,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.tools = tools or self._build_tool_list(settings)

    def _build_tool_list(self, settings: Settings) -> List[Dict[str, Any]]:
        """Combine core tools with selected registry tools, avoiding duplicates."""
        combined: List[Dict[str, Any]] = []
        used_names = set(LEGACY_TOOL_NAMES)
        for entry in DEFAULT_TOOLS:
            fn = entry.get("function", {})
            name = fn.get("name")
            if name in LEGACY_TOOL_NAMES:
                continue
            if name:
                used_names.add(name)
            combined.append(entry)

        combined.extend(self._build_registry_tools(settings, used_names))
        return combined

    def _build_registry_tools(self, settings: Settings, used_names: set[str]) -> List[Dict[str, Any]]:
        """Translate registry specs into LLM tool definitions for a safelist."""
        safelist = [
            "symbols.index",
            "code.search",
            "fs.read",
            "symbols.find",
            "ast.query",
            "exec",
            "fs.write_patch",
        ]
        tools: List[Dict[str, Any]] = []
        for name in safelist:
            try:
                spec = tool_registry.get(name)
            except KeyError:
                continue
            try:
                with spec.request_schema_path.open("r", encoding="utf-8") as handle:
                    params_schema = json.load(handle)
            except Exception:
                params_schema = {"type": "object", "properties": {}, "additionalProperties": True}

            description = spec.description or ""
            if name == "fs.write_patch" and not getattr(settings, "auto_approve_code", False):
                description = (description or "Apply a unified diff") + \
                              " (auto_approve_code is recommended for automated edits)"

            base_name = sanitize_tool_name(name)
            safe_name = base_name
            suffix = 1
            if safe_name in LEGACY_TOOL_NAMES:
                suffix += 1
                safe_name = f"{base_name}_{suffix}"
            while safe_name in used_names:
                suffix += 1
                safe_name = f"{base_name}_{suffix}"
            used_names.add(safe_name)

            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": safe_name,
                        "description": description,
                        "parameters": params_schema,
                    },
                }
            )
        return tools

    def route(self, prompt: str) -> IntentDecision:
        if not prompt.strip():
            raise IntentRoutingError("Empty prompt provided for intent routing.")

        # Require an LLM client; no keyword-based fallback
        if self.client is None:
            raise IntentRoutingError("No LLM client available; fallback routing is disabled.")

        messages = [
            Message(
                role="system",
                content=self._system_prompt(),
            ),
            Message(role="user", content=prompt.strip()),
        ]

        try:
            result: ToolCallResult = self.client.invoke_tools(
                messages,
                tools=self.tools,
                temperature=0.1,
            )
        except LLMError as exc:
            raise IntentRoutingError(f"LLM tool call failed: {exc}") from exc
        except Exception as exc:
            raise IntentRoutingError(f"Intent routing failed: {exc}") from exc

        if result.calls:
            call = result.calls[0]
            return IntentDecision(
                tool=call.name,
                arguments=call.arguments,
                rationale=(result.message_content or "").strip() or None,
            )

        if result.message_content:
            # No tool used; surface the response directly without invoking a handler.
            return IntentDecision(
                tool=None,
                arguments={"text": result.message_content.strip()},
            )

        raise IntentRoutingError("Model response did not include a tool call or content.")

    def _system_prompt(self) -> str:
        workspace = str(self.settings.workspace_root or ".")
        return (
            "You route developer requests for the DevAgent CLI. "
            "Always choose the most appropriate tool description based on the request. "
            "Locate files with 'code.search' or 'symbols.index' + 'symbols.find' before using 'fs.read'. "
            "Use 'fs.read' with specific paths and optional 'context_lines' to keep outputs small. "
            "Prefer registry tools over 'exec'; only run commands when no dedicated tool applies. "
            "The repository root is at "
            f"'{workspace}'. Return concise rationales when helpful."
        )

    # No keyword-based fallback routing


__all__ = ["IntentDecision", "IntentRouter", "IntentRoutingError", "DEFAULT_TOOLS"]

"""Natural-language intent routing leveraging LLM tool calling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .llm_provider.base import LLMClient, LLMError, Message, ToolCallResult
from .utils.config import Settings

DEFAULT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories relative to the repository root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to inspect. Defaults to '.'",
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Include hidden files when true.",
                        "default": False,
                    },
                    "detailed": {
                        "type": "boolean",
                        "description": "Return extra metadata (size, type).",
                        "default": False,
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_repository",
            "description": "Search for files or docs that mention the provided query string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords or phrase.",
                    },
                    "include_docs": {
                        "type": "boolean",
                        "description": "Search documentation files when true.",
                        "default": True,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_plan",
            "description": "Create a development plan for the given goal (auto-approve by default).",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Goal or feature request to plan for.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional Markdown output path relative to repo root.",
                    },
                    "write_markdown": {
                        "type": "boolean",
                        "description": "Persist the plan to Markdown when true.",
                        "default": False,
                    },
                },
                "required": ["goal"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a snippet of a file with optional line ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file."
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-based line number to start from."
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Inclusive 1-based line number to stop at."
                    },
                    "max_lines": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 200,
                        "description": "Maximum number of lines to return when end_line is not provided."
                    }
                },
                "required": ["path"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_repository",
            "description": "Answer a question about the repository using context gathering heuristics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question to answer.",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of files to prioritise in context gathering.",
                    },
                    "include_docs": {
                        "type": "boolean",
                        "default": True,
                        "description": "Allow documentation files in gathered context.",
                    },
                    "max_files": {
                        "type": "integer",
                        "default": 8,
                        "description": "Maximum number of files to gather for context.",
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_frontmatter_plan",
            "description": "Run a saved YAML-frontmatter plan step-by-step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_path": {
                        "type": "string",
                        "description": "Relative path to the plan document.",
                    },
                    "apply_changes": {
                        "type": "boolean",
                        "default": False,
                        "description": "Apply code changes instead of dry-run.",
                    },
                    "auto_commit": {
                        "type": "boolean",
                        "default": False,
                        "description": "Automatically commit after passing gates when true.",
                    },
                },
                "required": ["plan_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_against_plan",
            "description": "Verify repository state against a plan file and report divergences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_path": {
                        "type": "string",
                        "description": "Relative path to the plan for comparison.",
                    },
                    "fail_on_divergence": {
                        "type": "boolean",
                        "default": False,
                        "description": "Raise an error (exit 1) when divergence is detected.",
                    },
                },
                "required": ["plan_path"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class IntentDecision:
    """Result returned by the intent router."""

    tool: str
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
        self.tools = tools or DEFAULT_TOOLS

    def route(self, prompt: str) -> IntentDecision:
        if not prompt.strip():
            raise IntentRoutingError("Empty prompt provided for intent routing.")

        # If no LLM client available, use fallback routing immediately
        if self.client is None:
            fallback = self._try_fallback_routing(prompt.strip())
            if fallback:
                return fallback
            raise IntentRoutingError("No LLM client available and no fallback routing matched.")

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
            # Try to provide a fallback based on simple keyword matching
            fallback = self._try_fallback_routing(prompt.strip())
            if fallback:
                return fallback
            raise IntentRoutingError(f"LLM call failed and no fallback available: {exc}") from exc
        except Exception as exc:
            # Try to provide a fallback based on simple keyword matching
            fallback = self._try_fallback_routing(prompt.strip())
            if fallback:
                return fallback
            raise IntentRoutingError(f"Intent routing failed: {exc}") from exc

        if result.calls:
            call = result.calls[0]
            return IntentDecision(
                tool=call.name,
                arguments=call.arguments,
                rationale=(result.message_content or "").strip() or None,
            )

        if result.message_content:
            # No tool used; treat as direct natural language response.
            return IntentDecision(
                tool="respond_directly",
                arguments={"text": result.message_content.strip()},
            )

        raise IntentRoutingError("Model response did not include a tool call or content.")

    def _system_prompt(self) -> str:
        workspace = str(self.settings.workspace_root or ".")
        return (
            "You route developer requests for the DevAgent CLI. "
            "Always choose the most appropriate tool description based on the request. "
            "The repository root is at "
            f"'{workspace}'. Return concise rationales when helpful."
        )

    def _try_fallback_routing(self, prompt: str) -> Optional[IntentDecision]:
        """Provide simple keyword-based fallback routing when LLM is unavailable."""
        prompt_lower = prompt.lower()
        
        # Question patterns (highest priority - these are most common and need multi-step analysis)
        question_keywords = ["how", "what", "why", "explain", "describe"]
        if any(keyword in prompt_lower for keyword in question_keywords) or "?" in prompt:
            return IntentDecision(
                tool="ask_repository",
                arguments={"question": prompt.strip()},
                rationale="Fallback: Using keyword-based question routing (LLM unavailable)",
            )
        
        # Plan generation patterns
        plan_keywords = ["plan", "create", "generate", "implement", "add", "build"]
        if any(keyword in prompt_lower for keyword in plan_keywords):
            return IntentDecision(
                tool="generate_plan", 
                arguments={"goal": prompt.strip()},
                rationale="Fallback: Using keyword-based plan routing (LLM unavailable)",
            )
        
        # Search patterns (lower priority - only when it's clearly a search, not a question)
        search_keywords = ["где", "where", "find", "search", "locate"]
        if any(keyword in prompt_lower for keyword in search_keywords):
            # Extract potential search terms
            search_term = prompt.strip()
            return IntentDecision(
                tool="search_repository",
                arguments={"query": search_term, "include_docs": True},
                rationale="Fallback: Using keyword-based search routing (LLM unavailable)",
            )
        
        # List/show patterns  
        list_keywords = ["list", "show", "ls", "dir"]
        if any(keyword in prompt_lower for keyword in list_keywords):
            return IntentDecision(
                tool="list_directory",
                arguments={"path": ".", "detailed": True},
                rationale="Fallback: Using keyword-based directory listing (LLM unavailable)",
            )
        
        return None


__all__ = ["IntentDecision", "IntentRouter", "IntentRoutingError", "DEFAULT_TOOLS"]

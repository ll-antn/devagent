"""Tool invoker that routes ReAct tool calls to registry-backed implementations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from ai_dev_agent.tools.code.code_edit.editor import CodeEditor
from ai_dev_agent.engine.react.pipeline import PipelineCommands
from ai_dev_agent.tools.execution.testing.local_tests import TestRunner
from ai_dev_agent.tools import ToolContext, registry
from ai_dev_agent.core.utils.devagent_config import DevAgentConfig, load_devagent_yaml
from ai_dev_agent.core.utils.logger import get_logger
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.engine.metrics import MetricsCollector
from ai_dev_agent.tools.code.code_edit.tree_sitter_analysis import extract_symbols_from_outline
from .types import ActionRequest, Observation

LOGGER = get_logger(__name__)


class RegistryToolInvoker:
    """Invoke registered tools backed by the shared registry."""

    def __init__(
        self,
        workspace: Path,
        settings: Settings,
        code_editor: Optional[CodeEditor] = None,
        test_runner: Optional[TestRunner] = None,
        sandbox=None,
        collector: Optional[MetricsCollector] = None,
        pipeline_commands: Optional[PipelineCommands] = None,
        devagent_cfg: Optional[DevAgentConfig] = None,
    ) -> None:
        self.workspace = workspace
        self.settings = settings
        self.code_editor = code_editor
        self.test_runner = test_runner
        self.sandbox = sandbox
        self.collector = collector
        self.pipeline_commands = pipeline_commands
        self.devagent_cfg = devagent_cfg or load_devagent_yaml()
        self._structure_hints: Dict[str, Any] = {
            "symbols": set(),
            "files": {},
            "project_summary": None,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, action: ActionRequest) -> Observation:
        payload = action.args or {}
        tool_name = action.tool
        try:
            result = self._invoke_registry(tool_name, payload)
        except KeyError:
            return Observation(
                success=False,
                outcome=f"Unknown tool: {tool_name}",
                tool=tool_name,
                error=f"Tool '{tool_name}' is not registered",
            )
        except ValueError as exc:
            return Observation(
                success=False,
                outcome=f"Tool {tool_name} rejected input",
                tool=tool_name,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Tool %s execution failed", tool_name)
            return Observation(
                success=False,
                outcome=f"Tool {tool_name} failed",
                tool=tool_name,
                error=str(exc),
            )

        return self._wrap_result(tool_name, result)

    # ------------------------------------------------------------------
    # Registry invocation helpers
    # ------------------------------------------------------------------

    def _invoke_registry(self, tool_name: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ctx = ToolContext(
            repo_root=self.workspace,
            settings=self.settings,
            sandbox=self.sandbox,
            devagent_config=self.devagent_cfg,
            metrics_collector=self.collector,
            extra={
                "code_editor": self.code_editor,
                "test_runner": self.test_runner,
                "pipeline_commands": self.pipeline_commands,
                "structure_hints": self._export_structure_hints(),
            },
        )
        return registry.invoke(tool_name, payload, ctx)

    def _wrap_result(self, tool_name: str, result: Mapping[str, Any]) -> Observation:
        success = True
        outcome = f"Executed {tool_name}"
        metrics: Dict[str, Any] = {}
        artifacts: list[str] = []
        raw_output: Optional[str] = None

        if tool_name == "fs.read":
            files = result.get("files", [])
            outcome = f"Read {len(files)} file(s)"
            metrics = {"files": len(files)}
            artifacts = [entry.get("path") for entry in files if entry.get("path")]
            raw_output = json.dumps(result, indent=2)
        elif tool_name == "fs.write_patch":
            applied = bool(result.get("applied"))
            rejected = result.get("rejected_hunks") or []
            success = applied and not rejected
            outcome = "Patch applied" if applied else "Patch validated"
            metrics = {
                "applied": applied,
                "rejected_hunks": len(rejected),
                **(result.get("diff_stats") or {}),
            }
            artifacts = result.get("changed_files") or []
            raw_output = json.dumps(result, indent=2)
        elif tool_name == "code.search":
            matches = result.get("matches", [])
            success = len(matches) > 0
            outcome = f"Found {len(matches)} match(es)"
            metrics = {"matches": len(matches)}
            artifacts = [match.get("path") for match in matches if match.get("path")]
            raw_output = json.dumps(matches[:20], indent=2)
        elif tool_name == "symbols.index":
            stats = result.get("stats", {})
            outcome = "Symbol index updated"
            metrics = stats
            artifacts = [result.get("db_path")] if result.get("db_path") else []
        elif tool_name == "symbols.find":
            defs = result.get("defs", [])
            outcome = f"Found {len(defs)} definition(s)"
            metrics = {"definitions": len(defs)}
            artifacts = [entry.get("path") for entry in defs if entry.get("path")]
        elif tool_name == "ast.query":
            mode = result.get("mode", "query")
            if mode == "summary":
                summaries = result.get("summaries", [])
                outcome = f"Summarised {len(summaries)} file(s)"
                metrics = {"summaries": len(summaries)}
                project_summary = result.get("project_summary")
                preview = {
                    "summaries": summaries[:5],
                    "project_summary": project_summary,
                }
                raw_output = json.dumps(preview, indent=2)
                self._update_structure_hints_from_summary(summaries, project_summary)
            else:
                nodes = result.get("nodes", [])
                outcome = f"Matched {len(nodes)} node(s)"
                metrics = {"nodes": len(nodes)}
                raw_output = json.dumps(nodes[:10], indent=2)
        elif tool_name == "exec":
            exit_code = result.get("exit_code", 0)
            success = exit_code == 0
            outcome = f"Command exited with {exit_code}"
            metrics = {"exit_code": exit_code, "duration_ms": result.get("duration_ms")}
            raw_output = "STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}".format(
                stdout=result.get("stdout_tail", ""),
                stderr=result.get("stderr_tail", ""),
            )
        elif tool_name == "security.secrets_scan":
            findings = result.get("findings", 0)
            success = findings == 0
            outcome = "No secrets detected" if success else f"Found {findings} potential secret(s)"
            metrics = {"findings": findings}
            artifacts = [result.get("report_path")] if result.get("report_path") else []
        else:
            metrics = dict(result)
            raw_output = json.dumps(result, indent=2)

        observations_kwargs: Dict[str, Any] = {
            success=success,
            outcome=outcome,
            metrics=metrics,
            artifacts=[item for item in artifacts if item],
            tool=tool_name,
            raw_output=raw_output,
        }

        structure_payload = self._export_structure_hints()
        if structure_payload["symbols"] or structure_payload["files"] or structure_payload["project_summary"]:
            observations_kwargs["structure_hints"] = structure_payload

        return Observation(**observations_kwargs)

    def _update_structure_hints_from_summary(
        self, summaries: List[Mapping[str, Any]], project_summary: Optional[str]
    ) -> None:
        file_hints = self._structure_hints.setdefault("files", {})
        symbol_set: Set[str] = self._structure_hints.setdefault("symbols", set())

        for entry in summaries:
            path = entry.get("path")
            outline = entry.get("outline") or []
            if not path:
                continue
            symbols = extract_symbols_from_outline(outline)
            file_hints[path] = {
                "outline": outline,
                "symbols": symbols,
            }
            symbol_set.update(symbols)

        if project_summary:
            self._structure_hints["project_summary"] = project_summary

    def _export_structure_hints(self) -> Dict[str, Any]:
        files_payload: Dict[str, Any] = {}
        file_hints = self._structure_hints.get("files") or {}
        for path, info in file_hints.items():
            outline = info.get("outline") or []
            symbols = info.get("symbols") or []
            files_payload[path] = {
                "outline": outline[:20],
                "symbols": sorted(set(symbols))[:20],
            }

        symbols = sorted(set(self._structure_hints.get("symbols") or []))[:50]
        return {
            "symbols": symbols,
            "files": files_payload,
            "project_summary": self._structure_hints.get("project_summary"),
        }



def create_tool_invoker(
    workspace: Path,
    settings: Settings,
    code_editor: Optional[CodeEditor] = None,
    test_runner: Optional[TestRunner] = None,
    sandbox=None,
    collector: Optional[MetricsCollector] = None,
    pipeline_commands: Optional[PipelineCommands] = None,
    devagent_cfg: Optional[DevAgentConfig] = None,
) -> RegistryToolInvoker:
    """Factory to create a configured tool invoker."""

    return RegistryToolInvoker(
        workspace=workspace,
        settings=settings,
        code_editor=code_editor,
        test_runner=test_runner,
        sandbox=sandbox,
        collector=collector,
        pipeline_commands=pipeline_commands,
        devagent_cfg=devagent_cfg,
    )


__all__ = ["RegistryToolInvoker", "create_tool_invoker"]

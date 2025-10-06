"""Tool invoker that routes ReAct tool calls to registry-backed implementations."""
from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
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
from .types import ActionRequest, Observation, ToolCall, ToolResult

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
        # Check if this is a batch request
        if action.tool_calls:
            return self.invoke_batch(action.tool_calls)

        # Single-tool mode (backward compatible)
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

    def invoke_batch(self, tool_calls: List[ToolCall]) -> Observation:
        """Execute multiple tools in parallel and return aggregated observation."""
        if not tool_calls:
            return Observation(
                success=False,
                outcome="No tool calls provided",
                error="Empty batch request",
            )

        # Run batch execution synchronously using ThreadPoolExecutor
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, can't use asyncio.run
                # Fall back to sequential execution
                results = []
                for call in tool_calls:
                    results.append(self._execute_single_tool(call))
            else:
                # Use asyncio.run for concurrent execution
                results = asyncio.run(self._execute_batch_async(tool_calls))
        except RuntimeError:
            # No event loop available, execute sequentially
            results = []
            for call in tool_calls:
                results.append(self._execute_single_tool(call))

        # Aggregate results
        all_success = all(r.success for r in results)
        total_calls = len(results)
        success_count = sum(1 for r in results if r.success)

        outcome_parts = [f"Executed {total_calls} tool(s): {success_count} succeeded"]
        if success_count < total_calls:
            outcome_parts.append(f"{total_calls - success_count} failed")

        # Aggregate metrics
        aggregated_metrics: Dict[str, Any] = {
            "total_calls": total_calls,
            "successful_calls": success_count,
            "failed_calls": total_calls - success_count,
        }

        # Sum up costs and wall times
        total_wall_time = sum(r.wall_time for r in results if r.wall_time)
        if total_wall_time > 0:
            aggregated_metrics["total_wall_time"] = total_wall_time
            aggregated_metrics["max_wall_time"] = max((r.wall_time for r in results if r.wall_time), default=0)

        # Collect all artifacts
        all_artifacts = []
        for r in results:
            if r.metrics.get("artifacts"):
                all_artifacts.extend(r.metrics["artifacts"])

        return Observation(
            success=all_success,
            outcome=", ".join(outcome_parts),
            metrics=aggregated_metrics,
            artifacts=all_artifacts,
            tool=f"batch[{total_calls}]",
            results=results,
        )

    async def _execute_batch_async(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute tool calls concurrently using asyncio."""
        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 10)) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, self._execute_single_tool, call)
                for call in tool_calls
            ]
            return await asyncio.gather(*tasks)

    def _execute_single_tool(self, call: ToolCall) -> ToolResult:
        """Execute a single tool and return its result."""
        start_time = time.time()
        tool_name = call.tool
        payload = call.args or {}

        try:
            result = self._invoke_registry(tool_name, payload)
            observation = self._wrap_result(tool_name, result)

            return ToolResult(
                call_id=call.call_id,
                tool=tool_name,
                success=observation.success,
                outcome=observation.outcome,
                error=observation.error,
                metrics={
                    **observation.metrics,
                    "artifacts": observation.artifacts,
                },
                wall_time=time.time() - start_time,
            )
        except KeyError:
            return ToolResult(
                call_id=call.call_id,
                tool=tool_name,
                success=False,
                outcome=f"Unknown tool: {tool_name}",
                error=f"Tool '{tool_name}' is not registered",
                wall_time=time.time() - start_time,
            )
        except ValueError as exc:
            return ToolResult(
                call_id=call.call_id,
                tool=tool_name,
                success=False,
                outcome=f"Tool {tool_name} rejected input",
                error=str(exc),
                wall_time=time.time() - start_time,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Tool %s execution failed", tool_name)
            return ToolResult(
                call_id=call.call_id,
                tool=tool_name,
                success=False,
                outcome=f"Tool {tool_name} failed",
                error=str(exc),
                wall_time=time.time() - start_time,
            )

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
            "success": success,
            "outcome": outcome,
            "metrics": metrics,
            "artifacts": [item for item in artifacts if item],
            "tool": tool_name,
            "raw_output": raw_output,
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

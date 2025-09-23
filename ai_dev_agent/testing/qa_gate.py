"""Quality gate helpers for tests and ReAct runs."""
from __future__ import annotations

from typing import Union

from ..react import RunResult
from .local_tests import TestResult


def passes(result: Union[TestResult, RunResult]) -> bool:
    if isinstance(result, RunResult):
        return result.status == "success"
    return result.success


def summarize(result: Union[TestResult, RunResult]) -> str:
    if isinstance(result, RunResult):
        lines = [
            f"Status: {result.status}",
            f"Stop reason: {result.stop_reason or 'n/a'}",
            "Gates:",
        ]
        for name, passed in sorted(result.gates.items()):
            flag = "PASS" if passed else "FAIL"
            lines.append(f"  {flag:<4} {name}")
        metrics = result.metrics or {}
        if metrics:
            lines.append("Metrics:")
            for key in ("diff_lines", "diff_files", "patch_coverage", "secrets_found"):
                if key in metrics and metrics[key] is not None:
                    lines.append(f"  {key}: {metrics[key]}")
        return "\n".join(lines)

    status = "PASSED" if result.success else "FAILED"
    return f"{status}: {' '.join(result.command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


__all__ = ["passes", "summarize"]

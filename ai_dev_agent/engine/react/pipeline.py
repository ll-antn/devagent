"""Implementation of a quality gate pipeline used by the ReAct loop."""
from __future__ import annotations

import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ai_dev_agent.engine.metrics import MetricsCollector
from ai_dev_agent.tools.execution.sandbox import SandboxExecutor, SandboxViolation
from ai_dev_agent.tools.analysis.security import scan_for_secrets
from ai_dev_agent.tools.execution.testing.local_tests import TestResult
from ai_dev_agent.core.utils.logger import get_logger
from .types import Observation

LOGGER = get_logger(__name__)


@dataclass
class PipelineCommands:
    format: Sequence[str] | None = None
    lint: Sequence[str] | None = None
    typecheck: Sequence[str] | None = None
    compile: Sequence[str] | None = None
    test: Sequence[str] | None = None
    coverage_xml: Path | None = None
    perf: Sequence[str] | None = None
    flake_runs: int = 0


def run_quality_pipeline(
    repo_root: Path,
    sandbox: SandboxExecutor,
    collector: MetricsCollector,
    commands: PipelineCommands,
    *,
    run_tests: bool = True,
    tokens_cost: Optional[float] = None,
) -> Observation:
    """Execute the quality pipeline and return an observation with metrics."""

    start = time.perf_counter()
    format_errors = _run_optional_command(sandbox, commands.format, "format")
    lint_errors = _run_optional_command(sandbox, commands.lint, "lint")
    type_errors = _run_optional_command(sandbox, commands.typecheck, "type-check")
    compile_errors = _run_optional_command(sandbox, commands.compile, "compile")

    test_result: TestResult | None = None
    test_logs: Dict[str, str] | None = None
    flaky_count = 0
    if run_tests and commands.test:
        # Primary test run
        test_process = sandbox.run(list(commands.test), capture_output=True, text=True, check=False)
        test_result = TestResult(
            command=list(commands.test),
            returncode=test_process.returncode,
            stdout=test_process.stdout,
            stderr=test_process.stderr,
        )
        test_logs = {
            "stdout": test_process.stdout,
            "stderr": test_process.stderr,
        }
        # Optional flake re-runs
        additional_runs = max(0, int(commands.flake_runs) - 1)
        for _ in range(additional_runs):
            proc = sandbox.run(list(commands.test), capture_output=True, text=True, check=False)
            # Count as flaky if results disagree
            if (proc.returncode == 0) != (test_result.returncode == 0):
                flaky_count += 1
    elif run_tests and not commands.test:
        LOGGER.warning("Test command not provided; tests will be considered skipped.")

    snapshot = collector.collect(
        test_result=test_result,
        lint_errors=lint_errors,
        type_errors=type_errors,
        format_errors=format_errors,
        compile_errors=compile_errors,
        coverage_xml=commands.coverage_xml,
        secrets_found=None,
        sandbox_violations=sandbox.stats.violations,
        flaky_tests=flaky_count,
        tokens_cost=tokens_cost,
        wall_time=time.perf_counter() - start,
    )

    diff_files = snapshot.raw.get("diff_files", [])
    secret_scan = scan_for_secrets(repo_root, diff_files)
    snapshot.secrets_found = secret_scan.count
    snapshot.raw.setdefault("secret_findings", [finding.__dict__ for finding in secret_scan.findings])

    snapshot.sandbox_violations = sandbox.stats.violations

    # Optional performance harness
    if commands.perf:
        try:
            perf_proc = sandbox.run(list(commands.perf), capture_output=True, text=True, check=False)
            snapshot.raw.setdefault("perf_ok", perf_proc.returncode == 0)
            if perf_proc.stdout:
                snapshot.raw.setdefault("perf_stdout", perf_proc.stdout[:4000])
            if perf_proc.stderr:
                snapshot.raw.setdefault("perf_stderr", perf_proc.stderr[:2000])
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.warning("Perf harness failed: %s", exc)

    outcome_parts = [
        "Executed quality pipeline",
        f"format={'pass' if format_errors == 0 else 'fail' if format_errors is not None else 'skip'}",
        f"lint={'pass' if lint_errors == 0 else 'fail' if lint_errors is not None else 'skip'}",
        f"types={'pass' if type_errors == 0 else 'fail' if type_errors is not None else 'skip'}",
    ]
    if compile_errors is not None:
        outcome_parts.append(f"compile={'pass' if compile_errors == 0 else 'fail'}")
    if run_tests and commands.test:
        outcome_parts.append(
            f"tests={'pass' if test_result and test_result.success else 'fail'}"
        )

    raw_output: List[str] = []
    if test_logs:
        raw_output.append("Test stdout:\n" + test_logs["stdout"])
        raw_output.append("Test stderr:\n" + test_logs["stderr"])

    observation = Observation(
        success=(snapshot.tests_passed is True),
        outcome="; ".join(outcome_parts),
        metrics=snapshot.model_dump(),
        artifacts=[],
        tool="qa.pipeline",
        raw_output="\n\n".join(raw_output) if raw_output else None,
    )
    return observation


def _run_optional_command(sandbox: SandboxExecutor, command: Sequence[str] | None, label: str) -> Optional[int]:
    if not command:
        return None
    LOGGER.debug("Running %s command: %s", label, " ".join(command))
    try:
        process = sandbox.run(list(command), capture_output=True, text=True, check=False)
    except SandboxViolation as exc:
        LOGGER.error("Sandbox violation during %s: %s", label, exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("%s command failed: %s", label, exc)
        return 1
    if process.returncode != 0:
        LOGGER.debug("%s command stderr: %s", label, process.stderr.strip())
        return 1
    return 0


def parse_command(command: str | Sequence[str] | None) -> Sequence[str] | None:
    if command is None:
        return None
    if isinstance(command, (list, tuple)):
        return list(command)
    text = str(command).strip()
    if not text:
        return None
    return shlex.split(text)


__all__ = [
    "PipelineCommands",
    "parse_command",
    "run_quality_pipeline",
]

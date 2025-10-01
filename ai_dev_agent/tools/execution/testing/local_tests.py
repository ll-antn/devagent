"""Local testing utilities."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ai_dev_agent.core.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class TestExecutionResult:
    """Result of a test execution.

    Note: Not a pytest test class - used to store test command results.
    """
    __test__ = False  # Tell pytest this is not a test class

    command: List[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0


# Backward compatibility alias
TestResult = TestExecutionResult


class TestRunner:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def run(self, command: Iterable[str]) -> TestExecutionResult:
        cmd = list(command)
        LOGGER.info("Running tests: %s", " ".join(cmd))
        process = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return TestExecutionResult(command=cmd, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)


__all__ = ["TestExecutionResult", "TestResult", "TestRunner"]

"""Local testing utilities."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ai_dev_agent.core.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class TestResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0


class TestRunner:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def run(self, command: Iterable[str]) -> TestResult:
        cmd = list(command)
        LOGGER.info("Running tests: %s", " ".join(cmd))
        process = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return TestResult(command=cmd, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)

    def run_pytest(self, extra_args: Iterable[str] | None = None) -> TestResult:
        args = ["pytest"]
        if extra_args:
            args.extend(extra_args)
        return self.run(args)


__all__ = ["TestResult", "TestRunner"]

"""Sandboxed command execution with basic allowlisting and resource caps."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from ..utils.logger import get_logger

LOGGER = get_logger(__name__)

try:  # pragma: no cover - platform dependent
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore


class SandboxViolation(RuntimeError):
    """Raised when a command violates sandbox policy."""


@dataclass
class SandboxConfig:
    allowlist: Sequence[str] = field(default_factory=list)
    default_timeout: float = 120.0
    cpu_time_limit: Optional[int] = 120
    memory_limit_mb: Optional[int] = 2048


@dataclass
class SandboxStats:
    violations: int = 0
    last_command: Sequence[str] | None = None
    last_returncode: Optional[int] = None


class SandboxExecutor:
    """Execute shell commands with allowlisting and resource limits."""

    def __init__(self, repo_root: Path, config: SandboxConfig | None = None) -> None:
        self.repo_root = Path(repo_root)
        self.config = config or SandboxConfig()
        self.allowlist = set(self.config.allowlist)
        self.stats = SandboxStats()

    def run(
        self,
        command: Sequence[str],
        *,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        capture_output: bool = True,
        text: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if not command:
            raise ValueError("Command must not be empty.")
        if not self._is_allowed(command):
            self.stats.violations += 1
            self.stats.last_command = command
            raise SandboxViolation(f"Command '{command[0]}' not allowed by sandbox policy.")

        run_cwd = Path(cwd) if cwd else self.repo_root
        run_cwd = run_cwd if run_cwd.is_dir() else self.repo_root
        timeout = timeout or self.config.default_timeout

        LOGGER.debug("Sandbox running: %s", " ".join(command))
        try:
            process = subprocess.run(
                list(command),
                cwd=str(run_cwd),
                env=self._prepare_env(env),
                capture_output=capture_output,
                text=text,
                timeout=timeout,
                check=check,
                preexec_fn=self._preexec_limits(),
            )
        except subprocess.TimeoutExpired as exc:
            self.stats.last_command = command
            self.stats.last_returncode = None
            LOGGER.error("Sandbox command timed out: %s", command)
            raise

        self.stats.last_command = command
        self.stats.last_returncode = process.returncode
        return process

    def _is_allowed(self, command: Sequence[str]) -> bool:
        if not self.allowlist:
            return True
        executable = Path(command[0]).name
        return executable in self.allowlist

    def _preexec_limits(self):  # pragma: no cover - platform dependent
        if resource is None or (self.config.cpu_time_limit is None and self.config.memory_limit_mb is None):
            return None

        def apply_limits() -> None:
            if self.config.cpu_time_limit is not None:
                resource.setrlimit(resource.RLIMIT_CPU, (self.config.cpu_time_limit, self.config.cpu_time_limit))
            if self.config.memory_limit_mb is not None:
                bytes_limit = self.config.memory_limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))

        return apply_limits

    def _prepare_env(self, overrides: Mapping[str, str] | None) -> Mapping[str, str]:
        base = os.environ.copy()
        if overrides:
            base.update(overrides)
        base.setdefault("PYTHONUNBUFFERED", "1")
        return base


__all__ = ["SandboxExecutor", "SandboxConfig", "SandboxStats", "SandboxViolation"]

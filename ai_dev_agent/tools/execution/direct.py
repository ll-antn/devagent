"""Direct command execution tool without sandbox restrictions."""
from __future__ import annotations

import os
import subprocess
import shlex
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..registry import ToolContext, ToolSpec, registry

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas" / "tools"


def _build_command(cmd: str, args: Sequence[str] | None) -> list[str]:
    if args:
        return [cmd, *args]
    return shlex.split(cmd)


def _exec_command(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    repo_root = context.repo_root

    cmd = payload["cmd"]
    args = payload.get("args")
    cwd_value = payload.get("cwd")
    cwd = repo_root if cwd_value is None else (repo_root / cwd_value).resolve()
    if repo_root not in cwd.parents and cwd != repo_root:
        raise ValueError("cwd must be within repository root")
    timeout = payload.get("timeout_sec")
    env_overrides = payload.get("env")
    env = None
    if env_overrides:
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in env_overrides.items()})

    command = _build_command(cmd, args)
    start = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=str(cwd),
        timeout=float(timeout) if timeout is not None else None,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    duration_ms = int((time.perf_counter() - start) * 1000)

    stdout_tail = process.stdout[-4000:] if process.stdout else ""
    stderr_tail = process.stderr[-2000:] if process.stderr else ""

    return {
        "exit_code": process.returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "log_path": None,
        "duration_ms": duration_ms,
    }


registry.register(
    ToolSpec(
        name="exec",
        handler=_exec_command,
        request_schema_path=SCHEMA_DIR / "exec.request.json",
        response_schema_path=SCHEMA_DIR / "exec.response.json",
        description=(
            "Execute a command directly. Provide 'cmd' (string) and optionally 'args' (list), "
            "'cwd' (string within the repo), and 'timeout_sec' (int)."
        ),
    )
)


__all__ = ["_exec_command"]

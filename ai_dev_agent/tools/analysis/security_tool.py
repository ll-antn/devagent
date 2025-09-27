"""Security related tool wrappers."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .security import scan_for_secrets
from ..registry import ToolSpec, ToolContext, registry

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas" / "tools"


def _collect_paths(repo_root: Path, paths: list[str] | None) -> list[str]:
    if paths:
        return paths
    process = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        return []
    return [line.strip() for line in process.stdout.splitlines() if line.strip()]


def _secrets_scan(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    repo_root = context.repo_root
    target_paths = _collect_paths(repo_root, payload.get("paths"))
    result = scan_for_secrets(repo_root, target_paths)

    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "secrets_scan.json"
    report_path.write_text(
        json.dumps({"findings": [finding.__dict__ for finding in result.findings]}, indent=2),
        encoding="utf-8",
    )

    try:
        rel_path = str(report_path.relative_to(repo_root))
    except ValueError:
        rel_path = str(report_path)

    return {"findings": result.count, "report_path": rel_path}


registry.register(
    ToolSpec(
        name="security.secrets_scan",
        handler=_secrets_scan,
        request_schema_path=SCHEMA_DIR / "security.secrets_scan.request.json",
        response_schema_path=SCHEMA_DIR / "security.secrets_scan.response.json",
        description="Scan files for potential secrets",
    )
)


__all__ = ["_secrets_scan"]

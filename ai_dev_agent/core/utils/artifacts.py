"""Helpers for persisting large tool outputs as artifacts."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

ARTIFACTS_ROOT = Path(".devagent") / "artifacts"


def write_artifact(content: str, *, suffix: str = ".txt", root: Optional[Path] = None) -> Path:
    """Persist content to an artifact file and return the path.

    Files are stored under `.devagent/artifacts` by default using a timestamp and
    hash-based naming scheme to avoid collisions."""

    base_dir = (root or Path.cwd()) / ARTIFACTS_ROOT
    base_dir.mkdir(parents=True, exist_ok=True)

    data = content.encode("utf-8", errors="replace")
    digest = hashlib.sha1(data).hexdigest()[:12]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"artifact_{timestamp}_{digest}{suffix}"

    path = base_dir / filename
    path.write_bytes(data)
    return path


__all__ = ["write_artifact", "ARTIFACTS_ROOT"]

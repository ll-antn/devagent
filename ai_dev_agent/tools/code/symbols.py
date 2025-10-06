"""Symbol indexing and lookup tools using Universal Ctags."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import json as _json
import shutil
from typing import Any, Dict, Iterable, Mapping

from ..registry import ToolSpec, ToolContext, registry

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas" / "tools"


def _ctags_command(context: ToolContext, targets: Iterable[str]) -> list[str]:
    cfg = getattr(context.devagent_config, "ctags_cmd", None)
    if cfg:
        if isinstance(cfg, list):
            cmd = list(cfg)
        else:
            cmd = str(cfg).split()
    else:
        cmd = [
            "ctags",
            "-R",
            "--output-format=json",
            "--fields=+n",
            "--fields=+r",
            "-f",
            "-",
        ]
    cmd.extend(targets)
    return cmd


def _ctags_db_path(context: ToolContext) -> Path:
    custom = getattr(context.devagent_config, "ctags_db", None)
    repo_root = context.repo_root
    if custom:
        return (repo_root / custom).resolve()
    return repo_root / ".devagent" / "index" / "ctags.jsonl"


def _ctags_meta_path(context: ToolContext) -> Path:
    db = _ctags_db_path(context)
    return db.with_name("ctags.meta.json")


def _load_meta(path: Path) -> dict | None:
    try:
        if path.exists():
            return _json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _write_meta(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        # Best effort; metadata is optional
        pass


def _git_head(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root), capture_output=True, text=True)
        if proc.returncode == 0:
            return proc.stdout.strip() or None
    except Exception:
        return None
    return None


def _working_tree_dirty(repo_root: Path) -> bool:
    try:
        proc = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo_root), capture_output=True, text=True)
        if proc.returncode == 0:
            return bool(proc.stdout.strip())
    except Exception:
        return False
    return False


def _symbols_index(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    repo_root = context.repo_root
    # Verify Universal Ctags availability up-front for clearer errors
    if not _has_universal_ctags():
        raise RuntimeError(
            "Universal Ctags is required for symbols.index. Install it (e.g., brew install universal-ctags)"
        )
    paths = payload.get("paths") or ["."]
    command = _ctags_command(context, paths)
    process = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        # Common failure: non-Universal ctags that doesn't support JSON/output format
        message = process.stderr.strip() or process.stdout.strip() or "ctags failed"
        raise RuntimeError(message)

    db_path = _ctags_db_path(context)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    db_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    files_indexed = set()
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        path = entry.get("path")
        if path:
            files_indexed.add(path)

    try:
        db_rel = str(db_path.relative_to(repo_root))
    except ValueError:
        db_rel = str(db_path)

    # Write metadata for refresh decisions
    meta_path = _ctags_meta_path(context)
    metadata = {
        "git_head": _git_head(repo_root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paths": list(paths),
        "stats": {
            "files_indexed": len(files_indexed),
            "symbols": len(lines),
        },
        "db_path": db_rel,
        "version": 1,
    }
    _write_meta(meta_path, metadata)

    return {
        "db_path": db_rel,
        "stats": {
            "files_indexed": len(files_indexed),
            "symbols": len(lines),
        },
    }


def _symbols_find(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    repo_root = context.repo_root
    db_path = _ctags_db_path(context)

    # Ensure index is present and reasonably fresh
    _ensure_index_fresh(context)

    name = payload["name"]
    kind = payload.get("kind")
    lang = payload.get("lang")

    defs: list[Dict[str, Any]] = []
    with db_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("name") != name:
                continue
            if kind and entry.get("kind") != kind:
                continue
            if lang and entry.get("language") != lang:
                continue
            defs.append(
                {
                    "path": entry.get("path"),
                    "line": entry.get("line", 1),
                    "kind": entry.get("kind", ""),
                    "scope": entry.get("scope"),
                }
            )

    return {"defs": defs, "refs": []}


def _has_universal_ctags() -> bool:
    path = shutil.which("ctags")
    if not path:
        return False
    proc = subprocess.run([path, "--version"], capture_output=True, text=True)
    return "Universal Ctags" in (proc.stdout or "")


def _ensure_index_fresh(context: ToolContext) -> None:
    """Ensure the ctags index exists and is refreshed when source has changed.

    Strategy:
    - Build index if missing.
    - If git HEAD changed since last index, rebuild.
    - Optionally, if configured refresh interval elapsed and working tree dirty, rebuild.
    """
    repo_root = context.repo_root
    db_path = _ctags_db_path(context)
    meta_path = _ctags_meta_path(context)

    # Build if missing
    if not db_path.exists():
        _symbols_index({}, context)
        return

    # Read metadata
    meta = _load_meta(meta_path) or {}
    last_head = meta.get("git_head")
    current_head = _git_head(repo_root)
    if last_head and current_head and last_head != current_head:
        _symbols_index({}, context)
        return

    # Time-based refresh when working tree is dirty and refresh interval set
    refresh_sec = getattr(context.devagent_config, "ctags_refresh_sec", None) if context.devagent_config else None
    if refresh_sec and _working_tree_dirty(repo_root):
        try:
            ts_text = meta.get("timestamp")
            if ts_text:
                last = datetime.fromisoformat(ts_text)
            else:
                last = None
        except Exception:
            last = None
        if last is None:
            _symbols_index({}, context)
            return
        now = datetime.now(timezone.utc)
        if (now - (last if last.tzinfo else last.replace(tzinfo=timezone.utc))).total_seconds() >= refresh_sec:
            _symbols_index({}, context)


registry.register(
    ToolSpec(
        name="symbols.index",
        handler=_symbols_index,
        request_schema_path=SCHEMA_DIR / "symbols.index.request.json",
        response_schema_path=SCHEMA_DIR / "symbols.index.response.json",
        description=(
            "Build or refresh the ctags symbol index for the repository. Run this ONCE at session start "
            "to enable fast symbol lookups. Accepts optional 'paths' (list of directories/files) to scope "
            "indexing. Required before using symbols.find. Fast indexing of functions, classes, variables."
        ),
    )
)

registry.register(
    ToolSpec(
        name="symbols.find",
        handler=_symbols_find,
        request_schema_path=SCHEMA_DIR / "symbols.find.request.json",
        response_schema_path=SCHEMA_DIR / "symbols.find.response.json",
        description=(
            "Look up symbol definitions using the ctags index (requires symbols.index to run first). "
            "Provide 'name' (string) to find. Optional: 'kind' (function, class, variable, method, etc.) "
            "and 'lang' (py, js, etc.) to refine search. Returns file path and line number of definitions. "
            "Faster than code.search for finding where symbols are defined."
        ),
    )
)


__all__ = ["_symbols_index", "_symbols_find"]

"""code.search implementation backed by ripgrep."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, List

from ..registry import ToolSpec, ToolContext, registry

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas" / "tools"


def _code_search(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    repo_root = context.repo_root
    query = payload["query"]
    regex = payload.get("regex", False)
    max_results = payload.get("max_results", 100)
    where = payload.get("where") or []

    if not shutil.which("rg"):
        matches = _fallback_search(repo_root, query, where, max_results)
        return {"matches": matches}

    command = ["rg", "--json", "--color", "never", "--max-columns", "500"]
    if not regex:
        command.append("--fixed-strings")
    command.append(query)
    if where:
        command.extend(where)

    process = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    if process.returncode not in {0, 1}:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "ripgrep failed")

    matches: list[dict[str, Any]] = []
    for line in process.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data", {})
        path = data.get("path", {}).get("text")
        if not path:
            continue
        subs = data.get("submatches") or []
        if not subs:
            continue
        sub = subs[0]
        preview = sub.get("match", {}).get("text", "")
        start = sub.get("start", 0)
        line_number = data.get("line_number", 1) or 1
        matches.append(
            {
                "path": path,
                "line": int(line_number),
                "col": int(start) + 1,
                "preview": preview,
            }
        )
        if len(matches) >= max_results:
            break

    return {"matches": matches}


def _fallback_search(repo_root: Path, query: str, where: List[str], max_results: int) -> List[dict[str, Any]]:
    """Simple Python fallback when ripgrep is not available."""
    query_str = str(query)
    # Determine directories to search
    search_roots: List[Path]
    if where:
        search_roots = []
        for rel in where:
            candidate = (repo_root / rel).resolve()
            if repo_root not in candidate.parents and candidate != repo_root:
                continue
            if candidate.exists():
                search_roots.append(candidate)
        if not search_roots:
            search_roots = [repo_root]
    else:
        search_roots = [repo_root]

    matches: List[dict[str, Any]] = []
    for root in search_roots:
        for path in root.rglob("*"):
            if len(matches) >= max_results:
                break
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            lines = text.splitlines()
            for idx, line in enumerate(lines, start=1):
                col = line.find(query_str)
                if col != -1:
                    try:
                        rel_path = str(path.relative_to(repo_root))
                    except ValueError:
                        rel_path = str(path)
                    matches.append(
                        {
                            "path": rel_path,
                            "line": idx,
                            "col": col + 1,
                            "preview": line.strip(),
                        }
                    )
                    break
    return matches[:max_results]


registry.register(
    ToolSpec(
        name="code.search",
        handler=_code_search,
        request_schema_path=SCHEMA_DIR / "code.search.request.json",
        response_schema_path=SCHEMA_DIR / "code.search.response.json",
        description=(
            "Search repository text for the provided query. By DEFAULT uses FIXED STRING matching - "
            "regex patterns like 'def.*test' will be treated as literal strings. Set 'regex': true "
            "to enable regex patterns. Supports optional 'where' (list of directories/files), and "
            "'max_results' (int). Automatically falls back to a Python-based scan when ripgrep is unavailable."
        ),
    )
)


__all__ = ["_code_search"]

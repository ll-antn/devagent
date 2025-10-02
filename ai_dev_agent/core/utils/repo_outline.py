"""Generate lightweight repository structure outlines for prompts."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .constants import DEFAULT_IGNORED_REPO_DIRS


def generate_repo_outline(
    root: Path,
    *,
    max_entries: int = 512,
    max_depth: int = 4,
    max_chars: int = 4_000,
    directories_only: bool = True,
) -> Optional[str]:
    """Return a repository outline capped by entry count/length via depth reduction."""

    try:
        root = root.resolve()
    except OSError:
        return None

    def iter_children(path: Path) -> List[Path]:
        try:
            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return []

        visible: List[Path] = []
        fallback_files: List[Path] = []
        for child in children:
            if child.name in DEFAULT_IGNORED_REPO_DIRS:
                continue
            if child.name.startswith('.'):
                continue
            if child.is_dir():
                visible.append(child)
            elif not directories_only:
                visible.append(child)
            else:
                fallback_files.append(child)

        if directories_only and not visible:
            visible = fallback_files[:3]

        return visible

    def build_outline(depth_limit: int) -> List[str]:
        outline: List[str] = [f"{root.name}/"]

        def walk(path: Path, current_depth: int, prefix: str) -> None:
            if current_depth > depth_limit:
                return

            children = iter_children(path)
            count = len(children)
            for index, child in enumerate(children):
                connector = "└── " if index == count - 1 else "├── "
                outline.append(f"{prefix}{connector}{child.name}{'/' if child.is_dir() else ''}")
                if child.is_dir():
                    extension = "    " if index == count - 1 else "│   "
                    walk(child, current_depth + 1, prefix + extension)

        walk(root, 1, "")
        return outline

    def outline_length(lines: List[str]) -> int:
        # Include newline characters between lines.
        return sum(len(line) + 1 for line in lines)

    for depth_limit in range(max_depth, 0, -1):
        entries = build_outline(depth_limit)
        if (
            len(entries) <= max_entries
            and outline_length(entries) <= max_chars
        ) or depth_limit == 1:
            return "\n".join(entries[:max_entries])

    return None


__all__ = ["generate_repo_outline"]

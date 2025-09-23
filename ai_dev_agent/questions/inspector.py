"""Repository inspection utilities for question answering."""
from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set

from ..utils.keywords import extract_keywords
from ..utils.logger import get_logger

LOGGER = get_logger(__name__)


_SKIP_DIRECTORIES = {
    ".git",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
}


_QUESTION_EXTRA_STOPWORDS = {
    "files",
    "directory",
    "directories",
    "root",
    "top",
    "level",
    "how",
    "does",
    "check",
    "see",
    "exist",
    "exists",
    "which",
    "list",
    "project",
    "what",
    "with",
    "for",
}


@dataclass(frozen=True)
class RepositoryFile:
    """Lightweight metadata describing a repository file."""

    path: Path
    size: int
    depth: int
    extension: str

    @property
    def relative(self) -> str:
        return str(self.path)


class RepositoryInspector:
    """Provides lazy discovery and querying of repository files."""

    def __init__(
        self,
        repo_root: Path,
        *,
        max_index_size: int = 2000,
        max_depth: int = 8,
    ) -> None:
        self.repo_root = repo_root
        self.max_index_size = max_index_size
        self.max_depth = max_depth
        self._indexed = False
        self._files: List[RepositoryFile] = []
        self._by_extension: dict[str, List[RepositoryFile]] = {}
        self._by_name: dict[str, List[RepositoryFile]] = {}

    def suggest_files(
        self,
        question: str,
        *,
        limit: int = 20,
    ) -> List[str]:
        """Suggest repository files that may help answer a question."""
        self._ensure_index()
        lowered = question.lower()
        suggestions: List[str] = []
        seen: Set[str] = set()

        explicit_paths = self._extract_explicit_paths(question)
        for candidate in explicit_paths:
            if candidate not in seen:
                seen.add(candidate)
                suggestions.append(candidate)
                if len(suggestions) >= limit:
                    return suggestions

        for directory in self._extract_directories(question):
            for repo_file in self._files_in_directory(directory, limit - len(suggestions)):
                rel = repo_file.relative
                if rel not in seen:
                    seen.add(rel)
                    suggestions.append(rel)
                    if len(suggestions) >= limit:
                        return suggestions

        for extension in self._detect_extensions(lowered):
            root_only = any(token in lowered for token in {"root", "top-level", "top level"})
            for repo_file in self._files_by_extension(extension, root_only):
                rel = repo_file.relative
                if rel not in seen:
                    seen.add(rel)
                    suggestions.append(rel)
                    if len(suggestions) >= limit:
                        return suggestions

        keywords = extract_keywords(
            question,
            extra_stopwords=_QUESTION_EXTRA_STOPWORDS,
        )
        for keyword in keywords:
            for repo_file in self._files_matching_keyword(keyword):
                rel = repo_file.relative
                if rel not in seen:
                    seen.add(rel)
                    suggestions.append(rel)
                    if len(suggestions) >= limit:
                        return suggestions

        return suggestions[:limit]

    # Index helpers -----------------------------------------------------

    def _ensure_index(self) -> None:
        if self._indexed:
            return
        self._build_index()
        self._indexed = True

    def _build_index(self) -> None:
        LOGGER.debug("Building repository file index for %s", self.repo_root)
        queue: deque[tuple[Path, int]] = deque([(self.repo_root, 0)])

        while queue and len(self._files) < self.max_index_size:
            current_dir, depth = queue.popleft()
            if depth > self.max_depth:
                continue

            try:
                entries = sorted(current_dir.iterdir())
            except (OSError, PermissionError) as exc:
                LOGGER.debug("Skipping directory %s: %s", current_dir, exc)
                continue

            for entry in entries:
                relative_path = entry.relative_to(self.repo_root)
                if entry.is_dir():
                    if entry.name in _SKIP_DIRECTORIES:
                        continue
                    queue.append((entry, depth + 1))
                    continue

                if not entry.is_file():
                    continue

                size = entry.stat().st_size if entry.exists() else 0
                repo_file = RepositoryFile(
                    path=relative_path,
                    size=size,
                    depth=depth,
                    extension=entry.suffix.lower().lstrip("."),
                )
                self._files.append(repo_file)
                self._by_extension.setdefault(repo_file.extension, []).append(repo_file)
                self._index_by_name(relative_path, repo_file)

                if len(self._files) >= self.max_index_size:
                    break

        LOGGER.debug("Indexed %d files for inspection", len(self._files))

    def _index_by_name(self, relative_path: Path, repo_file: RepositoryFile) -> None:
        fragments = set(relative_path.parts)
        fragments.add(relative_path.stem.lower())
        for fragment in fragments:
            key = fragment.lower()
            if not key:
                continue
            self._by_name.setdefault(key, []).append(repo_file)

    # Query helpers -----------------------------------------------------

    def _files_by_extension(self, extension: str, root_only: bool) -> Iterable[RepositoryFile]:
        matches = self._by_extension.get(extension, [])
        for repo_file in matches:
            if root_only and repo_file.depth != 0:
                continue
            yield repo_file

    def _files_matching_keyword(self, keyword: str) -> Iterable[RepositoryFile]:
        keyword_lower = keyword.lower()
        # Match exact fragment hits first
        for repo_file in self._by_name.get(keyword_lower, []):
            yield repo_file

        # Fall back to substring search in remaining files
        for repo_file in self._files:
            if keyword_lower in repo_file.relative.lower():
                yield repo_file

    def _files_in_directory(self, directory: str, limit: int) -> Iterable[RepositoryFile]:
        if limit <= 0:
            return []
        normalized = directory.rstrip("/")
        collected: List[RepositoryFile] = []
        for repo_file in self._files:
            if repo_file.relative.startswith(normalized):
                collected.append(repo_file)
                if len(collected) >= limit:
                    break
        return collected

    # Question parsing helpers -----------------------------------------

    def _extract_explicit_paths(self, question: str) -> List[str]:
        candidates: List[str] = []
        for match in re.findall(r"([A-Za-z0-9_.\-/]+\.[A-Za-z0-9_]+)", question):
            rel = Path(match.lstrip("/"))
            target = (self.repo_root / rel)
            if target.is_file():
                candidates.append(str(rel))
        return candidates

    def _extract_directories(self, question: str) -> List[str]:
        directories: Set[str] = set()
        for match in re.findall(r"([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)/", question):
            rel = Path(match.lstrip("/")).as_posix()
            if (self.repo_root / rel).is_dir():
                directories.add(rel)
        return list(directories)

    def _detect_extensions(self, lowered_question: str) -> List[str]:
        found: Set[str] = set()
        extension_keywords = {
            "toml": "toml",
            "yaml": "yaml",
            "yml": "yml",
            "json": "json",
            "md": "md",
            "rst": "rst",
            "ini": "ini",
        }
        for keyword, extension in extension_keywords.items():
            if keyword in lowered_question:
                found.add(extension)
        return list(found)

__all__ = ["RepositoryInspector", "RepositoryFile"]

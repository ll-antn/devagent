"""Parse unified diff patches into structured data for analysis."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ai_dev_agent.core.utils.logger import get_logger
from ai_dev_agent.tools.registry import ToolSpec, ToolContext, registry

LOGGER = get_logger(__name__)
SCHEMA_DIR = Path(__file__).resolve().parent / "schemas" / "tools"


class PatchParser:
    """Parse unified diff patches into structured data."""

    # Regex patterns for parsing
    DIFF_HEADER = re.compile(r'^diff --git a/(.*) b/(.*)$')
    FILE_HEADER_OLD = re.compile(r'^--- (?:a/)?(.*)$')  # a/ prefix is optional
    FILE_HEADER_NEW = re.compile(r'^\+\+\+ (?:b/)?(.*)$')  # b/ prefix is optional
    HUNK_HEADER = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$')
    COMMIT_HEADER = re.compile(r'^From ([0-9a-f]{40}) ')

    def __init__(self, patch_content: str, include_context: bool = False):
        """Initialize parser with patch content.

        Args:
            patch_content: The unified diff patch as a string
            include_context: Whether to include unchanged context lines
        """
        self.lines = patch_content.splitlines()
        self.include_context = include_context
        self.current_line = 0

    def parse(self, filter_pattern: Optional[str] = None) -> Dict[str, Any]:
        """Parse the entire patch.

        Args:
            filter_pattern: Optional regex to filter files by path

        Returns:
            Dictionary with patch_info, files, and summary
        """
        patch_info = self._extract_commit_info()
        files = []

        while self.current_line < len(self.lines):
            file_entry = self._parse_file()
            if file_entry:
                # Apply filter if provided
                if filter_pattern:
                    if re.search(filter_pattern, file_entry['path']):
                        files.append(file_entry)
                else:
                    files.append(file_entry)

        return {
            "patch_info": patch_info,
            "files": files,
            "summary": self._compute_summary(files)
        }

    def _extract_commit_info(self) -> Dict[str, str]:
        """Extract commit metadata from patch header."""
        info = {}
        # Check first few lines for git format-patch headers
        for i in range(min(20, len(self.lines))):
            line = self.lines[i]
            if match := self.COMMIT_HEADER.match(line):
                info['commit'] = match.group(1)
            elif line.startswith('From: '):
                info['author'] = line[6:].strip()
            elif line.startswith('Date: '):
                info['date'] = line[6:].strip()
            elif line.startswith('Subject: '):
                info['message'] = line[9:].strip()
        return info

    def _parse_file(self) -> Optional[Dict[str, Any]]:
        """Parse a single file's changes."""
        # Find next diff header
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line]
            if match := self.DIFF_HEADER.match(line):
                old_path = match.group(1)
                new_path = match.group(2)
                self.current_line += 1
                break
            self.current_line += 1
        else:
            return None  # No more files

        # Parse --- and +++ lines to determine actual change type
        old_file = None
        new_file = None

        # Look for --- line
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line]
            if match := self.FILE_HEADER_OLD.match(line):
                old_file = match.group(1)
                self.current_line += 1
                break
            self.current_line += 1

        # Look for +++ line
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line]
            if match := self.FILE_HEADER_NEW.match(line):
                new_file = match.group(1)
                self.current_line += 1
                break
            self.current_line += 1

        # Determine change type from --- +++ lines
        change_type = "modified"
        if old_file == "/dev/null":
            change_type = "added"
            actual_path = new_path
        elif new_file == "/dev/null":
            change_type = "deleted"
            actual_path = old_path
        elif old_path != new_path:
            change_type = "renamed"
            actual_path = new_path
        else:
            actual_path = new_path

        # Parse all hunks for this file
        hunks = []
        additions = 0
        deletions = 0

        while self.current_line < len(self.lines):
            line = self.lines[self.current_line]

            # Check if we've hit the next file
            if self.DIFF_HEADER.match(line):
                break

            # Parse hunk
            if hunk := self._parse_hunk():
                hunks.append(hunk)
                additions += len(hunk['added_lines'])
                deletions += len(hunk['removed_lines'])
            else:
                self.current_line += 1

        return {
            "path": actual_path,
            "change_type": change_type,
            "old_path": old_path if change_type == "renamed" else None,
            "language": self._detect_language(actual_path),
            "hunks": hunks,
            "stats": {
                "additions": additions,
                "deletions": deletions,
                "total_hunks": len(hunks)
            }
        }

    def _parse_hunk(self) -> Optional[Dict[str, Any]]:
        """Parse a single hunk."""
        if self.current_line >= len(self.lines):
            return None

        line = self.lines[self.current_line]
        match = self.HUNK_HEADER.match(line)
        if not match:
            return None

        old_start = int(match.group(1))
        old_count = int(match.group(2) or 1)
        new_start = int(match.group(3))
        new_count = int(match.group(4) or 1)
        header_context = match.group(5).strip()

        self.current_line += 1

        added_lines = []
        removed_lines = []
        context_lines = []

        new_line_num = new_start
        old_line_num = old_start

        # Parse hunk content
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line]

            # End of hunk markers
            if (line.startswith('@@') or
                line.startswith('diff --git') or
                (line.startswith('---') and self.FILE_HEADER_OLD.match(line)) or
                (line.startswith('+++') and self.FILE_HEADER_NEW.match(line))):
                break

            if line.startswith('+') and not line.startswith('+++'):
                # Added line
                content = line[1:]
                added_lines.append({
                    "line_number": new_line_num,
                    "content": content,
                    "indentation": self._get_indentation(content)
                })
                new_line_num += 1
            elif line.startswith('-') and not line.startswith('---'):
                # Removed line
                removed_lines.append({
                    "line_number": old_line_num,
                    "content": line[1:]
                })
                old_line_num += 1
            elif line.startswith(' ') or line == '':
                # Context line (including empty lines which are context)
                if self.include_context:
                    content = line[1:] if line else ''
                    context_lines.append({
                        "line_number": new_line_num,
                        "content": content
                    })
                new_line_num += 1
                old_line_num += 1
            elif line.startswith('\\'):
                # "\ No newline at end of file" - skip
                pass
            else:
                # Unknown line format - end of hunk
                break

            self.current_line += 1

        result = {
            "header": f"@@ -{old_start},{old_count} +{new_start},{new_count} @@ {header_context}",
            "old_start": old_start,
            "old_count": old_count,
            "new_start": new_start,
            "new_count": new_count,
            "added_lines": added_lines,
            "removed_lines": removed_lines
        }

        if self.include_context:
            result["context_lines"] = context_lines

        return result

    @staticmethod
    def _get_indentation(line: str) -> str:
        """Extract leading whitespace."""
        return line[:len(line) - len(line.lstrip())]

    @staticmethod
    def _detect_language(path: str) -> str:
        """Detect language from file extension."""
        ext = Path(path).suffix.lower()
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.ets': 'ets',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.sh': 'shell',
            '.md': 'markdown',
        }
        return lang_map.get(ext, ext[1:] if ext else 'unknown')

    @staticmethod
    def _compute_summary(files: List[Dict]) -> Dict[str, int]:
        """Compute overall patch statistics."""
        summary = {
            "total_files": len(files),
            "files_added": 0,
            "files_modified": 0,
            "files_deleted": 0,
            "total_additions": 0,
            "total_deletions": 0
        }

        for f in files:
            if f['change_type'] == 'added':
                summary['files_added'] += 1
            elif f['change_type'] == 'deleted':
                summary['files_deleted'] += 1
            else:
                summary['files_modified'] += 1

            summary['total_additions'] += f['stats']['additions']
            summary['total_deletions'] += f['stats']['deletions']

        return summary


# Tool function for registry
def parse_patch_handler(payload: Mapping[str, Any], context: ToolContext) -> Dict[str, Any]:
    """Parse a unified diff patch file into structured data.

    This tool parses unified diff patches (git format-patch, git diff output)
    into structured JSON showing added/removed lines per file with accurate
    line numbers. Use this instead of sed/grep for patch analysis.

    Args:
        payload: Tool input with 'path', optional 'include_context', 'filter_pattern'
        context: Tool context (provides repo_root, settings, etc.)

    Returns:
        Dictionary containing:
        - success: bool
        - patch_info: commit metadata (if available)
        - files: list of file changes with hunks and line-by-line diffs
        - summary: overall statistics
        - error: error message (if success=False)

    Example:
        parse_patch(path="changes.patch", filter_pattern="src/.*\\.py$")
    """
    path = payload.get("path", "")
    include_context = payload.get("include_context", False)
    filter_pattern = payload.get("filter_pattern")

    # Resolve path relative to repo root if needed
    patch_path = Path(path)
    if not patch_path.is_absolute():
        patch_path = context.repo_root / patch_path

    if not patch_path.exists():
        return {
            "success": False,
            "error": f"Patch file not found: {path}",
            "files": [],
            "summary": {
                "total_files": 0,
                "files_added": 0,
                "files_modified": 0,
                "files_deleted": 0,
                "total_additions": 0,
                "total_deletions": 0
            }
        }

    try:
        content = patch_path.read_text(encoding='utf-8')
        parser = PatchParser(content, include_context=include_context)
        result = parser.parse(filter_pattern=filter_pattern)

        return {
            "success": True,
            **result
        }
    except Exception as e:
        LOGGER.exception("Failed to parse patch file: %s", path)
        return {
            "success": False,
            "error": f"Failed to parse patch: {str(e)}",
            "files": [],
            "summary": {
                "total_files": 0,
                "files_added": 0,
                "files_modified": 0,
                "files_deleted": 0,
                "total_additions": 0,
                "total_deletions": 0
            }
        }


# Register the tool
registry.register(
    ToolSpec(
        name="parse_patch",
        handler=parse_patch_handler,
        request_schema_path=SCHEMA_DIR / "parse_patch.request.json",
        response_schema_path=SCHEMA_DIR / "parse_patch.response.json",
        description=(
            "Parse unified diff patches into structured JSON with ALL added/removed lines per file "
            "and accurate line numbers. This tool COMPLETELY PARSES the patch file - no need for additional "
            "sed/grep commands to extract content.\n\n"
            "IMPORTANT: This tool returns the COMPLETE content of every added line in the patch, with:\n"
            "- Exact line numbers in the new file (after patch is applied)\n"
            "- Full line content including code, documentation, exports, public/private declarations\n"
            "- File paths, change types (added/modified/deleted), and language detection\n\n"
            "Use this tool ONCE instead of dozens of sed/grep commands. The returned data contains everything "
            "you need to analyze the patch - just iterate through files[].hunks[].added_lines[].\n\n"
            "Examples:\n"
            "  parse_patch(path='changes.patch') - get ALL lines from ALL files\n"
            "  parse_patch(path='changes.patch', filter_pattern='stdlib/.*\\.ets$') - only .ets files in stdlib\n"
            "  parse_patch(path='changes.patch', include_context=True) - include unchanged lines too\n\n"
            "After calling this tool, you have the complete patch structure - no further extraction needed."
        ),
        category="analysis",
    )
)


__all__ = ["parse_patch_handler", "PatchParser"]

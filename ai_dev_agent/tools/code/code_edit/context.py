"""Enhanced context gathering with intelligent file discovery and analysis."""
from __future__ import annotations

import ast
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ai_dev_agent.core.utils.logger import get_logger
from .tree_sitter_analysis import TreeSitterProjectAnalyzer

LOGGER = get_logger(__name__)


@dataclass
class FileContext:
    path: Path
    content: str
    relevance_score: float = 0.0
    reason: str = "explicitly_requested"
    size_bytes: int = field(default=0, init=False)
    
    def __post_init__(self):
        self.size_bytes = len(self.content.encode('utf-8'))


@dataclass
class ContextGatheringOptions:
    """Configuration for context gathering behavior."""
    max_files: int = 20
    max_total_size: int = 100_000  # ~100KB of text
    include_tests: bool = True
    include_docs: bool = False
    search_depth: int = 3
    follow_imports: bool = True
    include_related_files: bool = True
    include_structure_summary: bool = True
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "*.pyo", "__pycache__/*", ".git/*", "node_modules/*",
        "*.min.js", "*.bundle.js", "dist/*", "build/*", ".venv/*",
        "venv/*", ".env", "*.log", "*.tmp"
    ])


class ContextGatherer:
    """Intelligent context gathering with file discovery heuristics."""
    
    def __init__(self, repo_root: Path, options: Optional[ContextGatheringOptions] = None):
        self.repo_root = repo_root
        self.options = options or ContextGatheringOptions()
        self._git_available = self._check_git_available()
        self._structure_analyzer = TreeSitterProjectAnalyzer(repo_root)
        
    def gather_contexts(
        self, 
        files: Iterable[str], 
        task_description: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> List[FileContext]:
        """Gather file contexts with intelligent discovery."""
        requested_files = set(files)
        all_contexts = {}
        
        # Start with explicitly requested files
        for rel_path in requested_files:
            try:
                context = self._load_file_context(rel_path, "explicitly_requested", 1.0)
                if context:
                    all_contexts[rel_path] = context
            except Exception as exc:
                LOGGER.warning("Failed to load requested file %s: %s", rel_path, exc)
        
        # Discover additional relevant files
        if self.options.include_related_files:
            discovered = self._discover_related_files(requested_files, task_description, keywords)
            for rel_path, reason, score in discovered:
                if rel_path not in all_contexts:
                    try:
                        context = self._load_file_context(rel_path, reason, score)
                        if context:
                            all_contexts[rel_path] = context
                    except Exception as exc:
                        LOGGER.debug("Failed to load discovered file %s: %s", rel_path, exc)
        
        # Sort by relevance and apply limits
        contexts = list(all_contexts.values())

        if self.options.include_structure_summary:
            summary_context = self._build_structure_summary(contexts)
            if summary_context:
                contexts.append(summary_context)

        contexts.sort(key=lambda c: c.relevance_score, reverse=True)

        return self._apply_size_limits(contexts)
    
    def search_files(self, pattern: str, file_types: Optional[List[str]] = None) -> List[str]:
        """Search for files containing a pattern."""
        if self._git_available:
            return self._git_grep_search(pattern, file_types)
        else:
            return self._fallback_search(pattern, file_types)
    
    def find_symbol_references(self, symbol: str, file_types: Optional[List[str]] = None) -> List[Tuple[str, int]]:
        """Find files and line numbers where a symbol is referenced."""
        pattern = rf'\b{re.escape(symbol)}\b'
        results = []
        
        if self._git_available:
            try:
                cmd = ["git", "grep", "-n", pattern]
                if file_types:
                    for ft in file_types:
                        cmd.extend(["--", f"*.{ft}"])
                
                result = subprocess.run(
                    cmd, 
                    cwd=self.repo_root, 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if ':' in line:
                            parts = line.split(':', 2)
                            if len(parts) >= 2:
                                file_path = parts[0]
                                line_num = int(parts[1]) if parts[1].isdigit() else 0
                                results.append((file_path, line_num))
            except Exception as exc:
                LOGGER.debug("Git grep failed: %s", exc)
        
        return results
    
    def find_related_files(self, file_path: str) -> List[Tuple[str, str]]:
        """Find files related to the given file."""
        related = []
        path = Path(file_path)
        
        # Test files
        if self.options.include_tests:
            test_files = self._find_test_files(path)
            related.extend([(f, "test_file") for f in test_files])
        
        # Import dependencies
        if self.options.follow_imports and path.suffix == '.py':
            imports = self._find_python_imports(path)
            related.extend([(f, "import_dependency") for f in imports])
        
        # Configuration files
        config_files = self._find_config_files(path)
        related.extend([(f, "config_file") for f in config_files])
        
        # Files in same directory
        sibling_files = self._find_sibling_files(path)
        related.extend([(f, "sibling_file") for f in sibling_files])
        
        return related
    
    def _discover_related_files(
        self, 
        base_files: Set[str], 
        task_description: Optional[str], 
        keywords: Optional[List[str]]
    ) -> List[Tuple[str, str, float]]:
        """Discover files related to the base set."""
        discovered = []
        
        # Find files related to each base file
        for file_path in base_files:
            related = self.find_related_files(file_path)
            for rel_path, reason in related:
                if rel_path not in base_files:
                    score = self._calculate_relevance_score(rel_path, reason, task_description, keywords)
                    if score > 0.1:  # Minimum relevance threshold
                        discovered.append((rel_path, reason, score))
        
        # Search for files mentioned in task description
        if task_description and keywords:
            for keyword in keywords:
                matches = self.search_files(keyword, ["py", "js", "ts", "java", "cpp", "h"])
                for match in matches[:5]:  # Limit search results
                    if match not in base_files:
                        score = self._calculate_relevance_score(match, "keyword_match", task_description, keywords)
                        if score > 0.2:
                            discovered.append((match, f"keyword_match({keyword})", score))
        
        return discovered
    
    def _calculate_relevance_score(
        self, 
        file_path: str, 
        reason: str, 
        task_description: Optional[str], 
        keywords: Optional[List[str]]
    ) -> float:
        """Calculate relevance score for a file."""
        base_score = {
            "explicitly_requested": 1.0,
            "test_file": 0.8,
            "import_dependency": 0.7,
            "sibling_file": 0.5,
            "config_file": 0.4,
            "keyword_match": 0.6,
        }.get(reason, 0.3)
        
        # Boost score based on file type and task
        path = Path(file_path)
        
        # Python files get higher score for Python tasks
        if path.suffix == '.py' and task_description:
            if any(word in task_description.lower() for word in ['python', 'function', 'class', 'module']):
                base_score *= 1.2
        
        # Test files get higher score if task mentions testing
        if 'test' in path.name and task_description:
            if any(word in task_description.lower() for word in ['test', 'testing', 'unittest', 'pytest']):
                base_score *= 1.3
        
        # Reduce score for large files
        try:
            file_size = (self.repo_root / file_path).stat().st_size
            if file_size > 50000:  # 50KB
                base_score *= 0.7
            elif file_size > 20000:  # 20KB
                base_score *= 0.8
        except (OSError, FileNotFoundError):
            pass
        
        return min(base_score, 1.0)
    
    def _load_file_context(self, rel_path: str, reason: str, score: float) -> Optional[FileContext]:
        """Load a single file context."""
        full_path = (self.repo_root / rel_path).resolve()
        
        if not full_path.exists():
            LOGGER.debug("File not found: %s", rel_path)
            return None
        
        if not self._should_include_file(full_path):
            return None
        
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            return FileContext(
                path=full_path,
                content=content,
                relevance_score=score,
                reason=reason
            )
        except Exception as exc:
            LOGGER.warning("Failed to read file %s: %s", rel_path, exc)
            return None
    
    def _should_include_file(self, file_path: Path) -> bool:
        """Check if file should be included based on patterns and size."""
        rel_path = str(file_path.relative_to(self.repo_root))
        
        # Check exclude patterns
        for pattern in self.options.exclude_patterns:
            if Path(rel_path).match(pattern):
                return False
        
        # Check file size
        try:
            if file_path.stat().st_size > 200_000:  # 200KB limit per file
                LOGGER.debug("Skipping large file: %s", rel_path)
                return False
        except OSError:
            return False
        
        # Check if it's a text file
        try:
            with file_path.open('rb') as f:
                chunk = f.read(1024)
                if b'\x00' in chunk:  # Binary file detection
                    return False
        except OSError:
            return False
        
        return True
    
    def _apply_size_limits(self, contexts: List[FileContext]) -> List[FileContext]:
        """Apply size and count limits to contexts."""
        limited_contexts = []
        total_size = 0
        
        for context in contexts:
            if len(limited_contexts) >= self.options.max_files:
                break
            if total_size + context.size_bytes > self.options.max_total_size:
                # Try to include partial content for important files
                if context.relevance_score > 0.8 and len(limited_contexts) < 5:
                    remaining_size = self.options.max_total_size - total_size
                    if remaining_size > 1000:  # At least 1KB
                        truncated_content = context.content[:remaining_size] + "\n... [truncated]"
                        limited_contexts.append(FileContext(
                            path=context.path,
                            content=truncated_content,
                            relevance_score=context.relevance_score,
                            reason=f"{context.reason} (truncated)"
                        ))
                break
            
            limited_contexts.append(context)
            total_size += context.size_bytes
        
        LOGGER.info(
            "Gathered %d files (%d bytes) out of %d candidates",
            len(limited_contexts), total_size, len(contexts)
        )
        
        return limited_contexts
    
    def _find_test_files(self, file_path: Path) -> List[str]:
        """Find test files related to the given file."""
        test_files = []
        full_path = file_path if file_path.is_absolute() else self.repo_root / file_path

        try:
            rel_path = full_path.relative_to(self.repo_root)
        except ValueError:
            return test_files
        
        # Common test patterns
        test_patterns = [
            f"test_{rel_path.stem}.py",
            f"{rel_path.stem}_test.py",
            f"tests/test_{rel_path.stem}.py",
            f"test/test_{rel_path.stem}.py",
            f"{rel_path.parent}/test_{rel_path.name}",
        ]
        
        for pattern in test_patterns:
            test_path = self.repo_root / pattern
            if test_path.exists():
                try:
                    rel_test_path = str(test_path.relative_to(self.repo_root))
                    test_files.append(rel_test_path)
                except ValueError:
                    continue
        
        return test_files
    
    def _find_python_imports(self, file_path: Path) -> List[str]:
        """Find Python files imported by the given file."""
        imports = []
        
        try:
            full_path = file_path if file_path.is_absolute() else self.repo_root / file_path
            content = full_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(full_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_path = self._resolve_python_import(alias.name)
                        if import_path:
                            imports.append(import_path)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        import_path = self._resolve_python_import(node.module)
                        if import_path:
                            imports.append(import_path)
                            
        except (SyntaxError, UnicodeDecodeError, OSError) as exc:
            LOGGER.debug("Failed to parse imports from %s: %s", file_path, exc)
        
        return imports
    
    def _resolve_python_import(self, module_name: str) -> Optional[str]:
        """Resolve a Python import to a file path."""
        # Handle relative imports and local modules
        if module_name.startswith('.'):
            return None  # Skip relative imports for now
        
        # Convert module name to file path
        parts = module_name.split('.')
        potential_paths = [
            Path(*parts) / "__init__.py",
            Path(*parts).with_suffix('.py'),
        ]
        
        for path in potential_paths:
            full_path = self.repo_root / path
            if full_path.exists():
                try:
                    return str(path)
                except ValueError:
                    continue
        
        return None

    def _build_structure_summary(self, contexts: List[FileContext]) -> Optional[FileContext]:
        """Construct a synthetic context summarising project structure via tree-sitter."""
        if not self._structure_analyzer.available:
            return None

        entries: List[Tuple[str, str]] = []

        for context in contexts:
            try:
                rel_path = str(context.path.relative_to(self.repo_root))
            except ValueError:
                continue

            if Path(rel_path).suffix not in self._structure_analyzer.supported_suffixes:
                continue

            entries.append((rel_path, context.content))

        if not entries:
            return None

        summary_text = self._structure_analyzer.build_project_summary(entries)
        if not summary_text:
            return None

        synthetic_path = (self.repo_root / "__project_structure__.md").resolve()
        return FileContext(
            path=synthetic_path,
            content=summary_text,
            relevance_score=0.98,
            reason="project_structure_summary"
        )

    def _find_config_files(self, file_path: Path) -> List[str]:
        """Find configuration files related to the given file."""
        config_files = []
        
        # Common config patterns
        config_patterns = [
            "pyproject.toml", "setup.py", "requirements.txt", "Pipfile",
            "package.json", "tsconfig.json", "webpack.config.js",
            "Makefile", "CMakeLists.txt", ".gitignore", "Dockerfile"
        ]
        
        full_path = file_path if file_path.is_absolute() else self.repo_root / file_path
        current_dir = full_path if full_path.is_dir() else full_path.parent

        for _ in range(3):  # Check up to 3 levels up
            try:
                rel_dir = current_dir.relative_to(self.repo_root)
            except ValueError:
                break

            for pattern in config_patterns:
                config_path = current_dir / pattern
                if config_path.exists():
                    rel_config_path = str(config_path.relative_to(self.repo_root))
                    config_files.append(rel_config_path)

            if current_dir == self.repo_root:
                break
            current_dir = current_dir.parent

        return config_files
    
    def _find_sibling_files(self, file_path: Path) -> List[str]:
        """Find sibling files in the same directory."""
        siblings = []
        
        full_path = file_path if file_path.is_absolute() else self.repo_root / file_path

        try:
            dir_path = full_path.parent
            if not dir_path.is_dir():
                return siblings
            
            for sibling in dir_path.iterdir():
                if (sibling.is_file() and 
                    sibling != full_path and
                    sibling.suffix in {'.py', '.js', '.ts', '.java', '.cpp', '.h', '.md'}):
                    try:
                        rel_sibling_path = str(sibling.relative_to(self.repo_root))
                        siblings.append(rel_sibling_path)
                    except ValueError:
                        continue
                        
        except OSError:
            pass
        
        return siblings[:5]  # Limit to 5 siblings
    
    def _git_grep_search(self, pattern: str, file_types: Optional[List[str]] = None) -> List[str]:
        """Search using git grep."""
        try:
            cmd = ["git", "grep", "-l", pattern]
            if file_types:
                for ft in file_types:
                    cmd.extend(["--", f"*.{ft}"])
            
            result = subprocess.run(
                cmd, 
                cwd=self.repo_root, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
        except Exception as exc:
            LOGGER.debug("Git grep search failed: %s", exc)
        
        return []
    
    def _fallback_search(self, pattern: str, file_types: Optional[List[str]] = None) -> List[str]:
        """Fallback search using file system traversal."""
        matches = []
        
        try:
            for root, dirs, files in os.walk(self.repo_root):
                # Skip common ignore patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'node_modules'}]
                
                for file in files:
                    if file_types and not any(file.endswith(f'.{ft}') for ft in file_types):
                        continue
                    
                    file_path = Path(root) / file
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        if pattern in content:
                            rel_path = str(file_path.relative_to(self.repo_root))
                            matches.append(rel_path)
                            
                        if len(matches) >= 20:  # Limit results
                            break
                    except (OSError, UnicodeDecodeError):
                        continue
                        
                if len(matches) >= 20:
                    break
                    
        except Exception as exc:
            LOGGER.debug("Fallback search failed: %s", exc)
        
        return matches
    
    def _check_git_available(self) -> bool:
        """Check if git is available and we're in a git repo."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_root,
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False


def gather_file_contexts(repo_root: Path, files: Iterable[str]) -> List[FileContext]:
    """Legacy function for backward compatibility."""
    gatherer = ContextGatherer(repo_root)
    return gatherer.gather_contexts(files)


__all__ = ["FileContext", "ContextGatherer", "ContextGatheringOptions", "gather_file_contexts"]

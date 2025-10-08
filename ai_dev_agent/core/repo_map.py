"""Repository mapping and caching system with PageRank-based intelligent ranking."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter

import logging
import networkx as nx


@dataclass
class FileInfo:
    """Information about a file in the repository."""

    path: str
    size: int
    modified_time: float
    language: Optional[str] = None
    symbols: List[str] = field(default_factory=list)  # Symbols defined in this file
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    references: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)  # symbol -> [(file, line)]
    symbols_used: List[str] = field(default_factory=list)  # Symbols referenced/used in this file


@dataclass
class RepoContext:
    """Context information about the repository."""

    root_path: Path
    files: Dict[str, FileInfo] = field(default_factory=dict)
    symbol_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    import_graph: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    file_rankings: Dict[str, float] = field(default_factory=dict)
    pagerank_scores: Dict[str, float] = field(default_factory=dict)  # PageRank scores
    dependency_graph: Optional[nx.DiGraph] = None  # NetworkX graph for PageRank
    last_updated: float = 0.0
    last_pagerank_update: float = 0.0  # Track when PageRank was last computed


class RepoMapManager:
    """Singleton manager to prevent re-scanning repositories."""
    _instances: Dict[str, 'RepoMap'] = {}

    @classmethod
    def get_instance(cls, root_path: Path) -> 'RepoMap':
        """Get or create a RepoMap instance for the given root path."""
        key = str(root_path.absolute())
        if key not in cls._instances:
            cls._instances[key] = RepoMap(root_path)
            # Only scan on first access
            if not cls._instances[key].context.files:
                cls._instances[key].scan_repository()
        return cls._instances[key]

    @classmethod
    def clear_instance(cls, root_path: Path) -> None:
        """Clear a specific instance (useful for testing)."""
        key = str(root_path.absolute())
        if key in cls._instances:
            del cls._instances[key]


class RepoMap:
    """Repository mapping with intelligent caching and ranking."""

    CACHE_DIR = ".devagent_cache"
    CACHE_VERSION = "1.0"

    def __init__(
        self,
        root_path: Optional[Path] = None,
        cache_enabled: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self.cache_enabled = cache_enabled
        self.logger = logger or logging.getLogger(__name__)

        self.context = RepoContext(root_path=self.root_path)
        self.cache_path = self.root_path / self.CACHE_DIR / "repo_map.json"

        # Load cache if available
        if cache_enabled:
            self._load_cache()

    def _load_cache(self) -> bool:
        """Load repository map from cache."""
        if not self.cache_path.exists():
            return False

        try:
            with open(self.cache_path, 'r') as f:
                data = json.load(f)

            # Check cache version
            if data.get('version') != self.CACHE_VERSION:
                return False

            # Restore context
            self.context.last_updated = data.get('last_updated', 0)

            # Restore files
            for file_data in data.get('files', []):
                file_info = FileInfo(
                    path=file_data['path'],
                    size=file_data['size'],
                    modified_time=file_data['modified_time'],
                    language=file_data.get('language'),
                    symbols=file_data.get('symbols', []),
                    imports=file_data.get('imports', []),
                    exports=file_data.get('exports', []),
                    dependencies=set(file_data.get('dependencies', [])),
                    references=file_data.get('references', {}),
                    symbols_used=file_data.get('symbols_used', [])
                )
                self.context.files[file_data['path']] = file_info

            # Restore PageRank scores if available
            self.context.pagerank_scores = data.get('pagerank_scores', {})
            self.context.last_pagerank_update = data.get('last_pagerank_update', 0.0)

            # Rebuild indices
            self._rebuild_indices()
            return True

        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return False

    def _save_cache(self) -> None:
        """Save repository map to cache."""
        if not self.cache_enabled:
            return

        try:
            # Ensure cache directory exists
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data
            data = {
                'version': self.CACHE_VERSION,
                'last_updated': self.context.last_updated,
                'last_pagerank_update': self.context.last_pagerank_update,
                'pagerank_scores': self.context.pagerank_scores,
                'files': []
            }

            for file_info in self.context.files.values():
                data['files'].append({
                    'path': file_info.path,
                    'size': file_info.size,
                    'modified_time': file_info.modified_time,
                    'language': file_info.language,
                    'symbols': file_info.symbols,
                    'imports': file_info.imports,
                    'exports': file_info.exports,
                    'dependencies': list(file_info.dependencies),
                    'references': file_info.references,
                    'symbols_used': file_info.symbols_used
                })

            # Write cache
            with open(self.cache_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def _rebuild_indices(self) -> None:
        """Rebuild symbol and import indices from file information."""
        self.context.symbol_index.clear()
        self.context.import_graph.clear()

        for file_path, file_info in self.context.files.items():
            # Build symbol index
            for symbol in file_info.symbols:
                self.context.symbol_index[symbol].add(file_path)

            # Build import graph
            for imp in file_info.imports:
                self.context.import_graph[file_path].add(imp)

    def scan_repository(self, force: bool = False) -> None:
        """Scan repository and build file map."""
        current_time = time.time()

        # Check if scan is needed
        if not force and self.context.last_updated > 0:
            time_since_update = current_time - self.context.last_updated
            if time_since_update < 300:  # 5 minutes
                return

        # Get all Python and TypeScript files
        extensions = {'.py', '.ts', '.tsx', '.js', '.jsx'}
        for ext in extensions:
            for file_path in self.root_path.rglob(f'*{ext}'):
                # Skip cache and common ignore directories
                if any(part.startswith('.') or part in {'node_modules', '__pycache__', 'venv', 'dist', 'build'}
                       for part in file_path.parts):
                    continue

                self._scan_file(file_path)

        self.context.last_updated = current_time
        self._rebuild_indices()
        self._save_cache()

    def _scan_file(self, file_path: Path) -> None:
        """Scan a single file and extract information."""
        try:
            stat = file_path.stat()
            relative_path = str(file_path.relative_to(self.root_path))

            # Check if file needs updating
            existing = self.context.files.get(relative_path)
            if existing and existing.modified_time >= stat.st_mtime:
                return

            # Detect language
            language = self._detect_language(file_path)

            # Create file info
            file_info = FileInfo(
                path=relative_path,
                size=stat.st_size,
                modified_time=stat.st_mtime,
                language=language
            )

            # Extract symbols and imports based on language
            if language == 'python':
                self._extract_python_info(file_path, file_info)
            elif language in {'typescript', 'javascript'}:
                self._extract_typescript_info(file_path, file_info)

            self.context.files[relative_path] = file_info

        except Exception as e:
            self.logger.debug(f"Failed to scan {file_path}: {e}")

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        ext = file_path.suffix.lower()
        return {
            '.py': 'python',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.js': 'javascript',
            '.jsx': 'javascript'
        }.get(ext)

    def _extract_python_info(self, file_path: Path, file_info: FileInfo) -> None:
        """Extract Python symbols, imports, and references."""
        try:
            import ast

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            # Track defined symbols at module level
            defined_in_file = set()

            for node in ast.walk(tree):
                # Extract function and class definitions
                if isinstance(node, ast.FunctionDef):
                    file_info.symbols.append(node.name)
                    defined_in_file.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    file_info.symbols.append(node.name)
                    defined_in_file.add(node.name)
                # Extract imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        file_info.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        file_info.imports.append(node.module)
                        # Track imported names as symbols used
                        for alias in node.names:
                            if alias.name != '*':
                                file_info.symbols_used.append(alias.name)
                # Extract symbol references (Name nodes)
                elif isinstance(node, ast.Name):
                    # Only track if it's not defined in this file and looks like a class/function
                    if (node.id not in defined_in_file and
                        node.id[0].isupper() or  # Likely class name
                        node.id in {'print', 'len', 'str', 'int', 'list', 'dict', 'set', 'tuple'}):  # Common funcs
                        if node.id not in file_info.symbols_used:
                            file_info.symbols_used.append(node.id)

        except Exception:
            pass

    def _extract_typescript_info(self, file_path: Path, file_info: FileInfo) -> None:
        """Extract TypeScript/JavaScript symbols and imports."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            import re

            # Extract function declarations
            func_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)'
            file_info.symbols.extend(re.findall(func_pattern, content))

            # Extract class declarations
            class_pattern = r'(?:export\s+)?class\s+(\w+)'
            file_info.symbols.extend(re.findall(class_pattern, content))

            # Extract const/let/var declarations
            var_pattern = r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*='
            file_info.symbols.extend(re.findall(var_pattern, content))

            # Extract imports
            import_pattern = r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]'
            file_info.imports.extend(re.findall(import_pattern, content))

        except Exception:
            pass

    def get_ranked_files(
        self,
        mentioned_files: Set[str],
        mentioned_symbols: Set[str],
        max_files: int = 20
    ) -> List[Tuple[str, float]]:
        """Get ranked list of relevant files."""

        rankings = {}

        for file_path, file_info in self.context.files.items():
            score = 0.0

            # Direct file mention - highest score
            if file_path in mentioned_files:
                score += 10.0

            # Symbol matches
            matching_symbols = set(file_info.symbols) & mentioned_symbols
            score += len(matching_symbols) * 2.0

            # Import relationships
            for mentioned in mentioned_files:
                if mentioned in file_info.dependencies:
                    score += 1.0
                if file_path in self.context.files.get(mentioned, FileInfo('', 0, 0)).dependencies:
                    score += 1.0

            # File size penalty (prefer smaller files)
            if file_info.size > 10000:
                score *= 0.9
            if file_info.size > 50000:
                score *= 0.8

            if score > 0:
                rankings[file_path] = score

        # Sort by score
        sorted_files = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        return sorted_files[:max_files]

    def get_file_summary(self, file_path: str) -> Optional[str]:
        """Get a summary of a file."""
        file_info = self.context.files.get(file_path)
        if not file_info:
            return None

        summary_parts = [f"File: {file_path}"]

        if file_info.language:
            summary_parts.append(f"Language: {file_info.language}")

        if file_info.symbols:
            symbols_preview = file_info.symbols[:10]
            summary_parts.append(f"Symbols: {', '.join(symbols_preview)}")
            if len(file_info.symbols) > 10:
                summary_parts.append(f"  ... and {len(file_info.symbols) - 10} more")

        if file_info.imports:
            imports_preview = file_info.imports[:5]
            summary_parts.append(f"Imports: {', '.join(imports_preview)}")
            if len(file_info.imports) > 5:
                summary_parts.append(f"  ... and {len(file_info.imports) - 5} more")

        return "\n".join(summary_parts)

    def find_symbol(self, symbol: str) -> List[str]:
        """Find files containing a symbol."""
        return list(self.context.symbol_index.get(symbol, set()))

    def get_dependencies(self, file_path: str) -> Set[str]:
        """Get files that a given file depends on."""
        file_info = self.context.files.get(file_path)
        if not file_info:
            return set()
        return file_info.dependencies

    def invalidate_file(self, file_path: str) -> None:
        """Invalidate cache for a specific file."""
        if file_path in self.context.files:
            del self.context.files[file_path]
            self._rebuild_indices()
            self._save_cache()

    def build_dependency_graph(self) -> nx.DiGraph:
        """Build dependency graph for PageRank computation."""
        G = nx.DiGraph()

        # Add all files as nodes
        for file_path in self.context.files.keys():
            G.add_node(file_path)

        # Build edges based on symbol usage
        for file_path, file_info in self.context.files.items():
            # For each symbol used in this file
            for symbol in file_info.symbols_used:
                # Find files that define this symbol
                defining_files = self.context.symbol_index.get(symbol, set())
                for definer in defining_files:
                    if definer != file_path:  # Don't self-reference
                        # Add edge from user to definer (this file depends on definer)
                        # Weight based on how many times the symbol is used
                        symbol_count = file_info.symbols_used.count(symbol)
                        weight = math.sqrt(symbol_count) if symbol_count > 1 else 1.0

                        if G.has_edge(file_path, definer):
                            G[file_path][definer]['weight'] += weight
                            G[file_path][definer]['symbols'].append(symbol)
                        else:
                            G.add_edge(file_path, definer, weight=weight, symbols=[symbol])

        self.context.dependency_graph = G
        return G

    def compute_pagerank(
        self,
        personalization: Optional[Dict[str, float]] = None,
        *,
        cache_results: bool = True,
    ) -> Dict[str, float]:
        """Compute PageRank scores for all files.

        When ``cache_results`` is True (default), the computed scores are stored on the
        RepoContext for reuse. Callers supplying a personalization vector should set
        ``cache_results`` to False so the base cache remains untouched.
        """
        # Build or get cached graph
        if self.context.dependency_graph is None:
            self.build_dependency_graph()

        G = self.context.dependency_graph

        if G.number_of_nodes() == 0:
            return {}

        try:
            # Compute PageRank
            pagerank_scores = nx.pagerank(
                G,
                alpha=0.85,
                personalization=personalization,
                weight='weight'
            )
        except nx.PowerIterationFailedConvergence:
            # Fallback to unweighted if convergence fails
            pagerank_scores = nx.pagerank(
                G,
                alpha=0.85,
                personalization=personalization
            )

        if cache_results and personalization is None:
            self.context.pagerank_scores = pagerank_scores
            self.context.last_pagerank_update = time.time()

        return pagerank_scores

    def get_file_rank(self, file_path: str) -> float:
        """Get PageRank score for a single file."""
        if not self.context.pagerank_scores:
            self.compute_pagerank()
        return self.context.pagerank_scores.get(file_path, 0.0)

    def get_ranked_files_pagerank(
        self,
        mentioned_files: Set[str],
        mentioned_symbols: Set[str],
        conversation_files: Optional[Set[str]] = None,
        max_files: int = 20
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Get files ranked by PageRank with context boosting.

        Returns: List of (file_path, score, metadata)
        """
        conversation_files = conversation_files or set()

        # Build personalization vector for PageRank
        personalization = {}
        if conversation_files:
            # Files already in conversation get high personalization
            for file in conversation_files:
                if file in self.context.files:
                    personalization[file] = 100.0 / len(conversation_files)

        # Compute or get cached PageRank scores
        # Ensure we have baseline scores cached
        if not self.context.pagerank_scores:
            base_scores = self.compute_pagerank()
        else:
            base_scores = self.context.pagerank_scores

        # Run personalized ranking without mutating the cached baseline
        if personalization:
            score_source = self.compute_pagerank(personalization, cache_results=False)
        else:
            score_source = base_scores

        # Apply dynamic weight boosting
        adjusted_scores = {}
        for file_path, base_score in score_source.items():
            file_info = self.context.files.get(file_path)
            if not file_info:
                continue

            score = base_score

            # Boost if directly mentioned
            if file_path in mentioned_files:
                score *= 10.0

            # Boost for matching symbols
            matching_symbols = set(file_info.symbols) & mentioned_symbols
            if matching_symbols:
                score *= (1.0 + len(matching_symbols))

            # Boost for files referencing mentioned symbols
            referenced_mentioned = set(file_info.symbols_used) & mentioned_symbols
            if referenced_mentioned:
                score *= (1.0 + 0.5 * len(referenced_mentioned))

            # Boost long meaningful identifiers in symbols
            long_symbols = sum(
                1
                for symbol in file_info.symbols
                if len(symbol) >= 8 and ('_' in symbol or symbol[0].isupper())
            )
            if long_symbols:
                score *= 1.0 + 0.1 * min(long_symbols, 5)

            # Penalize very large files
            if file_info.size > 50000:
                score *= 0.7
            elif file_info.size > 10000:
                score *= 0.9

            adjusted_scores[file_path] = score

        # Sort and prepare results
        sorted_files = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for file_path, score in sorted_files[:max_files]:
            file_info = self.context.files[file_path]
            metadata = {
                'base_pagerank': self.context.pagerank_scores.get(file_path, 0.0),
                'adjusted_score': score,
                'symbols': file_info.symbols[:5],  # Top 5 symbols
                'size': file_info.size,
                'language': file_info.language
            }

            # Add graph info if available
            if self.context.dependency_graph and file_path in self.context.dependency_graph:
                G = self.context.dependency_graph
                metadata['incoming_edges'] = G.in_degree(file_path)
                metadata['outgoing_edges'] = G.out_degree(file_path)

            results.append((file_path, score, metadata))

        return results

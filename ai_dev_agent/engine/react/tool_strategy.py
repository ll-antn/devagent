"""Intelligent tool selection strategy for enhanced exploration efficiency."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


_DEFAULT_ITERATION_BUDGET = 25


def _get_relative_threshold(base_limit: Optional[int], ratio: float, minimum: int = 1) -> int:
    """Scale a threshold proportionally to a base iteration budget."""

    if not isinstance(base_limit, int) or base_limit <= 0:
        base = _DEFAULT_ITERATION_BUDGET
    else:
        base = base_limit

    ratio = max(0.0, min(ratio, 1.0))
    threshold = max(minimum, int(round(base * ratio)))
    return threshold  # Remove the cap that prevents proper scaling


class TaskType(Enum):
    """Types of tasks that require different tool selection strategies."""
    CODE_EXPLORATION = "code_exploration"
    DEBUGGING = "debugging"
    FILE_MODIFICATION = "file_modification"
    TESTING = "testing"
    RESEARCH = "research"
    BUILD_AUTOMATION = "build_automation"


@dataclass
class ToolContext:
    """Context information for making intelligent tool selection decisions."""
    task_type: TaskType
    language: Optional[str] = None
    has_symbol_index: bool = False
    files_discovered: Set[str] = field(default_factory=set)
    tools_used: List[str] = field(default_factory=list)
    last_tool_success: bool = True
    iteration_count: int = 0
    task_keywords: Set[str] = field(default_factory=set)
    repository_size: Optional[int] = None  # Number of files
    iteration_budget: int = _DEFAULT_ITERATION_BUDGET


class ToolSelectionStrategy:
    """Intelligent tool selection engine that improves exploration efficiency."""
    
    # Tool dependency graph - tools that must run before others
    TOOL_DEPENDENCIES = {
        "symbols.find": ["symbols.index"],
        "ast.query": [],  # Can run standalone but benefits from file discovery
        "fs.write_patch": ["fs.read"],  # Must read before patching
    }
    
    # Tool priorities by task type (first = highest priority)
    TASK_TOOL_PRIORITIES = {
        TaskType.CODE_EXPLORATION: [
            "symbols.index",      # Always build index first for fast lookups
            "symbols.find",       # Fast symbol-based navigation
            "ast.query",         # Structural code analysis
            "code.search",       # Text search fallback
            "fs.read",          # Targeted file reading
            "exec",      # Command execution when needed
        ],
        TaskType.DEBUGGING: [
            "exec",      # Run failing test to understand issue
            "code.search",       # Find error locations and patterns
            "symbols.find",      # Navigate to symbol definitions
            "fs.read",          # Read error context
            "ast.query",        # Analyze code structure around errors
            "fs.write_patch",   # Apply fixes
            "symbols.index",     # Build index if needed
        ],
        TaskType.FILE_MODIFICATION: [
            "fs.read",          # Read current state
            "symbols.find",      # Locate symbols to modify
            "ast.query",        # Understand code structure
            "fs.write_patch",   # Apply changes
            "exec",     # Validate changes
            "code.search",      # Search for related code
        ],
        TaskType.TESTING: [
            "exec",      # Run tests
            "fs.read",          # Read test files
            "code.search",      # Find test patterns
            "symbols.find",      # Locate test functions
            "fs.write_patch",   # Fix failing tests
            "ast.query",        # Analyze test structure
        ],
        TaskType.RESEARCH: [
            "code.search",       # Broad pattern search
            "symbols.index",     # Build comprehensive index
            "symbols.find",      # Find relevant definitions
            "fs.read",          # Deep dive into files
            "ast.query",        # Analyze code patterns
            "exec",     # Execute research queries
        ],
        TaskType.BUILD_AUTOMATION: [
            "exec",      # Run build commands
            "fs.read",          # Read build files
            "code.search",       # Find build patterns
            "fs.write_patch",   # Update build files
            "symbols.find",      # Locate build symbols
        ]
    }
    
    # Language-specific tool preferences
    LANGUAGE_TOOL_PREFERENCES = {
        "cpp": ["ast.query", "symbols.find", "code.search"],  # AST excellent for C++
        "c": ["ast.query", "symbols.find", "code.search"],
        "python": ["symbols.find", "ast.query", "code.search"],  # Great symbol support
        "javascript": ["ast.query", "code.search", "symbols.find"],
        "typescript": ["ast.query", "code.search", "symbols.find"],
        "java": ["symbols.find", "ast.query", "code.search"],
        "go": ["symbols.find", "code.search", "ast.query"],
        "rust": ["symbols.find", "ast.query", "code.search"],
    }
    
    def __init__(self):
        """Initialize the tool selection strategy."""
        self.tool_performance: Dict[str, List[Tuple[bool, float]]] = {}
        self.context_history: List[ToolContext] = []
        self.redundant_tool_threshold = 3  # Max same tool in a row
        
    def select_next_tool(self, context: ToolContext) -> str:
        """Select the optimal tool for the current context."""
        
        # Get base priority list for task type
        priority_list = self._get_base_priorities(context)
        
        # Filter to available tools (dependencies satisfied)
        available_tools = self._filter_available_tools(priority_list, context)
        
        # Apply contextual adjustments
        adjusted_tools = self._apply_context_rules(available_tools, context)
        
        # Apply language-specific preferences
        if context.language:
            adjusted_tools = self._apply_language_preferences(adjusted_tools, context.language)
        
        # Return best candidate or fallback
        return adjusted_tools[0] if adjusted_tools else "code.search"
    
    def _get_base_priorities(self, context: ToolContext) -> List[str]:
        """Get base tool priority list for task type."""
        return self.TASK_TOOL_PRIORITIES.get(
            context.task_type,
            self.TASK_TOOL_PRIORITIES[TaskType.CODE_EXPLORATION]  # Default
        ).copy()
    
    def _filter_available_tools(self, tools: List[str], context: ToolContext) -> List[str]:
        """Filter tools to only those whose dependencies are satisfied."""
        available = []
        
        for tool in tools:
            dependencies = self.TOOL_DEPENDENCIES.get(tool, [])
            
            # Check if all dependencies are satisfied
            if all(dep in context.tools_used for dep in dependencies):
                # Special case: symbols.find needs index
                if tool == "symbols.find" and not context.has_symbol_index:
                    continue
                available.append(tool)
            elif not dependencies:  # No dependencies
                available.append(tool)
                
        return available
    
    def _apply_context_rules(self, tools: List[str], context: ToolContext) -> List[str]:
        """Apply intelligent context-based filtering and reordering."""
        
        # Rule 1: If no symbol index and exploring code, prioritize indexing
        if (context.task_type == TaskType.CODE_EXPLORATION and 
            not context.has_symbol_index and 
            "symbols.index" in tools):
            
            # Move symbols.index to front unless we're in a very small repo
            if not context.repository_size or context.repository_size > 10:
                tools = ["symbols.index"] + [t for t in tools if t != "symbols.index"]
        
        # Rule 2: Avoid tool repetition (anti-loop protection)
        if context.tools_used and len(context.tools_used) >= self.redundant_tool_threshold:
            recent_tools = context.tools_used[-self.redundant_tool_threshold:]
            if len(set(recent_tools)) == 1:  # Same tool repeated
                repeated_tool = recent_tools[0]
                tools = [t for t in tools if t != repeated_tool] + [repeated_tool]
        
        # Rule 3: If last tool failed, try alternatives
        if not context.last_tool_success and context.tools_used:
            failed_tool = context.tools_used[-1]
            # Move failed tool to end of priority
            tools = [t for t in tools if t != failed_tool] + [failed_tool]
        
        # Rule 4: Task keyword optimization
        if context.task_keywords:
            if any(kw in context.task_keywords for kw in ["node", "ast", "tree", "parse"]):
                # AST-related task - prioritize ast.query
                if "ast.query" in tools:
                    tools = ["ast.query"] + [t for t in tools if t != "ast.query"]
            
            if any(kw in context.task_keywords for kw in ["symbol", "function", "class", "method"]):
                # Symbol-related task - prioritize symbols tools
                if "symbols.find" in tools:
                    tools = ["symbols.find"] + [t for t in tools if t != "symbols.find"]
        
        # Rule 5: Iteration count adjustments
        late_iteration_threshold = _get_relative_threshold(context.iteration_budget, 0.75, minimum=5)
        if context.iteration_count > late_iteration_threshold:
            # Late in exploration - prefer more targeted tools
            targeted_tools = ["symbols.find", "ast.query", "fs.read"]
            tools = [t for t in tools if t in targeted_tools] + [t for t in tools if t not in targeted_tools]

        return tools
    
    def _apply_language_preferences(self, tools: List[str], language: str) -> List[str]:
        """Adjust tool order based on language-specific strengths."""
        
        language = language.lower()
        if language in self.LANGUAGE_TOOL_PREFERENCES:
            preferred = self.LANGUAGE_TOOL_PREFERENCES[language]
            
            # Reorder tools to prioritize language preferences
            reordered = []
            for pref_tool in preferred:
                if pref_tool in tools:
                    reordered.append(pref_tool)
            
            # Add remaining tools
            for tool in tools:
                if tool not in reordered:
                    reordered.append(tool)
                    
            return reordered
        
        return tools
    
    # Complete AST query templates for all supported languages
    AST_QUERY_TEMPLATES = {
        "cpp": {
            "find_classes": "(class_declaration name: (type_identifier) @name)",
            "find_functions": "(function_definition declarator: (function_declarator) @func)",
            "find_methods": "(field_declaration declarator: (function_declarator) @method)",
            "find_constructors": "(function_definition declarator: (function_declarator declarator: (destructor_name) @ctor))",
            "find_allocations": "(new_expression type: (_) @type)",
            "find_templates": "(template_declaration) @template",
            "find_namespaces": "(namespace_definition name: (identifier) @namespace)",
        },
        "c": {
            "find_functions": "(function_definition declarator: (function_declarator) @func)",
            "find_structs": "(struct_specifier name: (type_identifier) @name)",
            "find_typedefs": "(type_definition declarator: (type_identifier) @name)",
            "find_macros": "(preproc_def name: (identifier) @macro)",
            "find_enums": "(enum_specifier name: (type_identifier) @name)",
        },
        "python": {
            "find_classes": "(class_definition name: (identifier) @name)",
            "find_functions": "(function_definition name: (identifier) @name)",
            "find_methods": "(function_definition name: (identifier) @method)",
            "find_decorators": "(decorator) @decorator",
            "find_imports": "(import_statement) @import",
            "find_async": "(function_definition (async) @async)",
        },
        "javascript": {
            "find_functions": "(function_declaration name: (identifier) @name)",
            "find_arrow_functions": "(arrow_function) @arrow",
            "find_classes": "(class_declaration name: (identifier) @name)",
            "find_methods": "(method_definition key: (property_identifier) @name)",
            "find_imports": "(import_statement) @import",
            "find_exports": "(export_statement) @export",
            "find_async": "(function_declaration (async) @async)",
        },
        "typescript": {
            "find_interfaces": "(interface_declaration name: (type_identifier) @name)",
            "find_types": "(type_alias_declaration name: (type_identifier) @name)",
            "find_enums": "(enum_declaration name: (identifier) @name)",
            "find_functions": "(function_declaration name: (identifier) @name)",
            "find_classes": "(class_declaration name: (type_identifier) @name)",
            "find_generics": "(type_parameters) @generics",
        },
        "java": {
            "find_classes": "(class_declaration name: (identifier) @name)",
            "find_interfaces": "(interface_declaration name: (identifier) @name)",
            "find_methods": "(method_declaration name: (identifier) @name)",
            "find_fields": "(field_declaration declarator: (variable_declarator name: (identifier) @name))",
            "find_annotations": "(annotation) @annotation",
            "find_enums": "(enum_declaration name: (identifier) @name)",
        },
        "go": {
            "find_functions": "(function_declaration name: (identifier) @name)",
            "find_methods": "(method_declaration name: (field_identifier) @name)",
            "find_structs": "(type_spec name: (type_identifier) @name type: (struct_type))",
            "find_interfaces": "(type_spec name: (type_identifier) @name type: (interface_type))",
            "find_imports": "(import_declaration) @import",
        },
        "rust": {
            "find_functions": "(function_item name: (identifier) @name)",
            "find_structs": "(struct_item name: (type_identifier) @name)",
            "find_enums": "(enum_item name: (type_identifier) @name)",
            "find_traits": "(trait_item name: (type_identifier) @name)",
            "find_impls": "(impl_item) @impl",
            "find_macros": "(macro_invocation macro: (identifier) @name)",
        },
        "ruby": {
            "find_classes": "(class name: (constant) @name)",
            "find_modules": "(module name: (constant) @name)",
            "find_methods": "(method name: (identifier) @name)",
            "find_blocks": "(block) @block",
            "find_requires": "(call method: (identifier) @req (#eq? @req \"require\"))",
        },
        "csharp": {
            "find_classes": "(class_declaration name: (identifier) @name)",
            "find_interfaces": "(interface_declaration name: (identifier) @name)",
            "find_methods": "(method_declaration name: (identifier) @name)",
            "find_properties": "(property_declaration name: (identifier) @name)",
            "find_namespaces": "(namespace_declaration name: (_) @name)",
        },
        "php": {
            "find_classes": "(class_declaration name: (name) @name)",
            "find_functions": "(function_definition name: (name) @name)",
            "find_methods": "(method_declaration name: (name) @name)",
            "find_traits": "(trait_declaration name: (name) @name)",
            "find_namespaces": "(namespace_definition name: (namespace_name) @name)",
        }
    }

    def get_safe_ast_query(self, language: str, intent: str) -> str:
        """Get a safe, validated AST query for the language and intent."""
        
        # Normalize language name
        lang_map = {
            "c++": "cpp", "c#": "csharp", "c-sharp": "csharp",
            "typescript": "typescript", "ts": "typescript",
            "javascript": "javascript", "js": "javascript"
        }
        language = lang_map.get(language.lower(), language.lower())
        
        # Find matching query template
        lang_templates = self.AST_QUERY_TEMPLATES.get(language, {})
        
        # Match intent to template
        intent_lower = intent.lower()
        for template_key, query in lang_templates.items():
            if any(keyword in intent_lower for keyword in template_key.split("_")[1:]):
                return query
        
        # Default safe queries per language
        defaults = {
            "cpp": "(class_declaration) @class",
            "c": "(function_definition) @func",
            "python": "(class_definition) @class",
            "javascript": "(function_declaration) @func",
            "typescript": "(interface_declaration) @interface",
            "java": "(class_declaration) @class",
            "go": "(function_declaration) @func",
            "rust": "(function_item) @func",
            "ruby": "(method) @method",
            "csharp": "(class_declaration) @class",
            "php": "(class_declaration) @class"
        }
        
        return defaults.get(language, "(ERROR) @error")

    def get_tool_hints(self, tool: str, context: ToolContext) -> Dict[str, any]:
        """Provide optimization hints for specific tool usage."""
        
        hints = {}
        
        if tool == "code.search":
            # Language-specific search patterns
            if context.language == "cpp" or context.language == "c":
                if any(kw in context.task_keywords for kw in ["node", "ast", "allocator"]):
                    hints["regex"] = True
                    hints["patterns"] = [
                        r"allocator\s*->\s*New<",
                        r"ProgramAllocator\(\)\s*->\s*New<",
                        r"::Create.*Node",
                        r"Make.*Node"
                    ]
            
            # Adjust max results based on exploration phase
            early_threshold = _get_relative_threshold(context.iteration_budget, 0.4, minimum=5)
            if context.iteration_count < early_threshold:
                hints["max_results"] = 30  # Broader search early
            else:
                hints["max_results"] = 50  # More results when focused
                
        elif tool == "fs.read":
            # Optimal context for LLM processing
            hints["context_lines"] = 200
            
        elif tool == "symbols.find":
            # Language and task-specific symbol search
            if context.language:
                hints["lang"] = context.language
                
        elif tool == "ast.query":
            # Provide safe AST query based on context
            if context.task_keywords and context.language:
                # Try to infer intent from keywords
                intent_keywords = list(context.task_keywords)
                if intent_keywords:
                    safe_query = self.get_safe_ast_query(context.language, " ".join(intent_keywords))
                    if safe_query != "(ERROR) @error":
                        hints["suggested_query"] = safe_query
            
            # Prevent overwhelming AST output
            hints["max_nodes"] = 100
            
        elif tool == "symbols.index":
            # Index optimization
            if context.repository_size and context.repository_size > 1000:
                hints["paths"] = ["."]  # Index everything for large repos
            else:
                # For smaller repos, might target specific directories
                pass
                
        return hints
    
    def record_tool_result(self, tool: str, success: bool, duration: float):
        """Track tool performance for adaptive learning."""
        if tool not in self.tool_performance:
            self.tool_performance[tool] = []
            
        self.tool_performance[tool].append((success, duration))
        
        # Keep only recent history for adaptive behavior
        if len(self.tool_performance[tool]) > 100:
            self.tool_performance[tool] = self.tool_performance[tool][-50:]
    
    def get_tool_success_rate(self, tool: str) -> float:
        """Get success rate for a tool."""
        if tool not in self.tool_performance or not self.tool_performance[tool]:
            return 0.5  # Neutral assumption
            
        results = self.tool_performance[tool]
        successes = sum(1 for success, _ in results if success)
        return successes / len(results)
    
    def get_tool_avg_duration(self, tool: str) -> float:
        """Get average duration for a tool."""
        if tool not in self.tool_performance or not self.tool_performance[tool]:
            return 1.0  # Default assumption
            
        results = self.tool_performance[tool]
        total_duration = sum(duration for _, duration in results)
        return total_duration / len(results)
    
    def should_build_index(self, context: ToolContext) -> bool:
        """Determine if building symbol index is worthwhile."""
        
        # Don't build index if already built
        if context.has_symbol_index:
            return False
            
        # Always build for code exploration tasks
        if context.task_type == TaskType.CODE_EXPLORATION:
            return True
            
        # Build if repository is large enough to benefit
        if context.repository_size and context.repository_size > 50:
            return True
            
        # Build if we've done many searches already
        search_count = context.tools_used.count("code.search")
        if search_count > 5:
            return True
            
        return False
    
    def detect_task_type(self, task_description: str, goal: str = "") -> TaskType:
        """Detect task type from description and goal."""
        
        text = (task_description + " " + goal).lower()
        
        # Debugging keywords
        if any(kw in text for kw in ["debug", "error", "bug", "fix", "failing", "broken", "issue"]):
            return TaskType.DEBUGGING
            
        # File modification keywords
        if any(kw in text for kw in ["modify", "change", "update", "edit", "patch", "implement", "add"]):
            return TaskType.FILE_MODIFICATION
            
        # Testing keywords  
        if any(kw in text for kw in ["test", "testing", "unittest", "pytest", "spec"]):
            return TaskType.TESTING
            
        # Build keywords
        if any(kw in text for kw in ["build", "compile", "make", "cmake", "cargo", "npm"]):
            return TaskType.BUILD_AUTOMATION
            
        # Research keywords
        if any(kw in text for kw in ["find", "search", "locate", "understand", "analyze", "explore", "investigate"]):
            # Could be research or code exploration
            if any(kw in text for kw in ["code", "function", "class", "method", "symbol", "implementation"]):
                return TaskType.CODE_EXPLORATION
            else:
                return TaskType.RESEARCH
        
        # Default to code exploration for code-related tasks
        return TaskType.CODE_EXPLORATION
    
    def detect_language(self, file_paths: List[str] = None, content_sample: str = "") -> Optional[str]:
        """Detect programming language from file paths or content."""
        
        if file_paths:
            # Count file extensions
            ext_counts = {}
            for path in file_paths:
                if "." in path:
                    ext = path.split(".")[-1].lower()
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
            
            # Map extensions to languages
            ext_to_lang = {
                "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "c++": "cpp",
                "c": "c", "h": "c",  # Heuristic - could be C++
                "py": "python",
                "js": "javascript", "jsx": "javascript",
                "ts": "typescript", "tsx": "typescript",
                "java": "java",
                "go": "go",
                "rs": "rust",
                "rb": "ruby",
                "cs": "csharp",
                "php": "php"
            }
            
            # Find most common language
            lang_counts = {}
            for ext, count in ext_counts.items():
                if ext in ext_to_lang:
                    lang = ext_to_lang[ext]
                    lang_counts[lang] = lang_counts.get(lang, 0) + count
            
            if lang_counts:
                return max(lang_counts, key=lang_counts.get)
        
        # Content-based detection as fallback
        if content_sample:
            content_lower = content_sample.lower()
            
            if any(kw in content_lower for kw in ["#include", "std::", "namespace", "template<"]):
                return "cpp"
            elif any(kw in content_lower for kw in ["def ", "import ", "python", "__init__"]):
                return "python"
            elif any(kw in content_lower for kw in ["function", "const", "let", "var", "=>"]):
                return "javascript"
        
        return None


__all__ = ["ToolSelectionStrategy", "ToolContext", "TaskType"]

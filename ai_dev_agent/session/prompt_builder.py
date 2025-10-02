"""Helpers for constructing consistent system prompts across DevAgent surfaces."""
from __future__ import annotations

from typing import Dict, List, Optional

from ai_dev_agent.providers.llm.base import Message

_LANGUAGE_HINTS: Dict[str, str] = {
    "python": "\n- Use ast_query for Python code structure\n- Check requirements.txt/setup.py for dependencies\n- Use import analysis for module relationships",
    "javascript": "\n- Consider package.json for dependencies\n- Use ast_query for JS/TS structure\n- Check for .eslintrc for code standards",
    "typescript": "\n- Check tsconfig.json for compilation settings\n- Use ast_query for TypeScript analysis\n- Consider type definitions in .d.ts files",
    "java": "\n- Check pom.xml or build.gradle for dependencies\n- Use ast_query for class hierarchies\n- Consider package structure for organization",
    "c++": "\n- Check CMakeLists.txt or Makefile for build config\n- Look for .h/.hpp headers separately from .cpp/.cc files\n- Use compile_commands.json if available",
    "c": "\n- Check Makefile or CMakeLists.txt for build setup\n- Analyze header files (.h) for interfaces\n- Use grep for macro definitions",
    "go": "\n- Check go.mod for module dependencies\n- Use go tools for analysis\n- Consider internal vs external packages",
    "rust": "\n- Check Cargo.toml for dependencies\n- Use cargo commands for analysis\n- Consider module structure in lib.rs/main.rs",
    "ruby": "\n- Check Gemfile for dependencies\n- Look for rake tasks\n- Consider Rails structure if applicable",
    "php": "\n- Check composer.json for dependencies\n- Look for autoload configurations\n- Consider framework structure (Laravel, Symfony)",
    "c#": "\n- Check .csproj or .sln files\n- Use NuGet packages info\n- Consider namespace organization",
    "swift": "\n- Check Package.swift for dependencies\n- Look for .xcodeproj/xcworkspace\n- Consider iOS/macOS target differences",
    "kotlin": "\n- Check build.gradle.kts for configuration\n- Consider Android vs JVM targets\n- Use Gradle for dependency info",
    "scala": "\n- Check build.sbt for dependencies\n- Use sbt commands for analysis\n- Consider Play/Akka frameworks if present",
    "dart": "\n- Check pubspec.yaml for dependencies\n- Consider Flutter structure if applicable\n- Use dart analyze for code issues",
}


def build_system_messages(
    *,
    iteration_cap: Optional[int] = None,
    repository_language: Optional[str] = None,
    include_react_guidance: bool = True,
    extra_messages: Optional[List[str]] = None,
) -> List[Message]:
    """Produce baseline system messages reused across DevAgent entry points."""

    messages: List[Message] = []

    if include_react_guidance:
        cap = iteration_cap or 120
        core = (
            "You are a helpful assistant for the devagent CLI tool, specialized in efficient software development tasks.\n\n"
            "## MISSION\n"
            "Complete the user's task efficiently using available tools within {iteration_cap} iterations.\n\n"
            "## CORE PRINCIPLES\n"
            "1. EFFICIENCY: Choose the most appropriate tool for each task\n"
            "2. AVOID REDUNDANCY: Never repeat identical tool calls\n"
            "3. BULK OPERATIONS: Prefer batch operations over individual file reads\n"
            "4. EARLY TERMINATION: Stop when you have sufficient information\n"
            "5. ADAPTIVE STRATEGY: Change approach if tools fail\n"
            "6. SCRIPT GENERATION: Create scripts for complex computations\n\n"
            "## ITERATION MANAGEMENT\n"
            "- Current budget: {iteration_cap} iterations\n"
            "- At 75% usage ({iteration_cap_75} iterations): Begin consolidating findings\n"
            "- At 90% usage ({iteration_cap_90} iterations): Finalize answer\n"
        ).format(
            iteration_cap=cap,
            iteration_cap_75=int(cap * 0.75),
            iteration_cap_90=int(cap * 0.9),
        )

        if repository_language:
            hint = _LANGUAGE_HINTS.get(repository_language.lower())
            if hint:
                core += f"\nLANGUAGE-SPECIFIC ({repository_language}):{hint}\n"

        tool_semantics = (
            "\nTOOL SEMANTICS:\n"
            "- exec: Runs commands in POSIX shell; pipes, globs, redirects work\n"
            "- Prefer machine-parsable output (e.g., find -print0) over formatted listings\n"
            "- Minimize tool calls - stop once you have the answer\n"
        )

        output_discipline = (
            "\nOUTPUT REQUIREMENTS:\n"
            "- State scope explicitly (depth, hidden files, symlinks)\n"
            "- Ensure counts match actual listed items\n"
            "- Stop executing once you have sufficient information\n"
        )

        fs_recipes = (
            "\nCOMMON OPERATIONS:\n"
            "Count files: `find . -maxdepth 1 -type f | wc -l`\n"
            "List safely: `find . -maxdepth 1 -type f -print0 | xargs -0 -n1 basename`\n"
            "Verify location: `pwd` and `ls -la`\n"
        )

        failure_handling = (
            "\nFAILURE HANDLING:\n"
            "- First failure: Adjust parameters\n"
            "- Second failure: Switch tools/approach\n"
            "- System blocks identical calls after 2 failures\n"
            "- 3+ consecutive failures trigger termination\n"
        )

        tool_guidance = (
            "\nUNIVERSAL TOOL STRATEGIES:\n"
            "- For counting/metrics: Use shell commands with find/grep/wc\n"
            "- For pattern search: Start with code_search, then read specific files\n"
            "- For exploration: Use ast_query for structure, symbols for definitions\n"
            "- For bulk operations: Generate and execute scripts\n"
            "- For file operations: Prefer find/xargs over individual reads\n"
            "- Verify unexpected results (especially counts <=1) with pwd and ls -la\n"
        )

        tool_priority = (
            "\nTOOL SELECTION GUIDE:\n"
            "- Counting/metrics -> exec with scripts\n"
            "- Pattern search -> code_search\n"
            "- Code structure -> ast_query\n"
            "- Specific files -> fs.read\n"
            "- Bulk operations -> exec with find/xargs\n"
            "- Complex analysis -> Generate analysis scripts\n"
        )

        instructions = (
            core
            + tool_semantics
            + output_discipline
            + fs_recipes
            + failure_handling
            + tool_guidance
            + tool_priority
        )

        messages.append(Message(role="system", content=instructions))

    if extra_messages:
        for entry in extra_messages:
            if entry:
                messages.append(Message(role="system", content=entry))

    return messages

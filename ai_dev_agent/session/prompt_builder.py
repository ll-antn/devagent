"""Helpers for constructing consistent system prompts across DevAgent surfaces."""
from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from ai_dev_agent.core.utils.config import DEFAULT_MAX_ITERATIONS, Settings
from ai_dev_agent.core.utils.context_budget import summarize_text
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

_PROVIDER_PREAMBLES: Dict[str, str] = {
    "anthropic": "Anthropic models favour concise tool arguments and defer to user confirmations when tooling is denied.",
    "deepseek": "DeepSeek models can emit multi-step tool calls; keep responses grounded in repository evidence and summarise final answers clearly.",
    "openai": "OpenAI GPT models support JSON function calls; prefer structured outputs when sharing lists or diagnostics.",
    "google": "Gemini models may rate-limit long outputs; batch tool calls and stream concise updates when possible.",
}

_DEFAULT_INSTRUCTION_GLOBS: Sequence[str] = (
    "AGENTS.md",
    "CLAUDE.md",
    "CONTEXT.md",
    ".devagent/instructions/*.md",
)

_GLOBAL_INSTRUCTION_CANDIDATES: Sequence[Path] = (
    Path.home() / ".devagent" / "AGENTS.md",
    Path.home() / ".config" / "devagent" / "instructions.md",
)


def build_system_messages(
    *,
    iteration_cap: Optional[int] = None,
    repository_language: Optional[str] = None,
    include_react_guidance: bool = True,
    extra_messages: Optional[List[str]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    workspace_root: Optional[Path] = None,
    settings: Optional[Settings] = None,
    instruction_paths: Optional[Sequence[str]] = None,
) -> List[Message]:
    """Produce baseline system messages reused across DevAgent entry points."""

    root = _resolve_workspace_root(workspace_root, settings)

    guidance_sections: List[str] = []
    context_sections: List[str] = []

    provider_preamble = _provider_preamble(provider or (getattr(settings, "provider", None) or ""), model or getattr(settings, "model", None))
    if provider_preamble:
        guidance_sections.append(provider_preamble)

    if include_react_guidance:
        guidance_sections.append(
            _react_guidance(iteration_cap, repository_language, settings=settings)
        )

    environment_snapshot = _environment_snapshot(root)
    if environment_snapshot:
        context_sections.append(environment_snapshot)

    instruction_blocks = _instruction_overlays(root, instruction_paths, settings)
    if instruction_blocks:
        context_sections.append("Additional instructions:\n" + "\n\n".join(instruction_blocks))

    if extra_messages:
        context_sections.append("\n".join(entry.strip() for entry in extra_messages if entry).strip())

    primary_text = "\n\n".join(section.strip() for section in guidance_sections if section).strip()

    messages: List[Message] = []
    if primary_text:
        messages.append(Message(role="system", content=primary_text))

    for section in context_sections:
        text = section.strip()
        if text:
            messages.append(Message(role="system", content=text))

    if not messages:
        fallback = "You are a helpful assistant for the devagent CLI tool. Prioritise accurate reasoning and safe tool usage."
        messages.append(Message(role="system", content=fallback))

    return messages


def _resolve_workspace_root(workspace_root: Optional[Path], settings: Optional[Settings]) -> Path:
    candidate = workspace_root or getattr(settings, "workspace_root", None) or Path.cwd()
    try:
        return candidate.resolve()
    except OSError:
        return Path.cwd()


def _provider_preamble(provider: str, model: Optional[str]) -> str:
    key = provider.lower().strip()
    if not key:
        return ""
    base = _PROVIDER_PREAMBLES.get(key)
    if not base:
        return f"You are running on the {provider} provider{f' using {model}' if model else ''}. Optimise tool usage and produce grounded answers."
    if model:
        return f"Model: {model} ({provider}). {base}"
    return f"Provider: {provider}. {base}"


def _react_guidance(
    iteration_cap: Optional[int],
    repository_language: Optional[str],
    *,
    settings: Optional[Settings] = None,
) -> str:
    core = (
        "You are a helpful assistant for the devagent CLI tool, specialised in efficient software development tasks.\n\n"
        "## MISSION\n"
        "Complete the user's task efficiently using available tools.\n"
        "Pay attention to budget status messages that guide your execution strategy.\n\n"
        "## CORE PRINCIPLES\n"
        "1. EFFICIENCY: Choose the most appropriate tool for each task\n"
        "2. AVOID REDUNDANCY: Never repeat identical tool calls\n"
        "3. BULK OPERATIONS: Prefer batch operations over individual file reads\n"
        "4. EARLY TERMINATION: Stop when you have sufficient information\n"
        "5. ADAPTIVE STRATEGY: Change approach if tools fail\n"
        "6. SCRIPT GENERATION: Create scripts for complex computations\n"
        "\n"
        "## COMMUNICATION STYLE\n"
        "Be concise and direct. Minimize output tokens while maintaining accuracy.\n"
        "- For simple questions, give direct answers without preamble\n"
        "- One-word or one-sentence answers are preferred when appropriate\n"
        "- Avoid unnecessary explanations unless asked\n"
        "- Don't add phrases like 'The answer is...' or 'Based on...'\n"
        "\nExamples of proper conciseness:\n"
        "User: 'What is 2+2?' -> You: '4'\n"
        "User: 'Is 11 prime?' -> You: 'Yes'\n"
        "User: 'Which file has the User class?' -> You: 'models/user.py'\n"
        "User: 'How many Python files in src/?' -> You: '23'\n"
        "\nFor complex tasks:\n"
        "- Provide complete information but be succinct\n"
        "- Match detail level to task complexity\n"
        "- Explain non-trivial commands before running them\n"
    )

    if repository_language:
        hint = _LANGUAGE_HINTS.get(str(repository_language).lower())
        if hint:
            core += f"\nLANGUAGE-SPECIFIC ({repository_language}):{hint}\n"

    tool_semantics = (
        "\nTOOL SEMANTICS:\n"
        "- exec: Runs commands in POSIX shell; pipes, globs, redirects work\n"
        "- Prefer machine-parsable output (e.g. find -print0) over formatted listings\n"
        "- Minimise tool calls – stop once you have the answer\n"
        "\nPARALLEL TOOL EXECUTION:\n"
        "- You have the capability to call multiple tools in a single response\n"
        "- When multiple independent pieces of information are requested, batch your tool calls for optimal performance\n"
        "- Independent tool calls will execute concurrently, providing 3-5x speedup\n"
        "- Examples:\n"
        "  • Reading 3 different files: batch into one response instead of 3 sequential calls\n"
        "  • Multiple searches: run code_search queries in parallel\n"
        "  • File read + search: execute simultaneously if independent\n"
        "- IMPORTANT: Only batch truly independent operations (don't batch if one depends on another's result)\n"
    )

    output_discipline = (
        "\nOUTPUT REQUIREMENTS:\n"
        "- State scope explicitly (depth, hidden files, symlinks)\n"
        "- Ensure counts match actual listed items\n"
        "- Stop executing once you have sufficient information\n"
    )

    code_edit_guidance = (
        "\nCODE EDITING BEST PRACTICES:\n"
        "\nWhen modifying code:\n"
        "1. Read the file first to understand context and conventions\n"
        "2. Follow existing code style (indentation, naming, imports)\n"
        "3. Use fs.write_patch for surgical changes to existing files\n"
        "4. Don't modify unrelated code - stay focused on the task\n"
        "5. Verify library imports exist in the project before using them\n"
        "\nFormat for file changes:\n"
        "- Use unified diff format (fs.write_patch) for existing files\n"
        "- Show context lines around changes for clarity\n"
        "- Make multiple small patches rather than one large rewrite\n"
        "\nExample patch structure:\n"
        "  --- a/file.py\n"
        "  +++ b/file.py\n"
        "  @@ -10,6 +10,7 @@\n"
        "   def existing_function():\n"
        "       existing_line\n"
        "  +    new_line_added\n"
        "       another_existing_line\n"
        "\nDon't:\n"
        "- Leave TODO/FIXME comments without implementing the code\n"
        "- Add unnecessary comments explaining obvious code\n"
        "- Modify code outside the scope of the user's request\n"
        "- Assume libraries are available without checking\n"
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
        "- System blocks identical calls after two failures\n"
        "- Three or more consecutive failures should trigger termination\n"
    )

    anti_patterns = (
        "\nANTI-PATTERNS TO AVOID:\n"
        "\nCode quality:\n"
        "- NEVER leave TODO or FIXME comments - implement the code completely\n"
        "- Don't add placeholder functions that need to be 'filled in later'\n"
        "- Don't write incomplete implementations with '# rest of code here' comments\n"
        "- Avoid over-commenting obvious code\n"
        "\nScope discipline:\n"
        "- Don't refactor or 'improve' code outside the user's request\n"
        "- Don't fix unrelated bugs or style issues unless asked\n"
        "- Don't add extra features beyond what was requested\n"
        "- Stay focused on the specific task at hand\n"
        "\nTool usage:\n"
        "- Don't search for the same pattern multiple times\n"
        "- Don't read files you've already read\n"
        "- Don't run tests without knowing the test command\n"
        "- Don't assume build tools or frameworks are available\n"
        "\nSecurity:\n"
        "- Never commit API keys, passwords, or secrets\n"
        "- Don't log sensitive information\n"
        "- Run security.secrets_scan before committing if unsure\n"
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

    tool_descriptions = (
        "\nDETAILED TOOL DESCRIPTIONS:\n"
        "\nfs.read:\n"
        "  Purpose: Read file contents from the repository\n"
        "  When to use: When you need to examine specific files you already know exist\n"
        "  Parameters: paths (list of strings), optional context_lines or byte_range\n"
        "  Example: Use after code_search to read the files that matched your search\n"
        "\ncode.search:\n"
        "  Purpose: Search repository text for patterns (fixed string or regex)\n"
        "  When to use: When you need to find files containing specific text or code patterns\n"
        "  Parameters: query (string), optional regex=true, file_type, max_results\n"
        "  Example: Search for function definitions before reading the full file\n"
        "  Note: Default is FIXED STRING matching; use regex=true for patterns\n"
        "\nsymbols.index:\n"
        "  Purpose: Build or refresh the ctags symbol index\n"
        "  When to use: Once at the start of a session, or after significant file changes\n"
        "  Parameters: None\n"
        "  Note: Run this before using symbols.find\n"
        "\nsymbols.find:\n"
        "  Purpose: Look up symbol definitions (functions, classes, variables)\n"
        "  When to use: When you need to find where a specific symbol is defined\n"
        "  Parameters: name (string), optional kind (function, class, variable, etc.)\n"
        "  Example: Find the definition of 'DatabaseConnection' class\n"
        "\nast.query:\n"
        "  Purpose: Run tree-sitter queries for precise code structure analysis\n"
        "  When to use: When you need to find specific code patterns across the syntax tree\n"
        "  Parameters: path (string), query (tree-sitter query string)\n"
        "  Supported: Python, JavaScript, TypeScript, Go, Rust, C, C++, Java\n"
        "  Example: Find all function definitions with specific decorators\n"
        "\nexec:\n"
        "  Purpose: Execute shell commands directly\n"
        "  When to use: For git operations, running tests, building, or complex file operations\n"
        "  Parameters: cmd (string), optional args (list)\n"
        "  Caution: Always verify commands are safe before execution\n"
        "\nfs.write_patch:\n"
        "  Purpose: Apply unified diff patches to files\n"
        "  When to use: When making precise, reviewable changes to existing files\n"
        "  Parameters: patches (list of {path, patch_text})\n"
        "  Note: Prefer this over rewriting entire files\n"
        "\nsecurity.secrets_scan:\n"
        "  Purpose: Scan files for potential secrets, keys, credentials\n"
        "  When to use: Before committing changes, or when reviewing security\n"
        "  Parameters: paths (list of strings)\n"
        "  Note: Helps prevent accidental credential leaks\n"
    )

    tool_workflows = (
        "\nCOMMON TOOL WORKFLOWS:\n"
        "\nDiscovering code:\n"
        "  1. code_search for keywords -> 2. fs.read matching files -> 3. ast.query for details\n"
        "\nFinding and modifying a function:\n"
        "  1. symbols.index -> 2. symbols.find 'function_name' -> 3. fs.read -> 4. fs.write_patch\n"
        "\nUnderstanding project structure:\n"
        "  1. exec 'find . -type f -name \"*.py\" | head -20' -> 2. fs.read key files\n"
        "\nRefactoring across files:\n"
        "  1. code_search for usage -> 2. fs.read all matches -> 3. fs.write_patch each file\n"
        "\nDon't repeat tool calls:\n"
        "  - If code_search found no results, don't search again with same query\n"
        "  - If fs.read succeeded, don't read the same file again\n"
        "  - If symbols.index just ran, don't run it again in the same session\n"
    )

    return (
        core
        + tool_semantics
        + output_discipline
        + code_edit_guidance
        + fs_recipes
        + failure_handling
        + anti_patterns
        + tool_guidance
        + tool_priority
        + tool_descriptions
        + tool_workflows
    )


def _environment_snapshot(root: Path) -> str:
    lines = ["Environment snapshot:"]
    lines.append(f"  Workspace: {root}")
    lines.append(f"  Python: {platform.python_version()}")
    lines.append(f"  Platform: {platform.platform()}")
    lines.append(f"  Timestamp: {datetime.utcnow().isoformat()}Z")

    git_lines = _git_context(root)
    lines.extend(f"  {entry}" for entry in git_lines)

    return "\n".join(lines)


def _git_context(root: Path) -> List[str]:
    args = ["git", "rev-parse", "--is-inside-work-tree"]
    try:
        probe = subprocess.run(args, cwd=root, capture_output=True, text=True, check=False)
    except (OSError, ValueError):
        return ["Git: unavailable"]

    if probe.returncode != 0 or probe.stdout.strip().lower() != "true":
        return ["Git: not a repository"]

    context: List[str] = []
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    if branch:
        context.append(f"Git branch: {branch}")

    head = _run_git(["rev-parse", "--short", "HEAD"], root)
    if head:
        context.append(f"Git commit: {head}")

    status_raw = _run_git(["status", "--short"], root)
    if status_raw is not None:
        changes = [line for line in status_raw.splitlines() if line.strip()]
        if changes:
            sample = ", ".join(changes[:4])
            if len(changes) > 4:
                sample += ", …"
            context.append(f"Git status: {len(changes)} change(s) ({sample})")
        else:
            context.append("Git status: clean")
    return context


def _run_git(arguments: Sequence[str], root: Path) -> Optional[str]:
    try:
        result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True, check=False)
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _instruction_overlays(
    root: Path,
    instruction_paths: Optional[Sequence[str]],
    settings: Optional[Settings],
    *,
    max_chars: int = 4_000,
) -> List[str]:
    candidates: List[str] = []
    seen: Set[Path] = set()

    for pattern in _DEFAULT_INSTRUCTION_GLOBS:
        candidates.extend(_expand_instruction_glob(root, pattern))

    for global_candidate in _GLOBAL_INSTRUCTION_CANDIDATES:
        if global_candidate.is_file():
            candidates.append(str(global_candidate))

    if settings:
        provider_cfg = getattr(settings, "provider_config", {})
        if isinstance(provider_cfg, dict):
            extra = provider_cfg.get("prompt_instructions")
            if isinstance(extra, str):
                candidates.extend([extra])
            elif isinstance(extra, Iterable):
                for item in extra:
                    if isinstance(item, str):
                        candidates.append(item)

    env_instructions = os.getenv("DEVAGENT_PROMPT_INSTRUCTIONS")
    if env_instructions:
        for token in env_instructions.split(os.pathsep):
            token = token.strip()
            if token:
                candidates.append(token)

    if instruction_paths:
        candidates.extend(list(instruction_paths))

    blocks: List[str] = []
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = (root / path).resolve()
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            text = resolved.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        text = summarize_text(text.strip(), max_chars)
        if text:
            blocks.append(f"[{resolved.name}]\n{text}")
    return blocks


def _expand_instruction_glob(root: Path, pattern: str) -> List[str]:
    if "*" not in pattern:
        return [pattern]
    try:
        matches = list(root.glob(pattern))
    except OSError:
        return []
    return [str(match) for match in matches if match.is_file()]

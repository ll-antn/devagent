"""Tool invoker that maps LLM Actor tool requests to actual implementations."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..code_edit.context import ContextGatherer, ContextGatheringOptions
from ..code_edit.editor import CodeEditor
from ..react.pipeline import run_quality_pipeline
from ..testing.local_tests import TestRunner
from ..utils.keywords import extract_keywords
from .types import ActionRequest, Observation


class ReActToolInvoker:
    """Maps tool names to implementations for ReAct execution."""
    
    def __init__(self, workspace: Path, code_editor: CodeEditor, test_runner: TestRunner, sandbox, collector, pipeline_commands):
        self.workspace = workspace
        self.code_editor = code_editor
        self.test_runner = test_runner
        self.sandbox = sandbox
        self.collector = collector
        self.pipeline_commands = pipeline_commands
        
    def __call__(self, action: ActionRequest) -> Observation:
        """Invoke the requested tool and return observation."""
        
        tool_name = action.tool
        args = action.args
        
        try:
            # Route to appropriate tool implementation
            if tool_name == "analyze_code":
                return self._analyze_code(args)
            elif tool_name == "write_patch":
                return self._write_patch(args)
            elif tool_name == "run_tests":
                return self._run_tests(args)
            elif tool_name == "fix_tests":
                return self._fix_tests(args)
            elif tool_name == "add_tests":
                return self._add_tests(args)
            elif tool_name == "run_linter":
                return self._run_linter(args)
            elif tool_name == "search_codebase":
                return self._search_codebase(args)
            elif tool_name == "refactor_code":
                return self._refactor_code(args)
            elif tool_name == "check_dependencies":
                return self._check_dependencies(args)
            elif tool_name == "generate_docs":
                return self._generate_docs(args)
            elif tool_name == "qa_pipeline":
                return self._qa_pipeline(args)
            else:
                return Observation(
                    success=False,
                    outcome=f"Unknown tool: {tool_name}",
                    error=f"Tool '{tool_name}' is not implemented",
                    tool=tool_name
                )
                
        except Exception as e:
            return Observation(
                success=False,
                outcome=f"Tool {tool_name} failed with exception",
                error=str(e),
                tool=tool_name,
                raw_output=str(e)
            )
    
    def _analyze_code(self, args: Dict[str, Any]) -> Observation:
        """Analyze code structure and identify issues."""
        
        files = args.get("files", [])
        focus_areas = args.get("focus_areas", ["structure", "issues"])
        
        if not files:
            # Auto-detect Python files
            files = list(self.workspace.rglob("*.py"))
            files = [str(f.relative_to(self.workspace)) for f in files[:10]]  # Limit to avoid overload
        
        # Use context gatherer to analyze files
        options = ContextGatheringOptions(
            max_files=len(files),
            include_docs=False,
            include_related_files=True,
            include_structure_summary=True,
            include_tests=True
        )
        
        gatherer = ContextGatherer(self.workspace, options)
        
        # Analyze each file
        issues = []
        structures = []
        
        for file_path in files:
            try:
                full_path = self.workspace / file_path
                if full_path.exists() and full_path.is_file():
                    content = full_path.read_text(encoding='utf-8')
                    
                    # Basic analysis
                    lines = content.splitlines()
                    
                    # Check for common issues
                    if "structure" in focus_areas:
                        if len(lines) > 500:
                            issues.append(f"{file_path}: Very long file ({len(lines)} lines)")
                        
                        # Check for functions without docstrings
                        func_lines = [i for i, line in enumerate(lines) if line.strip().startswith("def ")]
                        for func_line in func_lines:
                            if func_line + 1 < len(lines):
                                next_line = lines[func_line + 1].strip()
                                if not next_line.startswith('"""') and not next_line.startswith("'''"):
                                    func_name = lines[func_line].split("def ")[1].split("(")[0]
                                    issues.append(f"{file_path}:{func_line + 1}: Function '{func_name}' missing docstring")
                    
                    if "issues" in focus_areas:
                        # Look for potential issues
                        for i, line in enumerate(lines):
                            if "TODO" in line or "FIXME" in line or "HACK" in line:
                                issues.append(f"{file_path}:{i + 1}: {line.strip()}")
                    
                    structures.append(f"{file_path}: {len(lines)} lines, {len(func_lines)} functions")
                    
            except Exception as e:
                issues.append(f"{file_path}: Failed to analyze - {e}")
        
        return Observation(
            success=True,
            outcome=f"Analyzed {len(files)} files, found {len(issues)} issues",
            metrics={
                "files_analyzed": len(files),
                "issues_found": len(issues),
                "analysis_focus": focus_areas
            },
            artifacts=files,
            tool="analyze_code",
            raw_output="\n".join(["STRUCTURES:"] + structures + ["ISSUES:"] + issues)
        )
    
    def _write_patch(self, args: Dict[str, Any]) -> Observation:
        """Generate code changes using the code editor."""
        
        files = args.get("files", [])
        changes_description = args.get("changes_description", "")
        test_first = args.get("test_first", False)
        
        if not files:
            return Observation(
                success=False,
                outcome="No files specified for patch",
                error="files argument is required",
                tool="write_patch"
            )
        
        if not changes_description:
            return Observation(
                success=False,
                outcome="No changes description provided",
                error="changes_description argument is required",
                tool="write_patch"
            )
        
        try:
            # Use the existing code editor to generate and apply changes
            success, attempts = self.code_editor.apply_diff_with_fixes(
                changes_description,
                files,
                extra_instructions="",
                test_command=None,
                dry_run=False
            )
            
            if success and attempts:
                last_attempt = attempts[-1]
                return Observation(
                    success=True,
                    outcome=f"Successfully applied patch to {len(files)} files",
                    metrics={
                        "files_modified": len(files),
                        "attempts": len(attempts),
                        "test_passed": last_attempt.test_result.success if last_attempt.test_result else None
                    },
                    artifacts=files,
                    tool="write_patch"
                )
            else:
                error_msg = attempts[-1].error_message if attempts else "Unknown error"
                return Observation(
                    success=False,
                    outcome="Failed to apply patch",
                    error=error_msg,
                    tool="write_patch"
                )
                
        except Exception as e:
            return Observation(
                success=False,
                outcome="Patch generation failed",
                error=str(e),
                tool="write_patch"
            )
    
    def _run_tests(self, args: Dict[str, Any]) -> Observation:
        """Execute tests and return results."""
        
        test_pattern = args.get("test_pattern", "")
        verbose = args.get("verbose", False)
        coverage = args.get("coverage", True)
        
        # Build test command
        command = ["pytest"]
        
        if test_pattern:
            command.extend(["-k", test_pattern])
        
        if verbose:
            command.append("-v")
        
        if coverage:
            command.extend(["--cov=.", "--cov-report=xml"])
        
        try:
            result = self.test_runner.run(command)
            
            return Observation(
                success=result.success,
                outcome=f"Tests {'passed' if result.success else 'failed'}",
                metrics={
                    "tests_passed": result.success,
                    "exit_code": result.exit_code,
                    "command": " ".join(result.command)
                },
                tool="run_tests",
                raw_output=f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )
            
        except Exception as e:
            return Observation(
                success=False,
                outcome="Test execution failed",
                error=str(e),
                tool="run_tests"
            )
    
    def _fix_tests(self, args: Dict[str, Any]) -> Observation:
        """Fix failing tests based on error output."""
        
        test_output = args.get("test_output", "")
        files_to_fix = args.get("files_to_fix", [])
        
        if not test_output:
            return Observation(
                success=False,
                outcome="No test output provided",
                error="test_output argument is required",
                tool="fix_tests"
            )
        
        # Extract error information from test output
        fix_description = f"Fix failing tests based on this output:\n{test_output[:1000]}"
        
        if not files_to_fix:
            # Try to extract file names from test output
            lines = test_output.split('\n')
            for line in lines:
                if ".py:" in line and ("FAILED" in line or "ERROR" in line):
                    file_part = line.split(".py:")[0] + ".py"
                    if file_part not in files_to_fix:
                        files_to_fix.append(file_part)
        
        if not files_to_fix:
            files_to_fix = ["tests/"]  # Default to tests directory
        
        return self._write_patch({
            "files": files_to_fix,
            "changes_description": fix_description,
            "test_first": False
        })
    
    def _add_tests(self, args: Dict[str, Any]) -> Observation:
        """Generate new tests to improve coverage."""
        
        target_files = args.get("target_files", [])
        coverage_gaps = args.get("coverage_gaps", [])
        test_style = args.get("test_style", "pytest")
        
        if not target_files:
            return Observation(
                success=False,
                outcome="No target files specified",
                error="target_files argument is required",
                tool="add_tests"
            )
        
        # Generate test files for target files
        test_files = []
        for target_file in target_files:
            # Convert src/module.py to tests/test_module.py
            path_parts = Path(target_file).parts
            if path_parts[0] == "src":
                test_path = Path("tests") / f"test_{Path(target_file).name}"
            else:
                test_path = Path("tests") / f"test_{Path(target_file).stem}.py"
            test_files.append(str(test_path))
        
        changes_description = f"""Generate comprehensive {test_style} tests for {', '.join(target_files)}.

Focus on:
- Testing all public functions and methods
- Edge cases and error conditions
- {', '.join(coverage_gaps) if coverage_gaps else 'Improving overall coverage'}

Follow {test_style} best practices and include:
- Clear test names
- Good test isolation
- Appropriate fixtures
- Mock external dependencies where needed

"""
        
        return self._write_patch({
            "files": test_files,
            "changes_description": changes_description,
            "test_first": True
        })
    
    def _run_linter(self, args: Dict[str, Any]) -> Observation:
        """Run linter and get style issues."""
        
        files = args.get("files", ["."])
        fix_auto = args.get("fix_auto", False)
        
        try:
            # Run ruff (modern Python linter)
            cmd = ["ruff", "check"] + files
            if fix_auto:
                cmd.append("--fix")
            
            result = subprocess.run(
                cmd,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            error_count = result.stdout.count('\n') if result.stdout else 0
            
            return Observation(
                success=result.returncode == 0,
                outcome=f"Linting {'passed' if result.returncode == 0 else 'found issues'}, {error_count} issues",
                metrics={
                    "lint_errors": error_count,
                    "auto_fixed": fix_auto,
                    "exit_code": result.returncode
                },
                tool="run_linter",
                raw_output=f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )
            
        except Exception as e:
            return Observation(
                success=False,
                outcome="Linter execution failed",
                error=str(e),
                tool="run_linter"
            )
    
    def _search_codebase(self, args: Dict[str, Any]) -> Observation:
        """Search for patterns in the codebase."""
        
        query = args.get("query", "")
        file_types = args.get("file_types", ["py"])
        include_tests = args.get("include_tests", True)
        
        if not query:
            return Observation(
                success=False,
                outcome="No search query provided",
                error="query argument is required",
                tool="search_codebase"
            )
        
        # Use context gatherer to search
        options = ContextGatheringOptions(
            include_docs=True,
            include_tests=include_tests
        )
        
        gatherer = ContextGatherer(self.workspace, options)
        
        # Search files
        matches = gatherer.search_files(query, file_types)
        
        return Observation(
            success=len(matches) > 0,
            outcome=f"Found {len(matches)} matches for '{query}'",
            metrics={
                "matches_found": len(matches),
                "query": query,
                "file_types": file_types
            },
            artifacts=matches[:20],  # Limit results
            tool="search_codebase",
            raw_output="\n".join(matches[:50])  # Show more in raw output
        )
    
    def _refactor_code(self, args: Dict[str, Any]) -> Observation:
        """Refactor code for better structure."""
        
        files = args.get("files", [])
        refactoring_type = args.get("refactoring_type", "improve_structure")
        preserve_tests = args.get("preserve_tests", True)
        
        if not files:
            return Observation(
                success=False,
                outcome="No files specified for refactoring",
                error="files argument is required",
                tool="refactor_code"
            )
        
        refactor_description = f"""Refactor the code in {', '.join(files)} to {refactoring_type}.

Guidelines:
- Improve code structure and readability
- Extract reusable functions/classes
- Remove code duplication
- Follow Python best practices
- {'Preserve existing test behavior' if preserve_tests else 'Update tests as needed'}

Specific refactoring: {refactoring_type}

"""
        
        return self._write_patch({
            "files": files,
            "changes_description": refactor_description,
            "test_first": False
        })
    
    def _check_dependencies(self, args: Dict[str, Any]) -> Observation:
        """Analyze and fix dependency issues."""
        
        check_security = args.get("check_security", True)
        update_outdated = args.get("update_outdated", False)
        
        issues = []
        
        try:
            # Check for requirements.txt or pyproject.toml
            req_files = []
            if (self.workspace / "requirements.txt").exists():
                req_files.append("requirements.txt")
            if (self.workspace / "pyproject.toml").exists():
                req_files.append("pyproject.toml")
            
            if not req_files:
                return Observation(
                    success=False,
                    outcome="No dependency files found",
                    error="No requirements.txt or pyproject.toml found",
                    tool="check_dependencies"
                )
            
            # Basic dependency check using pip
            if check_security:
                try:
                    result = subprocess.run(
                        ["pip", "list", "--outdated"],
                        cwd=self.workspace,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.stdout:
                        outdated_count = len(result.stdout.strip().split('\n')) - 2  # Minus header lines
                        issues.append(f"Found {max(0, outdated_count)} outdated packages")
                except Exception:
                    issues.append("Could not check for outdated packages")
            
            return Observation(
                success=True,
                outcome=f"Dependency check completed, {len(issues)} issues found",
                metrics={
                    "dependency_files": len(req_files),
                    "issues_found": len(issues),
                    "security_check": check_security
                },
                artifacts=req_files,
                tool="check_dependencies",
                raw_output="\n".join(issues)
            )
            
        except Exception as e:
            return Observation(
                success=False,
                outcome="Dependency check failed",
                error=str(e),
                tool="check_dependencies"
            )
    
    def _generate_docs(self, args: Dict[str, Any]) -> Observation:
        """Generate or update documentation."""
        
        files = args.get("files", [])
        doc_type = args.get("doc_type", "docstring")
        include_examples = args.get("include_examples", True)
        
        if not files:
            return Observation(
                success=False,
                outcome="No files specified for documentation",
                error="files argument is required",
                tool="generate_docs"
            )
        
        doc_description = f"""Add or improve {doc_type} documentation for {', '.join(files)}.

Requirements:
- Add comprehensive docstrings to all functions and classes
- Follow Google/NumPy docstring style
- Include parameter types and descriptions
- Include return value descriptions
- {'Add usage examples where appropriate' if include_examples else 'Focus on clear descriptions'}
- Document any exceptions raised

Documentation type: {doc_type}

"""
        
        return self._write_patch({
            "files": files,
            "changes_description": doc_description,
            "test_first": False
        })
    
    def _qa_pipeline(self, args: Dict[str, Any]) -> Observation:
        """Run full quality pipeline."""
        
        run_tests = args.get("run_tests", True)
        fix_issues = args.get("fix_issues", False)
        
        try:
            # Use existing pipeline
            observation = run_quality_pipeline(
                self.workspace,
                self.sandbox,
                self.collector,
                self.pipeline_commands,
                run_tests=run_tests
            )
            
            # Convert to our Observation format
            return Observation(
                success=observation.success,
                outcome=observation.outcome,
                metrics=observation.metrics or {},
                artifacts=observation.artifacts or [],
                tool="qa_pipeline",
                raw_output=observation.raw_output
            )
            
        except Exception as e:
            return Observation(
                success=False,
                outcome="QA pipeline failed",
                error=str(e),
                tool="qa_pipeline"
            )


def create_tool_invoker(workspace: Path, code_editor: CodeEditor, test_runner: TestRunner, sandbox, collector, pipeline_commands) -> ReActToolInvoker:
    """Factory function to create a configured tool invoker."""
    return ReActToolInvoker(workspace, code_editor, test_runner, sandbox, collector, pipeline_commands)


__all__ = ["ReActToolInvoker", "create_tool_invoker"]

"""Enhanced code editing with iterative fix loops and intelligent context gathering."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from ai_dev_agent.core.approval.approvals import ApprovalManager
from ai_dev_agent.providers.llm import LLMClient, LLMError, Message
from ai_dev_agent.tools.execution.testing.local_tests import TestResult, TestRunner
from ai_dev_agent.core.utils.keywords import extract_keywords
from ai_dev_agent.core.utils.logger import get_logger
from .context import ContextGatherer, ContextGatheringOptions, FileContext
from .diff_utils import DiffError, DiffPreview, DiffProcessor

LOGGER = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a precise senior software engineer. Given a task description and current file contents, "
    "produce a unified diff with the minimal changes required. Preserve existing formatting and style. "
    "Focus on solving the specific problem described."
)

USER_TEMPLATE = """
Task Description:
{task_description}

Additional Guidance:
{instructions}

File Contexts:
{file_blocks}

Respond with a unified diff enclosed in ```diff fences. Only include files that require changes.
"""

FIX_TEMPLATE = """
The previous implementation failed tests. Please analyze the error and provide a corrected diff.

Original Task:
{task_description}

Previous Attempt:
{previous_diff}

Test Results:
{test_output}

Current File Contents:
{file_blocks}

Provide a corrected unified diff that addresses the test failures.
"""

FILE_BLOCK_TEMPLATE = """
<file path="{path}" relevance="{relevance:.2f}" reason="{reason}">
{content}
</file>
"""


@dataclass
class FixAttempt:
    """Record of a fix attempt with its results and preview details."""

    attempt_number: int
    diff: str
    preview: DiffPreview | None = None
    validation_errors: List[str] = field(default_factory=list)
    test_result: Optional[TestResult] = None
    error_message: Optional[str] = None
    approved: Optional[bool] = None
    applied: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class DiffProposal:
    """Enhanced diff proposal with validation and preview."""
    diff: str
    raw_response: str
    files: List[Path]
    preview: Optional[DiffPreview] = None
    validation_errors: List[str] = field(default_factory=list)
    fallback_reason: str | None = None
    fallback_guidance: str | None = None


@dataclass
class IterativeFixConfig:
    """Configuration for iterative fix behavior."""
    max_attempts: int = 3
    run_tests: bool = True
    test_timeout: float = 60.0
    require_test_success: bool = True
    enable_context_expansion: bool = True
    

class CodeEditor:
    """Enhanced code editor with iterative fix loops and intelligent context gathering."""
    
    def __init__(
        self, 
        repo_root: Path, 
        llm_client: LLMClient, 
        approvals: ApprovalManager,
        fix_config: Optional[IterativeFixConfig] = None
    ) -> None:
        self.repo_root = repo_root
        self.llm_client = llm_client
        self.approvals = approvals
        self.fix_config = fix_config or IterativeFixConfig()
        
        # Enhanced components
        self.context_gatherer = ContextGatherer(repo_root)
        self.diff_processor = DiffProcessor(repo_root)
        self.test_runner = TestRunner(repo_root) if fix_config and fix_config.run_tests else None
        
        # Fix attempt tracking
        self.fix_attempts: List[FixAttempt] = []

    def gather_context(
        self, 
        files: Iterable[str], 
        task_description: Optional[str] = None,
        options: Optional[ContextGatheringOptions] = None
    ) -> List[FileContext]:
        """Gather intelligent context for the given files and task."""
        # Extract keywords from task description for better context discovery
        keywords = (
            extract_keywords(task_description, include_special_terms=True)
            if task_description
            else None
        )
        
        return self.context_gatherer.gather_contexts(
            files=files,
            task_description=task_description,
            keywords=keywords
        )

    def propose_diff(
        self, 
        task_description: str, 
        files: Iterable[str], 
        extra_instructions: str = "",
        previous_attempts: Optional[List[FixAttempt]] = None
    ) -> DiffProposal:
        """Propose a diff with enhanced context and validation."""
        
        # Gather intelligent context
        contexts = self.gather_context(files, task_description)
        
        # Build the prompt
        if previous_attempts:
            # This is a fix attempt
            latest_attempt = previous_attempts[-1]
            file_blocks = self._build_file_blocks(contexts)
            prompt = FIX_TEMPLATE.format(
                task_description=task_description,
                previous_diff=latest_attempt.diff,
                test_output=self._format_test_output(latest_attempt.test_result),
                file_blocks=file_blocks,
            )
        else:
            # Initial implementation
            file_blocks = self._build_file_blocks(contexts)
            prompt = USER_TEMPLATE.format(
                task_description=task_description,
                instructions=extra_instructions or "No additional guidance",
                file_blocks=file_blocks,
            )
        
        # Get LLM response
        try:
            response = self.llm_client.complete(
                [
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=prompt),
                ],
                temperature=0.2 if not previous_attempts else 0.3,  # Slightly higher temp for fixes
            )
        except LLMError as exc:
            LOGGER.warning("Diff proposal fallback engaged: %s", exc)
            guidance = self._build_fallback_guidance(task_description, contexts, str(exc))
            return DiffProposal(
                diff="",
                raw_response="",
                files=[],
                validation_errors=[],
                fallback_reason=str(exc),
                fallback_guidance=guidance,
            )

        # Extract and validate diff
        try:
            diff_text, validation = self.diff_processor.extract_and_validate_diff(response)
            preview = self.diff_processor.create_preview(diff_text)
            files_touched = self._extract_files_from_diff(diff_text)

            return DiffProposal(
                diff=diff_text,
                raw_response=response,
                files=files_touched,
                preview=preview,
                validation_errors=validation.errors,
            )

        except DiffError as exc:
            LOGGER.error("Failed to extract valid diff: %s", exc)
            return DiffProposal(
                diff="",
                raw_response=response,
                files=[],
                validation_errors=[str(exc)]
            )

    def apply_diff_with_fixes(
        self,
        task_description: str,
        files: Iterable[str],
        extra_instructions: str = "",
        test_command: Optional[List[str]] = None,
        *,
        on_proposal: Optional[Callable[[DiffProposal, int], None]] = None,
    ) -> Tuple[bool, List[FixAttempt]]:
        """Apply diffs with iterative fixes until tests pass or attempts are exhausted."""

        attempts: List[FixAttempt] = []
        files_list = list(files)

        for attempt_num in range(1, self.fix_config.max_attempts + 1):
            LOGGER.info("Fix attempt %d/%d", attempt_num, self.fix_config.max_attempts)

            proposal = self.propose_diff(
                task_description=task_description,
                files=files_list,
                extra_instructions=extra_instructions,
                previous_attempts=attempts if attempts else None,
            )

            if proposal.fallback_reason:
                message = proposal.fallback_guidance or "LLM is unavailable; manual intervention required."
                LOGGER.warning("Stopping iterative fixes due to fallback: %s", proposal.fallback_reason)
                attempts.append(
                    FixAttempt(
                        attempt_number=attempt_num,
                        diff=proposal.diff,
                        preview=proposal.preview,
                        validation_errors=list(proposal.validation_errors),
                        error_message=message,
                        approved=None,
                    )
                )
                return False, attempts

            if proposal.validation_errors:
                error_msg = f"Diff validation failed: {'; '.join(proposal.validation_errors)}"
                LOGGER.error(error_msg)
                attempts.append(
                    FixAttempt(
                        attempt_number=attempt_num,
                        diff=proposal.diff,
                        preview=proposal.preview,
                        validation_errors=list(proposal.validation_errors),
                        error_message=error_msg,
                        approved=None,
                    )
                )
                continue

            if proposal.preview:
                LOGGER.info("Diff preview: %s", proposal.preview.summary)
                if proposal.preview.validation_result.warnings:
                    for warning in proposal.preview.validation_result.warnings:
                        LOGGER.warning("Diff warning: %s", warning)

            if on_proposal:
                try:
                    on_proposal(proposal, attempt_num)
                except Exception as exc:  # pragma: no cover - callback errors should not break fixes
                    LOGGER.warning("Diff preview callback failed: %s", exc)

            attempt_record = FixAttempt(
                attempt_number=attempt_num,
                diff=proposal.diff,
                preview=proposal.preview,
                validation_errors=list(proposal.validation_errors),
            )

            try:
                approved = self._apply_diff_with_approval(proposal)
            except Exception as exc:  # noqa: BLE001 - propagate as failure with context
                error_msg = f"Failed to apply diff: {exc}"
                LOGGER.error(error_msg)
                attempt_record.error_message = error_msg
                attempt_record.approved = None
                attempts.append(attempt_record)
                continue

            attempt_record.approved = approved
            if not approved:
                LOGGER.info("Diff application was not approved")
                attempt_record.error_message = "Diff application declined by user."
                attempts.append(attempt_record)
                return False, attempts

            attempt_record.applied = True
            attempts.append(attempt_record)

            should_run_tests = self.fix_config.run_tests and self.test_runner
            if should_run_tests:
                test_result = self._run_tests(test_command)
                attempt_record.test_result = test_result

                if test_result.success:
                    LOGGER.info("Tests passed! Fix successful on attempt %d", attempt_num)
                    return True, attempts

                LOGGER.warning("Tests failed on attempt %d", attempt_num)
                attempt_record.error_message = "Tests failed."

                if attempt_num < self.fix_config.max_attempts:
                    LOGGER.info("Attempting another fix after test failures")
                    continue

                LOGGER.error("Max attempts reached. Tests still failing.")
                return False, attempts

            LOGGER.info("Diff applied successfully without running tests")
            return True, attempts

        return False, attempts

    def propose_diff_simple(self, task_description: str, files: Iterable[str], extra_instructions: str = "") -> DiffProposal:
        """Simple diff proposal for backward compatibility."""
        return self.propose_diff(task_description, files, extra_instructions)

    def apply_diff(self, proposal: DiffProposal) -> None:
        """Apply diff for backward compatibility."""
        if proposal.fallback_reason:
            raise LLMError(
                "Cannot apply diff because the LLM fallback was activated. "
                + (proposal.fallback_guidance or proposal.fallback_reason or "")
            )
        if not self._apply_diff_with_approval(proposal):
            raise PermissionError("Code change not approved by user.")

    def _apply_diff_with_approval(self, proposal: DiffProposal) -> bool:
        """Apply diff with approval check."""
        if proposal.fallback_reason:
            raise LLMError(
                "Diff proposal unavailable: "
                + (proposal.fallback_reason or "LLM fallback active")
            )
        if not self.approvals.require("code", default=True):
            return False

        try:
            success = self.diff_processor.apply_diff_safely(proposal.diff)
            if success:
                LOGGER.info(
                    "Diff applied for files: %s",
                    ", ".join(str(path) for path in proposal.files) or "<unknown>",
                )
            return success
        except DiffError as exc:
            raise LLMError(f"Failed to apply diff: {exc}") from exc

    def _run_tests(self, test_command: Optional[List[str]] = None) -> TestResult:
        """Run tests and return results."""
        if not self.test_runner:
            raise RuntimeError("Test runner not configured")
        
        if test_command:
            return self.test_runner.run(test_command)
        else:
            return self.test_runner.run_pytest()

    def _build_file_blocks(self, contexts: List[FileContext]) -> str:
        """Build file blocks for the prompt."""
        return "\n".join(
            FILE_BLOCK_TEMPLATE.format(
                path=context.path.relative_to(self.repo_root),
                content=context.content,
                relevance=context.relevance_score,
                reason=context.reason
            )
            for context in contexts
        )

    def _format_test_output(self, test_result: Optional[TestResult]) -> str:
        """Format test output for inclusion in fix prompts."""
        if not test_result:
            return "No test results available"
        
        output_parts = []
        
        if test_result.exit_code != 0:
            output_parts.append(f"Exit code: {test_result.exit_code}")
        
        if test_result.stderr:
            output_parts.append(f"STDERR:\n{test_result.stderr[:2000]}...")  # Limit length
        
        if test_result.stdout:
            output_parts.append(f"STDOUT:\n{test_result.stdout[:2000]}...")
        
        return "\n".join(output_parts) if output_parts else "Tests failed but no output captured"

    def _extract_files_from_diff(self, diff_text: str) -> List[Path]:
        """Extract file paths from diff."""
        files: List[Path] = []
        for line in diff_text.splitlines():
            if line.startswith("+++"):
                path = line[4:].strip()
                if path.startswith("b/"):
                    path = path[2:]
                if path != "/dev/null":
                    files.append(Path(path))
        return files

    def _build_fallback_guidance(
        self,
        task_description: str,
        contexts: List[FileContext],
        reason: str,
    ) -> str:
        if contexts:
            suggestions = []
            for context in contexts[:5]:
                try:
                    rel_path = context.path.relative_to(self.repo_root)
                except ValueError:
                    rel_path = context.path
                suggestions.append(
                    f"- {rel_path}: {context.reason or 'Relevant code context'}"
                )
            suggestion_block = "\n".join(suggestions)
            return (
                "LLM-generated diff unavailable; manual edits required.\n"
                f"Reason: {reason}\n"
                "Review these files and implement the task manually, then rerun with tests:\n"
                f"{suggestion_block}"
            )
        details = task_description.strip() or "No task description provided"
        return (
            "LLM-generated diff unavailable; manual edits required.\n"
            f"Reason: {reason}\n"
            f"Task context: {details}"
        )


__all__ = ["CodeEditor", "DiffProposal", "FixAttempt", "IterativeFixConfig"]

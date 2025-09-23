"""Repository question answering helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from ..code_edit.context import ContextGatherer, ContextGatheringOptions, FileContext
from ..llm_provider import LLMClient, LLMError, Message
from ..utils.logger import get_logger
from .inspector import RepositoryInspector

LOGGER = get_logger(__name__)

QA_SYSTEM_PROMPT = (
    "You are a pragmatic senior software engineer helping a teammate understand a codebase. "
    "Provided with repository context, answer questions accurately and concisely. "
    "Cite file paths and line numbers when they appear in the context. "
    "If the context does not contain enough information, say so explicitly and suggest where to look next. "
    "Do not invent tasks, action plans, or changes—focus purely on explaining the current code."  # noqa: E501
)

QA_USER_TEMPLATE = """
Question:
{question}

Repository Context:
{file_blocks}

Answer:
"""

FILE_BLOCK_TEMPLATE = """
<file path="{path}" reason="{reason}" score="{score:.2f}">
{content}
</file>
""".strip()

NO_CONTEXT_PLACEHOLDER = (
    "No repository context matched the question. Explain what additional files or information would be helpful."
)


@dataclass
class QAResult:
    """Result of a repository question."""

    answer: str
    contexts: List[FileContext]
    raw_response: str
    fallback_reason: str | None = None


class QuestionAnswerer:
    """High-level helper for answering repository questions."""

    def __init__(
        self,
        repo_root: Path,
        llm_client: LLMClient,
        options: ContextGatheringOptions | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.llm_client = llm_client
        self.context_gatherer = ContextGatherer(repo_root, options)
        self.inspector = RepositoryInspector(repo_root)

    def gather_context(
        self,
        question: str,
        files: Iterable[str] | None = None,
        keywords: Sequence[str] | None = None,
    ) -> List[FileContext]:
        """Return file contexts that may answer the question."""
        file_iterable = list(files or [])
        inspector_suggestions = self.inspector.suggest_files(question, limit=20)
        for candidate in inspector_suggestions:
            if candidate not in file_iterable:
                file_iterable.append(candidate)
        keyword_list = list(keywords) if keywords else None
        contexts = self.context_gatherer.gather_contexts(
            files=file_iterable,
            task_description=question,
            keywords=keyword_list,
        )
        LOGGER.info("Gathered %d context files for question", len(contexts))
        return contexts

    def answer(
        self,
        question: str,
        files: Iterable[str] | None = None,
        keywords: Sequence[str] | None = None,
    ) -> QAResult:
        """Generate an answer for the given question."""
        contexts = self.gather_context(question, files, keywords)
        file_blocks = self._build_file_blocks(contexts)
        prompt = QA_USER_TEMPLATE.format(
            question=question.strip(),
            file_blocks=file_blocks or NO_CONTEXT_PLACEHOLDER,
        )
        try:
            response = self.llm_client.complete(
                [
                    Message(role="system", content=QA_SYSTEM_PROMPT),
                    Message(role="user", content=prompt),
                ],
                temperature=0.15,
            )
        except LLMError as exc:
            LOGGER.warning("Question answering fallback engaged: %s", exc)
            fallback = self._build_fallback_answer(question, contexts, str(exc))
            return QAResult(
                answer=fallback,
                contexts=contexts,
                raw_response="",
                fallback_reason=str(exc),
            )
        answer = response.strip()
        LOGGER.debug("LLM answered question with %d characters", len(answer))
        return QAResult(answer=answer, contexts=contexts, raw_response=response)

    def _build_file_blocks(self, contexts: List[FileContext]) -> str:
        """Render file contexts into prompt blocks."""
        blocks: List[str] = []
        for context in contexts:
            try:
                rel_path = context.path.relative_to(self.repo_root)
            except ValueError:
                rel_path = context.path
            block = FILE_BLOCK_TEMPLATE.format(
                path=rel_path,
                reason=context.reason,
                score=context.relevance_score,
                content=context.content,
            )
            blocks.append(block)
        return "\n\n".join(blocks)

    def _build_fallback_answer(self, question: str, contexts: List[FileContext], reason: str) -> str:
        if contexts:
            suggestions = []
            for context in contexts[:5]:
                try:
                    rel_path = context.path.relative_to(self.repo_root)
                except ValueError:
                    rel_path = context.path
                suggestions.append(
                    f"- {rel_path}: {context.reason or 'Relevant content'}"
                )
            suggestion_block = "\n".join(suggestions)
            return (
                "LLM service is unavailable; unable to generate an answer automatically.\n"
                f"Reason: {reason}\n"
                "Review these context files manually and rerun once connectivity is restored:\n"
                f"{suggestion_block}"
            )
        return (
            "LLM service is unavailable and no context files were gathered.\n"
            f"Reason: {reason}\n"
            "Search the repository manually for keywords from your question and retry after the LLM becomes available."
        )


__all__ = ["QuestionAnswerer", "QAResult"]

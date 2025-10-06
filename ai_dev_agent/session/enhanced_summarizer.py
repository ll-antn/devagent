"""Enhanced conversation summarizer with multi-model fallback support."""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple
from dataclasses import dataclass

from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.core.utils.context_budget import estimate_tokens


@dataclass
class SummarizationConfig:
    """Configuration for enhanced summarization."""

    max_tokens: int = 1024
    min_split_messages: int = 4
    max_recursion_depth: int = 3
    preserve_assistant_endings: bool = True
    model_buffer_tokens: int = 512

    # Critical information preservation
    preserve_function_names: bool = True
    preserve_file_paths: bool = True
    preserve_error_messages: bool = True


class EnhancedSummarizer:
    """Multi-model conversation summarizer with intelligent fallback."""

    def __init__(
        self,
        models: List,
        config: Optional[SummarizationConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        if not models:
            raise ValueError("At least one model must be provided")

        self.models = models if isinstance(models, list) else [models]
        self.config = config or SummarizationConfig()
        self.logger = logger or logging.getLogger(__name__)

        # Use first model for token counting
        self.primary_model = self.models[0]

    def summarize(
        self,
        messages: Sequence[Message],
        max_chars: Optional[int] = None
    ) -> str:
        """Summarize messages with multi-model fallback."""

        # Try each model in sequence
        for model in self.models:
            try:
                summary = self._summarize_with_model(messages, model, max_chars)
                if summary and summary.strip():
                    return summary
            except Exception as e:
                self.logger.warning(f"Summarization failed with {model}: {e}")
                continue

        # Final fallback: heuristic summarization
        return self._heuristic_summarize(messages, max_chars)

    def _summarize_with_model(
        self,
        messages: Sequence[Message],
        model,
        max_chars: Optional[int] = None
    ) -> str:
        """Attempt summarization with a specific model."""

        # Build summarization prompt
        prompt = self._build_summarization_prompt(messages)

        system_prompt = """You are a conversation summarizer. Create a concise summary that:
1. Preserves all function names, file paths, and error messages
2. Maintains chronological flow of the conversation
3. Emphasizes recent messages more than older ones
4. Uses first person from the user's perspective
5. Keeps technical details and specific code references

Start with "I asked you..." and write as the user addressing the assistant."""

        summary_messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=prompt)
        ]

        # Get summary from model
        response = model.complete(summary_messages)

        if max_chars and len(response) > max_chars:
            response = response[:max_chars-3] + "..."

        return response

    def _build_summarization_prompt(self, messages: Sequence[Message]) -> str:
        """Build a prompt for summarization."""

        content_parts = []

        for msg in messages:
            if msg.role not in ("user", "assistant"):
                continue

            role_label = "USER" if msg.role == "user" else "ASSISTANT"
            content = msg.content or ""

            # Extract and preserve critical information
            if self.config.preserve_function_names:
                content = self._highlight_functions(content)
            if self.config.preserve_file_paths:
                content = self._highlight_paths(content)
            if self.config.preserve_error_messages:
                content = self._highlight_errors(content)

            content_parts.append(f"# {role_label}\n{content}\n")

        return "\n".join(content_parts)

    def _highlight_functions(self, text: str) -> str:
        """Highlight function names for preservation."""
        import re
        # Simple pattern for function detection
        pattern = r'\b(def|function|func|fn)\s+(\w+)'
        return re.sub(pattern, r'[FUNCTION: \2]', text)

    def _highlight_paths(self, text: str) -> str:
        """Highlight file paths for preservation."""
        import re
        # Pattern for file paths
        pattern = r'([a-zA-Z0-9_\-./]+\.(py|ts|js|tsx|jsx|java|go|rs|cpp|c|h))'
        return re.sub(pattern, r'[FILE: \1]', text)

    def _highlight_errors(self, text: str) -> str:
        """Highlight error messages for preservation."""
        import re
        # Pattern for common error indicators
        pattern = r'(error|exception|failed|failure|traceback)([:\s].*?)(?:\n|$)'
        return re.sub(pattern, r'[ERROR: \1\2]', text, flags=re.IGNORECASE)

    def _heuristic_summarize(
        self,
        messages: Sequence[Message],
        max_chars: Optional[int] = None
    ) -> str:
        """Fallback heuristic summarization."""

        summary_parts = []
        files_mentioned = set()
        functions_mentioned = set()
        errors_found = []

        for msg in messages:
            if msg.role not in ("user", "assistant"):
                continue

            content = msg.content or ""

            # Extract files
            import re
            file_pattern = r'([a-zA-Z0-9_\-./]+\.(py|ts|js|tsx|jsx|java|go|rs|cpp|c|h))'
            files = re.findall(file_pattern, content)
            files_mentioned.update(f[0] for f in files)

            # Extract functions
            func_pattern = r'\b(?:def|function|func|fn)\s+(\w+)'
            functions = re.findall(func_pattern, content)
            functions_mentioned.update(functions)

            # Extract errors
            if any(word in content.lower() for word in ['error', 'exception', 'failed']):
                # Take first line of error
                first_line = content.split('\n')[0][:100]
                errors_found.append(first_line)

        # Build summary
        if files_mentioned:
            summary_parts.append(f"Files discussed: {', '.join(list(files_mentioned)[:5])}")

        if functions_mentioned:
            summary_parts.append(f"Functions: {', '.join(list(functions_mentioned)[:5])}")

        if errors_found:
            summary_parts.append(f"Errors encountered: {errors_found[0]}")

        # Add last meaningful exchange
        for msg in reversed(messages):
            if msg.role == "assistant" and msg.content:
                last_content = msg.content[:200]
                summary_parts.append(f"Last action: {last_content}")
                break

        result = "\n".join(summary_parts)

        if max_chars and len(result) > max_chars:
            result = result[:max_chars-3] + "..."

        return result

    def smart_split_messages(
        self,
        messages: Sequence[Message],
        target_tokens: int
    ) -> Tuple[List[Message], List[Message]]:
        """Intelligently split messages at assistant boundaries."""

        if not messages:
            return [], []

        # Calculate token counts
        token_counts = []
        for msg in messages:
            tokens = estimate_tokens([msg])
            token_counts.append(tokens)

        # Find split point from end, keeping recent messages
        total_tokens = sum(token_counts)
        if total_tokens <= target_tokens:
            return [], list(messages)

        tail_tokens = 0
        split_index = len(messages)
        half_target = target_tokens // 2

        # Work backwards to find split point
        for i in range(len(messages) - 1, -1, -1):
            if tail_tokens + token_counts[i] < half_target:
                tail_tokens += token_counts[i]
                split_index = i
            else:
                break

        # Ensure split is at assistant message boundary
        if self.config.preserve_assistant_endings:
            while split_index > 0 and messages[split_index - 1].role != "assistant":
                split_index -= 1

        # Don't split too early
        if split_index < self.config.min_split_messages:
            return list(messages), []

        return list(messages[:split_index]), list(messages[split_index:])

    def recursive_summarize(
        self,
        messages: Sequence[Message],
        depth: int = 0
    ) -> List[Message]:
        """Recursively summarize messages with depth limiting."""

        if depth >= self.config.max_recursion_depth:
            # Force summarization at max depth
            summary = self.summarize(messages)
            return [Message(role="user", content=f"[Previous conversation summary]\n{summary}")]

        # Check if messages fit in target
        total_tokens = estimate_tokens(messages)
        if total_tokens <= self.config.max_tokens:
            return list(messages)

        # Split messages
        head, tail = self.smart_split_messages(messages, self.config.max_tokens)

        if not head:
            # Can't split further, summarize all
            summary = self.summarize(messages)
            return [Message(role="user", content=f"[Conversation summary]\n{summary}")]

        # Summarize head
        head_summary = self.summarize(head)
        summary_msg = Message(role="user", content=f"[Earlier conversation]\n{head_summary}")

        # Check if summary + tail fits
        combined = [summary_msg] + tail
        combined_tokens = estimate_tokens(combined)

        if combined_tokens <= self.config.max_tokens:
            return combined

        # Need to recurse
        return self.recursive_summarize(combined, depth + 1)
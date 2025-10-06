"""Wrapper to integrate EnhancedSummarizer with the existing summarizer interface."""
from __future__ import annotations

from typing import Optional, Sequence
import logging

from ai_dev_agent.providers.llm.base import Message
from .summarizer import ConversationSummarizer, HeuristicConversationSummarizer
from .enhanced_summarizer import EnhancedSummarizer, SummarizationConfig


class EnhancedSummarizerWrapper(ConversationSummarizer):
    """Wrapper that adapts EnhancedSummarizer to the ConversationSummarizer interface."""

    def __init__(
        self,
        client,  # LLMClient or list of models
        config: Optional[SummarizationConfig] = None,
        fallback: Optional[ConversationSummarizer] = None
    ):
        """Initialize the wrapper.

        Args:
            client: Either an LLMClient or list of models for multi-model fallback
            config: Configuration for enhanced summarization
            fallback: Fallback summarizer if enhanced fails
        """
        self.logger = logging.getLogger(__name__)
        self.fallback = fallback or HeuristicConversationSummarizer()

        # Try to create EnhancedSummarizer
        try:
            # If client is an LLMClient, wrap it in a list
            models = [client] if not isinstance(client, list) else client

            if not models:
                raise ValueError("No models provided")

            self.enhanced = EnhancedSummarizer(
                models=models,
                config=config or SummarizationConfig()
            )
            self.enabled = True
        except Exception as e:
            self.logger.debug(f"Could not initialize EnhancedSummarizer: {e}")
            self.enhanced = None
            self.enabled = False

    def summarize(self, messages: Sequence[Message], *, max_chars: int) -> str:
        """Summarize messages using enhanced summarizer or fallback."""

        if self.enabled and self.enhanced:
            try:
                # Use enhanced summarizer
                return self.enhanced.summarize(messages, max_chars=max_chars)
            except Exception as e:
                self.logger.warning(f"Enhanced summarization failed: {e}")
                # Fall through to fallback

        # Use fallback
        return self.fallback.summarize(messages, max_chars=max_chars)


def create_enhanced_summarizer(client) -> Optional[ConversationSummarizer]:
    """Factory function to create an enhanced summarizer if possible.

    Args:
        client: LLMClient instance

    Returns:
        EnhancedSummarizerWrapper if successful, None otherwise
    """
    try:
        from ai_dev_agent.core.config_defaults import get_feature_config

        # Check if feature is enabled
        config = get_feature_config("summarization")
        if not config.get("enabled", False):
            return None

        # Create configuration
        summarization_config = SummarizationConfig(
            max_tokens=config.get("max_tokens", 1024),
            preserve_function_names=config.get("preserve_function_names", True),
            preserve_file_paths=config.get("preserve_file_paths", True),
            preserve_error_messages=config.get("preserve_error_messages", True)
        )

        # Create wrapper
        wrapper = EnhancedSummarizerWrapper(
            client=client,
            config=summarization_config
        )

        if wrapper.enabled:
            return wrapper
        else:
            return None

    except ImportError:
        # Config defaults not available
        return None
    except Exception:
        # Any other error
        return None
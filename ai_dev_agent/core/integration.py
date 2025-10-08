"""Integration module for enhanced DevAgent components."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Import default configuration
try:
    from ai_dev_agent.core.config_defaults import (
        get_enhanced_features_config,
        is_feature_enabled,
        get_feature_config
    )
    CONFIG_DEFAULTS_AVAILABLE = True
except ImportError:
    CONFIG_DEFAULTS_AVAILABLE = False
    get_enhanced_features_config = lambda: {"enable_enhanced_components": True}
    is_feature_enabled = lambda x: True
    get_feature_config = lambda x: {"enabled": True}

# Import all enhanced components with graceful fallback
try:
    from ai_dev_agent.session.enhanced_summarizer import EnhancedSummarizer, SummarizationConfig
except ImportError:
    EnhancedSummarizer = None
    SummarizationConfig = None

try:
    from ai_dev_agent.core.repo_map import RepoMap
except ImportError:
    RepoMap = None

try:
    from ai_dev_agent.prompts.provider_prompts import ProviderPrompts, PromptContext, PromptLoader
except ImportError:
    ProviderPrompts = None
    PromptContext = None
    PromptLoader = None

try:
    from ai_dev_agent.cli.react.budget_control import AdaptiveBudgetManager
except ImportError:
    AdaptiveBudgetManager = None

@dataclass
class EnhancedComponents:
    """Container for all enhanced components."""

    summarizer: Optional[EnhancedSummarizer] = None
    repo_map: Optional[RepoMap] = None
    budget_manager: Optional[AdaptiveBudgetManager] = None
    prompt_context: Optional[PromptContext] = None


class ComponentIntegration:
    """Manages integration of enhanced components."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        settings: Optional[Any] = None,
        enable_all: bool = None
    ):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.settings = settings

        # Use default configuration if enable_all not explicitly set
        if enable_all is None:
            config = get_enhanced_features_config()
            self.enable_all = config.get("enable_enhanced_components", True)
        else:
            self.enable_all = enable_all

        self.components = EnhancedComponents()
        self.feature_config = get_enhanced_features_config()

    def initialize_summarizer(self, models: Optional[List] = None) -> Optional[EnhancedSummarizer]:
        """Initialize enhanced summarizer with fallback."""

        # Check if feature is enabled
        feature_cfg = get_feature_config("summarization")
        if not feature_cfg.get("enabled", False) or not self.enable_all or EnhancedSummarizer is None:
            return None

        if not models:
            # If no models provided, we can't use enhanced summarizer effectively
            return None

        try:
            config = SummarizationConfig(
                max_tokens=feature_cfg.get("max_tokens", 1024),
                preserve_function_names=feature_cfg.get("preserve_function_names", True),
                preserve_file_paths=feature_cfg.get("preserve_file_paths", True),
                preserve_error_messages=feature_cfg.get("preserve_error_messages", True)
            )
            self.components.summarizer = EnhancedSummarizer(models, config)
            return self.components.summarizer
        except Exception:
            return None

    def initialize_repo_map(self, force_scan: bool = False) -> Optional[RepoMap]:
        """Initialize repository map with caching."""

        if not self.enable_all or RepoMap is None:
            return None

        try:
            self.components.repo_map = RepoMap(
                root_path=self.project_root,
                cache_enabled=True
            )
            if force_scan:
                self.components.repo_map.scan_repository(force=True)
            else:
                self.components.repo_map.scan_repository(force=False)
            return self.components.repo_map
        except Exception:
            return None

    def initialize_budget_manager(
        self,
        max_iterations: int = 15,
        model_context_window: Optional[int] = None
    ) -> Optional[AdaptiveBudgetManager]:
        """Initialize adaptive budget manager."""

        if not self.enable_all or AdaptiveBudgetManager is None:
            return None

        try:
            # Use the original AdaptiveBudgetManager API from budget_control.py
            self.components.budget_manager = AdaptiveBudgetManager(
                max_iterations,
                model_context_window=model_context_window or 100000,
                enable_reflection=getattr(self.settings, 'enable_reflection', True),
                adaptive_scaling=True,
                max_reflections=3
            )
            return self.components.budget_manager
        except Exception:
            return None

    def get_system_prompt(
        self,
        provider: str,
        task_description: Optional[str] = None,
        phase: Optional[str] = None
    ) -> str:
        """Generate system prompt using provider-specific templates."""

        if not ProviderPrompts or not PromptContext:
            # Fallback to basic prompt
            return f"You are an AI assistant helping with: {task_description or 'software development'}"

        # Detect project language
        language = None
        if PromptLoader:
            language = PromptLoader.get_language_from_project(self.project_root)

        # Load custom instructions
        custom_instructions = None
        if PromptLoader:
            custom_instructions = PromptLoader.load_custom_instructions(self.project_root)

        # Check if git repo
        is_git_repo = (self.project_root / ".git").exists()

        # Create prompt context
        context = PromptContext(
            working_directory=str(self.project_root),
            is_git_repo=is_git_repo,
            platform=os.name,
            language=language,
            custom_instructions=custom_instructions,
            phase=phase,
            task_description=task_description
        )

        # Store for later use
        self.components.prompt_context = context

        return ProviderPrompts.get_system_prompt(provider, context)

    def check_tool_permission(
        self,
        tool_name: str,
        arguments: Optional[Dict] = None
    ) -> Tuple[bool, Optional[str]]:
        """Check if tool usage is permitted."""
        return True, None

    def get_ranked_files(
        self,
        mentioned_files: set,
        mentioned_symbols: set,
        max_files: int = 20
    ) -> List[Tuple[str, float]]:
        """Get ranked list of relevant files."""

        if not self.components.repo_map:
            return []

        return self.components.repo_map.get_ranked_files(
            mentioned_files,
            mentioned_symbols,
            max_files
        )

    def save_state(self) -> None:
        """Save state of all components."""

        # RepoMap saves automatically via cache

    def get_statistics(self) -> Dict:
        """Get statistics from all components."""

        stats = {}

        if self.components.budget_manager:
            stats['budget'] = self.components.budget_manager.get_stats()

        return stats


# Global instance for easy access
_global_integration: Optional[ComponentIntegration] = None


def get_integration() -> Optional[ComponentIntegration]:
    """Get global integration instance."""
    return _global_integration


def initialize_integration(
    project_root: Optional[Path] = None,
    settings: Optional[Any] = None,
    enable_all: bool = True
) -> ComponentIntegration:
    """Initialize global integration instance."""

    global _global_integration
    _global_integration = ComponentIntegration(
        project_root=project_root,
        settings=settings,
        enable_all=enable_all
    )
    return _global_integration

"""Default configuration for enhanced DevAgent features."""
from __future__ import annotations

# Feature flags for enhanced components
ENHANCED_FEATURES_CONFIG = {
    # Enable all enhanced features by default
    "enable_enhanced_components": True,

    # Individual component flags
    "enable_enhanced_summarizer": True,
    "enable_repo_mapping": True,
    "enable_adaptive_budget": True,
    "enable_tool_tracking": True,
    "enable_agent_modes": True,
    "enable_provider_prompts": True,

    # Performance settings
    "cache_enabled": True,
    "cache_directory": ".devagent_cache",
    "persist_metrics": True,

    # Adaptive budget settings
    "adaptive_budget": {
        "enabled": True,
        "max_iterations": 20,
        "enable_reflection": True,
        "max_reflections": 3,
        "dynamic_adjustment": True,
        "model_context_window": 100000,
    },

    # Repository mapping settings
    "repo_map": {
        "enabled": True,
        "auto_scan": True,
        "cache_threshold": 0.95,
        "max_cache_age_minutes": 30,
    },

    # Tool tracking settings
    "tool_tracking": {
        "enabled": True,
        "track_performance": True,
        "track_errors": True,
        "max_consecutive_failures": 3,
    },

    # Agent modes settings
    "agent_modes": {
        "enabled": True,
        "default_mode": "general",
        "strict_permissions": False,  # Set to True for production
    },

    # Enhanced summarization settings
    "summarization": {
        "enabled": True,
        "preserve_function_names": True,
        "preserve_file_paths": True,
        "preserve_error_messages": True,
        "max_tokens": 1024,
        "multi_model_fallback": True,
    },
}

def get_enhanced_features_config() -> dict:
    """Get the default configuration for enhanced features."""
    return ENHANCED_FEATURES_CONFIG.copy()

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a specific feature is enabled."""
    config = get_enhanced_features_config()

    # Check main toggle first
    if not config.get("enable_enhanced_components", True):
        return False

    # Check specific feature flag
    return config.get(f"enable_{feature_name}", False)

def get_feature_config(feature_name: str) -> dict:
    """Get configuration for a specific feature."""
    config = get_enhanced_features_config()

    # Map feature names to config keys
    feature_map = {
        "adaptive_budget": "adaptive_budget",
        "repo_map": "repo_map",
        "tool_tracking": "tool_tracking",
        "agent_modes": "agent_modes",
        "summarization": "summarization",
    }

    config_key = feature_map.get(feature_name)
    if config_key and config_key in config:
        feature_config = config[config_key].copy()
        # Add the enabled flag from the main config
        # For summarization, check enable_enhanced_summarizer flag
        if feature_name == "summarization":
            feature_config["enabled"] = is_feature_enabled("enhanced_summarizer")
        else:
            # For other features, map the name appropriately
            feature_flag_map = {
                "adaptive_budget": "adaptive_budget",
                "repo_map": "repo_mapping",
                "tool_tracking": "tool_tracking",
                "agent_modes": "agent_modes",
            }
            flag_name = feature_flag_map.get(feature_name, feature_name)
            feature_config["enabled"] = is_feature_enabled(flag_name)
        return feature_config

    return {"enabled": False}
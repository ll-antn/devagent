import os
from pathlib import Path

from ai_dev_agent.utils.config import Settings, load_settings


def test_env_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text("provider = 'deepseek'\nmodel = 'deepseek-lite'\n")
    monkeypatch.setenv("DEVAGENT_API_KEY", "abc123")
    monkeypatch.setenv("DEVAGENT_AUTO_APPROVE_PLAN", "true")
    settings = load_settings(config_path)
    assert settings.api_key == "abc123"
    assert settings.auto_approve_plan is True
    assert isinstance(settings.state_file, Path)

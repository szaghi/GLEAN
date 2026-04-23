"""Tests for `glean.config` (M1e).

Covers:
    - defaults when no config file exists
    - TOML parsing with various partial/complete configs
    - validation rejections (bad endpoint, unknown fields, type errors)
    - XDG config path resolution
    - starter config writer
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from glean.config import (
    CloudConfig,
    Config,
    OllamaConfig,
    config_path,
    load_config,
    write_starter_config,
)

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------


class TestDefaults:
    def test_load_nonexistent_returns_defaults(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "does_not_exist.toml")
        assert isinstance(cfg, Config)
        assert cfg.default_repo is None
        assert cfg.ollama.endpoint == "http://localhost:11434"
        assert cfg.ollama.model_fast == "llama3.2:3b"
        assert cfg.ollama.model_deep == "qwen2.5-coder:32b"
        assert cfg.cloud.enabled is False
        assert cfg.cloud.anthropic_model == "claude-opus-4-7"

    def test_editor_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("EDITOR", "nano")
        cfg = load_config(tmp_path / "missing.toml")
        assert cfg.editor == "nano"

    def test_editor_defaults_to_vi_when_env_unset(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("EDITOR", raising=False)
        cfg = load_config(tmp_path / "missing.toml")
        assert cfg.editor == "vi"


# -----------------------------------------------------------------------------
# TOML parsing
# -----------------------------------------------------------------------------


def _write_toml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(body)
    return p


class TestTomlParsing:
    def test_minimal_toml(self, tmp_path: Path) -> None:
        p = _write_toml(tmp_path, "")
        cfg = load_config(p)
        assert cfg.ollama.endpoint == "http://localhost:11434"

    def test_full_toml(self, tmp_path: Path) -> None:
        body = """\
default_repo = "~/notes"
editor = "hx"

[ollama]
endpoint = "http://gpu-box:11434"
model_fast = "qwen2.5:0.5b"
model_deep = "deepseek-r1:70b"

[cloud]
enabled = true
anthropic_model = "claude-sonnet-4-6"
"""
        p = _write_toml(tmp_path, body)
        cfg = load_config(p)
        assert cfg.default_repo == Path.home() / "notes"
        assert cfg.editor == "hx"
        assert cfg.ollama.endpoint == "http://gpu-box:11434"
        assert cfg.ollama.model_fast == "qwen2.5:0.5b"
        assert cfg.ollama.model_deep == "deepseek-r1:70b"
        assert cfg.cloud.enabled is True
        assert cfg.cloud.anthropic_model == "claude-sonnet-4-6"

    def test_partial_sections_merge_with_defaults(self, tmp_path: Path) -> None:
        """Only overriding one field in [ollama] leaves the others at defaults."""
        body = """\
[ollama]
model_fast = "tinyllama:1b"
"""
        p = _write_toml(tmp_path, body)
        cfg = load_config(p)
        assert cfg.ollama.model_fast == "tinyllama:1b"
        assert cfg.ollama.model_deep == "qwen2.5-coder:32b"  # default
        assert cfg.ollama.endpoint == "http://localhost:11434"  # default

    def test_default_repo_expands_tilde(self, tmp_path: Path) -> None:
        body = 'default_repo = "~/elsewhere"\n'
        p = _write_toml(tmp_path, body)
        cfg = load_config(p)
        assert cfg.default_repo == Path.home() / "elsewhere"

    def test_default_repo_expands_env_vars(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_NOTES_DIR", "/srv/notes")
        body = 'default_repo = "$MY_NOTES_DIR"\n'
        p = _write_toml(tmp_path, body)
        cfg = load_config(p)
        assert cfg.default_repo == Path("/srv/notes")

    def test_empty_default_repo_string_becomes_none(self, tmp_path: Path) -> None:
        body = 'default_repo = ""\n'
        p = _write_toml(tmp_path, body)
        cfg = load_config(p)
        assert cfg.default_repo is None


# -----------------------------------------------------------------------------
# Validation rejections
# -----------------------------------------------------------------------------


class TestValidation:
    def test_rejects_malformed_toml(self, tmp_path: Path) -> None:
        p = _write_toml(tmp_path, "not = valid = toml\n")
        with pytest.raises(ValueError, match=r"not valid TOML"):
            load_config(p)

    def test_rejects_unknown_top_level_field(self, tmp_path: Path) -> None:
        p = _write_toml(tmp_path, "rogue_setting = 42\n")
        with pytest.raises(ValidationError, match=r"Extra inputs are not permitted"):
            load_config(p)

    def test_rejects_unknown_field_in_ollama(self, tmp_path: Path) -> None:
        body = """\
[ollama]
typo_field = "oops"
"""
        p = _write_toml(tmp_path, body)
        with pytest.raises(ValidationError, match=r"Extra inputs are not permitted"):
            load_config(p)

    def test_rejects_non_http_endpoint(self, tmp_path: Path) -> None:
        body = """\
[ollama]
endpoint = "ws://localhost:11434"
"""
        p = _write_toml(tmp_path, body)
        with pytest.raises(ValidationError, match=r"http://"):
            load_config(p)

    def test_strips_trailing_slash_from_endpoint(self) -> None:
        oc = OllamaConfig(endpoint="http://localhost:11434/")
        assert oc.endpoint == "http://localhost:11434"

    def test_rejects_wrong_type_for_cloud_enabled(self, tmp_path: Path) -> None:
        body = """\
[cloud]
enabled = "yes please"
"""
        p = _write_toml(tmp_path, body)
        with pytest.raises(ValidationError):
            load_config(p)


# -----------------------------------------------------------------------------
# XDG path resolution
# -----------------------------------------------------------------------------


class TestConfigPath:
    def test_default_location(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        assert config_path() == Path.home() / ".config" / "glean" / "config.toml"

    def test_respects_xdg_config_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", "/custom/xdg")
        assert config_path() == Path("/custom/xdg") / "glean" / "config.toml"

    def test_empty_xdg_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unusual but possible: XDG_CONFIG_HOME set to empty string."""
        monkeypatch.setenv("XDG_CONFIG_HOME", "")
        # An empty string is falsy but os.environ.get returns ""; our code uses
        # `if xdg` which treats "" as unset. Verify the fallback path.
        assert config_path() == Path.home() / ".config" / "glean" / "config.toml"


# -----------------------------------------------------------------------------
# Starter config writer
# -----------------------------------------------------------------------------


class TestStarterConfig:
    def test_writes_to_path(self, tmp_path: Path) -> None:
        target = tmp_path / "glean" / "config.toml"
        result = write_starter_config(target)
        assert result == target
        assert target.exists()
        content = target.read_text()
        # Starter content should be parseable back as empty config (all lines commented).
        loaded = load_config(target)
        assert loaded.ollama.endpoint == "http://localhost:11434"  # the default
        # Verify key comment hints are present (useful guidance for users).
        assert "ANTHROPIC_API_KEY" in content
        assert "default_repo" in content

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "deeply" / "nested" / "path" / "config.toml"
        write_starter_config(target)
        assert target.exists()

    def test_refuses_to_overwrite_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text("existing content\n")
        with pytest.raises(FileExistsError, match=r"overwrite=True"):
            write_starter_config(target)

    def test_overwrite_flag(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text("existing content\n")
        write_starter_config(target, overwrite=True)
        assert "existing content" not in target.read_text()
        assert "ANTHROPIC_API_KEY" in target.read_text()


# -----------------------------------------------------------------------------
# Direct model tests (belt + suspenders)
# -----------------------------------------------------------------------------


class TestCloudConfig:
    def test_disabled_by_default(self) -> None:
        c = CloudConfig()
        assert c.enabled is False

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError, match=r"Extra inputs are not permitted"):
            CloudConfig.model_validate({"enabled": True, "api_key": "sk-ant-..."})

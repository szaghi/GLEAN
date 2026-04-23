"""User configuration loading for GLEAN (M1e).

Reads `$XDG_CONFIG_HOME/glean/config.toml` (default: `~/.config/glean/config.toml`)
and returns a typed `Config` object. Configuration is deliberately narrow at
v0.1: Ollama endpoint, two model tiers, default notes-repo path, and editor
command.

**No secrets in the config file.** The Anthropic API key is read from the
`ANTHROPIC_API_KEY` environment variable at backend-construction time (M2),
not from this config. This is a deliberate privacy and deployment choice:
config files get backed up, checked into dotfile repos, and shown in pair
programming; API keys should not follow them.

Typical config.toml shape:

    default_repo = "~/rossum"
    editor = "vi"

    [ollama]
    endpoint = "http://localhost:11434"
    model_fast = "llama3.2:3b"
    model_deep = "qwen2.5-coder:32b"

    [cloud]
    enabled = false
    # Anthropic API key is NOT stored here. Set ANTHROPIC_API_KEY in env.
    anthropic_model = "claude-opus-4-7"
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OllamaConfig(BaseModel):
    """Ollama backend configuration."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    endpoint: str = "http://localhost:11434"
    model_fast: str = "llama3.2:3b"
    model_deep: str = "qwen2.5-coder:32b"

    @field_validator("endpoint")
    @classmethod
    def _endpoint_must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"ollama endpoint must start with http:// or https://; got {v!r}")
        return v.rstrip("/")


class CloudConfig(BaseModel):
    """Cloud backend configuration.

    The API key is NOT stored here — it comes from `ANTHROPIC_API_KEY` env var
    at backend construction. `enabled` is a per-install toggle for whether the
    cloud door is even available; the `--cloud` CLI flag is the per-invocation
    opt-in that requires `enabled=true`.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    enabled: bool = False
    anthropic_model: str = "claude-opus-4-7"


class Config(BaseModel):
    """GLEAN user configuration, loaded from `~/.config/glean/config.toml`."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    default_repo: Path | None = None
    editor: str = Field(default_factory=lambda: os.environ.get("EDITOR", "vi"))
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)

    @field_validator("default_repo", mode="before")
    @classmethod
    def _expand_default_repo(cls, v: Any) -> Any:
        """Expand ~ and $ENV in user-provided repo paths."""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return Path(os.path.expandvars(v)).expanduser()
        return v


def config_path() -> Path:
    """Return the canonical path to GLEAN's config file.

    Respects `$XDG_CONFIG_HOME` if set; falls back to `~/.config/glean/config.toml`.
    Does NOT check whether the file exists.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "glean" / "config.toml"


def load_config(path: Path | None = None) -> Config:
    """Load config from disk, or return defaults if no file exists.

    Parameters
    ----------
    path
        Optional explicit path. Defaults to `config_path()`.

    Returns
    -------
    A validated `Config`. If the config file does not exist, returns a
    `Config` built from per-field defaults (Ollama at localhost, no cloud,
    editor from `$EDITOR`, default_repo unset).

    Raises
    ------
    ValueError
        When the TOML file exists but is malformed or violates the schema
        (unknown fields, bad types, invalid endpoint URL, etc.).
    """
    target = path if path is not None else config_path()
    if not target.exists():
        return Config()

    try:
        raw = tomllib.loads(target.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"config file at {target} is not valid TOML: {e}") from e

    return Config.model_validate(raw)


def write_starter_config(path: Path | None = None, *, overwrite: bool = False) -> Path:
    """Write a starter config.toml with all defaults shown as commented examples.

    Used by `glean init` to bootstrap the config file on first run. Does not
    overwrite an existing file unless `overwrite=True`.

    Returns the path written.

    Raises
    ------
    FileExistsError
        If the target already exists and `overwrite=False`.
    """
    target = path if path is not None else config_path()
    if target.exists() and not overwrite:
        raise FileExistsError(f"config already exists at {target}; pass overwrite=True to replace")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_STARTER_CONFIG)
    return target


_STARTER_CONFIG = """\
# GLEAN user configuration.
#
# Every field below is optional; commented-out values show the default.
# Secrets (API keys) are never stored here — the Anthropic API key must come
# from the ANTHROPIC_API_KEY environment variable at invocation time.

# default_repo = "~/rossum"       # default path for `glean` commands when -r not passed
# editor = "vi"                    # editor for source.yaml / claim-batch review (falls back to $EDITOR)

[ollama]
# endpoint = "http://localhost:11434"
# model_fast = "llama3.2:3b"      # used for classification, linking, lint
# model_deep = "qwen2.5-coder:32b" # used for claim extraction, wiki synthesis

[cloud]
# enabled = false                  # set true to allow --cloud flag on any command
# anthropic_model = "claude-opus-4-7"
"""

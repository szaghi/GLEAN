"""Shared pytest fixtures for GLEAN tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def empty_notes_repo(tmp_path: Path) -> Path:
    """A freshly scaffolded GLEAN notes repo in a tmp dir."""
    for layer in ("sources", "notebook", "claims", "wiki"):
        (tmp_path / layer).mkdir()
    (tmp_path / "AGENTS.md").write_text("# AGENTS (test fixture)\n")
    return tmp_path

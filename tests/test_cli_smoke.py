"""Smoke tests for the CLI entry point."""

from __future__ import annotations

from typer.testing import CliRunner

from glean.cli import app

runner = CliRunner()


def test_version_prints() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "glean" in result.stdout


def test_help_prints() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "GLEAN" in result.stdout or "glean" in result.stdout

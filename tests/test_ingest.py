"""Tests for `glean.ingest` — the top-level orchestrator (M3d).

These tests wire gates 1→2→3 together with mocked LLM and scripted $EDITOR.
Full end-to-end ingest against real Ollama belongs in tests/integration/
per D25.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from glean.cli import app
from glean.enums import SourceType
from glean.errors import GleanRepoError
from glean.ingest import abort_command, ingest_command

_GIT = shutil.which("git") or "git"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rossum_repo(tmp_path: Path) -> Path:
    """A fresh rossum repo, gpg-signing disabled."""
    repo = tmp_path / "rossum"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("# AGENTS v0.2 (fixture)\n")
    for layer in ("sources", "notebook", "claims", "wiki"):
        (repo / layer).mkdir()
        (repo / layer / ".gitkeep").write_text("")
    (repo / "wiki" / "log.md").write_text("# Log\n")
    (repo / "wiki" / "index.md").write_text("---\nkind: index\nupdated: 2026-04-23\n---\n\n# Index\n")

    subprocess.run([_GIT, "init", "-q", "-b", "main"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "user.email", "t@e.com"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "user.name", "Test"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "commit.gpgSign", "false"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "add", "-A"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "commit", "-q", "-m", "init"], cwd=repo, check=True)  # noqa: S603
    return repo


@pytest.fixture
def scripted_editor(tmp_path: Path) -> str:
    """A content-aware fake editor.

    The editor looks at the current file's content to decide what to do:
      - If it contains 'type: simulation' (gate-1 source.yaml), replace with
        a valid simulation source.yaml.
      - If it contains 'source: sim_2026_04_test_run' (gate-2 claim batch),
        leave the file unchanged (accept all drafts as-is).
      - Otherwise, leave unchanged.

    This models a real human who would type different things into different
    editor sessions depending on what they see.
    """
    sim_replacement = (
        "id: sim_2026_04_test_run\n"
        "type: simulation\n"
        "title: Test simulation\n"
        "authors: ['Zaghi, Stefano']\n"
        "year: 2026\n"
        "venue: local\n"
        "added: 2026-04-23\n"
        "confidence: high\n"
        "tags: []\n"
        "solver_repo_id: repo_foo_bar_abc1234\n"
        "solver_commit: abc1234+dirty\n"
        "input_files: ['input.ini']\n"
        "output_summary: output_summary.md\n"
        "run_date: 2026-04-14\n"
        "hardware: test\n"
    )
    safe = sim_replacement.replace("'", "'\\''")
    script = tmp_path / "fake_editor.sh"
    # Bash: if the file contains 'type: simulation' near the top AND does not
    # contain 'Batch review:' header, it's a gate-1 source.yaml. Replace it.
    # Otherwise, leave untouched.
    script.write_text(
        f"""#!/usr/bin/env bash
set -e
if grep -q 'type: simulation' "$1" && ! grep -q 'Batch review' "$1"; then
    printf '%s' '{safe}' > "$1"
fi
exit 0
"""
    )
    script.chmod(0o755)
    return str(script)


@pytest.fixture
def sim_run(tmp_path: Path) -> Path:
    """A simulation directory shaped to produce id 'sim_YYYY_MM_test_run'."""
    run = tmp_path / "test_run"
    run.mkdir()
    (run / "input.ini").write_text("[numerics]\nscheme=foo\n")
    return run


def _claim_block_yaml(slug: str) -> str:
    from datetime import date

    return (
        f"slug: {slug}\n"
        f"source: sim_2026_04_test_run\n"
        f'source_span: "§1"\n'
        f"quote: |\n  Verbatim.\n"
        f"claim: |\n  A paraphrase.\n"
        f"confidence: author_assertion\n"
        f"extracted: {date.today().isoformat()}\n"
        f"status: active\n"
    )


def _wiki_block(page_id: str = "entity_test") -> str:
    from datetime import date

    today = date.today().isoformat()
    return (
        f"===== {page_id} =====\n"
        f"---\n"
        f"id: {page_id}\n"
        f"kind: entity\n"
        f'title: "Test Entity"\n'
        f"created: {today}\n"
        f"updated: {today}\n"
        f"claim_count: 1\n"
        f"tags: [test]\n"
        f"---\n"
        f"\n"
        f"# Test Entity\n"
        f"\n"
        f"Content [[claim_2026_test_run_a_slug]].\n"
        f"===== end =====\n"
    )


# =============================================================================
# Full orchestrator end-to-end (mocked LLM)
# =============================================================================


class TestIngestCommandEndToEnd:
    def test_simulation_full_flow(
        self,
        rossum_repo: Path,
        sim_run: Path,
        scripted_editor: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Full 1→2→3 flow for a simulation ingest with mocked LLM and scripted editor."""
        monkeypatch.setenv("EDITOR", scripted_editor)

        # Patch the backend factory to return a mock for BOTH calls (gate 2 + gate 3).
        gate2_response = _claim_block_yaml("a_slug")
        gate3_response = _wiki_block("entity_test")

        mock_backend = MagicMock()
        mock_backend.complete = MagicMock(
            side_effect=[
                (gate2_response, MagicMock(to_log_bullets=lambda: ["- backend: fake"])),
                (gate3_response, MagicMock(to_log_bullets=lambda: ["- backend: fake"])),
            ]
        )

        with patch("glean.ingest.get_backend", return_value=mock_backend):
            ingest_command(
                source=str(sim_run),
                repo_path=rossum_repo,
                source_type=SourceType.SIMULATION,
                confirm_type=False,
            )

        # Verify: source directory committed, claims committed, wiki page in working tree.
        assert (rossum_repo / "sources" / "sim_2026_04_test_run" / "source.yaml").exists()
        claim_files = list((rossum_repo / "claims").glob("claim_*.md"))
        assert len(claim_files) == 1
        assert (rossum_repo / "wiki" / "entity_test.md").exists()
        # Log entry landed.
        log_content = (rossum_repo / "wiki" / "log.md").read_text()
        assert "ingest | sim_2026_04_test_run" in log_content

        # Gate 3's wiki changes should still be uncommitted per AGENTS.md §5.
        result = subprocess.run(  # noqa: S603
            [_GIT, "status", "--porcelain", "wiki/"],
            cwd=rossum_repo,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "wiki/entity_test.md" in result.stdout
        assert "wiki/log.md" in result.stdout or " M wiki/log.md" in result.stdout


# =============================================================================
# abort_command
# =============================================================================


class TestAbortCommand:
    def test_abort_clears_uncommitted(self, rossum_repo: Path) -> None:
        source_dir = rossum_repo / "sources" / "sim_2026_04_stale"
        source_dir.mkdir()
        (source_dir / "source.yaml").write_text("partial\n")
        abort_command(source_id="sim_2026_04_stale", repo_path=rossum_repo)
        assert not source_dir.exists()

    def test_abort_missing_raises(self, rossum_repo: Path) -> None:
        with pytest.raises(GleanRepoError, match=r"no gate-1 state"):
            abort_command(source_id="sim_ghost", repo_path=rossum_repo)


# =============================================================================
# CLI smoke tests
# =============================================================================


class TestCliIngest:
    def test_ingest_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "--type" in result.output
        assert "--cloud" in result.output
        assert "--pdf-extractor" in result.output
        assert "--no-network" in result.output
        assert "--resume" in result.output
        assert "--abort" in result.output

    def test_ingest_invalid_type_exits_1(self, tmp_path: Path) -> None:
        """Invalid --type should exit before reaching any filesystem access."""
        runner = CliRunner()
        dummy = tmp_path / "anything.pdf"
        result = runner.invoke(app, ["ingest", str(dummy), "--type", "bogus"])
        assert result.exit_code == 1
        assert "unknown --type" in result.output

    def test_ingest_abort_flag(self, rossum_repo: Path) -> None:
        source_dir = rossum_repo / "sources" / "sim_2026_04_stale"
        source_dir.mkdir()
        (source_dir / "source.yaml").write_text("partial\n")

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["ingest", "unused", "--abort", "sim_2026_04_stale", "--repo", str(rossum_repo)],
        )
        assert result.exit_code == 0
        assert "Aborted" in result.output
        assert not source_dir.exists()

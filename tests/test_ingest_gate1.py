"""Tests for `glean.ingest_gate1` (M3b).

$EDITOR is faked via a shell script that non-interactively modifies the
temp file (or leaves it unchanged). Crossref + PDF extraction are mocked.
Real git is used (tmp repos with gpg-signing disabled per conftest pattern).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from glean.adapters import DraftSource, IngestInput
from glean.config import Config
from glean.enums import SourceType
from glean.errors import GleanRepoError
from glean.ingest_gate1 import (
    _confirm_source_yaml,
    _gate1_complete,
    _gate1_partial,
    _prepend_error_marker,
    _strip_error_markers,
    abort_gate1,
    run_gate1,
)
from glean.repo import NotesRepo

_GIT = shutil.which("git") or "git"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rossum_repo(tmp_path: Path) -> Path:
    """A freshly-scaffolded, git-initialized rossum repo."""
    repo = tmp_path / "rossum"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("# AGENTS (fixture)\n")
    for layer in ("sources", "notebook", "claims", "wiki"):
        (repo / layer).mkdir()
        (repo / layer / ".gitkeep").write_text("")
    (repo / "wiki" / "log.md").write_text("# Log\n")

    subprocess.run([_GIT, "init", "-q", "-b", "main"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "user.email", "t@e.com"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "user.name", "Test"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "commit.gpgSign", "false"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "add", "-A"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "commit", "-q", "-m", "init"], cwd=repo, check=True)  # noqa: S603
    return repo


def _make_scripted_editor(tmp_path: Path, action: str) -> str:
    """Create a shell-script 'editor' that takes one file arg and performs `action`.

    Actions:
        "noop"                   — exit 0 without touching the file (unchanged save)
        "replace:<new_content>"  — overwrite the file with <new_content>
        "sed:<pattern>:<repl>"   — run sed to substitute in-place
        "fail"                   — exit 1 (editor-failed scenario)
    """
    script = tmp_path / "fake_editor.sh"
    if action == "noop":
        body = "#!/usr/bin/env bash\nexit 0\n"
    elif action.startswith("replace:"):
        new_content = action[len("replace:") :]
        # Write via printf to preserve newlines; escape single quotes.
        safe = new_content.replace("'", "'\\''")
        body = f"#!/usr/bin/env bash\nprintf '%s' '{safe}' > \"$1\"\n"
    elif action.startswith("sed:"):
        _, pattern, repl = action.split(":", 2)
        safe_p = pattern.replace("/", r"\/")
        safe_r = repl.replace("/", r"\/")
        body = f"#!/usr/bin/env bash\nsed -i 's/{safe_p}/{safe_r}/g' \"$1\"\n"
    elif action == "fail":
        body = "#!/usr/bin/env bash\nexit 1\n"
    else:
        raise ValueError(f"unknown action: {action}")

    script.write_text(body)
    script.chmod(0o755)
    return str(script)


# =============================================================================
# Error-marker helpers
# =============================================================================


class TestErrorMarkers:
    def test_prepend_and_strip_roundtrip(self) -> None:
        original = "id: x\ntitle: y\n"
        with_marker = _prepend_error_marker(original, "something failed")
        assert "# !! ERROR:" in with_marker
        stripped = _strip_error_markers(with_marker)
        # After strip, the error lines are gone but the original content is preserved.
        assert "id: x" in stripped
        assert "title: y" in stripped
        assert "# !! ERROR:" not in stripped

    def test_multiline_error(self) -> None:
        result = _prepend_error_marker("body\n", "line one\nline two")
        assert result.count("# !! ERROR:") == 3  # two error lines + one "fix the above"


# =============================================================================
# _confirm_source_yaml with fake editor
# =============================================================================


def _paper_draft(tmp_path: Path, pdf_name: str = "p.pdf") -> DraftSource:
    """Build a valid paper DraftSource for editor-confirmation tests."""
    pdf = tmp_path / pdf_name
    pdf.write_bytes(b"%PDF-1.4 fake\n")
    return DraftSource(
        source_type=SourceType.PAPER,
        proposed_id="paper_test_2026_sample",
        draft_yaml={
            "id": "paper_test_2026_sample",
            "type": "paper",
            "title": "Test paper",
            "authors": ["Test, Author"],
            "year": 2026,
            "venue": "J. Test",
            "doi": "10.1/abc",
            "url": None,
            "added": "2026-04-23",
            "confidence": "high",
            "tags": [],
            "bibtex_key": "test2026",
            "arxiv_id": None,
        },
        artifacts={"paper.md": "body"},
        files_to_copy={"paper.pdf": pdf},
    )


class TestConfirmSourceYaml:
    def test_noop_editor_accepts_draft(self, tmp_path: Path) -> None:
        """Per D26: saving without changes accepts the draft as-is."""
        editor = _make_scripted_editor(tmp_path, "noop")
        draft = _paper_draft(tmp_path)
        confirmed = _confirm_source_yaml(draft, editor)
        assert confirmed["id"] == "paper_test_2026_sample"
        assert confirmed["title"] == "Test paper"

    def test_editor_failure_raises(self, tmp_path: Path) -> None:
        editor = _make_scripted_editor(tmp_path, "fail")
        draft = _paper_draft(tmp_path)
        with pytest.raises(GleanRepoError, match=r"exited with error"):
            _confirm_source_yaml(draft, editor)

    def test_missing_editor_raises(self, tmp_path: Path) -> None:
        draft = _paper_draft(tmp_path)
        with pytest.raises(GleanRepoError, match=r"not found on PATH"):
            _confirm_source_yaml(draft, "/nonexistent/editor-binary-xyz")

    def test_user_edit_overriding_field(self, tmp_path: Path) -> None:
        """User changes the title via a sed edit; confirmed dict reflects it."""
        editor = _make_scripted_editor(tmp_path, "sed:Test paper:New Title")
        draft = _paper_draft(tmp_path)
        confirmed = _confirm_source_yaml(draft, editor)
        assert confirmed["title"] == "New Title"

    def test_invalid_yaml_retries_then_fails(self, tmp_path: Path) -> None:
        """If the editor keeps saving garbage, we fail after _EDITOR_RETRIES attempts."""
        # "replace" with a string that cannot be parsed as a YAML mapping.
        editor = _make_scripted_editor(tmp_path, "replace:not yaml: [broken")
        draft = _paper_draft(tmp_path)
        with pytest.raises(GleanRepoError, match=r"failed validation after"):
            _confirm_source_yaml(draft, editor)


# =============================================================================
# run_gate1 with a simulation ingest (simpler than paper — no PDF/Crossref)
# =============================================================================


@pytest.fixture
def sim_run(tmp_path: Path) -> Path:
    """Directory name drives the proposed slug — name it 'test_run' to match
    the target id `sim_YYYY_MM_test_run` in tests that hardcode the id."""
    run = tmp_path / "test_run"
    run.mkdir()
    (run / "input.ini").write_text("[numerics]\nscheme=foo\n")
    return run


def _config_with_editor(editor_path: str) -> Config:
    """Build a Config using the scripted editor as the editor command."""
    return Config(editor=editor_path)


class TestRunGate1Simulation:
    def test_happy_path_commits_source(self, rossum_repo: Path, sim_run: Path, tmp_path: Path) -> None:
        """A clean simulation ingest produces sources/<id>/ committed to git."""
        # Editor that fills the FILL markers to satisfy the simulation schema.
        # source.yaml must pass SimulationSource validation; fill id, authors,
        # solver_repo_id, solver_commit, hardware.
        replacement = (
            "id: sim_2026_04_test_run\n"
            "type: simulation\n"
            "title: Test simulation\n"
            "authors:\n  - 'Zaghi, Stefano'\n"
            "year: 2026\n"
            "venue: local\n"
            "added: 2026-04-23\n"
            "confidence: high\n"
            "tags: []\n"
            "solver_repo_id: repo_foo_bar_abc1234\n"
            "solver_commit: abc1234+dirty\n"
            "input_files:\n  - input.ini\n"
            "output_summary: output_summary.md\n"
            "run_date: 2026-04-14\n"
            "hardware: test\n"
        )
        editor = _make_scripted_editor(tmp_path, f"replace:{replacement}")
        repo = NotesRepo(rossum_repo)
        inp = IngestInput(input_spec=str(sim_run), type_override=SourceType.SIMULATION)
        config = _config_with_editor(editor)

        result = run_gate1(inp, repo=repo, config=config, confirm_type=False)

        assert result.source_type == SourceType.SIMULATION
        assert result.source_id == "sim_2026_04_test_run"
        assert result.commit_sha is not None
        assert not result.was_resumed
        # sources/<id>/source.yaml and input.ini should be committed.
        source_dir = rossum_repo / "sources" / "sim_2026_04_test_run"
        assert (source_dir / "source.yaml").exists()
        assert (source_dir / "input.ini").exists()
        assert (source_dir / "output_summary.md").exists()

    def test_resume_skips_when_gate1_complete(self, rossum_repo: Path, sim_run: Path, tmp_path: Path) -> None:
        """If sources/<id>/source.yaml is already committed, --resume returns immediately."""
        # First ingest.
        replacement = (
            "id: sim_2026_04_test_run\n"
            "type: simulation\n"
            "title: Test simulation\n"
            "authors:\n  - 'Zaghi, Stefano'\n"
            "year: 2026\n"
            "venue: local\n"
            "added: 2026-04-23\n"
            "confidence: high\n"
            "tags: []\n"
            "solver_repo_id: repo_foo_bar_abc1234\n"
            "solver_commit: abc1234+dirty\n"
            "input_files:\n  - input.ini\n"
            "output_summary: output_summary.md\n"
            "run_date: 2026-04-14\n"
            "hardware: test\n"
        )
        editor = _make_scripted_editor(tmp_path, f"replace:{replacement}")
        repo = NotesRepo(rossum_repo)
        inp = IngestInput(input_spec=str(sim_run), type_override=SourceType.SIMULATION)
        config = _config_with_editor(editor)

        run_gate1(inp, repo=repo, config=config, confirm_type=False)

        # Re-run with --resume.
        result = run_gate1(inp, repo=repo, config=config, resume=True, confirm_type=False)
        assert result.was_resumed
        assert result.source_id == "sim_2026_04_test_run"

    def test_refuses_without_resume_when_partial(self, rossum_repo: Path, sim_run: Path, tmp_path: Path) -> None:
        """Uncommitted sources/<id>/ without --resume raises a helpful error.

        The sim_run fixture's directory name is 'test_run', so the adapter
        proposes 'sim_YYYY_MM_test_run'. We plant a partial state under that
        exact id to trigger the refuse-without-resume path.
        """
        from datetime import date as _date

        repo = NotesRepo(rossum_repo)
        year_month = _date.today().strftime("%Y_%m")
        partial_id = f"sim_{year_month}_test_run"
        (rossum_repo / "sources" / partial_id).mkdir()
        (rossum_repo / "sources" / partial_id / "source.yaml").write_text("stale\n")

        editor = _make_scripted_editor(tmp_path, "noop")
        inp = IngestInput(input_spec=str(sim_run), type_override=SourceType.SIMULATION)
        config = _config_with_editor(editor)

        with pytest.raises(GleanRepoError, match=r"partial gate-1 state"):
            run_gate1(inp, repo=repo, config=config, resume=False, confirm_type=False)


# =============================================================================
# Notebook gate 1: self-critique invoked, file untouched
# =============================================================================


@pytest.fixture
def notebook_entry_path(rossum_repo: Path) -> Path:
    path = rossum_repo / "notebook" / "thought.md"
    path.write_text(
        "---\n"
        "id: note_2026_04_23_thought\n"
        "date: 2026-04-23\n"
        "topic: a thought\n"
        "status: draft\n"
        "tags: []\n"
        "---\n\n"
        "# About\n\nSome musings.\n"
    )
    return path


class TestRunGate1Notebook:
    def test_runs_critique_and_returns(self, rossum_repo: Path, notebook_entry_path: Path, tmp_path: Path) -> None:
        """Notebook gate 1 invokes the LLM critique and returns without a sources commit."""
        editor = _make_scripted_editor(tmp_path, "noop")
        repo = NotesRepo(rossum_repo)
        inp = IngestInput(input_spec=str(notebook_entry_path), type_override=SourceType.NOTEBOOK)
        config = _config_with_editor(editor)

        # Mock the Ollama backend's complete() to return a canned critique.
        with patch(
            "glean.ingest_gate1.OllamaBackend.complete",
            return_value=("No substantive weaknesses found.", None),
        ):
            result = run_gate1(inp, repo=repo, config=config, confirm_type=False)

        assert result.source_type == SourceType.NOTEBOOK
        assert result.source_id == "note_2026_04_23_thought"
        assert result.commit_sha is None  # no sources/ commit for notebooks
        # The notebook file itself is untouched.
        content = notebook_entry_path.read_text()
        assert "id: note_2026_04_23_thought" in content

    def test_llm_failure_does_not_block_ingest(
        self, rossum_repo: Path, notebook_entry_path: Path, tmp_path: Path
    ) -> None:
        """If Ollama is unreachable, we warn and proceed."""
        from glean.errors import GleanLLMError as _GLEANLLMError

        editor = _make_scripted_editor(tmp_path, "noop")
        repo = NotesRepo(rossum_repo)
        inp = IngestInput(input_spec=str(notebook_entry_path), type_override=SourceType.NOTEBOOK)
        config = _config_with_editor(editor)

        with patch(
            "glean.ingest_gate1.OllamaBackend.complete",
            side_effect=_GLEANLLMError("Ollama unreachable"),
        ):
            result = run_gate1(inp, repo=repo, config=config, confirm_type=False)

        assert result.source_type == SourceType.NOTEBOOK


# =============================================================================
# abort_gate1
# =============================================================================


class TestAbortGate1:
    def test_refuses_if_committed(self, rossum_repo: Path, sim_run: Path, tmp_path: Path) -> None:
        """After a successful ingest, abort must refuse."""
        replacement = (
            "id: sim_2026_04_test_run\n"
            "type: simulation\n"
            "title: Test simulation\n"
            "authors:\n  - 'Zaghi, Stefano'\n"
            "year: 2026\n"
            "venue: local\n"
            "added: 2026-04-23\n"
            "confidence: high\n"
            "tags: []\n"
            "solver_repo_id: repo_foo_bar_abc1234\n"
            "solver_commit: abc1234+dirty\n"
            "input_files:\n  - input.ini\n"
            "output_summary: output_summary.md\n"
            "run_date: 2026-04-14\n"
            "hardware: test\n"
        )
        editor = _make_scripted_editor(tmp_path, f"replace:{replacement}")
        repo = NotesRepo(rossum_repo)
        inp = IngestInput(input_spec=str(sim_run), type_override=SourceType.SIMULATION)
        config = _config_with_editor(editor)
        result = run_gate1(inp, repo=repo, config=config, confirm_type=False)

        with pytest.raises(GleanRepoError, match=r"committed files"):
            abort_gate1(repo, result.source_id)

    def test_clears_uncommitted(self, rossum_repo: Path) -> None:
        """abort_gate1 removes an uncommitted sources/<id>/ directory."""
        (rossum_repo / "sources" / "sim_2026_04_stale").mkdir()
        (rossum_repo / "sources" / "sim_2026_04_stale" / "source.yaml").write_text("partial\n")
        repo = NotesRepo(rossum_repo)

        abort_gate1(repo, "sim_2026_04_stale")
        assert not (rossum_repo / "sources" / "sim_2026_04_stale").exists()

    def test_no_state_to_abort_raises(self, rossum_repo: Path) -> None:
        repo = NotesRepo(rossum_repo)
        with pytest.raises(GleanRepoError, match=r"no gate-1 state"):
            abort_gate1(repo, "sim_2026_04_ghost")


# =============================================================================
# Type-confirmation prompt (non-interactive: always proceeds)
# =============================================================================


class TestTypeConfirmation:
    def test_type_override_skips_prompt(self, rossum_repo: Path, sim_run: Path, tmp_path: Path) -> None:
        """When --type is passed, no stdin prompt; adapter dispatches directly."""
        # A non-interactive stdin would block if confirm were attempted.
        replacement = (
            "id: sim_2026_04_test_run\n"
            "type: simulation\n"
            "title: T\n"
            "authors: ['A, B']\n"
            "year: 2026\n"
            "venue: v\n"
            "added: 2026-04-23\n"
            "confidence: high\n"
            "tags: []\n"
            "solver_repo_id: repo_a_b_abc1234\n"
            "solver_commit: abc1234\n"
            "input_files: ['input.ini']\n"
            "output_summary: output_summary.md\n"
            "run_date: 2026-04-14\n"
            "hardware: t\n"
        )
        editor = _make_scripted_editor(tmp_path, f"replace:{replacement}")
        repo = NotesRepo(rossum_repo)
        inp = IngestInput(input_spec=str(sim_run), type_override=SourceType.SIMULATION)
        config = _config_with_editor(editor)
        result = run_gate1(inp, repo=repo, config=config, confirm_type=False)
        assert result.source_type == SourceType.SIMULATION


# =============================================================================
# gate1_complete / gate1_partial
# =============================================================================


class TestGate1StateChecks:
    def test_complete_false_when_no_source_yaml(self, rossum_repo: Path) -> None:
        repo = NotesRepo(rossum_repo)
        draft = DraftSource(
            source_type=SourceType.SIMULATION,
            proposed_id="sim_2026_04_ghost",
            draft_yaml={},
            artifacts={},
            files_to_copy={},
        )
        assert not _gate1_complete(repo, draft)
        assert not _gate1_partial(repo, draft)

    def test_complete_true_for_committed_source(self, rossum_repo: Path) -> None:
        repo = NotesRepo(rossum_repo)
        source_dir = rossum_repo / "sources" / "sim_2026_04_x"
        source_dir.mkdir()
        (source_dir / "source.yaml").write_text("id: x\n")
        subprocess.run([_GIT, "add", "-A"], cwd=rossum_repo, check=True)  # noqa: S603
        subprocess.run(  # noqa: S603
            [_GIT, "commit", "-q", "-m", "test: add sim"], cwd=rossum_repo, check=True
        )
        draft = DraftSource(
            source_type=SourceType.SIMULATION,
            proposed_id="sim_2026_04_x",
            draft_yaml={},
            artifacts={},
            files_to_copy={},
        )
        assert _gate1_complete(repo, draft)
        assert not _gate1_partial(repo, draft)

    def test_partial_true_for_uncommitted_source(self, rossum_repo: Path) -> None:
        repo = NotesRepo(rossum_repo)
        (rossum_repo / "sources" / "sim_2026_04_y").mkdir()
        (rossum_repo / "sources" / "sim_2026_04_y" / "source.yaml").write_text("stale\n")
        draft = DraftSource(
            source_type=SourceType.SIMULATION,
            proposed_id="sim_2026_04_y",
            draft_yaml={},
            artifacts={},
            files_to_copy={},
        )
        assert not _gate1_complete(repo, draft)
        assert _gate1_partial(repo, draft)

    def test_notebook_never_complete(self, rossum_repo: Path) -> None:
        """Notebook state is always re-run; gate1_complete returns False."""
        repo = NotesRepo(rossum_repo)
        draft = DraftSource(
            source_type=SourceType.NOTEBOOK,
            proposed_id="note_2026_04_23_x",
            draft_yaml={},
            artifacts={},
            files_to_copy={},
        )
        assert not _gate1_complete(repo, draft)
        assert not _gate1_partial(repo, draft)

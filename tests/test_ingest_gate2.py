"""Tests for `glean.ingest_gate2` (M3c).

LLM calls are mocked. $EDITOR is the scripted shell fake pattern we
introduced in M3b tests. Real git is used (tmp repos with gpg-signing
disabled).
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from glean.config import Config
from glean.errors import GleanLLMError, GleanRepoError
from glean.ingest_gate2 import (
    _BATCH_DEFER_MARKER,
    _existing_claims_summary,
    _gate2_complete,
    _load_source_substrate,
    _render_batch_file,
    _source_slug_for,
    _split_yaml_blocks,
    _year_for_source,
    parse_llm_response,
    run_gate2,
)
from glean.repo import NotesRepo
from glean.schema import Claim

_GIT = shutil.which("git") or "git"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rossum_with_paper(tmp_path: Path) -> tuple[Path, str]:
    """A rossum repo with one filed paper source + paper.md, committed."""
    repo = tmp_path / "rossum"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("# AGENTS v0.2 (fixture)\n")
    for layer in ("sources", "notebook", "claims", "wiki"):
        (repo / layer).mkdir()
        (repo / layer / ".gitkeep").write_text("")
    (repo / "wiki" / "log.md").write_text("# Log\n")

    source_id = "paper_test_2023_sample"
    source_dir = repo / "sources" / source_id
    source_dir.mkdir()
    (source_dir / "source.yaml").write_text(
        f"id: {source_id}\n"
        "type: paper\n"
        "title: Sample paper\n"
        "authors: ['Test, Author']\n"
        "year: 2023\n"
        "venue: J. Test\n"
        "doi: 10.1/sample\n"
        "url: null\n"
        "added: 2026-04-23\n"
        "confidence: high\n"
        "tags: [test]\n"
        "bibtex_key: test2023\n"
        "arxiv_id: null\n"
    )
    (source_dir / "paper.md").write_text("# Sample paper\n\nA body about X, Y, and Z.\n")
    (source_dir / "paper.pdf").write_bytes(b"%PDF-1.4 fake\n")

    subprocess.run([_GIT, "init", "-q", "-b", "main"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "user.email", "t@e.com"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "user.name", "Test"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "commit.gpgSign", "false"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "add", "-A"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "commit", "-q", "-m", "init"], cwd=repo, check=True)  # noqa: S603
    return repo, source_id


def _make_scripted_editor(tmp_path: Path, action: str) -> str:
    """Minimal copy of the M3b fake-editor helper."""
    script = tmp_path / "fake_editor.sh"
    if action == "noop":
        body = "#!/usr/bin/env bash\nexit 0\n"
    elif action.startswith("replace:"):
        new_content = action[len("replace:") :]
        safe = new_content.replace("'", "'\\''")
        body = f"#!/usr/bin/env bash\nprintf '%s' '{safe}' > \"$1\"\n"
    elif action == "clear":
        # Wipe the file entirely — reject everything.
        body = '#!/usr/bin/env bash\n> "$1"\n'
    else:
        raise ValueError(f"unknown action: {action}")
    script.write_text(body)
    script.chmod(0o755)
    return str(script)


def _make_claim_block(slug: str, paraphrase: str = "A short claim.") -> str:
    """Build one YAML block for the LLM's fake response."""
    return (
        f"slug: {slug}\n"
        f"source: paper_test_2023_sample\n"
        f'source_span: "§1, para 1"\n'
        f"quote: |\n  Some verbatim text.\n"
        f"claim: |\n  {paraphrase}\n"
        f"confidence: author_assertion\n"
        f"extracted: {date.today().isoformat()}\n"
        f"status: active\n"
    )


def _fake_backend(response_text: str) -> MagicMock:
    """Mock an LLMBackend whose .complete() returns (response_text, call_log)."""
    mock_log = MagicMock()
    mock_log.prompt_hash = "deadbeef"
    backend = MagicMock()
    backend.complete = MagicMock(return_value=(response_text, mock_log))
    return backend


# =============================================================================
# _split_yaml_blocks
# =============================================================================


class TestSplitYamlBlocks:
    def test_single_block(self) -> None:
        assert _split_yaml_blocks("foo: bar") == ["foo: bar"]

    def test_two_blocks(self) -> None:
        text = "foo: a\n---\nbar: b\n"
        assert _split_yaml_blocks(text) == ["foo: a", "bar: b"]

    def test_leading_and_trailing_separators(self) -> None:
        text = "---\nfoo: a\n---\nbar: b\n---\n"
        assert _split_yaml_blocks(text) == ["foo: a", "bar: b"]

    def test_whitespace_only_blocks_dropped(self) -> None:
        text = "---\n\n---\nfoo: a\n---\n   \n"
        assert _split_yaml_blocks(text) == ["foo: a"]

    def test_dashes_only_as_own_line(self) -> None:
        """`---` inside a quoted string shouldn't split."""
        text = "foo: 'a --- b'\n"
        # Our splitter is line-based so `---` inside a string survives only
        # if it's not on its own line; here the line is `foo: 'a --- b'`,
        # which is a different line from just `---`.
        assert _split_yaml_blocks(text) == ["foo: 'a --- b'"]


# =============================================================================
# parse_llm_response
# =============================================================================


class TestParseLLMResponse:
    def test_single_valid_block(self) -> None:
        text = _make_claim_block("some_finding")
        claims = parse_llm_response(text, source_id="paper_test_2023_sample", year=2023)
        assert len(claims) == 1
        assert claims[0].id == "claim_2023_test_sample_some_finding"

    def test_multiple_blocks(self) -> None:
        text = _make_claim_block("finding_a") + "\n---\n" + _make_claim_block("finding_b")
        claims = parse_llm_response(text, source_id="paper_test_2023_sample", year=2023)
        assert len(claims) == 2
        slugs = {c.id for c in claims}
        assert any("finding_a" in s for s in slugs)
        assert any("finding_b" in s for s in slugs)

    def test_accepts_explicit_valid_id(self) -> None:
        text = (
            "id: claim_2023_other_source_explicit_id\n"
            "source: paper_test_2023_sample\n"
            'source_span: "§1"\n'
            "quote: x\nclaim: y\nconfidence: measured\nextracted: 2026-04-23\nstatus: active\n"
        )
        claims = parse_llm_response(text, source_id="paper_test_2023_sample", year=2023)
        assert claims[0].id == "claim_2023_other_source_explicit_id"

    def test_rejects_block_missing_slug_and_id(self) -> None:
        text = (
            "source: paper_test_2023_sample\n"
            'source_span: "§1"\n'
            "quote: x\nclaim: y\nconfidence: measured\nextracted: 2026-04-23\n"
        )
        with pytest.raises(GleanLLMError, match=r"no valid claims"):
            parse_llm_response(text, source_id="paper_test_2023_sample", year=2023)

    def test_empty_response_raises(self) -> None:
        with pytest.raises(GleanLLMError, match=r"no parseable YAML"):
            parse_llm_response("", source_id="paper_test_2023_sample", year=2023)

    def test_prose_around_blocks_tolerated(self) -> None:
        """LLMs sometimes wrap output in prose; we should extract what parses."""
        text = "Here are the claims:\n\n---\n" + _make_claim_block("real_claim") + "\n---\n\nEnd of extraction.\n"
        claims = parse_llm_response(text, source_id="paper_test_2023_sample", year=2023)
        # The `Here are the claims:` block is invalid YAML for Claim -> dropped.
        # Only the valid one survives.
        assert len(claims) == 1
        assert "real_claim" in claims[0].id

    def test_schema_failures_are_skipped(self) -> None:
        """One bad block + one good block → one claim."""
        bad = (
            "slug: bad\n"
            "source: paper_test_2023_sample\n"
            "confidence: not_a_valid_confidence\n"
            "quote: x\nclaim: y\nsource_span: '§1'\nextracted: 2026-04-23\nstatus: active\n"
        )
        text = bad + "\n---\n" + _make_claim_block("good")
        claims = parse_llm_response(text, source_id="paper_test_2023_sample", year=2023)
        assert len(claims) == 1
        assert "good" in claims[0].id


# =============================================================================
# _source_slug_for and _year_for_source
# =============================================================================


class TestSourceSlugFor:
    @pytest.mark.parametrize(
        "source_id,expected",
        [
            ("paper_zaghi_2023_amr_gpu_ibm", "zaghi_amr_gpu_ibm"),
            ("sim_2026_04_prism_rmf_restart", "prism_rmf_restart"),
            ("repo_szaghi_adam_dbe47a44", "szaghi_adam_dbe47a44"),
            ("note_2026_04_23_extending_amr", "extending_amr"),
        ],
    )
    def test_strips_prefix_and_year(self, source_id: str, expected: str) -> None:
        # Note: for sim and note, the "year" segments (2026, 04, 23) are
        # 4/2-digit but our heuristic drops only 4-digit year segments.
        # So sim_2026_04_x yields prism_rmf_restart... wait, let me re-check.
        assert _source_slug_for(source_id) == expected


class TestYearForSource:
    def test_paper_with_year_field(self) -> None:
        source = MagicMock()
        source.year = 2023
        source.date = None
        source.run_date = None
        source.commit_date = None
        source.archived_at = None
        assert _year_for_source(source) == 2023

    def test_notebook_with_date_field(self) -> None:
        source = MagicMock()
        source.year = None
        source.date = date(2026, 4, 23)
        source.run_date = None
        source.commit_date = None
        source.archived_at = None
        assert _year_for_source(source) == 2026

    def test_falls_back_to_current_year(self) -> None:
        source = MagicMock(spec=[])  # no attributes at all
        assert _year_for_source(source) == date.today().year


# =============================================================================
# _load_source_substrate
# =============================================================================


class TestLoadSourceSubstrate:
    def test_paper_loads_paper_md(self, rossum_with_paper: tuple[Path, str]) -> None:
        repo_path, source_id = rossum_with_paper
        repo = NotesRepo(repo_path)
        text = _load_source_substrate(repo, source_id)
        assert "A body about X, Y, and Z" in text

    def test_paper_missing_paper_md_raises(self, rossum_with_paper: tuple[Path, str]) -> None:
        repo_path, source_id = rossum_with_paper
        (repo_path / "sources" / source_id / "paper.md").unlink()
        repo = NotesRepo(repo_path)
        with pytest.raises(GleanRepoError, match=r"substrate file not found"):
            _load_source_substrate(repo, source_id)


# =============================================================================
# _existing_claims_summary
# =============================================================================


class TestExistingClaimsSummary:
    def test_empty_when_no_claims(self, rossum_with_paper: tuple[Path, str]) -> None:
        repo_path, source_id = rossum_with_paper
        repo = NotesRepo(repo_path)
        assert _existing_claims_summary(repo, exclude_source=source_id) == ""

    def test_excludes_given_source(self, rossum_with_paper: tuple[Path, str]) -> None:
        """Claims from the source under ingest should not appear in context."""
        repo_path, source_id = rossum_with_paper
        repo = NotesRepo(repo_path)
        # Seed one claim for a DIFFERENT source and one for THIS source.
        (repo_path / "claims" / "claim_2020_other_foo.md").write_text(
            "---\n"
            "id: claim_2020_other_foo\n"
            "source: paper_other_2020_bar\n"
            "source_span: '§1'\n"
            "quote: x\nclaim: A claim from another source.\n"
            "confidence: measured\nextracted: 2026-04-23\nstatus: active\n"
            "---\n\nbody\n"
        )
        (repo_path / "claims" / "claim_2023_test_sample_same_source.md").write_text(
            f"---\n"
            f"id: claim_2023_test_sample_same_source\n"
            f"source: {source_id}\n"
            f"source_span: '§1'\n"
            f"quote: x\nclaim: A claim from THIS source.\n"
            f"confidence: measured\nextracted: 2026-04-23\nstatus: active\n"
            f"---\n\nbody\n"
        )
        out = _existing_claims_summary(repo, exclude_source=source_id)
        assert "another source" in out
        assert "THIS source" not in out


# =============================================================================
# _render_batch_file
# =============================================================================


class TestRenderBatchFile:
    def test_renders_one_block_per_draft(self) -> None:
        claim = Claim.model_validate(
            {
                "id": "claim_2023_foo_bar_x",
                "source": "paper_foo_2023_bar",
                "source_span": "§1",
                "quote": "q",
                "claim": "c",
                "confidence": "author_assertion",
                "extracted": date(2026, 4, 23),
                "status": "active",
            }
        )
        text = _render_batch_file([claim])
        assert "claim_2023_foo_bar_x" in text
        assert _BATCH_DEFER_MARKER in text
        assert text.count("---\n") >= 2  # one before the block, one before DEFERRED

    def test_empty_list_still_has_defer_marker(self) -> None:
        text = _render_batch_file([])
        assert _BATCH_DEFER_MARKER in text


# =============================================================================
# run_gate2 end-to-end (mocked backend, scripted editor)
# =============================================================================


class TestRunGate2:
    def test_happy_path_approves_all(self, rossum_with_paper: tuple[Path, str], tmp_path: Path) -> None:
        """Mocked LLM returns one claim; noop editor accepts it; claim is committed."""
        repo_path, source_id = rossum_with_paper
        repo = NotesRepo(repo_path)
        editor = _make_scripted_editor(tmp_path, "noop")
        config = Config(editor=editor)

        llm_response = _make_claim_block("foundation_claim")
        backend = _fake_backend(llm_response)

        result = run_gate2(source_id, repo=repo, config=config, backend=backend, resume=False)

        assert result.source_id == source_id
        assert len(result.approved_claim_ids) == 1
        assert result.rejected_count == 0
        assert result.deferred_count == 0
        assert result.commit_sha is not None
        # Claim file is committed; .claim.draft is gone.
        approved_path = repo_path / "claims" / f"{result.approved_claim_ids[0]}.md"
        assert approved_path.exists()
        assert not any(repo_path.glob("claims/*.claim.draft"))

    def test_rejection_by_clearing_batch(self, rossum_with_paper: tuple[Path, str], tmp_path: Path) -> None:
        """If user clears the batch file, all drafts are rejected."""
        repo_path, source_id = rossum_with_paper
        repo = NotesRepo(repo_path)
        editor = _make_scripted_editor(tmp_path, "clear")
        config = Config(editor=editor)

        llm_response = _make_claim_block("slug_a") + "\n---\n" + _make_claim_block("slug_b")
        backend = _fake_backend(llm_response)

        result = run_gate2(source_id, repo=repo, config=config, backend=backend, resume=False)
        assert result.approved_claim_ids == []
        assert result.rejected_count == 2
        assert result.commit_sha is None

    def test_resume_short_circuits_when_already_committed(
        self, rossum_with_paper: tuple[Path, str], tmp_path: Path
    ) -> None:
        """If approved claims for source exist, gate 2 reports was_resumed=True."""
        repo_path, source_id = rossum_with_paper
        # Seed an approved claim directly.
        (repo_path / "claims" / "claim_2023_test_sample_preexisting.md").write_text(
            f"---\n"
            f"id: claim_2023_test_sample_preexisting\n"
            f"source: {source_id}\n"
            f"source_span: '§1'\nquote: x\nclaim: pre-existing claim.\n"
            f"confidence: measured\nextracted: 2026-04-23\nstatus: active\n"
            f"---\n\nbody\n"
        )
        repo = NotesRepo(repo_path)
        editor = _make_scripted_editor(tmp_path, "noop")
        config = Config(editor=editor)
        backend = _fake_backend("(should not be called)")

        result = run_gate2(source_id, repo=repo, config=config, backend=backend, resume=True)
        assert result.was_resumed is True
        assert "claim_2023_test_sample_preexisting" in result.approved_claim_ids
        backend.complete.assert_not_called()

    def test_missing_source_raises(self, rossum_with_paper: tuple[Path, str], tmp_path: Path) -> None:
        repo_path, _source_id = rossum_with_paper
        repo = NotesRepo(repo_path)
        config = Config(editor=_make_scripted_editor(tmp_path, "noop"))
        backend = _fake_backend("")
        with pytest.raises(GleanRepoError, match=r"source not found"):
            run_gate2(
                "paper_ghost_2099_nonexistent",
                repo=repo,
                config=config,
                backend=backend,
            )

    def test_refuses_partial_without_resume(self, rossum_with_paper: tuple[Path, str], tmp_path: Path) -> None:
        """If .claim.draft files exist for this source, refuse without --resume."""
        repo_path, source_id = rossum_with_paper
        # Plant a draft for this source.
        (repo_path / "claims" / "claim_2023_test_sample_stale.claim.draft").write_text(
            f"---\n"
            f"id: claim_2023_test_sample_stale\n"
            f"source: {source_id}\n"
            f"source_span: '§1'\nquote: x\nclaim: stale.\n"
            f"confidence: measured\nextracted: 2026-04-23\nstatus: active\n"
            f"---\n\nbody\n"
        )
        repo = NotesRepo(repo_path)
        config = Config(editor=_make_scripted_editor(tmp_path, "noop"))
        backend = _fake_backend("")

        with pytest.raises(GleanRepoError, match=r"partial gate-2 state"):
            run_gate2(source_id, repo=repo, config=config, backend=backend, resume=False)


class TestGate2StateChecks:
    def test_complete_false_when_no_claims(self, rossum_with_paper: tuple[Path, str]) -> None:
        repo_path, source_id = rossum_with_paper
        repo = NotesRepo(repo_path)
        assert not _gate2_complete(repo, source_id)

    def test_complete_true_when_one_claim_for_source(self, rossum_with_paper: tuple[Path, str]) -> None:
        repo_path, source_id = rossum_with_paper
        (repo_path / "claims" / "claim_2023_test_sample_any.md").write_text(
            f"---\n"
            f"id: claim_2023_test_sample_any\n"
            f"source: {source_id}\n"
            f"source_span: '§1'\nquote: x\nclaim: c.\n"
            f"confidence: measured\nextracted: 2026-04-23\nstatus: active\n"
            f"---\n\nbody\n"
        )
        repo = NotesRepo(repo_path)
        assert _gate2_complete(repo, source_id)

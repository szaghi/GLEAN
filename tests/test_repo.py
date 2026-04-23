"""Tests for `glean.repo` — NotesRepo I/O primitives (M2b).

Two tiers of tests:
    - TestAgainstRealRossum — exit-criterion tests. Point NotesRepo at the
      real rossum repo and verify every Phase 1 artifact loads cleanly.
    - All other classes — run against freshly-scaffolded tmp repos for write,
      roundtrip, and error paths that would be expensive/destructive against
      real rossum.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from glean.enums import ClaimStatus, Confidence, NotebookStatus, SourceConfidence, WikiKind
from glean.errors import GleanRepoError
from glean.repo import NotesRepo, _atomic_write, _split_frontmatter
from glean.schema import (
    Claim,
    LogEntry,
    NotebookSource,
    PaperSource,
    SimulationSource,
    WikiPage,
)

ROSSUM = Path.home() / "rossum"
_ROSSUM_AVAILABLE = (ROSSUM / "AGENTS.md").exists()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def empty_notes_repo(tmp_path: Path) -> Path:
    """Minimal valid rossum-shaped directory for write/roundtrip tests."""
    repo = tmp_path / "rossum"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("# AGENTS (test fixture)\n")
    for layer in ("sources", "notebook", "claims", "wiki"):
        (repo / layer).mkdir()
    # Seed wiki/log.md so append_log has a real file to work with.
    (repo / "wiki" / "log.md").write_text("# Log\n\n---\n")
    return repo


# =============================================================================
# Helper function tests
# =============================================================================


class TestSplitFrontmatter:
    def test_valid_frontmatter(self) -> None:
        text = "---\nid: x\ntitle: foo\n---\n\nbody content\n"
        fm, body = _split_frontmatter(text)
        assert fm == {"id": "x", "title": "foo"}
        assert body == "\nbody content\n"

    def test_no_opening_marker(self) -> None:
        with pytest.raises(GleanRepoError, match=r"does not begin"):
            _split_frontmatter("body without frontmatter\n")

    def test_no_closing_marker(self) -> None:
        with pytest.raises(GleanRepoError, match=r"no closing"):
            _split_frontmatter("---\nid: x\ntitle: y\n")

    def test_empty_frontmatter(self) -> None:
        fm, _body = _split_frontmatter("---\n---\n\nbody\n")
        assert fm == {}

    def test_malformed_yaml(self) -> None:
        with pytest.raises(GleanRepoError, match=r"YAML parse"):
            _split_frontmatter("---\nbad: : :\n---\n\nbody\n")

    def test_non_mapping_rejected(self) -> None:
        with pytest.raises(GleanRepoError, match=r"must be a YAML mapping"):
            _split_frontmatter("---\n- just_a_list\n---\n\nbody\n")


class TestAtomicWrite:
    def test_creates_file(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        _atomic_write(target, "hello\n")
        assert target.read_text() == "hello\n"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "path" / "out.txt"
        _atomic_write(target, "hello\n")
        assert target.read_text() == "hello\n"

    def test_replaces_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        target.write_text("old\n")
        _atomic_write(target, "new\n")
        assert target.read_text() == "new\n"

    def test_no_tmp_file_left_behind_on_success(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        _atomic_write(target, "hello\n")
        # Only the target file; no lingering .tmp
        assert list(tmp_path.iterdir()) == [target]


# =============================================================================
# Constructor validation
# =============================================================================


class TestNotesRepoConstruction:
    def test_valid_repo(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        assert repo.root == empty_notes_repo.resolve()

    def test_accepts_str_path(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(str(empty_notes_repo))
        assert repo.root == empty_notes_repo.resolve()

    def test_expands_user(self, empty_notes_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(empty_notes_repo.parent))
        repo = NotesRepo(f"~/{empty_notes_repo.name}")
        assert repo.root == empty_notes_repo.resolve()

    def test_rejects_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(GleanRepoError, match=r"not a directory"):
            NotesRepo(tmp_path / "nonexistent")

    def test_rejects_missing_agents_md(self, tmp_path: Path) -> None:
        repo = tmp_path / "incomplete"
        repo.mkdir()
        for layer in ("sources", "notebook", "claims", "wiki"):
            (repo / layer).mkdir()
        with pytest.raises(GleanRepoError, match=r"missing AGENTS.md"):
            NotesRepo(repo)

    def test_rejects_missing_layer_dir(self, tmp_path: Path) -> None:
        repo = tmp_path / "incomplete"
        repo.mkdir()
        (repo / "AGENTS.md").write_text("# foo\n")
        for layer in ("sources", "notebook", "claims"):
            (repo / layer).mkdir()
        # no wiki/
        with pytest.raises(GleanRepoError, match=r"'wiki'"):
            NotesRepo(repo)


# =============================================================================
# Source I/O — write + read roundtrips
# =============================================================================


def _sample_paper() -> PaperSource:
    return PaperSource.model_validate(
        {
            "id": "paper_test_2026_sample",
            "type": "paper",
            "title": "A sample paper",
            "authors": ["Test, Author"],
            "year": 2026,
            "venue": "J. Test",
            "doi": "10.1/abc",
            "url": None,
            "added": date(2026, 4, 23),
            "confidence": "high",
            "tags": [],
            "bibtex_key": "test2026",
            "arxiv_id": None,
        }
    )


def _sample_simulation() -> SimulationSource:
    return SimulationSource.model_validate(
        {
            "id": "sim_2026_04_sample_run",
            "type": "simulation",
            "title": "A sample run",
            "authors": ["Test, Author"],
            "year": 2026,
            "venue": "local",
            "added": date(2026, 4, 23),
            "confidence": "high",
            "tags": [],
            "solver_repo_id": "repo_foo_bar_dbe47a44",
            "solver_commit": "dbe47a44",
            "input_files": ["input.ini"],
            "output_summary": "output_summary.md",
            "run_date": date(2026, 4, 14),
            "hardware": "test",
        }
    )


def _sample_notebook() -> NotebookSource:
    return NotebookSource.model_validate(
        {
            "id": "note_2026_04_23_sample_thought",
            "type": "notebook",
            "date": date(2026, 4, 23),
            "topic": "A sample thought",
            "status": "draft",
            "tags": [],
        }
    )


class TestSourceIO:
    def test_save_and_load_paper(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        p = _sample_paper()
        yaml_path = repo.save_source(p)
        assert yaml_path.exists()
        loaded = repo.load_source(p.id)
        assert isinstance(loaded, PaperSource)
        assert loaded.id == p.id
        assert loaded.confidence == SourceConfidence.HIGH

    def test_save_and_load_simulation(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        s = _sample_simulation()
        repo.save_source(s)
        loaded = repo.load_source(s.id)
        assert isinstance(loaded, SimulationSource)
        assert loaded.solver_commit == "dbe47a44"

    def test_save_source_refuses_notebook(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        with pytest.raises(GleanRepoError, match=r"save_notebook"):
            repo.save_source(_sample_notebook())  # type: ignore[arg-type]

    def test_load_source_rejects_malformed_id(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        with pytest.raises(GleanRepoError, match=r"not a valid source ID"):
            repo.load_source("not-a-valid-id")

    def test_load_source_missing_raises(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        with pytest.raises(GleanRepoError, match=r"source.yaml not found"):
            repo.load_source("paper_ghost_2099_nonexistent")

    def test_load_notebook_scans_by_frontmatter_id(self, empty_notes_repo: Path) -> None:
        """Notebook files have free filenames; load_source must scan frontmatter."""
        repo = NotesRepo(empty_notes_repo)
        n = _sample_notebook()
        # Free filename — not the ID.
        (repo.notebook_dir / "random_filename.md").write_text(
            f"---\n"
            f"id: {n.id}\n"
            f"date: {n.date.isoformat()}\n"
            f"topic: {n.topic!r}\n"
            f"status: {n.status.value}\n"
            f"tags: []\n"
            f"---\n\nbody\n"
        )
        loaded = repo.load_source(n.id)
        assert isinstance(loaded, NotebookSource)
        assert loaded.id == n.id

    def test_load_notebook_not_found(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        with pytest.raises(GleanRepoError, match=r"notebook entry not found"):
            repo.load_source("note_2099_01_01_ghost")

    def test_list_sources_includes_notebook(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        repo.save_source(_sample_paper())
        n = _sample_notebook()
        (repo.notebook_dir / "any_name.md").write_text(
            f"---\nid: {n.id}\ndate: {n.date.isoformat()}\ntopic: x\nstatus: draft\ntags: []\n---\n\n"
        )
        ids = list(repo.list_sources())
        assert _sample_paper().id in ids
        assert n.id in ids


# =============================================================================
# Claim I/O
# =============================================================================


def _sample_claim() -> Claim:
    return Claim.model_validate(
        {
            "id": "claim_2026_test_sample",
            "source": "paper_test_2026_sample",
            "source_span": "§1",
            "quote": "Some quote.",
            "claim": "A paraphrase.",
            "confidence": "author_assertion",
            "extracted": date(2026, 4, 23),
            "status": "active",
        }
    )


class TestClaimIO:
    def test_save_draft_creates_gitignored_file(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        c = _sample_claim()
        path = repo.save_claim_draft(c, body="body notes")
        assert path.name.endswith(".claim.draft")

    def test_promote_draft(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        c = _sample_claim()
        repo.save_claim_draft(c)
        final = repo.promote_claim_draft(c.id)
        assert final.name == f"{c.id}.md"
        assert final.exists()
        assert not (repo.claims_dir / f"{c.id}.claim.draft").exists()

    def test_promote_missing_draft_raises(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        with pytest.raises(GleanRepoError, match=r"draft not found"):
            repo.promote_claim_draft("claim_2026_ghost")

    def test_promote_refuses_to_overwrite(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        c = _sample_claim()
        # Both an approved .md and a draft exist simultaneously.
        repo.save_claim(c, body="old approved")
        repo.save_claim_draft(c, body="new draft")
        with pytest.raises(GleanRepoError, match=r"target already exists"):
            repo.promote_claim_draft(c.id)

    def test_load_claim_roundtrip(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        c = _sample_claim()
        repo.save_claim(c, body="Extended context.\n")
        loaded, body = repo.load_claim(c.id)
        assert loaded.id == c.id
        assert loaded.confidence == Confidence.AUTHOR_ASSERTION
        assert loaded.status == ClaimStatus.ACTIVE
        assert "Extended context." in body

    def test_list_claims_excludes_drafts(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        c = _sample_claim()
        repo.save_claim_draft(c)
        assert list(repo.list_claims()) == []
        assert c.id in list(repo.list_claim_drafts())


# =============================================================================
# Wiki I/O
# =============================================================================


def _sample_page() -> WikiPage:
    return WikiPage.model_validate(
        {
            "id": "entity_test_thing",
            "kind": "entity",
            "title": "Test Thing",
            "created": date(2026, 4, 23),
            "updated": date(2026, 4, 23),
            "claim_count": 0,
            "tags": [],
        }
    )


class TestWikiIO:
    def test_save_and_load(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        p = _sample_page()
        repo.save_wiki_page(p, body="# Test Thing\n\nbody content.\n")
        loaded, body = repo.load_wiki_page(p.id)
        assert loaded.kind == WikiKind.ENTITY
        assert "body content" in body

    def test_list_excludes_index_and_log(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        # Create fake index.md and log.md (log already exists; create index).
        (repo.wiki_dir / "index.md").write_text("---\nkind: index\nupdated: 2026-04-23\n---\n")
        p = _sample_page()
        repo.save_wiki_page(p, body="body")
        pages = list(repo.list_wiki_pages())
        assert p.id in pages
        assert "index" not in pages
        assert "log" not in pages

    def test_load_wiki_page_rejects_index(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        with pytest.raises(GleanRepoError, match=r"not a regular wiki page"):
            repo.load_wiki_page("index")


# =============================================================================
# Log I/O
# =============================================================================


class TestLogIO:
    def test_append_log(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        entry = LogEntry(
            date=date(2026, 4, 23),
            op="ingest",
            subject="paper_test_2026_sample",
            body_lines=["- source: paper_test_2026_sample", "- claims: 5"],
        )
        repo.append_log(entry)
        content = repo.log_md_path.read_text()
        assert "## [2026-04-23] ingest | paper_test_2026_sample" in content
        assert "- source: paper_test_2026_sample" in content

    def test_append_preserves_prior_content(self, empty_notes_repo: Path) -> None:
        repo = NotesRepo(empty_notes_repo)
        original = repo.log_md_path.read_text()
        entry = LogEntry(date=date(2026, 4, 23), op="init", subject="test")
        repo.append_log(entry)
        content = repo.log_md_path.read_text()
        assert content.startswith(original.rstrip() + "\n")


# =============================================================================
# Exit-criterion tests: load every Phase 1 artifact from real rossum
# =============================================================================


@pytest.mark.skipif(not _ROSSUM_AVAILABLE, reason="rossum repo not present")
class TestAgainstRealRossum:
    def test_constructs(self) -> None:
        NotesRepo(ROSSUM)

    def test_loads_every_phase1_source(self) -> None:
        repo = NotesRepo(ROSSUM)
        expected = {
            "paper_zaghi_2023_amr_gpu_ibm",
            "sim_2026_04_prism_rmf_restart",
            "repo_szaghi_adam_dbe47a44",
            "note_2026_04_23_extending_amr_2to1_to_4to1",
        }
        for sid in expected:
            loaded = repo.load_source(sid)
            assert loaded.id == sid

    def test_list_sources_finds_all_four(self) -> None:
        repo = NotesRepo(ROSSUM)
        ids = set(repo.list_sources())
        assert "paper_zaghi_2023_amr_gpu_ibm" in ids
        assert "sim_2026_04_prism_rmf_restart" in ids
        assert "repo_szaghi_adam_dbe47a44" in ids
        assert "note_2026_04_23_extending_amr_2to1_to_4to1" in ids

    def test_loads_every_phase1_claim(self) -> None:
        repo = NotesRepo(ROSSUM)
        claim_ids = list(repo.list_claims())
        assert len(claim_ids) >= 31
        for cid in claim_ids:
            parsed, _body = repo.load_claim(cid)
            assert parsed.id == cid

    def test_loads_every_phase1_wiki_page(self) -> None:
        repo = NotesRepo(ROSSUM)
        page_ids = list(repo.list_wiki_pages())
        assert len(page_ids) >= 11
        for pid in page_ids:
            parsed, _body = repo.load_wiki_page(pid)
            assert parsed.id == pid

    def test_loads_wiki_index(self) -> None:
        repo = NotesRepo(ROSSUM)
        idx, _body = repo.load_wiki_index()
        assert idx.kind == "index"

    def test_loads_notebook_entry_by_id(self) -> None:
        repo = NotesRepo(ROSSUM)
        loaded = repo.load_source("note_2026_04_23_extending_amr_2to1_to_4to1")
        assert isinstance(loaded, NotebookSource)
        assert loaded.status == NotebookStatus.DRAFT

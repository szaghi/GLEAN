"""Tests for `glean.ingest_gate3` (M3d — gate 3).

LLM calls are mocked. Real git tmp repos (gpg-signing disabled per fixture).
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
from glean.ingest_gate3 import (
    _NO_WIKI_UPDATES,
    _assert_wiki_clean,
    parse_wiki_updates,
    run_gate3,
)
from glean.repo import NotesRepo

_GIT = shutil.which("git") or "git"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rossum_with_paper_and_claims(tmp_path: Path) -> tuple[Path, str, list[str]]:
    """A rossum repo with a paper source + 2 approved claims, committed."""
    repo = tmp_path / "rossum"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("# AGENTS v0.2 (fixture)\n")
    for layer in ("sources", "notebook", "claims", "wiki"):
        (repo / layer).mkdir()
        (repo / layer / ".gitkeep").write_text("")
    (repo / "wiki" / "log.md").write_text("# Log\n")
    (repo / "wiki" / "index.md").write_text("---\nkind: index\nupdated: 2026-04-23\n---\n\n# Index\n\n(empty)\n")

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
        "tags: []\n"
        "bibtex_key: test2023\n"
        "arxiv_id: null\n"
    )

    # Seed two approved claims.
    claim_ids = [
        "claim_2023_test_sample_finding_one",
        "claim_2023_test_sample_finding_two",
    ]
    for cid in claim_ids:
        (repo / "claims" / f"{cid}.md").write_text(
            f"---\n"
            f"id: {cid}\n"
            f"source: {source_id}\n"
            f"source_span: '§1'\n"
            f"quote: Verbatim text.\n"
            f"claim: A claim paraphrase.\n"
            f"confidence: author_assertion\n"
            f"extracted: 2026-04-23\n"
            f"status: active\n"
            f"---\n\nbody\n"
        )

    subprocess.run([_GIT, "init", "-q", "-b", "main"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "user.email", "t@e.com"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "user.name", "Test"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "commit.gpgSign", "false"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "add", "-A"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "commit", "-q", "-m", "init"], cwd=repo, check=True)  # noqa: S603
    return repo, source_id, claim_ids


def _fake_backend(response_text: str) -> MagicMock:
    mock_log = MagicMock()
    mock_log.prompt_hash = "deadbeef"
    mock_log.to_log_bullets = MagicMock(return_value=["- backend: fake"])
    backend = MagicMock()
    backend.complete = MagicMock(return_value=(response_text, mock_log))
    return backend


def _wiki_update_block(page_id: str, body: str) -> str:
    return f"===== {page_id} =====\n{body}===== end =====\n"


def _sample_page_body(page_id: str, kind: str = "entity", claim_count: int = 1) -> str:
    today = date.today().isoformat()
    return (
        f"---\n"
        f"id: {page_id}\n"
        f"kind: {kind}\n"
        f'title: "{page_id}"\n'
        f"created: {today}\n"
        f"updated: {today}\n"
        f"claim_count: {claim_count}\n"
        f"tags: [test]\n"
        f"---\n"
        f"\n"
        f"# {page_id}\n"
        f"\n"
        f"Some content [[claim_2023_test_sample_finding_one]].\n"
    )


# =============================================================================
# parse_wiki_updates
# =============================================================================


class TestParseWikiUpdates:
    def test_empty_input(self) -> None:
        assert parse_wiki_updates("") == {}

    def test_single_block(self) -> None:
        body = _sample_page_body("entity_foo")
        text = _wiki_update_block("entity_foo", body)
        result = parse_wiki_updates(text)
        assert "entity_foo" in result
        assert "id: entity_foo" in result["entity_foo"]

    def test_multiple_blocks(self) -> None:
        body_a = _sample_page_body("entity_a")
        body_b = _sample_page_body("concept_b", kind="concept")
        text = _wiki_update_block("entity_a", body_a) + _wiki_update_block("concept_b", body_b)
        result = parse_wiki_updates(text)
        assert set(result.keys()) == {"entity_a", "concept_b"}

    def test_missing_end_marker_still_extracts(self) -> None:
        body = _sample_page_body("entity_foo")
        text = f"===== entity_foo =====\n{body}"  # no end marker
        result = parse_wiki_updates(text)
        assert "entity_foo" in result

    def test_preamble_and_epilogue_ignored(self) -> None:
        """LLMs sometimes prepend/append prose despite the instructions."""
        body = _sample_page_body("entity_foo")
        text = "Here are the wiki updates:\n\n" + _wiki_update_block("entity_foo", body) + "\nThat's all.\n"
        result = parse_wiki_updates(text)
        assert list(result.keys()) == ["entity_foo"]

    def test_end_closes_block_explicitly(self) -> None:
        """When the LLM emits '===== end =====', the subsequent lines aren't part of the block."""
        body = _sample_page_body("entity_foo")
        # After end, add stray prose that should NOT be in entity_foo.
        text = _wiki_update_block("entity_foo", body) + "\nsome trailing prose\n"
        result = parse_wiki_updates(text)
        assert "trailing prose" not in result["entity_foo"]


# =============================================================================
# _assert_wiki_clean
# =============================================================================


class TestAssertWikiClean:
    def test_passes_when_clean(self, rossum_with_paper_and_claims: tuple[Path, str, list[str]]) -> None:
        repo_path, _source_id, _claim_ids = rossum_with_paper_and_claims
        repo = NotesRepo(repo_path)
        _assert_wiki_clean(repo)  # should not raise

    def test_raises_on_unstaged_wiki_change(self, rossum_with_paper_and_claims: tuple[Path, str, list[str]]) -> None:
        repo_path, _source_id, _claim_ids = rossum_with_paper_and_claims
        (repo_path / "wiki" / "index.md").write_text("modified\n")
        repo = NotesRepo(repo_path)
        with pytest.raises(GleanRepoError, match=r"wiki/ has uncommitted"):
            _assert_wiki_clean(repo)

    def test_raises_on_untracked_wiki_file(self, rossum_with_paper_and_claims: tuple[Path, str, list[str]]) -> None:
        repo_path, _source_id, _claim_ids = rossum_with_paper_and_claims
        (repo_path / "wiki" / "new_page.md").write_text("x\n")
        repo = NotesRepo(repo_path)
        with pytest.raises(GleanRepoError, match=r"wiki/ has uncommitted"):
            _assert_wiki_clean(repo)

    def test_tolerates_dirty_non_wiki_changes(self, rossum_with_paper_and_claims: tuple[Path, str, list[str]]) -> None:
        """Dirty notebook/ or sources/ must not block gate 3 (D21 scope)."""
        repo_path, _source_id, _claim_ids = rossum_with_paper_and_claims
        (repo_path / "notebook" / "stray.md").write_text("---\nid: x\n---\n")
        (repo_path / "sources" / "something_else.txt").write_text("x\n")
        repo = NotesRepo(repo_path)
        _assert_wiki_clean(repo)  # should NOT raise


# =============================================================================
# run_gate3 — happy paths and edge cases
# =============================================================================


class TestRunGate3:
    def test_happy_path_applies_pages_and_logs(self, rossum_with_paper_and_claims: tuple[Path, str, list[str]]) -> None:
        repo_path, source_id, claim_ids = rossum_with_paper_and_claims
        repo = NotesRepo(repo_path)
        config = Config()

        body = _sample_page_body("entity_sample", claim_count=1)
        llm_response = _wiki_update_block("entity_sample", body)
        backend = _fake_backend(llm_response)

        result = run_gate3(
            source_id,
            repo=repo,
            config=config,
            backend=backend,
            approved_claim_ids=claim_ids,
            gate1_sub_bullets=["- source_id: x"],
            gate2_sub_bullets=["- claims: 2 approved"],
        )

        assert result.source_id == source_id
        assert result.wiki_pages_created == ["entity_sample"]
        assert result.wiki_pages_updated == []
        assert result.log_entry_appended

        # Working-tree file exists; no commit.
        assert (repo_path / "wiki" / "entity_sample.md").exists()
        # Log entry landed.
        log_content = (repo_path / "wiki" / "log.md").read_text()
        assert f"ingest | {source_id}" in log_content
        assert "gate 3" in log_content

    def test_no_wiki_updates_response(self, rossum_with_paper_and_claims: tuple[Path, str, list[str]]) -> None:
        repo_path, source_id, claim_ids = rossum_with_paper_and_claims
        repo = NotesRepo(repo_path)
        config = Config()
        backend = _fake_backend(_NO_WIKI_UPDATES)

        result = run_gate3(
            source_id,
            repo=repo,
            config=config,
            backend=backend,
            approved_claim_ids=claim_ids,
            gate1_sub_bullets=[],
            gate2_sub_bullets=[],
        )
        assert result.wiki_pages_created == []
        assert result.wiki_pages_updated == []
        assert result.log_entry_appended
        log_content = (repo_path / "wiki" / "log.md").read_text()
        assert "LLM determined claims did not warrant" in log_content

    def test_empty_claims_skips_llm(self, rossum_with_paper_and_claims: tuple[Path, str, list[str]]) -> None:
        repo_path, source_id, _claim_ids = rossum_with_paper_and_claims
        repo = NotesRepo(repo_path)
        config = Config()
        backend = _fake_backend("(should not be called)")

        result = run_gate3(
            source_id,
            repo=repo,
            config=config,
            backend=backend,
            approved_claim_ids=[],  # no claims
            gate1_sub_bullets=[],
            gate2_sub_bullets=[],
        )
        assert result.wiki_pages_created == []
        assert result.log_entry_appended
        backend.complete.assert_not_called()
        log_content = (repo_path / "wiki" / "log.md").read_text()
        assert "no updates (gate 2 produced no claims)" in log_content

    def test_updates_existing_page(self, rossum_with_paper_and_claims: tuple[Path, str, list[str]]) -> None:
        repo_path, source_id, claim_ids = rossum_with_paper_and_claims
        # Pre-create a wiki page so the LLM's update targets it.
        existing_body = _sample_page_body("entity_sample", claim_count=0)
        (repo_path / "wiki" / "entity_sample.md").write_text(existing_body)
        subprocess.run([_GIT, "add", "wiki/entity_sample.md"], cwd=repo_path, check=True)  # noqa: S603
        subprocess.run(  # noqa: S603
            [_GIT, "commit", "-q", "-m", "seed page"], cwd=repo_path, check=True
        )

        repo = NotesRepo(repo_path)
        config = Config()
        new_body = _sample_page_body("entity_sample", claim_count=1)
        backend = _fake_backend(_wiki_update_block("entity_sample", new_body))

        result = run_gate3(
            source_id,
            repo=repo,
            config=config,
            backend=backend,
            approved_claim_ids=claim_ids,
            gate1_sub_bullets=[],
            gate2_sub_bullets=[],
        )
        assert result.wiki_pages_created == []
        assert result.wiki_pages_updated == ["entity_sample"]

    def test_dirty_wiki_aborts(self, rossum_with_paper_and_claims: tuple[Path, str, list[str]]) -> None:
        repo_path, source_id, claim_ids = rossum_with_paper_and_claims
        (repo_path / "wiki" / "dirty.md").write_text("---\nid: dirty\n---\n")
        repo = NotesRepo(repo_path)
        config = Config()
        backend = _fake_backend("(should not matter)")

        with pytest.raises(GleanRepoError, match=r"wiki/ has uncommitted"):
            run_gate3(
                source_id,
                repo=repo,
                config=config,
                backend=backend,
                approved_claim_ids=claim_ids,
                gate1_sub_bullets=[],
                gate2_sub_bullets=[],
            )

    def test_invalid_llm_response_raises(self, rossum_with_paper_and_claims: tuple[Path, str, list[str]]) -> None:
        repo_path, source_id, claim_ids = rossum_with_paper_and_claims
        repo = NotesRepo(repo_path)
        config = Config()
        backend = _fake_backend("just some prose, no blocks at all")

        with pytest.raises(GleanLLMError, match=r"no parseable wiki updates"):
            run_gate3(
                source_id,
                repo=repo,
                config=config,
                backend=backend,
                approved_claim_ids=claim_ids,
                gate1_sub_bullets=[],
                gate2_sub_bullets=[],
            )

    def test_page_id_mismatch_raises(self, rossum_with_paper_and_claims: tuple[Path, str, list[str]]) -> None:
        """If the delimiter ID and frontmatter ID disagree, reject."""
        repo_path, source_id, claim_ids = rossum_with_paper_and_claims
        repo = NotesRepo(repo_path)
        config = Config()
        # Delimiter says 'entity_one' but the body has id: entity_two.
        body = _sample_page_body("entity_two")
        backend = _fake_backend(_wiki_update_block("entity_one", body))

        with pytest.raises(GleanLLMError, match=r"disagrees with frontmatter"):
            run_gate3(
                source_id,
                repo=repo,
                config=config,
                backend=backend,
                approved_claim_ids=claim_ids,
                gate1_sub_bullets=[],
                gate2_sub_bullets=[],
            )

    def test_no_commit_during_gate3(self, rossum_with_paper_and_claims: tuple[Path, str, list[str]]) -> None:
        """Per AGENTS.md §5: gate 3 must never commit."""
        repo_path, source_id, claim_ids = rossum_with_paper_and_claims
        # Record HEAD before.
        result_before = subprocess.run(  # noqa: S603
            [_GIT, "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        head_before = result_before.stdout.strip()

        repo = NotesRepo(repo_path)
        config = Config()
        body = _sample_page_body("entity_sample")
        backend = _fake_backend(_wiki_update_block("entity_sample", body))

        run_gate3(
            source_id,
            repo=repo,
            config=config,
            backend=backend,
            approved_claim_ids=claim_ids,
            gate1_sub_bullets=[],
            gate2_sub_bullets=[],
        )

        # HEAD must not have advanced.
        result_after = subprocess.run(  # noqa: S603
            [_GIT, "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result_after.stdout.strip() == head_before

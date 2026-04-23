"""Tests for `glean.query` (M5).

Ollama is mocked. Two mocked calls per query: fast-tier page selection and
deep-tier synthesis.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from glean.cli import app
from glean.errors import GleanLLMError, GleanRepoError
from glean.query import (
    _NO_RELEVANT_PAGES,
    _build_page_catalog,
    _format_claim_line,
    _render_catalog,
    _select_pages,
    run_cli_query,
    run_query,
)
from glean.repo import NotesRepo
from glean.schema import Claim

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rossum_with_wiki(tmp_path: Path) -> Path:
    """A rossum repo with two wiki pages + three claims. No git init —
    query reads files directly and doesn't care about commit state."""
    repo = tmp_path / "rossum"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("# AGENTS v0.2\n")
    for layer in ("sources", "notebook", "claims", "wiki"):
        (repo / layer).mkdir()
        (repo / layer / ".gitkeep").write_text("")
    (repo / "wiki" / "log.md").write_text("# Log\n")
    (repo / "wiki" / "index.md").write_text("---\nkind: index\nupdated: 2026-04-23\n---\n\n# Index\n")

    source_id = "paper_test_2023_x"
    (repo / "sources" / source_id).mkdir()
    (repo / "sources" / source_id / "source.yaml").write_text(
        f"id: {source_id}\ntype: paper\ntitle: x\nauthors: ['A, B']\n"
        "year: 2023\nvenue: v\ndoi: 10.1/x\nurl: null\nadded: 2026-04-23\n"
        "confidence: high\ntags: []\nbibtex_key: ab2023\narxiv_id: null\n"
    )

    # Three claims.
    for slug, paraphrase in [
        ("alpha", "Alpha is the first letter."),
        ("beta", "Beta is the second letter."),
        ("gamma", "Gamma is the third letter."),
    ]:
        cid = f"claim_2023_test_x_{slug}"
        (repo / "claims" / f"{cid}.md").write_text(
            f"---\n"
            f"id: {cid}\n"
            f"source: {source_id}\n"
            f"source_span: '§1'\nquote: q\nclaim: {paraphrase}\n"
            f"confidence: author_assertion\nextracted: 2026-04-23\nstatus: active\n"
            f"---\n\nbody\n"
        )

    # Two wiki pages.
    (repo / "wiki" / "entity_alphabet.md").write_text(
        "---\n"
        "id: entity_alphabet\nkind: entity\ntitle: Alphabet\n"
        "created: 2026-04-23\nupdated: 2026-04-23\nclaim_count: 2\ntags: [letters]\n"
        "---\n\n"
        "# Alphabet\n\n"
        "The alphabet starts with alpha [[claim_2023_test_x_alpha]] "
        "and continues with beta [[claim_2023_test_x_beta]].\n"
    )
    (repo / "wiki" / "concept_ordering.md").write_text(
        "---\n"
        "id: concept_ordering\nkind: concept\ntitle: Ordering\n"
        "created: 2026-04-23\nupdated: 2026-04-23\nclaim_count: 1\ntags: [order]\n"
        "---\n\n"
        "# Ordering\n\n"
        "Gamma comes third [[claim_2023_test_x_gamma]].\n"
    )

    return repo


def _fake_backend(*responses: str) -> MagicMock:
    """Mock backend.complete() that returns each response in sequence."""
    logs = [MagicMock(to_log_bullets=lambda: ["- backend: fake"]) for _ in responses]
    backend = MagicMock()
    backend.complete = MagicMock(side_effect=[(r, log) for r, log in zip(responses, logs, strict=True)])
    return backend


# =============================================================================
# _build_page_catalog and _render_catalog
# =============================================================================


class TestPageCatalog:
    def test_build_catalog(self, rossum_with_wiki: Path) -> None:
        repo = NotesRepo(rossum_with_wiki)
        catalog = _build_page_catalog(repo)
        page_ids = {pid for pid, _ in catalog}
        assert page_ids == {"entity_alphabet", "concept_ordering"}

    def test_render_catalog_format(self, rossum_with_wiki: Path) -> None:
        repo = NotesRepo(rossum_with_wiki)
        catalog = _build_page_catalog(repo)
        text = _render_catalog(catalog)
        # Each line: id | kind | title | tags
        lines = text.splitlines()
        assert len(lines) == 2
        for line in lines:
            parts = line.split(" | ")
            assert len(parts) == 4


# =============================================================================
# _select_pages
# =============================================================================


class TestSelectPages:
    def test_parses_line_per_id(self, rossum_with_wiki: Path) -> None:
        repo = NotesRepo(rossum_with_wiki)
        catalog = _build_page_catalog(repo)
        backend = _fake_backend("entity_alphabet\nconcept_ordering\n")
        selected, _log = _select_pages(question="what letters are there?", page_catalog=catalog, backend=backend)
        assert selected == ["entity_alphabet", "concept_ordering"]

    def test_no_relevant_pages_returns_empty(self, rossum_with_wiki: Path) -> None:
        repo = NotesRepo(rossum_with_wiki)
        catalog = _build_page_catalog(repo)
        backend = _fake_backend(_NO_RELEVANT_PAGES)
        selected, _log = _select_pages(
            question="what is quantum chromodynamics?",
            page_catalog=catalog,
            backend=backend,
        )
        assert selected == []

    def test_ignores_bogus_ids(self, rossum_with_wiki: Path) -> None:
        """LLM invents a page ID; we filter to catalog-valid IDs."""
        repo = NotesRepo(rossum_with_wiki)
        catalog = _build_page_catalog(repo)
        backend = _fake_backend("entity_alphabet\nentity_ghost_nonexistent\n")
        selected, _log = _select_pages(question="letters?", page_catalog=catalog, backend=backend)
        assert selected == ["entity_alphabet"]

    def test_tolerates_list_prefixes(self, rossum_with_wiki: Path) -> None:
        """LLM adds bullets despite instructions; we strip them."""
        repo = NotesRepo(rossum_with_wiki)
        catalog = _build_page_catalog(repo)
        backend = _fake_backend("- entity_alphabet\n* concept_ordering\n")
        selected, _log = _select_pages(question="letters?", page_catalog=catalog, backend=backend)
        assert selected == ["entity_alphabet", "concept_ordering"]

    def test_dedupes(self, rossum_with_wiki: Path) -> None:
        """LLM repeats an ID; we emit it once."""
        repo = NotesRepo(rossum_with_wiki)
        catalog = _build_page_catalog(repo)
        backend = _fake_backend("entity_alphabet\nentity_alphabet\nentity_alphabet\n")
        selected, _log = _select_pages(question="letters?", page_catalog=catalog, backend=backend)
        assert selected == ["entity_alphabet"]


# =============================================================================
# _format_claim_line
# =============================================================================


class TestFormatClaimLine:
    def test_short_paraphrase(self) -> None:
        claim = Claim.model_validate(
            {
                "id": "claim_2023_test_x_alpha",
                "source": "paper_test_2023_x",
                "source_span": "§1",
                "quote": "q",
                "claim": "Alpha is the first letter.",
                "confidence": "author_assertion",
                "extracted": "2026-04-23",
                "status": "active",
            }
        )
        line = _format_claim_line("claim_2023_test_x_alpha", claim)
        assert "claim_2023_test_x_alpha" in line
        assert "author_assertion" in line
        assert "Alpha is the first letter" in line

    def test_long_paraphrase_truncated(self) -> None:
        long_prose = " ".join(["word"] * 100)
        claim = Claim.model_validate(
            {
                "id": "claim_2023_test_x_long",
                "source": "paper_test_2023_x",
                "source_span": "§1",
                "quote": "q",
                "claim": long_prose,
                "confidence": "measured",
                "extracted": "2026-04-23",
                "status": "active",
            }
        )
        line = _format_claim_line("claim_2023_test_x_long", claim)
        assert line.endswith("...")
        assert len(line) < 300


# =============================================================================
# run_query end-to-end (both LLM calls mocked)
# =============================================================================


class TestRunQuery:
    def test_happy_path(self, rossum_with_wiki: Path) -> None:
        repo = NotesRepo(rossum_with_wiki)
        # First call: page selection. Second call: synthesis.
        backend = _fake_backend(
            "entity_alphabet\n",
            "Alpha is the first letter [[claim_2023_test_x_alpha]].",
        )
        result = run_query("what is alpha?", repo=repo, backend=backend)
        assert result.selected_page_ids == ["entity_alphabet"]
        assert "Alpha is the first letter" in result.answer
        assert "[[claim_2023_test_x_alpha]]" in result.answer
        # Two LLM calls happened.
        assert backend.complete.call_count == 2

    def test_no_pages_selected_short_circuits(self, rossum_with_wiki: Path) -> None:
        """If selection returns nothing, no synthesis call is made."""
        repo = NotesRepo(rossum_with_wiki)
        backend = _fake_backend(_NO_RELEVANT_PAGES)
        result = run_query("what is unrelated?", repo=repo, backend=backend)
        assert result.selected_page_ids == []
        assert "does not contain any pages relevant" in result.answer
        # Only one LLM call — synthesis was skipped.
        assert backend.complete.call_count == 1

    def test_empty_question_raises(self, rossum_with_wiki: Path) -> None:
        repo = NotesRepo(rossum_with_wiki)
        backend = _fake_backend()
        with pytest.raises(GleanRepoError, match=r"empty question"):
            run_query("   ", repo=repo, backend=backend)

    def test_empty_wiki_raises(self, tmp_path: Path) -> None:
        # Build a rossum repo with no wiki pages.
        repo_path = tmp_path / "rossum"
        repo_path.mkdir()
        (repo_path / "AGENTS.md").write_text("# AGENTS\n")
        for layer in ("sources", "notebook", "claims", "wiki"):
            (repo_path / layer).mkdir()
            (repo_path / layer / ".gitkeep").write_text("")
        (repo_path / "wiki" / "log.md").write_text("")
        repo = NotesRepo(repo_path)
        backend = _fake_backend()
        with pytest.raises(GleanRepoError, match=r"no wiki pages"):
            run_query("anything", repo=repo, backend=backend)

    def test_empty_synthesis_response_raises(self, rossum_with_wiki: Path) -> None:
        repo = NotesRepo(rossum_with_wiki)
        backend = _fake_backend("entity_alphabet\n", "   \n")
        with pytest.raises(GleanLLMError, match=r"empty response"):
            run_query("what is alpha?", repo=repo, backend=backend)


# =============================================================================
# run_cli_query (exit code boundary)
# =============================================================================


class TestRunCliQuery:
    def test_success_returns_0_and_prints(self, rossum_with_wiki: Path, capsys: pytest.CaptureFixture[str]) -> None:
        repo = NotesRepo(rossum_with_wiki)
        backend = _fake_backend(
            "entity_alphabet\n",
            "Alpha [[claim_2023_test_x_alpha]].",
        )
        code = run_cli_query("what is alpha?", repo=repo, backend=backend)
        assert code == 0
        captured = capsys.readouterr()
        assert "Alpha" in captured.out

    def test_llm_error_returns_2(self, rossum_with_wiki: Path, capsys: pytest.CaptureFixture[str]) -> None:
        repo = NotesRepo(rossum_with_wiki)
        backend = _fake_backend("entity_alphabet\n", "")
        code = run_cli_query("what is alpha?", repo=repo, backend=backend)
        assert code == 2
        captured = capsys.readouterr()
        assert "LLM error" in captured.err

    def test_repo_error_returns_1(self, rossum_with_wiki: Path, capsys: pytest.CaptureFixture[str]) -> None:
        repo = NotesRepo(rossum_with_wiki)
        backend = _fake_backend()
        code = run_cli_query("  ", repo=repo, backend=backend)
        assert code == 1
        captured = capsys.readouterr()
        assert "Repo error" in captured.err


# =============================================================================
# CLI smoke
# =============================================================================


class TestCliQuery:
    def test_query_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0
        assert "--cloud" in result.output
        assert "--repo" in result.output

    def test_query_against_rossum(self, rossum_with_wiki: Path) -> None:
        """Full CLI invocation with both LLM calls mocked via get_backend."""
        backend = _fake_backend(
            "entity_alphabet\n",
            "Alpha [[claim_2023_test_x_alpha]].",
        )
        runner = CliRunner()
        # cli.query imports get_backend lazily from glean.llm; patch that module.
        with patch("glean.llm.get_backend", return_value=backend):
            result = runner.invoke(app, ["query", "what is alpha?", "--repo", str(rossum_with_wiki)])
        assert result.exit_code == 0
        assert "Alpha" in result.output

    def test_query_invalid_repo_exits_1(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["query", "q", "--repo", str(tmp_path / "not_a_repo")])
        assert result.exit_code == 1

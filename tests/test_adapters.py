"""Tests for `glean.adapters` — source ingesters (M3a).

Covers:
    - sniff_adapter dispatch for each handled input shape
    - PaperIngester (PDF metadata extraction, Crossref lookup, PDF text extraction) — mocked
    - SimulationIngester (directory sniffing, output_summary templating)
    - NotebookIngester (existing-file parsing)
    - WebArticleIngester stub raises with clear message
    - adapter_for and stage_source_directory helpers

Network calls and PDF extraction are fully mocked — no real Crossref, no real marker/pymupdf runs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from glean.adapters import (
    DraftSource,
    IngestInput,
    NotebookIngester,
    PaperIngester,
    SimulationIngester,
    SourceIngester,
    WebArticleIngester,
    _adapter_by_type,
    adapter_for,
    sniff_adapter,
    stage_source_directory,
)
from glean.enums import SourceType
from glean.errors import GleanRepoError

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fake_pdf(tmp_path: Path) -> Path:
    """A fake .pdf file; content is irrelevant because extractors are mocked."""
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake\n")
    return pdf


@pytest.fixture
def simulation_dir(tmp_path: Path) -> Path:
    """A directory shaped like a simulation run, with an input.ini."""
    run = tmp_path / "my_run"
    run.mkdir()
    (run / "input.ini").write_text("[numerics]\nscheme=foo\n")
    return run


@pytest.fixture
def notebook_file(tmp_path: Path) -> Path:
    """A file under a `notebook/` directory with valid frontmatter."""
    notebook_dir = tmp_path / "notebook"
    notebook_dir.mkdir()
    f = notebook_dir / "foo_thought.md"
    f.write_text(
        "---\nid: note_2026_04_23_foo_thought\ndate: 2026-04-23\ntopic: foo\nstatus: draft\ntags: []\n---\n\nbody\n"
    )
    return f


# =============================================================================
# sniff_adapter
# =============================================================================


class TestSniff:
    def test_pdf_path_routes_to_paper(self, fake_pdf: Path) -> None:
        assert sniff_adapter(str(fake_pdf)) is PaperIngester

    def test_directory_with_input_ini_routes_to_simulation(self, simulation_dir: Path) -> None:
        assert sniff_adapter(str(simulation_dir)) is SimulationIngester

    def test_notebook_file_routes_to_notebook(self, notebook_file: Path) -> None:
        assert sniff_adapter(str(notebook_file)) is NotebookIngester

    def test_http_url_routes_to_webarticle(self) -> None:
        assert sniff_adapter("https://example.com/post") is WebArticleIngester

    def test_unrecognized_raises(self, tmp_path: Path) -> None:
        (tmp_path / "random.txt").write_text("not a source")
        with pytest.raises(GleanRepoError, match=r"could not detect"):
            sniff_adapter(str(tmp_path / "random.txt"))

    def test_md_outside_notebook_not_recognized(self, tmp_path: Path) -> None:
        """A .md file not under notebook/ should not sniff as notebook."""
        (tmp_path / "random.md").write_text("---\nid: x\n---\n")
        with pytest.raises(GleanRepoError):
            sniff_adapter(str(tmp_path / "random.md"))


# =============================================================================
# PaperIngester
# =============================================================================


class TestPaperIngester:
    def test_can_handle_pdf(self, fake_pdf: Path) -> None:
        assert PaperIngester.can_handle(str(fake_pdf))

    def test_does_not_handle_non_pdf(self, tmp_path: Path) -> None:
        non_pdf = tmp_path / "foo.txt"
        non_pdf.write_text("x")
        assert not PaperIngester.can_handle(str(non_pdf))

    def test_does_not_handle_missing_file(self, tmp_path: Path) -> None:
        assert not PaperIngester.can_handle(str(tmp_path / "ghost.pdf"))

    def test_prepare_happy_path(self, fake_pdf: Path) -> None:
        """PaperIngester.prepare with mocked metadata + Crossref + extractor."""
        ingester = PaperIngester()

        with (
            patch(
                "glean.adapters._extract_pdf_metadata",
                return_value={
                    "title": "Fake Paper",
                    "authors": ["Zaghi, Stefano"],
                    "doi": "10.1/fake",
                    "arxiv_id": None,
                },
            ),
            patch(
                "glean.adapters._crossref_lookup",
                return_value={
                    "title": "Fake Paper (Crossref refined)",
                    "authors": ["Zaghi, Stefano"],
                    "year": 2026,
                    "venue": "J. Fake",
                    "url": "https://doi.org/10.1/fake",
                },
            ),
            patch("glean.adapters._extract_pdf_text", return_value="# Fake\n\nbody"),
        ):
            draft = ingester.prepare(IngestInput(input_spec=str(fake_pdf)))

        assert draft.source_type == SourceType.PAPER
        assert draft.proposed_id.startswith("paper_zaghi_2026_")
        assert draft.draft_yaml["title"] == "Fake Paper (Crossref refined)"
        assert draft.draft_yaml["year"] == 2026
        assert draft.draft_yaml["venue"] == "J. Fake"
        assert draft.draft_yaml["doi"] == "10.1/fake"
        assert draft.artifacts == {"paper.md": "# Fake\n\nbody"}
        assert "paper.pdf" in draft.files_to_copy
        assert draft.files_to_copy["paper.pdf"] == fake_pdf

    def test_prepare_offline_skips_crossref(self, fake_pdf: Path) -> None:
        ingester = PaperIngester()
        with (
            patch(
                "glean.adapters._extract_pdf_metadata",
                return_value={"title": "Offline Paper", "doi": "10.1/x"},
            ),
            patch("glean.adapters._crossref_lookup") as mock_crossref,
            patch("glean.adapters._extract_pdf_text", return_value="body"),
        ):
            ingester.prepare(IngestInput(input_spec=str(fake_pdf), offline=True))

        mock_crossref.assert_not_called()

    def test_prepare_missing_file_raises(self, tmp_path: Path) -> None:
        ingester = PaperIngester()
        with pytest.raises(GleanRepoError, match=r"PDF not found"):
            ingester.prepare(IngestInput(input_spec=str(tmp_path / "ghost.pdf")))

    def test_prepare_handles_empty_crossref(self, fake_pdf: Path) -> None:
        """Crossref returns {} on failure; we should still produce a DraftSource."""
        ingester = PaperIngester()
        with (
            patch(
                "glean.adapters._extract_pdf_metadata",
                return_value={"title": "Only PDF", "doi": "10.1/y"},
            ),
            patch("glean.adapters._crossref_lookup", return_value={}),
            patch("glean.adapters._extract_pdf_text", return_value="x"),
        ):
            draft = ingester.prepare(IngestInput(input_spec=str(fake_pdf)))
        assert draft.draft_yaml["title"] == "Only PDF"


class TestCrossrefLookup:
    """Tests for _crossref_lookup against mocked httpx."""

    def test_parses_well_formed_response(self) -> None:
        from glean.adapters import _crossref_lookup

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "title": ["A Paper Title"],
                "author": [{"family": "Zaghi", "given": "Stefano"}],
                "issued": {"date-parts": [[2023]]},
                "container-title": ["Computers & Fluids"],
                "URL": "https://doi.org/10.1/x",
            }
        }
        with patch("glean.adapters.httpx.get", return_value=mock_response):
            result = _crossref_lookup("10.1/x")
        assert result["title"] == "A Paper Title"
        assert result["authors"] == ["Zaghi, Stefano"]
        assert result["year"] == 2023
        assert result["venue"] == "Computers & Fluids"

    def test_returns_empty_on_404(self) -> None:
        from glean.adapters import _crossref_lookup

        mock_response = MagicMock()
        mock_response.status_code = 404
        with patch("glean.adapters.httpx.get", return_value=mock_response):
            assert _crossref_lookup("10.1/missing") == {}

    def test_returns_empty_on_network_error(self) -> None:
        from glean.adapters import _crossref_lookup

        with patch("glean.adapters.httpx.get", side_effect=httpx.ConnectError("no")):
            assert _crossref_lookup("10.1/x") == {}


# =============================================================================
# SimulationIngester
# =============================================================================


class TestSimulationIngester:
    def test_can_handle_dir_with_input_ini(self, simulation_dir: Path) -> None:
        assert SimulationIngester.can_handle(str(simulation_dir))

    def test_can_handle_dir_with_nml(self, tmp_path: Path) -> None:
        d = tmp_path / "run"
        d.mkdir()
        (d / "config.nml").write_text("")
        assert SimulationIngester.can_handle(str(d))

    def test_does_not_handle_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert not SimulationIngester.can_handle(str(empty))

    def test_does_not_handle_file(self, fake_pdf: Path) -> None:
        assert not SimulationIngester.can_handle(str(fake_pdf))

    def test_prepare_builds_draft(self, simulation_dir: Path) -> None:
        ingester = SimulationIngester()
        draft = ingester.prepare(IngestInput(input_spec=str(simulation_dir)))

        assert draft.source_type == SourceType.SIMULATION
        assert draft.proposed_id.startswith("sim_")
        assert draft.draft_yaml["input_files"] == ["input.ini"]
        assert draft.draft_yaml["output_summary"] == "output_summary.md"
        assert "output_summary.md" in draft.artifacts
        assert "input.ini" in draft.files_to_copy
        # Template should contain FILL markers
        assert "<FILL" in draft.artifacts["output_summary.md"]

    def test_prepare_missing_dir_raises(self, tmp_path: Path) -> None:
        ingester = SimulationIngester()
        with pytest.raises(GleanRepoError, match=r"not found"):
            ingester.prepare(IngestInput(input_spec=str(tmp_path / "ghost")))

    def test_prepare_rejects_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        ingester = SimulationIngester()
        with pytest.raises(GleanRepoError, match=r"no input files"):
            ingester.prepare(IngestInput(input_spec=str(empty)))


# =============================================================================
# NotebookIngester
# =============================================================================


class TestNotebookIngester:
    def test_can_handle_notebook_file(self, notebook_file: Path) -> None:
        assert NotebookIngester.can_handle(str(notebook_file))

    def test_does_not_handle_md_outside_notebook(self, tmp_path: Path) -> None:
        f = tmp_path / "stray.md"
        f.write_text("---\nid: x\n---\n")
        assert not NotebookIngester.can_handle(str(f))

    def test_does_not_handle_pdf(self, fake_pdf: Path) -> None:
        assert not NotebookIngester.can_handle(str(fake_pdf))

    def test_prepare_reads_frontmatter(self, notebook_file: Path) -> None:
        ingester = NotebookIngester()
        draft = ingester.prepare(IngestInput(input_spec=str(notebook_file)))
        assert draft.source_type == SourceType.NOTEBOOK
        assert draft.proposed_id == "note_2026_04_23_foo_thought"
        assert draft.draft_yaml["type"] == "notebook"
        assert draft.artifacts == {}
        assert draft.files_to_copy == {}

    def test_prepare_missing_id_raises(self, tmp_path: Path) -> None:
        nd = tmp_path / "notebook"
        nd.mkdir()
        bad = nd / "bad.md"
        bad.write_text("---\ntopic: x\ndate: 2026-04-23\n---\n\nbody\n")
        ingester = NotebookIngester()
        with pytest.raises(GleanRepoError, match=r"missing `id:`"):
            ingester.prepare(IngestInput(input_spec=str(bad)))


# =============================================================================
# WebArticleIngester stub
# =============================================================================


class TestWebArticleStub:
    def test_can_handle_urls(self) -> None:
        assert WebArticleIngester.can_handle("https://example.com")
        assert WebArticleIngester.can_handle("http://example.com")
        assert not WebArticleIngester.can_handle("ftp://example.com")
        assert not WebArticleIngester.can_handle("/local/path")

    def test_prepare_raises_not_implemented(self) -> None:
        ingester = WebArticleIngester()
        with pytest.raises(NotImplementedError, match=r"not implemented at v0.1"):
            ingester.prepare(IngestInput(input_spec="https://example.com"))


# =============================================================================
# adapter_for + _adapter_by_type
# =============================================================================


class TestAdapterFor:
    def test_respects_type_override(self, fake_pdf: Path) -> None:
        """Even if sniffing would pick PaperIngester, a --type override wins."""
        # Use a PDF file but force type=simulation — this isn't realistic but
        # exercises the override path.
        inp = IngestInput(input_spec=str(fake_pdf), type_override=SourceType.NOTEBOOK)
        adapter = adapter_for(inp)
        assert isinstance(adapter, NotebookIngester)

    def test_sniffs_when_no_override(self, fake_pdf: Path) -> None:
        adapter = adapter_for(IngestInput(input_spec=str(fake_pdf)))
        assert isinstance(adapter, PaperIngester)

    def test_by_type_rejects_unhandled(self) -> None:
        with pytest.raises(GleanRepoError, match=r"no adapter at v0.1"):
            _adapter_by_type(SourceType.DATASET)


# =============================================================================
# stage_source_directory
# =============================================================================


class TestStageSourceDirectory:
    def test_writes_artifacts(self, tmp_path: Path) -> None:
        draft = DraftSource(
            source_type=SourceType.PAPER,
            proposed_id="paper_x_2026_y",
            draft_yaml={},
            artifacts={"paper.md": "content"},
            files_to_copy={},
        )
        dest = tmp_path / "paper_x_2026_y"
        stage_source_directory(draft, dest)
        assert (dest / "paper.md").read_text() == "content"

    def test_copies_files(self, tmp_path: Path) -> None:
        source_pdf = tmp_path / "orig.pdf"
        source_pdf.write_bytes(b"PDF content")
        draft = DraftSource(
            source_type=SourceType.PAPER,
            proposed_id="paper_x_2026_y",
            draft_yaml={},
            artifacts={},
            files_to_copy={"paper.pdf": source_pdf},
        )
        dest = tmp_path / "paper_x_2026_y"
        stage_source_directory(draft, dest)
        assert (dest / "paper.pdf").read_bytes() == b"PDF content"

    def test_noop_for_empty_draft(self, tmp_path: Path) -> None:
        """NotebookIngester returns empty artifacts and files; staging is a no-op."""
        draft = DraftSource(
            source_type=SourceType.NOTEBOOK,
            proposed_id="note_x",
            draft_yaml={},
            artifacts={},
            files_to_copy={},
        )
        dest = tmp_path / "note_x"
        stage_source_directory(draft, dest)
        # dest is created (mkdir) but empty.
        assert dest.is_dir()
        assert list(dest.iterdir()) == []


# =============================================================================
# Protocol conformance
# =============================================================================


class TestProtocolConformance:
    @pytest.mark.parametrize(
        "cls",
        [PaperIngester, SimulationIngester, NotebookIngester, WebArticleIngester],
    )
    def test_declares_source_type(self, cls: type[SourceIngester]) -> None:
        assert hasattr(cls, "source_type")
        assert cls.source_type in SourceType

    @pytest.mark.parametrize(
        "cls",
        [PaperIngester, SimulationIngester, NotebookIngester, WebArticleIngester],
    )
    def test_can_handle_is_class_method(self, cls: type[SourceIngester]) -> None:
        # Must be callable without an instance.
        assert callable(cls.can_handle)

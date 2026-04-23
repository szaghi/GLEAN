"""Source adapters (M3a).

One `SourceIngester` subclass per source type, responsible for the type-specific
gate-1 work: reading the input artifact, extracting metadata, producing the
citable substrate, and assembling a draft `source.yaml`.

All adapters share a common interface (`SourceIngester` ABC). Gate-2 and
gate-3 logic is uniform across types and lives in separate modules
(`ingest_gate2.py`, `ingest_gate3.py`), called by the top-level ingest command.

Scope at M3a per D24 decision:
    - PaperIngester: functional (marker/pymupdf4llm, Crossref+PDF-meta fallback)
    - SimulationIngester: functional (input-deck copy, output_summary template)
    - NotebookIngester: functional (parse existing entry, prepare for critique)
    - WebArticleIngester: stub raising NotImplementedError with schema pointer
    - Other types: not implemented; ingest dispatches to unhandled-type error
"""

from __future__ import annotations

import contextlib
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

from glean.enums import SourceConfidence, SourceType
from glean.errors import GleanRepoError
from glean.ids import slugify, source_id_for

# =============================================================================
# Common types
# =============================================================================


@dataclass
class IngestInput:
    """Everything we need to start an ingest: the input path/URL, optional
    type override from `--type`, optional `--no-network` flag, optional
    `--pdf-extractor` choice.
    """

    input_spec: str  # path or URL from the CLI
    type_override: SourceType | None = None
    offline: bool = False
    pdf_extractor: str = "marker"  # "marker" | "pymupdf"


@dataclass
class DraftSource:
    """What an adapter produces before gate-1 `$EDITOR` confirmation.

    `draft_yaml` is a dict (not a Pydantic model) because the user may edit
    it — validation happens post-edit, not pre-edit. `artifacts` maps
    relative filenames to content (for adapters that produce substrate:
    paper.md, output_summary.md, etc.) — these are staged in the source
    directory after confirmation.
    """

    source_type: SourceType
    proposed_id: str
    draft_yaml: dict[str, object]
    artifacts: dict[str, str | bytes]  # relative filename -> content
    # Extra files to copy as-is (e.g. the PDF, input.ini) — keyed by
    # relative destination path, value is source path.
    files_to_copy: dict[str, Path]


# =============================================================================
# Adapter ABC
# =============================================================================


class SourceIngester(ABC):
    """Common interface for source-type adapters.

    Subclass responsibilities:
        1. `can_handle(input_spec)`: return True if this adapter should handle
           the given path/URL. Used for sniffing per D10.
        2. `prepare(input_spec, options)`: perform type-specific gate-1 work.
           Returns a DraftSource that the gate-1 orchestrator presents to the
           user for $EDITOR confirmation.
    """

    source_type: SourceType  # concrete subclasses override

    @classmethod
    @abstractmethod
    def can_handle(cls, input_spec: str) -> bool:
        """Return True if `input_spec` looks like the kind of input this adapter handles."""

    @abstractmethod
    def prepare(self, inp: IngestInput) -> DraftSource:
        """Extract metadata and artifacts; return a DraftSource for gate-1 confirmation."""


# =============================================================================
# Sniffing
# =============================================================================


def sniff_adapter(input_spec: str) -> type[SourceIngester]:
    """Return the adapter class that best handles `input_spec`.

    Order matters: more-specific checks first. Raises `GleanRepoError` if
    no adapter claims the input.
    """
    candidates: list[type[SourceIngester]] = [
        NotebookIngester,  # file already under notebook/ takes priority
        SimulationIngester,  # directory with input.ini
        PaperIngester,  # .pdf path
        WebArticleIngester,  # http:// or https://
    ]
    for cls in candidates:
        if cls.can_handle(input_spec):
            return cls
    raise GleanRepoError(
        f"could not detect source type for input: {input_spec!r}. "
        f"Pass --type explicitly (paper | simulation | notebook | web_article)."
    )


# =============================================================================
# PaperIngester
# =============================================================================


_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


class PaperIngester(SourceIngester):
    """Paper ingest: PDF → extracted markdown + metadata lookup."""

    source_type = SourceType.PAPER

    @classmethod
    def can_handle(cls, input_spec: str) -> bool:
        p = Path(input_spec).expanduser()
        return p.is_file() and p.suffix.lower() == ".pdf"

    def prepare(self, inp: IngestInput) -> DraftSource:
        pdf_path = Path(inp.input_spec).expanduser().resolve()
        if not pdf_path.is_file():
            raise GleanRepoError(f"PDF not found: {pdf_path}")

        pdf_metadata = _extract_pdf_metadata(pdf_path)
        doi = pdf_metadata.get("doi")
        crossref_metadata: dict[str, object] = {}
        if doi and not inp.offline:
            crossref_metadata = _crossref_lookup(doi)

        merged = {**pdf_metadata, **crossref_metadata}  # Crossref overrides PDF

        # Best-effort ID construction. User will confirm/change in $EDITOR.
        first_author_slug = "unknown"
        if merged.get("authors"):
            authors_list = merged["authors"]
            if isinstance(authors_list, list) and authors_list:
                first_author = authors_list[0]
                if isinstance(first_author, str):
                    # "Surname, Given" -> "surname"
                    first_author_slug = slugify(first_author.split(",")[0])
        year = merged.get("year") or date.today().year
        title = merged.get("title", "untitled")
        title_slug = slugify(str(title))[:40] if title else "untitled"

        try:
            proposed_id = source_id_for(
                SourceType.PAPER,
                first_author=first_author_slug,
                year=int(year),
                slug=title_slug,
            )
        except ValueError:
            proposed_id = f"paper_{first_author_slug}_{year}_untitled"

        paper_md = _extract_pdf_text(pdf_path, extractor=inp.pdf_extractor)

        draft_yaml: dict[str, object] = {
            "id": proposed_id,
            "type": SourceType.PAPER.value,
            "title": merged.get("title", "<FILL: paper title>"),
            "authors": merged.get("authors", ["<FILL: Surname, Given>"]),
            "year": int(year),
            "venue": merged.get("venue", "<FILL: journal / conference>"),
            "doi": doi,
            "url": merged.get("url"),
            "added": date.today(),
            "confidence": SourceConfidence.HIGH.value,
            "tags": [],
            "bibtex_key": f"{first_author_slug}{year}",
            "arxiv_id": merged.get("arxiv_id"),
        }

        return DraftSource(
            source_type=SourceType.PAPER,
            proposed_id=proposed_id,
            draft_yaml=draft_yaml,
            artifacts={"paper.md": paper_md},
            files_to_copy={"paper.pdf": pdf_path},
        )


def _extract_pdf_metadata(pdf_path: Path) -> dict[str, object]:
    """Extract what we can from the PDF's own metadata dictionary.

    Returns whatever subset of (title, authors, doi, arxiv_id) we can pull.
    Never raises; missing fields are simply omitted from the dict.
    """
    out: dict[str, object] = {}
    try:
        import pypdf  # local import: optional dep under [pdf] extra
    except ImportError as e:
        raise GleanRepoError("pypdf not installed; paper ingest requires `pip install glean[pdf]`") from e

    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as e:
        raise GleanRepoError(f"failed to open PDF {pdf_path}: {e}") from e

    info = reader.metadata or {}
    if info.title:
        out["title"] = info.title
    if info.author:
        # PDF metadata authors are typically a single string; split best-effort.
        out["authors"] = [info.author]

    # DOI and arXiv ID often appear in the PDF's "Subject" or first page text.
    blob = f"{info.title or ''} {info.author or ''} {info.subject or ''}"
    # Add the first page's text for a better shot at catching DOI.
    if reader.pages:
        with contextlib.suppress(Exception):
            blob += " " + (reader.pages[0].extract_text() or "")

    doi_match = _DOI_RE.search(blob)
    if doi_match:
        out["doi"] = doi_match.group(0).rstrip(".").rstrip(",")

    arxiv_match = _ARXIV_ID_RE.search(blob)
    if arxiv_match:
        out["arxiv_id"] = arxiv_match.group(1)

    return out


def _crossref_lookup(doi: str) -> dict[str, object]:
    """Query Crossref for a DOI. Returns {} on any failure (network, 404, parse error).

    Never raises. Caller merges this over PDF-metadata defaults.
    """
    try:
        resp = httpx.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=5.0,
            headers={"User-Agent": "glean/0.1 (https://github.com/szaghi/GLEAN)"},
        )
        if resp.status_code != 200:
            return {}
        data = resp.json().get("message", {})
    except (httpx.HTTPError, ValueError):
        return {}

    out: dict[str, object] = {}
    if titles := data.get("title"):
        out["title"] = titles[0] if isinstance(titles, list) else titles

    if authors := data.get("author"):
        formatted = []
        for a in authors:
            family = a.get("family", "")
            given = a.get("given", "")
            if family:
                formatted.append(f"{family}, {given}".rstrip(", "))
        if formatted:
            out["authors"] = formatted

    issued = data.get("issued", {}).get("date-parts", [[None]])
    if issued and issued[0] and issued[0][0]:
        out["year"] = issued[0][0]

    container = data.get("container-title")
    if container:
        out["venue"] = container[0] if isinstance(container, list) else container

    if url := data.get("URL"):
        out["url"] = url

    return out


def _extract_pdf_text(pdf_path: Path, *, extractor: str) -> str:
    """Extract PDF text to markdown using the chosen extractor."""
    if extractor == "marker":
        return _extract_pdf_marker(pdf_path)
    if extractor == "pymupdf":
        return _extract_pdf_pymupdf(pdf_path)
    raise GleanRepoError(f"unknown PDF extractor: {extractor!r}. Use 'marker' or 'pymupdf'.")


def _extract_pdf_marker(pdf_path: Path) -> str:
    """Extract with marker. First run triggers ~2 GB model download."""
    try:
        from marker.converters.pdf import PdfConverter  # local import: heavy dep
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError as e:
        raise GleanRepoError(
            "marker not installed; paper ingest with --pdf-extractor=marker requires "
            "`pip install glean[pdf]` plus `pip install marker-pdf`"
        ) from e

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(pdf_path))
    text, _, _ = text_from_rendered(rendered)
    return text


def _extract_pdf_pymupdf(pdf_path: Path) -> str:
    """Extract with pymupdf4llm. Fast, but fails on complex layouts."""
    try:
        import pymupdf4llm  # local import: optional dep
    except ImportError as e:
        raise GleanRepoError(
            "pymupdf4llm not installed; paper ingest with --pdf-extractor=pymupdf requires `pip install glean[pdf]`"
        ) from e
    return pymupdf4llm.to_markdown(str(pdf_path))


# =============================================================================
# SimulationIngester
# =============================================================================


class SimulationIngester(SourceIngester):
    """Simulation ingest: directory with input files → source + output_summary skeleton."""

    source_type = SourceType.SIMULATION

    @classmethod
    def can_handle(cls, input_spec: str) -> bool:
        p = Path(input_spec).expanduser()
        if not p.is_dir():
            return False
        # Heuristic: directory containing a .ini / .nml / .yaml input file.
        return any(any(p.glob(pattern)) for pattern in ("input.ini", "*.nml", "input.yaml", "inputs.yaml"))

    def prepare(self, inp: IngestInput) -> DraftSource:
        run_dir = Path(inp.input_spec).expanduser().resolve()
        if not run_dir.is_dir():
            raise GleanRepoError(f"simulation directory not found: {run_dir}")

        input_files = sorted(
            [p.name for pattern in ("input.ini", "*.nml", "input.yaml", "inputs.yaml") for p in run_dir.glob(pattern)]
        )
        if not input_files:
            raise GleanRepoError(f"no input files found in {run_dir}")

        today = date.today()
        year_month = today.strftime("%Y_%m")
        slug = slugify(run_dir.name)[:40]

        try:
            proposed_id = source_id_for(SourceType.SIMULATION, year_month=year_month, slug=slug)
        except ValueError:
            proposed_id = f"sim_{year_month}_unnamed"

        from glean.llm import load_prompt  # local import: avoid circular dep

        template = load_prompt("output_summary_template", suffix="md").safe_substitute(
            source_title="<FILL>",
            solver_name="<FILL>",
            solver_commit="<FILL>+dirty",
            input_files=", ".join(input_files),
            hardware="<FILL: hardware description>",
            run_date=today.isoformat(),
        )

        draft_yaml: dict[str, object] = {
            "id": proposed_id,
            "type": SourceType.SIMULATION.value,
            "title": "<FILL: run title>",
            "authors": ["<FILL: Surname, Given>"],
            "year": today.year,
            "venue": f"Local run, {run_dir}",
            "added": today,
            "confidence": SourceConfidence.HIGH.value,
            "tags": [],
            "solver_repo_id": "<FILL: repo_org_name_commit>",
            "solver_commit": "<FILL: hex+dirty if applicable>",
            "input_files": input_files,
            "output_summary": "output_summary.md",
            "run_date": today,
            "hardware": "<FILL>",
        }

        files_to_copy: dict[str, Path] = {name: run_dir / name for name in input_files}

        return DraftSource(
            source_type=SourceType.SIMULATION,
            proposed_id=proposed_id,
            draft_yaml=draft_yaml,
            artifacts={"output_summary.md": template},
            files_to_copy=files_to_copy,
        )


# =============================================================================
# NotebookIngester
# =============================================================================


class NotebookIngester(SourceIngester):
    """Notebook ingest: parse an already-written entry under `notebook/`.

    Notebook authorship is out-of-scope for M3a; users write the entry
    themselves in their editor, then invoke `glean ingest notebook/<file>.md`.
    The adapter reads, validates frontmatter, and stages for gate 2.

    Note: notebook files DO NOT land in `sources/<id>/` per AGENTS.md v0.2 §2.1
    and §2.4. The gate-1 orchestrator treats this adapter's DraftSource as a
    pass-through: the file already lives in `notebook/`, no copy needed.
    """

    source_type = SourceType.NOTEBOOK

    @classmethod
    def can_handle(cls, input_spec: str) -> bool:
        p = Path(input_spec).expanduser()
        if not p.is_file() or p.suffix.lower() != ".md":
            return False
        # The file must already live under a `notebook/` directory to be recognized.
        return "notebook" in p.parts

    def prepare(self, inp: IngestInput) -> DraftSource:
        entry_path = Path(inp.input_spec).expanduser().resolve()
        if not entry_path.is_file():
            raise GleanRepoError(f"notebook entry not found: {entry_path}")

        from glean.repo import _split_frontmatter  # local import: avoid circular

        body = entry_path.read_text()
        frontmatter, _ = _split_frontmatter(body)

        # Frontmatter per §2.4 does not carry `type:`; inject for dispatch.
        frontmatter.setdefault("type", SourceType.NOTEBOOK.value)

        note_id = frontmatter.get("id")
        if not isinstance(note_id, str):
            raise GleanRepoError(f"notebook entry {entry_path} missing `id:` in frontmatter")

        return DraftSource(
            source_type=SourceType.NOTEBOOK,
            proposed_id=note_id,
            draft_yaml=frontmatter,
            # No artifacts to stage — the notebook file is the substrate and
            # already lives in notebook/.
            artifacts={},
            files_to_copy={},
        )


# =============================================================================
# WebArticleIngester (stub per D24)
# =============================================================================


class WebArticleIngester(SourceIngester):
    """Stub for web_article sources. v0.1 M3 does not implement URL ingest.

    Per D24 decision: defer functional WebArticle until Phase 2 surfaces
    concrete need. The stub raises `NotImplementedError` with a pointer at
    AGENTS.md §2.2 so the user knows the schema shape when they implement it
    manually or await the v0.2 feature.
    """

    source_type = SourceType.WEB_ARTICLE

    @classmethod
    def can_handle(cls, input_spec: str) -> bool:
        return input_spec.startswith(("http://", "https://"))

    def prepare(self, inp: IngestInput) -> DraftSource:
        raise NotImplementedError(
            "web_article ingest is not implemented at v0.1. Per AGENTS.md v0.2 §2.2, "
            "a web_article source requires: url, archived_at (date), and optionally "
            "archived_snapshot_path. File the source manually in sources/web_<id>/ "
            "or wait for the v0.2 web_article adapter. See PLAN.md v2 D24."
        )


# =============================================================================
# Public helper: wrap an input into an adapter instance
# =============================================================================


def adapter_for(inp: IngestInput) -> SourceIngester:
    """Pick the adapter for the given input.

    Honors `--type` override; otherwise sniffs via `sniff_adapter`.
    """
    if inp.type_override is not None:
        adapter_cls = _adapter_by_type(inp.type_override)
    else:
        adapter_cls = sniff_adapter(inp.input_spec)
    return adapter_cls()


def _adapter_by_type(source_type: SourceType) -> type[SourceIngester]:
    """Return the adapter class for a given SourceType.

    Raises GleanRepoError for unhandled types (Preprint, Dataset, Talk, etc.)
    per D24 — those are schema-valid but have no adapter at v0.1.
    """
    mapping: dict[SourceType, type[SourceIngester]] = {
        SourceType.PAPER: PaperIngester,
        SourceType.SIMULATION: SimulationIngester,
        SourceType.NOTEBOOK: NotebookIngester,
        SourceType.WEB_ARTICLE: WebArticleIngester,
    }
    cls = mapping.get(source_type)
    if cls is None:
        raise GleanRepoError(
            f"source type {source_type.value!r} is schema-valid but has no adapter at v0.1. "
            f"Per PLAN.md v2 D24, only paper/simulation/notebook/web_article are wired up."
        )
    return cls


# Copy utility used by the gate-1 orchestrator after confirmation.
def stage_source_directory(draft: DraftSource, dest: Path) -> None:
    """Write the DraftSource's artifacts and copy its files into `dest`.

    Called by gate-1 orchestrator after the user confirms the source.yaml.
    Creates `dest` if missing; writes each artifact as a file; copies each
    listed file verbatim. For notebook sources where both dicts are empty,
    this is a no-op.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for rel_path, content in draft.artifacts.items():
        target = dest / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            target.write_text(content)
        else:
            target.write_bytes(content)
    for rel_path, src_path in draft.files_to_copy.items():
        target = dest / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, target)

"""Pydantic models for the GLEAN schema.

This module is the executable specification of what a valid rossum repo looks
like on disk. It mirrors AGENTS.md v0.2 §2.1 (common source fields), §2.2
(source type taxonomy), and §2.5 (claim and notebook shapes).

Implementation strategy:
    - `SourceCommon` holds fields shared by all source types per §2.1.
    - One concrete subclass per source type (`PaperSource`, `SimulationSource`,
      etc.) that pins `type:` as a `Literal` and declares type-specific fields.
    - `SourceManifest` is a discriminated union on `type:`, the entry point
      for parsing a `source.yaml` file of unknown type.

M1b scope: Paper, Simulation, Repository — the three types Phase 1 exercised.
M1c-minimal scope: adds WebArticle (likely next Phase-2 type) and Notebook
(exercised in Phase 1 but with deliberate asymmetry from SourceCommon).
M1d scope: adds `Claim`, `WikiPage`, `WikiIndex`, `LogEntry` — the remaining
data types needed to fully model a rossum repo.
Remaining types (Preprint, Dataset, Talk, Book, Standard, PersonalComm) are
deferred until Phase 2 surfaces concrete need.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from glean.enums import (
    ClaimStatus,
    Confidence,
    NotebookStatus,
    SourceConfidence,
    SourceType,
    WikiKind,
)
from glean.ids import is_valid_claim_id, is_valid_source_id, is_valid_wiki_page_id

_HASH_RE = re.compile(r"[a-f0-9]{7,40}")


class SourceCommon(BaseModel):
    """Fields shared by every source per AGENTS.md v0.2 §2.1.

    Concrete source types inherit from this and add type-specific fields.
    Fields are declared in the order they appear in §2.1's `source.yaml`
    example, which matches on-disk ordering produced by the ingest flow.
    """

    model_config = ConfigDict(
        extra="forbid",  # reject unknown fields; keep schema strict
        str_strip_whitespace=True,
        validate_default=True,
    )

    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int = Field(ge=1000, le=9999)
    venue: str = ""
    doi: str | None = None
    url: str | None = None
    added: date
    confidence: SourceConfidence
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_must_be_well_formed(cls, v: str) -> str:
        if not is_valid_source_id(v):
            raise ValueError(f"source id {v!r} does not match any known source-type pattern")
        return v


class PaperSource(SourceCommon):
    """A peer-reviewed or formally published paper.

    Per AGENTS.md v0.2 §2.2: `doi` OR `arxiv_id` required (not both empty),
    plus `bibtex_key`.
    """

    type: Literal[SourceType.PAPER] = SourceType.PAPER
    bibtex_key: str
    arxiv_id: str | None = None

    @model_validator(mode="after")
    def _must_have_doi_or_arxiv(self) -> PaperSource:
        if not (self.doi or self.arxiv_id):
            raise ValueError("paper source must have either 'doi' or 'arxiv_id' populated")
        return self


class SimulationSource(SourceCommon):
    """A local simulation run. Load-bearing for CFD work; see AGENTS.md v0.2 §2.2.

    Provenance requires:
        - `solver_repo_id`: points at a filed `repository` source
        - `solver_commit`: HEAD at RUN TIME, optionally `+dirty` marker
        - either `input_files` (checked-in inputs) or `generation_spec`
          (procedural generation) — at least one
        - `output_summary`: must name an existing `.md` file in the source dir
        - `run_date`, `hardware`: context
    """

    type: Literal[SourceType.SIMULATION] = SourceType.SIMULATION
    solver_repo_id: str
    solver_commit: str
    input_files: list[str] | None = None
    generation_spec: dict[str, object] | None = None
    output_summary: str = "output_summary.md"
    run_date: date
    hardware: str
    raw_data_location: list[str] = Field(default_factory=list)

    @field_validator("solver_repo_id")
    @classmethod
    def _solver_repo_id_must_be_well_formed(cls, v: str) -> str:
        if not is_valid_source_id(v, SourceType.REPOSITORY):
            raise ValueError(f"solver_repo_id {v!r} must be a valid repository source ID")
        return v

    @field_validator("solver_commit")
    @classmethod
    def _solver_commit_must_be_hash_optionally_dirty(cls, v: str) -> str:
        """Per AGENTS.md v0.2 §2.2: hash-at-run-time, optional `+dirty` marker."""
        base, sep, marker = v.partition("+")
        if not _HASH_RE.fullmatch(base):
            raise ValueError(
                f"solver_commit base must be a 7-40 char hex hash; got {base!r}. "
                f"Per AGENTS.md §2.2, 'last commit' without a hash is ambiguous and forbidden."
            )
        if sep and marker not in {"dirty"}:
            raise ValueError(f"solver_commit marker after '+' must be 'dirty'; got {marker!r}")
        return v

    @model_validator(mode="after")
    def _must_have_input_files_or_generation_spec(self) -> SimulationSource:
        if not self.input_files and not self.generation_spec:
            raise ValueError(
                "simulation source must have either 'input_files' (literal files) "
                "or 'generation_spec' (procedural). At least one must be present "
                "per AGENTS.md v0.2 §2.2."
            )
        return self


class RepositorySource(SourceCommon):
    """A git repository as a source, e.g. a solver referenced by a simulation.

    Per AGENTS.md v0.2 §2.2, `url` and `commit` are required. `path_in_repo`,
    `commit_date`, `commit_subject`, `local_clone` are optional but all useful
    for human reading and reproducibility.
    """

    type: Literal[SourceType.REPOSITORY] = SourceType.REPOSITORY
    commit: str
    path_in_repo: str | None = None
    commit_date: date | None = None
    commit_subject: str | None = None
    local_clone: str | None = None

    @field_validator("commit")
    @classmethod
    def _commit_must_be_full_or_short_hash(cls, v: str) -> str:
        if not _HASH_RE.fullmatch(v.lower()):
            raise ValueError(f"repository commit must be a 7-40 char hex hash; got {v!r}")
        return v.lower()

    @model_validator(mode="after")
    def _url_required_for_repository(self) -> RepositorySource:
        # Common field url is Optional by default; for repositories we tighten.
        if not self.url:
            raise ValueError("repository source requires a 'url' field")
        return self


class WebArticleSource(SourceCommon):
    """A web article or blog post as a source.

    Per AGENTS.md v0.2 §2.2: `url`, `archived_at` required; `archived_snapshot_path`
    optional. The snapshot path, when present, points at a local archive file
    (HTML or markdown) inside the source directory — insurance against link rot.
    """

    type: Literal[SourceType.WEB_ARTICLE] = SourceType.WEB_ARTICLE
    archived_at: date
    archived_snapshot_path: str | None = None

    @model_validator(mode="after")
    def _url_required_for_web_article(self) -> WebArticleSource:
        if not self.url:
            raise ValueError("web_article source requires a 'url' field")
        return self


class NotebookSource(BaseModel):
    """A Stefano-authored notebook entry.

    **Asymmetry from SourceCommon is deliberate.** Notebook entries live in
    `notebook/`, not `sources/`, and their frontmatter shape is defined by
    AGENTS.md v0.2 §2.4, not §2.1. Common fields like `title`, `authors`,
    `venue`, `added`, `confidence` do not apply — the entry IS authored by
    Stefano, the date IS the added-date, and source-level confidence is
    meaningless when every extracted claim gets `author_hypothesis` or
    `author_reasoning` per §2.5.

    Notebook entries are therefore a peer shape to SourceCommon-descended
    sources, dispatched into SourceManifest by the same `type:` discriminator.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    id: str
    type: Literal[SourceType.NOTEBOOK] = SourceType.NOTEBOOK
    date: date
    topic: str
    status: NotebookStatus = NotebookStatus.DRAFT
    superseded_by: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_must_be_notebook_form(cls, v: str) -> str:
        if not is_valid_source_id(v, SourceType.NOTEBOOK):
            raise ValueError(f"notebook id {v!r} must match pattern note_YYYY_MM_DD_<slug>")
        return v

    @field_validator("superseded_by")
    @classmethod
    def _superseded_by_must_be_notebook_id(cls, v: str | None) -> str | None:
        if v is not None and not is_valid_source_id(v, SourceType.NOTEBOOK):
            raise ValueError(f"superseded_by {v!r} must be a valid notebook source ID")
        return v

    @model_validator(mode="after")
    def _superseded_status_requires_superseded_by(self) -> NotebookSource:
        if self.status == NotebookStatus.SUPERSEDED and not self.superseded_by:
            raise ValueError("notebook with status=superseded must name the successor in 'superseded_by'")
        if self.status != NotebookStatus.SUPERSEDED and self.superseded_by:
            raise ValueError("'superseded_by' is only valid when status=superseded")
        return self


# Discriminated union entry point. When parsing a `source.yaml` of unknown type,
# use `SourceManifest`:
#
#     SourceManifest.validate_python(yaml_dict)
#
# Pydantic will dispatch to the concrete class based on the `type:` field.
# Note: NotebookSource is included here even though notebook frontmatter lives in
# a different file location (notebook/<id>.md, not sources/<id>/source.yaml);
# the discriminated union is purely on the `type:` field semantics.
SourceManifest = Annotated[
    PaperSource | SimulationSource | RepositorySource | WebArticleSource | NotebookSource,
    Field(discriminator="type"),
]


# =============================================================================
# Claim model (M1d)
# =============================================================================


class ExternalRef(BaseModel):
    """A reference to an external source not yet filed in rossum.

    Per AGENTS.md v0.2 §2.5: when a claim's body cites an external source (paper,
    report, dataset) not yet in `sources/`, it should be declared in the
    `external_refs:` frontmatter field. Lint (M4) reports externals cited by ≥2
    distinct rossum sources as promotion candidates.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    citation: str
    kind: SourceType
    promotion_candidate: bool = False


class Claim(BaseModel):
    """An atomic, source-anchored assertion per AGENTS.md v0.2 §2.5.

    Every claim has a frontmatter block (these fields) and a body (free-form
    markdown for context, limitations, scope of validity). Body is not modeled
    here — it is the extracted text content of the `.md` file after the
    frontmatter block.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    id: str
    source: str
    source_span: str
    quote: str
    claim: str
    confidence: Confidence
    extracted: date
    status: ClaimStatus = ClaimStatus.ACTIVE

    # Status-dependent cross-references.
    disputed_by: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)

    # v0.2 additions.
    run: str | None = None  # sub-experiment label for multi-experiment simulation sources
    external_refs: list[ExternalRef] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_must_be_claim_form(cls, v: str) -> str:
        if not is_valid_claim_id(v):
            raise ValueError(f"claim id {v!r} must match pattern claim_YYYY_<slug>")
        return v

    @field_validator("source")
    @classmethod
    def _source_must_be_valid_source_id(cls, v: str) -> str:
        """Source field points at either a sources/<id>/ directory or a notebook/<id>.md file.

        Per AGENTS.md v0.2 §2.1 and §2.4, notebook entries are referenced by their
        note_... ID even though they live outside sources/. is_valid_source_id
        accepts both via the type-agnostic check.
        """
        if not is_valid_source_id(v):
            raise ValueError(f"source {v!r} must be a valid source ID (any type, including notebook)")
        return v

    @field_validator("disputed_by", "supersedes")
    @classmethod
    def _references_must_be_valid_claim_ids(cls, v: list[str]) -> list[str]:
        for ref in v:
            if not is_valid_claim_id(ref):
                raise ValueError(f"{ref!r} is not a valid claim ID")
        return v

    @model_validator(mode="after")
    def _status_coherence(self) -> Claim:
        """Status field must be coherent with disputed_by / supersedes per AGENTS.md v0.2 §2.5."""
        # A claim with status=disputed should have disputed_by populated.
        # However, a claim that DISPUTES another is itself active while the
        # older one becomes disputed — so disputed_by can be populated on the
        # OLDER claim; the newer claim's disputed_by references the old claim's ID.
        # We don't over-enforce this direction; lint catches contradictions.
        if self.status == ClaimStatus.SUPERSEDED and not self.supersedes and not self.disputed_by:
            # Pure soft-warning territory — not enforced here.
            pass
        return self


# =============================================================================
# Wiki page models (M1d)
# =============================================================================


class WikiPage(BaseModel):
    """A wiki page per AGENTS.md v0.2 §2.6.

    Frontmatter-only model; the body is free-form markdown with `[[claim_...]]`
    citations. Citation-rule-by-kind (strict for entity/concept/method, licensed
    for synthesis/comparison) is enforced at lint time, not at schema time.

    Used for ordinary wiki pages. `wiki/index.md` has a different, minimal
    frontmatter shape and is modeled separately as `WikiIndex`. `wiki/log.md`
    is unstructured (a sequence of `## [YYYY-MM-DD]` sections) and is modeled
    by `LogEntry` at the entry granularity.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    id: str
    kind: WikiKind
    title: str
    created: date
    updated: date
    claim_count: int = Field(ge=0)
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_must_be_wiki_page_form(cls, v: str) -> str:
        if not is_valid_wiki_page_id(v):
            raise ValueError(f"wiki page id {v!r} must be lowercase snake_case")
        return v

    @model_validator(mode="after")
    def _updated_not_before_created(self) -> WikiPage:
        if self.updated < self.created:
            raise ValueError(f"wiki page 'updated' ({self.updated}) precedes 'created' ({self.created})")
        return self

    @model_validator(mode="after")
    def _id_prefix_matches_kind_when_present(self) -> WikiPage:
        """If the ID carries a kind prefix (e.g. 'entity_...'), it must match `kind:`.

        IDs may also be bare slugs for well-known terms (per AGENTS.md §2.6),
        in which case no check fires. The lint layer handles the per-kind
        citation rule separately.
        """
        known_prefixes = {k.value + "_" for k in WikiKind}
        for prefix in known_prefixes:
            if self.id.startswith(prefix):
                expected = prefix[:-1]  # strip trailing underscore
                if self.kind.value != expected:
                    raise ValueError(
                        f"wiki page id {self.id!r} has prefix '{expected}_' but kind is "
                        f"{self.kind.value!r}; prefix and kind must agree when a prefix is present"
                    )
                break
        return self


class WikiIndex(BaseModel):
    """The wiki catalog file `wiki/index.md` per AGENTS.md v0.2 §2.6.

    Deliberately separate model from WikiPage: the index is a catalog, not a
    content page, and its frontmatter is minimal (no id, no title, no
    claim_count). The body is structured as sections-by-category with bullet
    lists of `[[page_id]] - summary` — body structure is enforced at lint time.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["index"] = "index"
    updated: date


# =============================================================================
# Log entry model (M1d)
# =============================================================================


class LogEntry(BaseModel):
    """One entry in `wiki/log.md` per AGENTS.md v0.2 §3.1.

    `log.md` is an append-only markdown file, NOT a single frontmatter model.
    Each entry is a `## [YYYY-MM-DD] <op> | <subject>` header followed by
    bulleted body lines. This model represents ONE such entry; the ingest and
    lint layers parse/emit them as a sequence.

    Grep-parseable form (the subject is free text, but date/op are structured):
        ## [2026-04-23] ingest | paper_zaghi_2023_amr_gpu_ibm
        - source: paper_zaghi_2023_amr_gpu_ibm (type=paper)
        - claims: 18 approved / 0 rejected
        - ...
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    date: date
    op: str  # e.g. "ingest", "lint", "query", "schema-change", "init"
    subject: str  # free-form, typically a source_id or short description
    body_lines: list[str] = Field(default_factory=list)

    @field_validator("op")
    @classmethod
    def _op_must_be_single_word(cls, v: str) -> str:
        """Op is grep-parseable; must be a single lowercase word (hyphens allowed)."""
        import re as _re

        if not _re.fullmatch(r"[a-z][a-z0-9-]*", v):
            raise ValueError(f"op {v!r} must be lowercase alphanumeric (hyphens allowed)")
        return v

    def to_markdown(self) -> str:
        """Emit this entry in the grep-parseable markdown shape."""
        header = f"## [{self.date.isoformat()}] {self.op} | {self.subject}"
        body = "\n".join(self.body_lines)
        if body:
            return f"{header}\n\n{body}\n"
        return f"{header}\n"

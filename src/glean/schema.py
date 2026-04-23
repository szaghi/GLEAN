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

M1b scope: Paper, Simulation, Repository only — the three types Phase 1
exercised. Other types land in M1c.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from glean.enums import SourceConfidence, SourceType
from glean.ids import is_valid_source_id

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


# Discriminated union entry point. When parsing a `source.yaml` of unknown type,
# use `SourceManifest`:
#
#     SourceManifest.validate_python(yaml_dict)
#
# Pydantic will dispatch to the concrete class based on the `type:` field.
SourceManifest = Annotated[
    PaperSource | SimulationSource | RepositorySource,
    Field(discriminator="type"),
]

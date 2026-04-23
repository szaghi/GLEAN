"""Enum types for the GLEAN schema.

All enums are `StrEnum` (Python 3.11+), so their `.value` is the canonical string
that appears in YAML / markdown frontmatter. Adding a new variant here is a
schema change — coordinate with `rossum/AGENTS.md` and bump the schema version.

The enum values mirror AGENTS.md §2.2 (SourceType), §2.5 (Confidence, ClaimStatus),
and §2.6 (WikiKind) as of schema v0.2.
"""

from __future__ import annotations

from enum import StrEnum


class SourceType(StrEnum):
    """Source types enumerated in AGENTS.md v0.2 §2.2."""

    PAPER = "paper"
    PREPRINT = "preprint"
    REPOSITORY = "repository"
    DATASET = "dataset"
    SIMULATION = "simulation"
    TALK = "talk"
    BOOK = "book"
    STANDARD = "standard"
    PERSONAL_COMM = "personal_comm"
    WEB_ARTICLE = "web_article"
    NOTEBOOK = "notebook"


class SourceConfidence(StrEnum):
    """Source-level confidence per AGENTS.md v0.2 §2.1.

    This is Stefano's trust in the source itself, distinct from `Confidence`
    (which is claim-level and records how the source itself frames each
    assertion). A high-confidence source can still host speculative claims.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Confidence(StrEnum):
    """Claim confidence values per AGENTS.md v0.2 §2.5.

    `AUTHOR_ASSERTION`, `MEASURED`, `DERIVED`: appropriate for external sources.
    `AUTHOR_HYPOTHESIS`, `AUTHOR_REASONING`: appropriate for notebook sources.
    `SPECULATIVE`: flagged as tentative by the source itself.

    When choosing between `AUTHOR_HYPOTHESIS` and `AUTHOR_REASONING`: a claim
    refutable by a single experimental outcome is a hypothesis; one refutable
    only by argument or a new reasoned position is reasoning.
    """

    AUTHOR_ASSERTION = "author_assertion"
    MEASURED = "measured"
    DERIVED = "derived"
    AUTHOR_HYPOTHESIS = "author_hypothesis"
    AUTHOR_REASONING = "author_reasoning"
    SPECULATIVE = "speculative"


class ClaimStatus(StrEnum):
    """Claim lifecycle status per AGENTS.md v0.2 §2.5."""

    ACTIVE = "active"
    DISPUTED = "disputed"
    RETRACTED = "retracted"
    SUPERSEDED = "superseded"


class WikiKind(StrEnum):
    """Wiki page kinds per AGENTS.md v0.2 §2.6.

    `ENTITY`, `CONCEPT`, `METHOD` pages describe and require strict per-sentence
    claim citation. `SYNTHESIS` and `COMPARISON` pages argue and are licensed for
    derived prose connecting cited claims. The lint layer enforces the distinction;
    schema does not.
    """

    ENTITY = "entity"
    CONCEPT = "concept"
    METHOD = "method"
    SYNTHESIS = "synthesis"
    COMPARISON = "comparison"


class NotebookStatus(StrEnum):
    """Notebook entry status per AGENTS.md v0.2 §2.4."""

    DRAFT = "draft"
    SETTLED = "settled"
    SUPERSEDED = "superseded"


# Page kinds whose wiki prose is licensed to contain derived sentences
# (arithmetic, connective prose, paraphrase) without a per-sentence claim citation.
# Consumed by the lint layer (M4) and by wiki-update prompt templates (M3).
DERIVED_PROSE_KINDS: frozenset[WikiKind] = frozenset({WikiKind.SYNTHESIS, WikiKind.COMPARISON})
STRICT_CITATION_KINDS: frozenset[WikiKind] = frozenset({WikiKind.ENTITY, WikiKind.CONCEPT, WikiKind.METHOD})

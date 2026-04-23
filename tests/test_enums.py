"""Tests for `glean.enums`.

These are structural tests — enum values, membership, the derived-prose / strict-
citation partition of WikiKind. Semantic correctness (that the enum matches
AGENTS.md v0.2) is validated by the downstream schema tests that round-trip
real rossum content.
"""

from __future__ import annotations

import pytest

from glean.enums import (
    DERIVED_PROSE_KINDS,
    STRICT_CITATION_KINDS,
    ClaimStatus,
    Confidence,
    NotebookStatus,
    SourceType,
    WikiKind,
)


class TestSourceType:
    def test_values_match_agents_md_v0_2(self) -> None:
        expected = {
            "paper",
            "preprint",
            "repository",
            "dataset",
            "simulation",
            "talk",
            "book",
            "standard",
            "personal_comm",
            "web_article",
            "notebook",
        }
        assert {s.value for s in SourceType} == expected

    def test_str_enum_backing(self) -> None:
        assert SourceType.PAPER == "paper"
        assert SourceType.SIMULATION.value == "simulation"


class TestConfidence:
    def test_values_match_agents_md_v0_2(self) -> None:
        expected = {
            "author_assertion",
            "measured",
            "derived",
            "author_hypothesis",
            "author_reasoning",
            "speculative",
        }
        assert {c.value for c in Confidence} == expected

    @pytest.mark.parametrize(
        "value",
        ["author_assertion", "measured", "author_hypothesis", "author_reasoning"],
    )
    def test_phase_1_confidence_values_accepted(self, value: str) -> None:
        """Every confidence value used in Phase 1 rossum is a valid enum member."""
        assert Confidence(value).value == value


class TestClaimStatus:
    def test_values(self) -> None:
        assert {s.value for s in ClaimStatus} == {"active", "disputed", "retracted", "superseded"}


class TestWikiKind:
    def test_values(self) -> None:
        assert {k.value for k in WikiKind} == {"entity", "concept", "method", "synthesis", "comparison"}

    def test_partition_covers_all_kinds(self) -> None:
        """Every WikiKind is either strict-citation or derived-prose, never both."""
        all_kinds = set(WikiKind)
        assert all_kinds == DERIVED_PROSE_KINDS | STRICT_CITATION_KINDS
        assert set() == DERIVED_PROSE_KINDS & STRICT_CITATION_KINDS

    def test_partition_matches_agents_md_v0_2(self) -> None:
        """AGENTS.md v0.2 §2.6: entity/concept/method are strict; synthesis/comparison are licensed."""
        assert {WikiKind.SYNTHESIS, WikiKind.COMPARISON} == DERIVED_PROSE_KINDS
        assert {WikiKind.ENTITY, WikiKind.CONCEPT, WikiKind.METHOD} == STRICT_CITATION_KINDS


class TestNotebookStatus:
    def test_values(self) -> None:
        assert {s.value for s in NotebookStatus} == {"draft", "settled", "superseded"}

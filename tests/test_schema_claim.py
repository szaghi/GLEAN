"""Tests for `glean.schema.Claim` — claim frontmatter model (M1d).

The exit-criterion test: every one of the 31 claim files in rossum
round-trips through the Claim model. Structural tests cover the v0.2 additions
(run: key, external_refs field, meta tag, ExternalRef shape).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from glean.enums import ClaimStatus, Confidence, SourceType
from glean.schema import Claim, ExternalRef

ROSSUM = Path.home() / "rossum"
_ROSSUM_AVAILABLE = (ROSSUM / "AGENTS.md").exists()


def _parse_claim_file(path: Path) -> dict[str, object]:
    """Extract YAML frontmatter from a `<claim_id>.md` file."""
    body = path.read_text()
    parts = body.split("---\n", 2)
    assert len(parts) == 3, f"{path} must start with '---'-delimited frontmatter"
    return yaml.safe_load(parts[1])


# -----------------------------------------------------------------------------
# Round-trip tests against real rossum content (the exit-criterion test)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not _ROSSUM_AVAILABLE, reason="rossum repo not present")
class TestRossumClaimsRoundTrip:
    """Every claim .md file in rossum/claims/ must parse cleanly via Claim."""

    def test_all_phase_1_claims_validate(self) -> None:
        claim_files = sorted((ROSSUM / "claims").glob("claim_*.md"))
        assert len(claim_files) >= 31, f"expected ≥31 Phase 1 claims; found {len(claim_files)}"

        failures: list[tuple[str, str]] = []
        for path in claim_files:
            try:
                raw = _parse_claim_file(path)
                parsed = Claim.model_validate(raw)
                # Cross-check that the filename matches the parsed id.
                assert path.stem == parsed.id, f"filename/id mismatch in {path}: stem={path.stem}, id={parsed.id}"
            except (ValidationError, AssertionError, yaml.YAMLError) as e:
                failures.append((path.name, str(e)))

        if failures:
            msg = "\n".join(f"  {name}: {err}" for name, err in failures)
            pytest.fail(f"{len(failures)}/{len(claim_files)} Phase 1 claims failed to parse:\n{msg}")

    def test_phase_1_contains_run_tagged_claims(self) -> None:
        """Verify the schema accepts the real run: A / run: B claims from sim source 2."""
        for claim_name in [
            "claim_2026_prism_rmf_run_a_artifacts_deleted_provenance.md",
            "claim_2026_prism_restart_matches_baseline_at_machine_precision.md",
        ]:
            raw = _parse_claim_file(ROSSUM / "claims" / claim_name)
            parsed = Claim.model_validate(raw)
            assert parsed.run in {"A", "B"}, f"{claim_name}: expected run A/B, got {parsed.run!r}"


# -----------------------------------------------------------------------------
# Minimal Claim + structural tests
# -----------------------------------------------------------------------------


def _minimal_claim() -> dict[str, object]:
    return {
        "id": "claim_2026_foo_bar",
        "source": "paper_foo_2026_bar",
        "source_span": "§2.1",
        "quote": "Some verbatim text.",
        "claim": "A short paraphrase.",
        "confidence": "author_assertion",
        "extracted": date(2026, 4, 23),
        "status": "active",
    }


class TestClaimMinimal:
    def test_validates(self) -> None:
        c = Claim.model_validate(_minimal_claim())
        assert c.confidence == Confidence.AUTHOR_ASSERTION
        assert c.status == ClaimStatus.ACTIVE
        assert c.run is None
        assert c.external_refs == []
        assert c.disputed_by == []
        assert c.supersedes == []
        assert c.tags == []

    def test_rejects_malformed_id(self) -> None:
        d = _minimal_claim()
        d["id"] = "not-a-claim-id"
        with pytest.raises(ValidationError, match=r"claim_YYYY"):
            Claim.model_validate(d)

    def test_rejects_malformed_source(self) -> None:
        d = _minimal_claim()
        d["source"] = "not-a-source"
        with pytest.raises(ValidationError, match=r"valid source ID"):
            Claim.model_validate(d)

    def test_accepts_notebook_source(self) -> None:
        """Claim.source can point into notebook/ via a note_... ID."""
        d = _minimal_claim()
        d["source"] = "note_2026_04_23_foo_bar"
        c = Claim.model_validate(d)
        assert c.source == "note_2026_04_23_foo_bar"

    def test_rejects_unknown_confidence(self) -> None:
        d = _minimal_claim()
        d["confidence"] = "high"  # valid SourceConfidence, NOT valid Confidence
        with pytest.raises(ValidationError):
            Claim.model_validate(d)

    def test_rejects_extra_field(self) -> None:
        d = _minimal_claim()
        d["unknown"] = "nope"
        with pytest.raises(ValidationError, match=r"Extra inputs are not permitted"):
            Claim.model_validate(d)


class TestClaimReferences:
    def test_disputed_by_accepts_claim_ids(self) -> None:
        d = _minimal_claim()
        d["disputed_by"] = ["claim_2026_other_claim"]
        c = Claim.model_validate(d)
        assert c.disputed_by == ["claim_2026_other_claim"]

    def test_disputed_by_rejects_non_claim_ids(self) -> None:
        d = _minimal_claim()
        d["disputed_by"] = ["paper_zaghi_2023_amr_gpu_ibm"]
        with pytest.raises(ValidationError, match=r"valid claim ID"):
            Claim.model_validate(d)

    def test_supersedes_accepts_claim_ids(self) -> None:
        d = _minimal_claim()
        d["supersedes"] = ["claim_2026_old_version"]
        c = Claim.model_validate(d)
        assert c.supersedes == ["claim_2026_old_version"]


class TestClaimRunKey:
    def test_run_accepts_string_label(self) -> None:
        d = _minimal_claim()
        d["run"] = "A"
        c = Claim.model_validate(d)
        assert c.run == "A"

    def test_run_accepts_descriptive_label(self) -> None:
        """AGENTS.md v0.2 §2.5 allows 'A, B, or a descriptive label'."""
        d = _minimal_claim()
        d["run"] = "baseline"
        c = Claim.model_validate(d)
        assert c.run == "baseline"

    def test_run_optional(self) -> None:
        c = Claim.model_validate(_minimal_claim())
        assert c.run is None


class TestClaimExternalRefs:
    def test_empty_default(self) -> None:
        c = Claim.model_validate(_minimal_claim())
        assert c.external_refs == []

    def test_accepts_structured_external_refs(self) -> None:
        d = _minimal_claim()
        d["external_refs"] = [
            {
                "citation": "Cichocki et al. 2025 - IEPC-2025-291",
                "kind": "paper",
                "promotion_candidate": True,
            }
        ]
        c = Claim.model_validate(d)
        assert len(c.external_refs) == 1
        ref = c.external_refs[0]
        assert ref.kind == SourceType.PAPER
        assert ref.promotion_candidate is True

    def test_rejects_external_ref_with_bad_kind(self) -> None:
        d = _minimal_claim()
        d["external_refs"] = [{"citation": "x", "kind": "not_a_type"}]
        with pytest.raises(ValidationError):
            Claim.model_validate(d)

    def test_external_ref_promotion_candidate_defaults_false(self) -> None:
        r = ExternalRef.model_validate({"citation": "x", "kind": "paper"})
        assert r.promotion_candidate is False


class TestClaimTags:
    def test_meta_tag_accepted(self) -> None:
        """AGENTS.md v0.2 §2.5: meta-claims carry `tags: [meta]`. Schema does not special-case it."""
        d = _minimal_claim()
        d["tags"] = ["meta", "provenance"]
        c = Claim.model_validate(d)
        assert "meta" in c.tags

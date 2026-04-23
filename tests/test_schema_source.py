"""Tests for `glean.schema` — source models.

M1b scope: Paper, Simulation, Repository discriminated union. The strongest
correctness test is round-tripping the three actual `source.yaml` files from
rossum (Phase 1 commit `eed0a80` and later). If a real v0.2-valid source.yaml
does not parse, the schema is wrong.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from glean.enums import SourceConfidence, SourceType
from glean.schema import (
    PaperSource,
    RepositorySource,
    SimulationSource,
    SourceManifest,
)

# Path to the real rossum repo, used for the round-trip tests.
# If rossum is not present (clean CI, different user), the round-trip tests skip.
ROSSUM = Path.home() / "rossum"
_ROSSUM_AVAILABLE = (ROSSUM / "AGENTS.md").exists()

# TypeAdapter gives us the discriminated-union dispatch at the module level.
SOURCE_MANIFEST = TypeAdapter(SourceManifest)


# -----------------------------------------------------------------------------
# Round-trip tests against real rossum content (the exit-criterion test)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not _ROSSUM_AVAILABLE, reason="rossum repo not present")
class TestRossumRoundTrip:
    """Every source.yaml in rossum must parse cleanly via SourceManifest."""

    @pytest.mark.parametrize(
        "source_dir,expected_type",
        [
            ("paper_zaghi_2023_amr_gpu_ibm", SourceType.PAPER),
            ("sim_2026_04_prism_rmf_restart", SourceType.SIMULATION),
            ("repo_szaghi_adam_dbe47a44", SourceType.REPOSITORY),
        ],
    )
    def test_parses_real_source_yaml(self, source_dir: str, expected_type: SourceType) -> None:
        yaml_path = ROSSUM / "sources" / source_dir / "source.yaml"
        assert yaml_path.exists(), f"rossum source.yaml missing at {yaml_path}"
        raw = yaml.safe_load(yaml_path.read_text())
        parsed = SOURCE_MANIFEST.validate_python(raw)
        assert parsed.type == expected_type
        assert parsed.id == source_dir


# -----------------------------------------------------------------------------
# Common-fields tests (SourceCommon behavior)
# -----------------------------------------------------------------------------


def _minimal_paper() -> dict[str, object]:
    """A minimal valid paper source as a dict; base for mutation tests."""
    return {
        "id": "paper_zaghi_2023_amr_gpu_ibm",
        "type": "paper",
        "title": "A paper",
        "authors": ["Zaghi, Stefano"],
        "year": 2023,
        "venue": "J. Foo",
        "doi": "10.1/abc",
        "url": None,
        "added": date(2026, 4, 23),
        "confidence": "high",
        "tags": [],
        "bibtex_key": "zaghi2023",
        "arxiv_id": None,
    }


class TestSourceCommon:
    def test_minimal_paper_validates(self) -> None:
        p = PaperSource.model_validate(_minimal_paper())
        assert p.id == "paper_zaghi_2023_amr_gpu_ibm"
        assert p.confidence == SourceConfidence.HIGH

    def test_rejects_unknown_field(self) -> None:
        d = _minimal_paper()
        d["unknown_field"] = "nope"
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            PaperSource.model_validate(d)

    def test_rejects_malformed_id(self) -> None:
        d = _minimal_paper()
        d["id"] = "not-a-valid-id"
        with pytest.raises(ValidationError, match="does not match any known source-type pattern"):
            PaperSource.model_validate(d)

    @pytest.mark.parametrize("year", [999, 10000, -1, 0])
    def test_rejects_invalid_year(self, year: int) -> None:
        d = _minimal_paper()
        d["year"] = year
        with pytest.raises(ValidationError):
            PaperSource.model_validate(d)

    def test_rejects_invalid_confidence(self) -> None:
        d = _minimal_paper()
        d["confidence"] = "very-high"
        with pytest.raises(ValidationError):
            PaperSource.model_validate(d)


# -----------------------------------------------------------------------------
# PaperSource-specific tests
# -----------------------------------------------------------------------------


class TestPaperSource:
    def test_requires_doi_or_arxiv(self) -> None:
        d = _minimal_paper()
        d["doi"] = None
        d["arxiv_id"] = None
        with pytest.raises(ValidationError, match=r"doi.*arxiv_id"):
            PaperSource.model_validate(d)

    def test_arxiv_only_is_fine(self) -> None:
        d = _minimal_paper()
        d["doi"] = None
        d["arxiv_id"] = "2310.12345"
        d["id"] = "arxiv_2310_12345"  # ID must also match arxiv form
        p = PaperSource.model_validate(d)
        assert p.arxiv_id == "2310.12345"

    def test_missing_bibtex_key_rejected(self) -> None:
        d = _minimal_paper()
        del d["bibtex_key"]
        with pytest.raises(ValidationError, match="bibtex_key"):
            PaperSource.model_validate(d)


# -----------------------------------------------------------------------------
# SimulationSource-specific tests
# -----------------------------------------------------------------------------


def _minimal_sim() -> dict[str, object]:
    return {
        "id": "sim_2026_04_prism_rmf_restart",
        "type": "simulation",
        "title": "A sim",
        "authors": ["Zaghi, Stefano"],
        "year": 2026,
        "venue": "local",
        "added": date(2026, 4, 23),
        "confidence": "high",
        "tags": [],
        "solver_repo_id": "repo_szaghi_adam_dbe47a44",
        "solver_commit": "dbe47a44",
        "input_files": ["input.ini"],
        "output_summary": "output_summary.md",
        "run_date": date(2026, 4, 14),
        "hardware": "local workstation",
    }


class TestSimulationSource:
    def test_minimal_validates(self) -> None:
        s = SimulationSource.model_validate(_minimal_sim())
        assert s.solver_commit == "dbe47a44"

    def test_accepts_dirty_marker(self) -> None:
        d = _minimal_sim()
        d["solver_commit"] = "dbe47a440907c2aee466f84f206262488803bad0+dirty"
        s = SimulationSource.model_validate(d)
        assert s.solver_commit.endswith("+dirty")

    def test_rejects_bare_last_commit(self) -> None:
        d = _minimal_sim()
        d["solver_commit"] = "last"
        with pytest.raises(ValidationError, match="hex hash"):
            SimulationSource.model_validate(d)

    def test_rejects_bogus_dirty_variant(self) -> None:
        d = _minimal_sim()
        d["solver_commit"] = "abc1234+stale"
        with pytest.raises(ValidationError, match=r"marker after.*dirty"):
            SimulationSource.model_validate(d)

    def test_rejects_malformed_solver_repo_id(self) -> None:
        d = _minimal_sim()
        d["solver_repo_id"] = "not-a-repo-id"
        with pytest.raises(ValidationError, match="valid repository source ID"):
            SimulationSource.model_validate(d)

    def test_accepts_generation_spec_without_input_files(self) -> None:
        d = _minimal_sim()
        del d["input_files"]
        d["generation_spec"] = {
            "repo_id": "repo_szaghi_adam_dbe47a44",
            "commit": "dbe47a44",
            "case_name": "kt02",
        }
        s = SimulationSource.model_validate(d)
        assert s.input_files is None
        assert s.generation_spec is not None

    def test_rejects_neither_input_files_nor_generation_spec(self) -> None:
        d = _minimal_sim()
        del d["input_files"]
        with pytest.raises(ValidationError, match=r"input_files.*generation_spec"):
            SimulationSource.model_validate(d)


# -----------------------------------------------------------------------------
# RepositorySource-specific tests
# -----------------------------------------------------------------------------


def _minimal_repo() -> dict[str, object]:
    return {
        "id": "repo_szaghi_adam_dbe47a44",
        "type": "repository",
        "title": "adam",
        "authors": ["Zaghi, Stefano"],
        "year": 2026,
        "venue": "GitHub",
        "url": "https://github.com/szaghi/adam",
        "added": date(2026, 4, 23),
        "confidence": "high",
        "tags": [],
        "commit": "dbe47a440907c2aee466f84f206262488803bad0",
    }


class TestRepositorySource:
    def test_minimal_validates(self) -> None:
        r = RepositorySource.model_validate(_minimal_repo())
        assert r.commit.startswith("dbe47a44")

    def test_requires_url(self) -> None:
        d = _minimal_repo()
        d["url"] = None
        with pytest.raises(ValidationError, match="requires a 'url' field"):
            RepositorySource.model_validate(d)

    def test_optional_fields_accepted(self) -> None:
        d = _minimal_repo()
        d["commit_date"] = date(2026, 4, 14)
        d["commit_subject"] = "feat: stuff"
        d["local_clone"] = "~/fortran/adam"
        d["path_in_repo"] = "."
        r = RepositorySource.model_validate(d)
        assert r.commit_subject == "feat: stuff"

    def test_rejects_malformed_commit(self) -> None:
        d = _minimal_repo()
        d["commit"] = "not-a-hash"
        with pytest.raises(ValidationError, match="hex hash"):
            RepositorySource.model_validate(d)


# -----------------------------------------------------------------------------
# Discriminated-union dispatch tests
# -----------------------------------------------------------------------------


class TestDiscriminatedUnion:
    def test_paper_dispatches_correctly(self) -> None:
        parsed = SOURCE_MANIFEST.validate_python(_minimal_paper())
        assert isinstance(parsed, PaperSource)

    def test_simulation_dispatches_correctly(self) -> None:
        parsed = SOURCE_MANIFEST.validate_python(_minimal_sim())
        assert isinstance(parsed, SimulationSource)

    def test_repository_dispatches_correctly(self) -> None:
        parsed = SOURCE_MANIFEST.validate_python(_minimal_repo())
        assert isinstance(parsed, RepositorySource)

    def test_rejects_unknown_type(self) -> None:
        d = _minimal_paper()
        d["type"] = "dataset"  # valid SourceType but not in M1b union
        with pytest.raises(ValidationError):
            SOURCE_MANIFEST.validate_python(d)

    def test_rejects_missing_type(self) -> None:
        d = _minimal_paper()
        del d["type"]
        with pytest.raises(ValidationError):
            SOURCE_MANIFEST.validate_python(d)

"""Tests for `glean.ids`.

Covers:
    - slugify edge cases
    - deterministic source-ID construction per type
    - validation of real Phase 1 rossum IDs (the strongest correctness test)
    - rejection of malformed IDs
"""

from __future__ import annotations

from datetime import date

import pytest

from glean.enums import SourceType
from glean.ids import (
    claim_id_for,
    is_valid_claim_id,
    is_valid_source_id,
    is_valid_wiki_page_id,
    notebook_id_for,
    slugify,
    source_id_for,
)


class TestSlugify:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Hello, World!", "hello_world"),
            ("  A--B--C  ", "a_b_c"),
            ("AmrGpuIbm", "amrgpuibm"),
            ("amr gpu ibm", "amr_gpu_ibm"),
            ("multi__underscore___runs", "multi_underscore_runs"),
            ("Zaghi, Stefano", "zaghi_stefano"),
        ],
    )
    def test_basic(self, text: str, expected: str) -> None:
        assert slugify(text) == expected

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty slug"):
            slugify("!!!")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty slug"):
            slugify("   ")


class TestSourceIdConstruction:
    def test_paper_with_first_author(self) -> None:
        assert (
            source_id_for(SourceType.PAPER, first_author="Zaghi", year=2023, slug="amr gpu ibm")
            == "paper_zaghi_2023_amr_gpu_ibm"
        )

    def test_paper_with_arxiv_id(self) -> None:
        assert source_id_for(SourceType.PAPER, arxiv_id="2310.12345") == "arxiv_2310_12345"

    def test_preprint(self) -> None:
        assert source_id_for(SourceType.PREPRINT, arxiv_id="2310.00001") == "arxiv_2310_00001"

    def test_repository(self) -> None:
        assert (
            source_id_for(SourceType.REPOSITORY, org="szaghi", name="adam", commit="dbe47a44")
            == "repo_szaghi_adam_dbe47a44"
        )

    def test_repository_rejects_short_commit(self) -> None:
        with pytest.raises(ValueError, match=r"6-12 hex chars"):
            source_id_for(SourceType.REPOSITORY, org="a", name="b", commit="abc")

    def test_repository_rejects_non_hex_commit(self) -> None:
        with pytest.raises(ValueError, match=r"6-12 hex chars"):
            source_id_for(SourceType.REPOSITORY, org="a", name="b", commit="ghijklmn")

    def test_simulation(self) -> None:
        assert (
            source_id_for(SourceType.SIMULATION, year_month="2026_04", slug="prism rmf restart")
            == "sim_2026_04_prism_rmf_restart"
        )

    def test_simulation_rejects_bad_year_month(self) -> None:
        with pytest.raises(ValueError, match="YYYY_MM"):
            source_id_for(SourceType.SIMULATION, year_month="2026-04", slug="foo")

    def test_notebook(self) -> None:
        assert (
            notebook_id_for(date(2026, 4, 23), "extending amr 2to1 to 4to1")
            == "note_2026_04_23_extending_amr_2to1_to_4to1"
        )

    def test_dataset(self) -> None:
        assert source_id_for(SourceType.DATASET, slug="my dataset", year=2024) == "data_my_dataset_2024"

    def test_talk(self) -> None:
        assert source_id_for(SourceType.TALK, speaker="Zaghi", venue="iccfd", year=2025) == "talk_zaghi_iccfd_2025"

    def test_book(self) -> None:
        assert (
            source_id_for(SourceType.BOOK, first_author="Toro", year=2009, slug="riemann solvers")
            == "book_toro_2009_riemann_solvers"
        )

    def test_standard(self) -> None:
        assert source_id_for(SourceType.STANDARD, body="iso", number="13485") == "std_iso_13485"

    def test_personal_comm(self) -> None:
        assert (
            source_id_for(SourceType.PERSONAL_COMM, date_ymd="2026_03_15", correspondents_slug="smith_call")
            == "comm_2026_03_15_smith_call"
        )

    def test_web_article(self) -> None:
        assert (
            source_id_for(
                SourceType.WEB_ARTICLE,
                domain="example.com",
                date_ymd="2026_04_01",
                slug="some post",
            )
            == "web_example_com_2026_04_01_some_post"
        )

    def test_missing_kwarg_raises(self) -> None:
        with pytest.raises(ValueError, match="missing or non-string"):
            source_id_for(SourceType.PAPER, first_author="Zaghi", year=2023)  # no slug


class TestClaimId:
    def test_basic(self) -> None:
        assert (
            claim_id_for(2023, "zaghi_amr_gpu_ibm", "adam_hashmap_octree_tree")
            == "claim_2023_zaghi_amr_gpu_ibm_adam_hashmap_octree_tree"
        )

    def test_slugifies_inputs(self) -> None:
        assert claim_id_for(2023, "Zaghi AMR-GPU", "Adam Hash-Map") == "claim_2023_zaghi_amr_gpu_adam_hash_map"

    @pytest.mark.parametrize("year", [0, 100, 999, 10000])
    def test_rejects_bad_year(self, year: int) -> None:
        with pytest.raises(ValueError, match="4 digits"):
            claim_id_for(year, "a", "b")


class TestIdValidators:
    @pytest.mark.parametrize(
        "source_id,source_type",
        [
            # Real IDs from Phase 1 rossum — the strongest correctness test.
            ("paper_zaghi_2023_amr_gpu_ibm", SourceType.PAPER),
            ("sim_2026_04_prism_rmf_restart", SourceType.SIMULATION),
            ("repo_szaghi_adam_dbe47a44", SourceType.REPOSITORY),
            ("note_2026_04_23_extending_amr_2to1_to_4to1", SourceType.NOTEBOOK),
        ],
    )
    def test_phase_1_rossum_ids_validate(self, source_id: str, source_type: SourceType) -> None:
        assert is_valid_source_id(source_id, source_type)
        assert is_valid_source_id(source_id)  # type-agnostic form

    def test_is_valid_source_id_rejects_malformed(self) -> None:
        assert not is_valid_source_id("not_a_valid_id")
        assert not is_valid_source_id("paper-with-dashes")
        assert not is_valid_source_id("UPPERCASE_ID")
        assert not is_valid_source_id("")

    def test_is_valid_source_id_rejects_mismatched_type(self) -> None:
        """A paper ID given with type=SIMULATION should be rejected."""
        assert not is_valid_source_id("paper_zaghi_2023_amr_gpu_ibm", SourceType.SIMULATION)

    @pytest.mark.parametrize(
        "claim_id",
        [
            # Real claim IDs from Phase 1.
            "claim_2023_zaghi_adam_hashmap_octree_tree",
            "claim_2023_zaghi_amr_update_becomes_dominant_at_fine_resolution",
            "claim_2026_prism_restart_matches_baseline_at_machine_precision",
            "claim_2026_zaghi_amr_4to1_interpolation_order_hypothesis",
        ],
    )
    def test_phase_1_claim_ids_validate(self, claim_id: str) -> None:
        assert is_valid_claim_id(claim_id)

    def test_is_valid_claim_id_rejects_malformed(self) -> None:
        assert not is_valid_claim_id("not_a_claim")
        assert not is_valid_claim_id("claim_no_year_here")
        assert not is_valid_claim_id("claim_23_too_short_year")

    @pytest.mark.parametrize(
        "page_id",
        [
            # Real wiki page IDs from Phase 1.
            "entity_adam_framework",
            "entity_nasto_application",
            "entity_prism_application",
            "concept_octree_amr",
            "concept_gpu_data_layout_coalescing",
            "method_morton_load_balancing",
            "method_ibm_signed_distance_eikonal",
            "method_prism_checkpoint_restart",
            "synthesis_adam_gpu_performance_envelope",
            "synthesis_prism_rmf_validation_status",
            "synthesis_adam_amr_refinement_ratio_investigation",
        ],
    )
    def test_phase_1_wiki_page_ids_validate(self, page_id: str) -> None:
        assert is_valid_wiki_page_id(page_id)

    def test_is_valid_wiki_page_id_rejects_dashes(self) -> None:
        assert not is_valid_wiki_page_id("entity-with-dashes")

    def test_is_valid_wiki_page_id_rejects_uppercase(self) -> None:
        assert not is_valid_wiki_page_id("Entity_Camel_Case")

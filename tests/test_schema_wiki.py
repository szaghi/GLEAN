"""Tests for `glean.schema.WikiPage` and `WikiIndex` (M1d).

The exit-criterion test: every one of the 11 wiki pages in rossum round-trips
through the appropriate model.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from glean.enums import WikiKind
from glean.schema import WikiIndex, WikiPage

ROSSUM = Path.home() / "rossum"
_ROSSUM_AVAILABLE = (ROSSUM / "AGENTS.md").exists()


def _parse_frontmatter(path: Path) -> dict[str, object]:
    body = path.read_text()
    parts = body.split("---\n", 2)
    assert len(parts) == 3, f"{path} must start with '---'-delimited frontmatter"
    return yaml.safe_load(parts[1])


# -----------------------------------------------------------------------------
# Round-trip tests against real rossum content
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not _ROSSUM_AVAILABLE, reason="rossum repo not present")
class TestRossumWikiRoundTrip:
    def test_all_phase_1_wiki_pages_validate(self) -> None:
        # Ordinary wiki pages: every *.md in wiki/ except index.md and log.md.
        all_md = sorted((ROSSUM / "wiki").glob("*.md"))
        page_files = [p for p in all_md if p.name not in {"index.md", "log.md"}]
        assert len(page_files) >= 11, f"expected ≥11 Phase 1 wiki pages; found {len(page_files)}"

        failures: list[tuple[str, str]] = []
        for path in page_files:
            try:
                raw = _parse_frontmatter(path)
                parsed = WikiPage.model_validate(raw)
                assert path.stem == parsed.id, f"filename/id mismatch: {path.stem} vs {parsed.id}"
            except (ValidationError, AssertionError, yaml.YAMLError) as e:
                failures.append((path.name, str(e)))

        if failures:
            msg = "\n".join(f"  {name}: {err}" for name, err in failures)
            pytest.fail(f"{len(failures)}/{len(page_files)} Phase 1 wiki pages failed to parse:\n{msg}")

    def test_index_md_validates_as_wiki_index(self) -> None:
        raw = _parse_frontmatter(ROSSUM / "wiki" / "index.md")
        parsed = WikiIndex.model_validate(raw)
        assert parsed.kind == "index"


# -----------------------------------------------------------------------------
# WikiPage structural tests
# -----------------------------------------------------------------------------


def _minimal_page() -> dict[str, object]:
    return {
        "id": "entity_foo",
        "kind": "entity",
        "title": "Foo",
        "created": date(2026, 4, 23),
        "updated": date(2026, 4, 23),
        "claim_count": 0,
        "tags": [],
    }


class TestWikiPageMinimal:
    def test_validates(self) -> None:
        p = WikiPage.model_validate(_minimal_page())
        assert p.kind == WikiKind.ENTITY
        assert p.claim_count == 0

    def test_rejects_malformed_id(self) -> None:
        d = _minimal_page()
        d["id"] = "Invalid-ID"
        with pytest.raises(ValidationError, match=r"lowercase snake_case"):
            WikiPage.model_validate(d)

    def test_rejects_negative_claim_count(self) -> None:
        d = _minimal_page()
        d["claim_count"] = -1
        with pytest.raises(ValidationError):
            WikiPage.model_validate(d)

    def test_rejects_updated_before_created(self) -> None:
        d = _minimal_page()
        d["created"] = date(2026, 4, 23)
        d["updated"] = date(2026, 4, 22)
        with pytest.raises(ValidationError, match=r"precedes 'created'"):
            WikiPage.model_validate(d)

    def test_rejects_extra_field(self) -> None:
        d = _minimal_page()
        d["rogue"] = "nope"
        with pytest.raises(ValidationError, match=r"Extra inputs are not permitted"):
            WikiPage.model_validate(d)


class TestWikiPageIdPrefixKindCoherence:
    @pytest.mark.parametrize(
        "page_id,kind",
        [
            ("entity_foo", "entity"),
            ("concept_bar", "concept"),
            ("method_baz", "method"),
            ("synthesis_qux", "synthesis"),
            ("comparison_corge", "comparison"),
        ],
    )
    def test_matching_prefix_and_kind(self, page_id: str, kind: str) -> None:
        d = _minimal_page()
        d["id"] = page_id
        d["kind"] = kind
        WikiPage.model_validate(d)  # should succeed

    def test_rejects_prefix_kind_mismatch(self) -> None:
        d = _minimal_page()
        d["id"] = "entity_foo"
        d["kind"] = "method"  # mismatch
        with pytest.raises(ValidationError, match=r"prefix and kind must agree"):
            WikiPage.model_validate(d)

    def test_bare_slug_id_any_kind_accepted(self) -> None:
        """AGENTS.md §2.6: bare slugs (no prefix) are valid for well-known terms."""
        d = _minimal_page()
        d["id"] = "navier_stokes"
        d["kind"] = "concept"
        WikiPage.model_validate(d)


# -----------------------------------------------------------------------------
# WikiIndex tests
# -----------------------------------------------------------------------------


class TestWikiIndex:
    def test_minimal_validates(self) -> None:
        idx = WikiIndex.model_validate({"kind": "index", "updated": date(2026, 4, 23)})
        assert idx.kind == "index"

    def test_rejects_non_index_kind(self) -> None:
        with pytest.raises(ValidationError):
            WikiIndex.model_validate({"kind": "entity", "updated": date(2026, 4, 23)})

    def test_rejects_extra_fields(self) -> None:
        """WikiIndex frontmatter is deliberately minimal — reject regular page fields."""
        with pytest.raises(ValidationError, match=r"Extra inputs are not permitted"):
            WikiIndex.model_validate({"kind": "index", "updated": date(2026, 4, 23), "title": "Catalog"})

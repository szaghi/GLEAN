"""Tests for `glean.schema.LogEntry` (M1d).

LogEntry represents ONE entry in wiki/log.md. The file itself is a sequence
of entries separated by '## [YYYY-MM-DD] ...' headers plus free surrounding
markdown. Parsing the file is an ingest concern (M3); this tests the per-entry
model only.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from glean.schema import LogEntry


def _minimal_entry() -> dict[str, object]:
    return {
        "date": date(2026, 4, 23),
        "op": "ingest",
        "subject": "paper_zaghi_2023_amr_gpu_ibm",
        "body_lines": [],
    }


class TestLogEntryMinimal:
    def test_validates(self) -> None:
        e = LogEntry.model_validate(_minimal_entry())
        assert e.op == "ingest"
        assert e.body_lines == []

    def test_with_body_lines(self) -> None:
        d = _minimal_entry()
        d["body_lines"] = ["- source: paper_foo (type=paper)", "- claims: 18 approved"]
        e = LogEntry.model_validate(d)
        assert len(e.body_lines) == 2

    def test_rejects_bad_op(self) -> None:
        d = _minimal_entry()
        d["op"] = "Ingest"  # uppercase not allowed
        with pytest.raises(ValidationError, match=r"lowercase alphanumeric"):
            LogEntry.model_validate(d)

    def test_accepts_hyphenated_op(self) -> None:
        d = _minimal_entry()
        d["op"] = "schema-change"
        e = LogEntry.model_validate(d)
        assert e.op == "schema-change"

    def test_rejects_op_with_spaces(self) -> None:
        d = _minimal_entry()
        d["op"] = "ingest and lint"
        with pytest.raises(ValidationError, match=r"lowercase alphanumeric"):
            LogEntry.model_validate(d)


class TestLogEntryMarkdown:
    def test_to_markdown_header_only(self) -> None:
        e = LogEntry.model_validate(_minimal_entry())
        md = e.to_markdown()
        assert md.startswith("## [2026-04-23] ingest | paper_zaghi_2023_amr_gpu_ibm")

    def test_to_markdown_with_body(self) -> None:
        d = _minimal_entry()
        d["body_lines"] = ["- source: paper_foo", "- claims: 18 approved / 0 rejected"]
        e = LogEntry.model_validate(d)
        md = e.to_markdown()
        assert "## [2026-04-23] ingest | paper_zaghi_2023_amr_gpu_ibm" in md
        assert "- source: paper_foo" in md
        assert "- claims: 18 approved / 0 rejected" in md

    def test_to_markdown_grep_parseable(self) -> None:
        """AGENTS.md §3.1 step 3: entries must be grep-parseable by date.

        `grep "^## \\[" log.md` should match every entry header.
        """
        e = LogEntry.model_validate(_minimal_entry())
        md = e.to_markdown()
        first_line = md.split("\n")[0]
        assert first_line.startswith("## [")

"""Wiki consistency checks (M4).

`glean lint` runs 15 checks across a rossum repo per AGENTS.md v0.2 §3.3
plus the v0.2 additions and the F14 "strict-parse" check from PHASE1_NOTES.

Output: human-readable `file:line: severity: message` by default; `--json`
for structured output. Exit code 0 if no errors (warnings allowed unless
`--strict`), nonzero otherwise.

Architecture: each check is a function taking a `NotesRepo` and returning a
`list[LintFinding]`. The top-level `run_lint()` invokes them in a fixed order
from the `_CHECKS` registry and aggregates findings.

Check registry is static (D28): adding a check means adding one entry to the
list plus the function. No plugins, no decorators — if v0.2 wants extensibility
we rebuild.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import StrEnum
from pathlib import Path

import yaml

from glean.enums import DERIVED_PROSE_KINDS, SourceType
from glean.errors import GleanRepoError
from glean.git import status as git_status
from glean.repo import NotesRepo, _split_frontmatter


class Severity(StrEnum):
    """Lint finding severity. Errors always fail; warnings fail only with --strict."""

    ERROR = "error"
    WARNING = "warning"


@dataclass
class LintFinding:
    """One lint result."""

    check: str  # short check name, e.g. "uncited_sentences"
    severity: Severity
    message: str
    path: str | None = None  # repo-relative, if the finding has a specific location
    line: int | None = None  # 1-indexed, if applicable


@dataclass
class LintReport:
    """Aggregated result of a full lint run."""

    findings: list[LintFinding] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    def errors(self) -> list[LintFinding]:
        return [f for f in self.findings if f.severity == Severity.ERROR]

    def warnings(self) -> list[LintFinding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]


# =============================================================================
# Regex patterns reused by several checks
# =============================================================================

# [[claim_...]] citations in wiki body prose.
_CITATION_RE = re.compile(r"\[\[(claim_[a-z0-9_]+)\]\]")

# [[<anything>]] — any wikilink, used to distinguish claim citations from page links.
_WIKILINK_RE = re.compile(r"\[\[([a-z0-9_]+)\]\]")

# Rough sentence splitter: split on `.`, `!`, `?` followed by whitespace or EOL,
# OR on a newline. Deliberately conservative — we'd rather flag borderline
# cases than miss real ones.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")

# A sentence is considered "non-trivial" if it contains at least one letter
# and is longer than a threshold (~15 chars); sentences shorter than that are
# typically headers, captions, bullet fragments, or single-word lines.
_MIN_SENTENCE_LEN = 15


# =============================================================================
# Individual check functions
# =============================================================================


def check_frontmatter_parse(repo: NotesRepo) -> list[LintFinding]:
    """F14: every frontmatter block must strict-parse as YAML mapping.

    Three Phase 1 claim files had malformed YAML that no runtime tool caught.
    Lint catches it before ingest-time errors propagate.
    """
    findings: list[LintFinding] = []

    candidates: list[tuple[Path, str]] = []
    for path in repo.claims_dir.glob("*.md"):
        candidates.append((path, "claim"))
    for path in repo.wiki_dir.glob("*.md"):
        if path.name in {"log.md"}:
            continue  # log.md has no frontmatter by design
        candidates.append((path, "wiki"))
    for path in repo.notebook_dir.glob("*.md"):
        candidates.append((path, "notebook"))
    for source_dir in repo.sources_dir.iterdir():
        if source_dir.is_dir():
            ypath = source_dir / "source.yaml"
            if ypath.is_file():
                candidates.append((ypath, "source"))

    for path, kind in candidates:
        rel = str(path.relative_to(repo.root))
        try:
            text = path.read_text()
        except OSError as e:
            findings.append(
                LintFinding(
                    check="frontmatter_parse",
                    severity=Severity.ERROR,
                    message=f"{kind}: cannot read file ({e})",
                    path=rel,
                )
            )
            continue

        if path.suffix == ".yaml":
            # source.yaml is pure YAML, no frontmatter delimiters.
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError as e:
                findings.append(
                    LintFinding(
                        check="frontmatter_parse",
                        severity=Severity.ERROR,
                        message=f"{kind}: YAML parse failed ({e})",
                        path=rel,
                    )
                )
                continue
            if not isinstance(data, dict):
                findings.append(
                    LintFinding(
                        check="frontmatter_parse",
                        severity=Severity.ERROR,
                        message=f"{kind}: YAML is not a mapping (got {type(data).__name__})",
                        path=rel,
                    )
                )
            continue

        # Markdown files with frontmatter: use the canonical splitter.
        try:
            _split_frontmatter(text)
        except GleanRepoError as e:
            findings.append(
                LintFinding(
                    check="frontmatter_parse",
                    severity=Severity.ERROR,
                    message=f"{kind}: frontmatter parse failed ({e})",
                    path=rel,
                )
            )

    return findings


def check_dangling_claim_citations(repo: NotesRepo) -> list[LintFinding]:
    """Every [[claim_...]] in wiki prose must point at an existing claim .md file."""
    findings: list[LintFinding] = []
    known = set(repo.list_claims())

    for page_id in repo.list_wiki_pages():
        page_path = repo.wiki_dir / f"{page_id}.md"
        rel = str(page_path.relative_to(repo.root))
        try:
            body = page_path.read_text()
        except OSError:
            continue
        for lineno, line in enumerate(body.splitlines(), start=1):
            for m in _CITATION_RE.finditer(line):
                claim_id = m.group(1)
                if claim_id not in known:
                    findings.append(
                        LintFinding(
                            check="dangling_claim_citations",
                            severity=Severity.ERROR,
                            message=f"cites unknown claim {claim_id!r}",
                            path=rel,
                            line=lineno,
                        )
                    )
    return findings


def check_dangling_source_references(repo: NotesRepo) -> list[LintFinding]:
    """Every claim's `source:` must point at a real source (sources/ or notebook/)."""
    findings: list[LintFinding] = []
    known_sources = set(repo.list_sources())
    for claim_id in repo.list_claims():
        try:
            claim, _body = repo.load_claim(claim_id)
        except GleanRepoError:
            continue  # covered by frontmatter_parse
        if claim.source not in known_sources:
            findings.append(
                LintFinding(
                    check="dangling_source_references",
                    severity=Severity.ERROR,
                    message=f"claim cites unknown source {claim.source!r}",
                    path=f"claims/{claim_id}.md",
                )
            )
    return findings


def check_orphan_claims(repo: NotesRepo) -> list[LintFinding]:
    """Warn when a claim is not cited by any wiki page.

    Orphan status is a warning, not an error: newly-extracted claims may be
    legitimately orphaned until a future gate-3 run integrates them.
    """
    findings: list[LintFinding] = []
    cited: set[str] = set()
    for page_id in repo.list_wiki_pages():
        page_path = repo.wiki_dir / f"{page_id}.md"
        try:
            body = page_path.read_text()
        except OSError:
            continue
        cited.update(_CITATION_RE.findall(body))

    for claim_id in repo.list_claims():
        if claim_id not in cited:
            findings.append(
                LintFinding(
                    check="orphan_claims",
                    severity=Severity.WARNING,
                    message=f"claim {claim_id} is not cited by any wiki page",
                    path=f"claims/{claim_id}.md",
                )
            )
    return findings


def check_uncited_sentences(repo: NotesRepo) -> list[LintFinding]:
    """Per-page-kind citation rule per AGENTS.md v0.2 §2.6.

    WARNING (not ERROR) on entity/concept/method pages: sentences without a
    claim citation OR any wikilink get flagged. Derived prose in
    synthesis/comparison pages is licensed and not flagged.

    Rationale for WARNING rather than ERROR: the citation rule is a writing
    discipline that humans apply with judgment — "see [[method_foo]] for X"
    is a legitimate pointer, as are equations, list-item headers, and
    connective prose between cited claims. Automated sentence-level checking
    will always have false positives on natural-language variants. Flag as
    WARNING so the user sees the candidates but doesn't fail CI on style.
    """
    findings: list[LintFinding] = []
    for page_id in repo.list_wiki_pages():
        try:
            page, body = repo.load_wiki_page(page_id)
        except GleanRepoError:
            continue
        if page.kind in DERIVED_PROSE_KINDS:
            continue  # synthesis/comparison pages may contain derived prose
        page_rel = f"wiki/{page_id}.md"
        for lineno, line in enumerate(body.splitlines(), start=1):
            # Skip structural / non-prose lines.
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("#", ">", "-", "*", "|", "```", ":", "$", "=")):
                continue
            # Skip lines containing ANY wikilink — pointer patterns like
            # "See [[page_id]] for detail" aren't claim assertions.
            if _WIKILINK_RE.search(stripped):
                continue
            # Skip lines that are mostly symbols (equations, math notation).
            letter_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
            if letter_ratio < 0.5:
                continue

            sentences = _split_into_sentences(stripped)
            for sentence in sentences:
                if len(sentence) < _MIN_SENTENCE_LEN:
                    continue
                if _CITATION_RE.search(sentence):
                    continue
                findings.append(
                    LintFinding(
                        check="uncited_sentences",
                        severity=Severity.WARNING,
                        message=f"uncited sentence candidate in {page.kind.value} page: {sentence[:80]!r}",
                        path=page_rel,
                        line=lineno,
                    )
                )
    return findings


def _split_into_sentences(text: str) -> list[str]:
    """Conservative sentence splitter."""
    parts = _SENTENCE_END_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def check_contradictions(repo: NotesRepo) -> list[LintFinding]:
    """Flag claim pairs where both are active and one disputes the other.

    Per AGENTS.md v0.2 §4: contradictions must be surfaced. If claim A says
    `disputed_by: [claim_B]` and both are `status: active`, that's an
    unresolved contradiction the human should mark.
    """
    findings: list[LintFinding] = []

    # Load all claims into a map for cross-reference.
    claims: dict[str, object] = {}
    for cid in repo.list_claims():
        try:
            claim, _body = repo.load_claim(cid)
        except GleanRepoError:
            continue
        claims[cid] = claim

    for cid, claim in claims.items():
        disputed_by = getattr(claim, "disputed_by", [])
        claim_status = getattr(claim, "status", None)
        if claim_status is None or getattr(claim_status, "value", None) != "active":
            continue
        for other_id in disputed_by:
            other = claims.get(other_id)
            if other is None:
                continue
            other_status = getattr(other, "status", None)
            if other_status is not None and getattr(other_status, "value", None) == "active":
                findings.append(
                    LintFinding(
                        check="contradictions",
                        severity=Severity.ERROR,
                        message=(
                            f"unresolved contradiction: {cid} (active) is disputed by "
                            f"{other_id} (also active). Mark one as status: retracted "
                            f"or supersede via explicit claim."
                        ),
                        path=f"claims/{cid}.md",
                    )
                )
    return findings


def check_stale_frontmatter(repo: NotesRepo) -> list[LintFinding]:
    """Warn when a wiki page's `updated` is older than the most-recent claim it cites."""
    findings: list[LintFinding] = []
    for page_id in repo.list_wiki_pages():
        try:
            page, body = repo.load_wiki_page(page_id)
        except GleanRepoError:
            continue
        cited_claim_ids = set(_CITATION_RE.findall(body))
        if not cited_claim_ids:
            continue
        latest: date | None = None
        for cid in cited_claim_ids:
            try:
                claim, _b = repo.load_claim(cid)
            except GleanRepoError:
                continue
            if latest is None or claim.extracted > latest:
                latest = claim.extracted
        if latest is not None and page.updated < latest:
            findings.append(
                LintFinding(
                    check="stale_frontmatter",
                    severity=Severity.WARNING,
                    message=(
                        f"wiki page {page_id} updated={page.updated} is older than "
                        f"most recent cited claim (extracted={latest})"
                    ),
                    path=f"wiki/{page_id}.md",
                )
            )
    return findings


def check_index_freshness(repo: NotesRepo) -> list[LintFinding]:
    """Every wiki page should be listed in wiki/index.md."""
    findings: list[LintFinding] = []
    if not repo.index_md_path.is_file():
        findings.append(
            LintFinding(
                check="index_freshness",
                severity=Severity.WARNING,
                message="wiki/index.md missing",
                path="wiki/index.md",
            )
        )
        return findings
    try:
        _idx, body = repo.load_wiki_index()
    except GleanRepoError:
        return findings  # covered by frontmatter_parse
    for page_id in repo.list_wiki_pages():
        # Index references pages via [[page_id]] wikilinks.
        if not re.search(rf"\[\[{re.escape(page_id)}\]\]", body):
            findings.append(
                LintFinding(
                    check="index_freshness",
                    severity=Severity.WARNING,
                    message=f"wiki/index.md does not link to {page_id}",
                    path="wiki/index.md",
                )
            )
    return findings


def check_log_completeness(repo: NotesRepo) -> list[LintFinding]:
    """Every ingest commit in git history should have a matching log.md entry.

    Uses a heuristic: commits with message starting 'source:', 'claims:', or
    'wiki: ingest' are ingest commits. log.md is scanned for a matching
    `## [YYYY-MM-DD] ingest | <subject>` header. One warning per source id
    that appears in commit history but not in log.
    """
    findings: list[LintFinding] = []
    log_path = repo.log_md_path
    if not log_path.is_file():
        findings.append(
            LintFinding(
                check="log_completeness",
                severity=Severity.WARNING,
                message="wiki/log.md missing",
                path="wiki/log.md",
            )
        )
        return findings
    log_text = log_path.read_text()

    # Collect source IDs from git log subjects matching 'source: add <id>'.
    # We read git log via a narrow shell-out to avoid coupling lint to the
    # full git.py surface. Failing to read git is non-fatal for lint.
    from glean.git import _run  # internal: reuse the git-base subprocess helper

    try:
        result = _run(("log", "--format=%s"), cwd=repo.root, check=False)
    except GleanRepoError:
        return findings
    commit_subjects = result.stdout.splitlines()

    expected_source_ids: set[str] = set()
    for subj in commit_subjects:
        m = re.match(r"source:\s*add\s+(\S+)", subj)
        if m:
            expected_source_ids.add(m.group(1))

    for sid in expected_source_ids:
        # Look for 'ingest | <sid>' header in log.md.
        if not re.search(rf"^## \[.*\] ingest \| {re.escape(sid)}", log_text, re.MULTILINE):
            findings.append(
                LintFinding(
                    check="log_completeness",
                    severity=Severity.WARNING,
                    message=f"source {sid} has a commit but no log.md entry",
                    path="wiki/log.md",
                )
            )
    return findings


def check_id_collisions(repo: NotesRepo) -> list[LintFinding]:
    """No two sources/claims/wiki-pages may share an ID."""
    findings: list[LintFinding] = []
    seen: dict[str, list[str]] = {}
    # Sources.
    for sid in repo.list_sources():
        seen.setdefault(sid, []).append(f"source:{sid}")
    # Claims.
    for cid in repo.list_claims():
        seen.setdefault(cid, []).append(f"claim:{cid}")
    # Wiki pages.
    for pid in repo.list_wiki_pages():
        seen.setdefault(pid, []).append(f"wiki:{pid}")
    for id_str, instances in seen.items():
        if len(instances) > 1:
            findings.append(
                LintFinding(
                    check="id_collisions",
                    severity=Severity.ERROR,
                    message=f"ID {id_str!r} appears in multiple locations: {instances}",
                )
            )
    return findings


def check_draft_leakage(repo: NotesRepo) -> list[LintFinding]:
    """.claim.draft files should be gitignored. Warn if any are tracked."""
    findings: list[LintFinding] = []
    drafts = list(repo.claims_dir.glob("*.claim.draft"))
    if not drafts:
        return findings

    # Check whether any draft is tracked by git.
    from glean.git import is_tracked

    for draft in drafts:
        rel = str(draft.relative_to(repo.root))
        if is_tracked(repo.root, rel):
            findings.append(
                LintFinding(
                    check="draft_leakage",
                    severity=Severity.ERROR,
                    message=(
                        f"{rel} is tracked by git. .claim.draft files must be gitignored per AGENTS.md v0.2 §2.5."
                    ),
                    path=rel,
                )
            )
    # Also warn if there are drafts lingering (even untracked) — the user may
    # have forgotten gate 2 promotion.
    untracked = [d for d in drafts if not is_tracked(repo.root, str(d.relative_to(repo.root)))]
    if untracked:
        names = ", ".join(d.name for d in untracked[:3])
        if len(untracked) > 3:
            names += f", ... (+{len(untracked) - 3} more)"
        findings.append(
            LintFinding(
                check="draft_leakage",
                severity=Severity.WARNING,
                message=f"{len(untracked)} pending .claim.draft file(s): {names}",
            )
        )
    return findings


def check_external_refs_promotions(repo: NotesRepo) -> list[LintFinding]:
    """v0.2 new: report externals cited by >=2 distinct rossum sources as promotion candidates."""
    findings: list[LintFinding] = []
    # Map citation text -> set of source IDs that cite it via a claim.
    citation_sources: dict[str, set[str]] = {}
    for cid in repo.list_claims():
        try:
            claim, _b = repo.load_claim(cid)
        except GleanRepoError:
            continue
        for ref in getattr(claim, "external_refs", []):
            key = getattr(ref, "citation", None)
            if not isinstance(key, str):
                continue
            citation_sources.setdefault(key, set()).add(claim.source)

    for citation, sources in citation_sources.items():
        if len(sources) >= 2:
            sample = ", ".join(sorted(sources)[:3])
            findings.append(
                LintFinding(
                    check="external_refs_promotions",
                    severity=Severity.WARNING,
                    message=(
                        f"external reference {citation!r} is cited by {len(sources)} "
                        f"distinct sources ({sample}). Consider promoting to a filed source."
                    ),
                )
            )
    return findings


def check_simulation_output_summary(repo: NotesRepo) -> list[LintFinding]:
    """v0.2 new: every simulation source must have a non-trivial output_summary.md."""
    findings: list[LintFinding] = []
    for sid in repo.list_sources():
        try:
            source = repo.load_source(sid)
        except GleanRepoError:
            continue
        if source.type != SourceType.SIMULATION:
            continue
        summary_name = getattr(source, "output_summary", "output_summary.md")
        summary_path = repo.sources_dir / sid / summary_name
        if not summary_path.is_file():
            findings.append(
                LintFinding(
                    check="simulation_output_summary",
                    severity=Severity.ERROR,
                    message=f"simulation source missing {summary_name}",
                    path=f"sources/{sid}/",
                )
            )
            continue
        text = summary_path.read_text()
        if "<FILL" in text:
            findings.append(
                LintFinding(
                    check="simulation_output_summary",
                    severity=Severity.ERROR,
                    message=f"{summary_name} still contains <FILL> markers",
                    path=f"sources/{sid}/{summary_name}",
                )
            )
        elif len(text) < 200:
            findings.append(
                LintFinding(
                    check="simulation_output_summary",
                    severity=Severity.WARNING,
                    message=f"{summary_name} is suspiciously short ({len(text)} chars)",
                    path=f"sources/{sid}/{summary_name}",
                )
            )
    return findings


def check_solver_commit_validity(repo: NotesRepo) -> list[LintFinding]:
    """v0.2 new: every simulation source's solver_commit must be a hex hash
    (optionally with '+dirty' marker) per AGENTS.md v0.2 §2.2."""
    findings: list[LintFinding] = []
    pattern = re.compile(r"^[a-f0-9]{7,40}(\+dirty)?$")
    for sid in repo.list_sources():
        try:
            source = repo.load_source(sid)
        except GleanRepoError:
            continue
        if source.type != SourceType.SIMULATION:
            continue
        commit = getattr(source, "solver_commit", "")
        if not pattern.fullmatch(commit):
            findings.append(
                LintFinding(
                    check="solver_commit_validity",
                    severity=Severity.ERROR,
                    message=f"solver_commit {commit!r} not a hex hash (optionally +dirty)",
                    path=f"sources/{sid}/source.yaml",
                )
            )
    return findings


def check_meta_claim_co_citation(repo: NotesRepo) -> list[LintFinding]:
    """v0.2 new: when a wiki page cites a content claim that has a governing
    meta-claim (same source, tags contains 'meta'), the page should also cite
    the meta-claim. Warning, not error."""
    findings: list[LintFinding] = []
    # Index: source -> list of meta-claim IDs from that source.
    meta_by_source: dict[str, list[str]] = {}
    for cid in repo.list_claims():
        try:
            claim, _b = repo.load_claim(cid)
        except GleanRepoError:
            continue
        if "meta" in getattr(claim, "tags", []):
            meta_by_source.setdefault(claim.source, []).append(cid)

    if not meta_by_source:
        return findings

    # For each wiki page, collect cited claims; if any share a source with a
    # meta-claim and the page does not also cite that meta-claim, warn.
    for page_id in repo.list_wiki_pages():
        page_path = repo.wiki_dir / f"{page_id}.md"
        try:
            body = page_path.read_text()
        except OSError:
            continue
        cited = set(_CITATION_RE.findall(body))
        if not cited:
            continue
        for claim_id in cited:
            try:
                claim, _b = repo.load_claim(claim_id)
            except GleanRepoError:
                continue
            metas = meta_by_source.get(claim.source, [])
            missing = [m for m in metas if m != claim_id and m not in cited]
            for m in missing:
                findings.append(
                    LintFinding(
                        check="meta_claim_co_citation",
                        severity=Severity.WARNING,
                        message=(
                            f"page cites {claim_id} but not the meta-claim {m} "
                            f"from the same source (co-citation recommended per "
                            f"AGENTS.md v0.2 §2.5)"
                        ),
                        path=f"wiki/{page_id}.md",
                    )
                )
    return findings


def check_wiki_clean(repo: NotesRepo) -> list[LintFinding]:
    """Warn when wiki/ has uncommitted changes.

    Not strictly a schema violation, but useful signal: if lint is run during
    an in-progress ingest, the user should know that wiki/ is dirty before
    interpreting results.
    """
    findings: list[LintFinding] = []
    raw = git_status(repo.root)
    wiki_lines = [line for line in raw.splitlines() if len(line) > 3 and "wiki/" in line[3:]]
    if wiki_lines:
        findings.append(
            LintFinding(
                check="wiki_clean",
                severity=Severity.WARNING,
                message=f"wiki/ has {len(wiki_lines)} uncommitted change(s); lint results may not reflect committed state",
            )
        )
    return findings


# =============================================================================
# Check registry (D28)
# =============================================================================


_CHECKS: list[tuple[str, Callable[[NotesRepo], list[LintFinding]]]] = [
    ("frontmatter_parse", check_frontmatter_parse),
    ("id_collisions", check_id_collisions),
    ("dangling_claim_citations", check_dangling_claim_citations),
    ("dangling_source_references", check_dangling_source_references),
    ("uncited_sentences", check_uncited_sentences),
    ("contradictions", check_contradictions),
    ("orphan_claims", check_orphan_claims),
    ("stale_frontmatter", check_stale_frontmatter),
    ("index_freshness", check_index_freshness),
    ("log_completeness", check_log_completeness),
    ("draft_leakage", check_draft_leakage),
    ("external_refs_promotions", check_external_refs_promotions),
    ("simulation_output_summary", check_simulation_output_summary),
    ("solver_commit_validity", check_solver_commit_validity),
    ("meta_claim_co_citation", check_meta_claim_co_citation),
    ("wiki_clean", check_wiki_clean),
]


# =============================================================================
# Top-level entry point
# =============================================================================


def run_lint(
    repo: NotesRepo,
    *,
    only: Iterable[str] | None = None,
) -> LintReport:
    """Run all lint checks (or a filtered subset) against the given repo."""
    report = LintReport()
    for name, func in _CHECKS:
        if only is not None and name not in only:
            continue
        try:
            findings = func(repo)
        except Exception as e:
            findings = [
                LintFinding(
                    check=name,
                    severity=Severity.ERROR,
                    message=f"check crashed with {type(e).__name__}: {e}",
                )
            ]
        report.findings.extend(findings)
        report.checks_run.append(name)
    return report


# =============================================================================
# Output formatters
# =============================================================================


def format_human(report: LintReport) -> str:
    """Human-readable stream: 'path:line: severity: check: message'."""
    lines: list[str] = []
    for f in report.findings:
        loc = ""
        if f.path:
            loc = f.path
            if f.line is not None:
                loc += f":{f.line}"
            loc += ": "
        lines.append(f"{loc}{f.severity.value}: {f.check}: {f.message}")
    err_count = len(report.errors())
    warn_count = len(report.warnings())
    lines.append("")
    lines.append(f"{err_count} error(s), {warn_count} warning(s), {len(report.checks_run)} check(s) run.")
    return "\n".join(lines)


def format_json(report: LintReport) -> str:
    """Structured JSON output for CI."""
    return json.dumps(
        {
            "findings": [asdict(f) for f in report.findings],
            "checks_run": report.checks_run,
            "error_count": len(report.errors()),
            "warning_count": len(report.warnings()),
        },
        default=str,  # handles enums cleanly
        indent=2,
    )

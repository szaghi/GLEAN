"""Gate-3 orchestration: wiki updates + log entry (M3d).

Gate 3 is the wiki-integration step per AGENTS.md v0.2 §3.1. The flow:

    1. Precondition: wiki/ working tree must be clean (D21). If not, abort
       with a commit-or-stash instruction.
    2. Load approved claims for this source.
    3. Assemble prompt context (AGENTS.md + wiki/index.md + existing-page
       summaries + approved claims) via ContextBuilder.
    4. Call the LLM (deep tier) with the wiki_update template.
    5. Parse the LLM output as '===== <page_id> =====' delimited blocks
       per D20.
    6. Apply each block to the working tree: write/overwrite the wiki page.
       Never git add or git commit — diffs land in working tree for human
       review per AGENTS.md §5.
    7. Append a log entry to wiki/log.md (D22) summarizing the whole ingest
       (source, claims, wiki pages touched, LLM calls).

Resume (D12 filesystem-is-state):
    - Gate 3 has no durable "I ran already" signal in the filesystem —
      uncommitted wiki changes aren't the same as "gate 3 ran". A caller
      that runs gate 3 twice will regenerate the wiki diff from the LLM.
      Idempotency is the user's responsibility via --resume flag semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import yaml
from pydantic import ValidationError

from glean.config import Config
from glean.errors import GleanLLMError, GleanRepoError
from glean.git import diff as git_diff
from glean.llm import (
    ContextBuilder,
    ContextPriority,
    LLMBackend,
    LLMCallLog,
    ModelTier,
    load_prompt,
)
from glean.repo import NotesRepo
from glean.schema import Claim, LogEntry, WikiPage

# Token budget for the wiki-update prompt. Slightly larger than gate 2 because
# existing-page summaries can grow quickly; LLM response also needs room to
# emit multiple full page bodies.
_WIKI_PROMPT_BUDGET = 14000

# Delimiter literals for the LLM output format (D20).
_PAGE_DELIM_PREFIX = "===== "
_PAGE_DELIM_SUFFIX = " ====="
_PAGE_END_MARKER = "===== end ====="
_NO_WIKI_UPDATES = "NO_WIKI_UPDATES"


@dataclass
class Gate3Result:
    """What gate 3 produces on success."""

    source_id: str
    wiki_pages_created: list[str] = field(default_factory=list)
    wiki_pages_updated: list[str] = field(default_factory=list)
    log_entry_appended: bool = False
    llm_call_log: LLMCallLog | None = None


# =============================================================================
# Main entry
# =============================================================================


def run_gate3(
    source_id: str,
    *,
    repo: NotesRepo,
    config: Config,
    backend: LLMBackend,
    approved_claim_ids: list[str],
    gate1_sub_bullets: list[str],
    gate2_sub_bullets: list[str],
) -> Gate3Result:
    """Execute gate 3 for the given source.

    Parameters
    ----------
    source_id
        Source being integrated into the wiki.
    repo, config, backend
        Standard gate dependencies.
    approved_claim_ids
        The claim IDs approved in gate 2 that the LLM will integrate.
    gate1_sub_bullets, gate2_sub_bullets
        Bullet-line fragments describing what happened in each prior gate,
        for the per-gate detail of the log entry (D22).
    """
    _ = config  # reserved; editor not needed at gate 3 (no interactive review)

    # D21: abort if wiki/ is dirty. Contaminating the gate-3 diff review
    # with unrelated pending changes defeats its purpose.
    _assert_wiki_clean(repo)

    # Load the approved claims. Empty list = gate 2 produced nothing; gate 3
    # becomes a no-op except for a log entry recording the fact.
    approved_claims: list[Claim] = []
    for cid in approved_claim_ids:
        claim, _body = repo.load_claim(cid)
        approved_claims.append(claim)

    # If there are no approved claims, skip the LLM call entirely.
    if not approved_claims:
        _append_log_entry(
            repo=repo,
            source_id=source_id,
            gate1_sub_bullets=gate1_sub_bullets,
            gate2_sub_bullets=gate2_sub_bullets,
            gate3_sub_bullets=["- wiki: no updates (gate 2 produced no claims)"],
        )
        return Gate3Result(
            source_id=source_id,
            log_entry_appended=True,
        )

    # Call the LLM.
    raw_output, call_log = _extract_wiki_updates(
        source_id=source_id,
        repo=repo,
        backend=backend,
        approved_claims=approved_claims,
    )

    # NO_WIKI_UPDATES is a valid response meaning "no changes needed".
    if raw_output.strip() == _NO_WIKI_UPDATES:
        gate3_bullets = ["- wiki: no updates (LLM determined claims did not warrant wiki changes)"]
        _append_log_entry(
            repo=repo,
            source_id=source_id,
            gate1_sub_bullets=gate1_sub_bullets,
            gate2_sub_bullets=gate2_sub_bullets,
            gate3_sub_bullets=gate3_bullets,
        )
        return Gate3Result(
            source_id=source_id,
            log_entry_appended=True,
            llm_call_log=call_log,
        )

    # Parse + apply page updates.
    updates = parse_wiki_updates(raw_output)
    if not updates:
        raise GleanLLMError(
            "LLM response contained no parseable wiki updates and was not NO_WIKI_UPDATES. "
            f"Response starts: {raw_output[:200]!r}"
        )

    created: list[str] = []
    updated: list[str] = []
    for page_id, body in updates.items():
        was_new = not (repo.wiki_dir / f"{page_id}.md").exists()
        _apply_wiki_page(page_id, body, repo=repo)
        (created if was_new else updated).append(page_id)

    # Construct gate-3 sub-bullets for the log entry.
    gate3_bullets: list[str] = [
        f"- wiki: {len(created)} new, {len(updated)} updated",
    ]
    for pid in created:
        gate3_bullets.append(f"- wiki-page created: {pid}")
    for pid in updated:
        gate3_bullets.append(f"- wiki-page updated: {pid}")
    if call_log is not None:
        gate3_bullets.extend(f"  {b}" for b in call_log.to_log_bullets())

    _append_log_entry(
        repo=repo,
        source_id=source_id,
        gate1_sub_bullets=gate1_sub_bullets,
        gate2_sub_bullets=gate2_sub_bullets,
        gate3_sub_bullets=gate3_bullets,
    )

    return Gate3Result(
        source_id=source_id,
        wiki_pages_created=created,
        wiki_pages_updated=updated,
        log_entry_appended=True,
        llm_call_log=call_log,
    )


# =============================================================================
# Preconditions
# =============================================================================


def _assert_wiki_clean(repo: NotesRepo) -> None:
    """Raise if wiki/ has any uncommitted changes per D21."""
    wiki_diff = git_diff(repo.root, paths=["wiki/"])
    wiki_staged = git_diff(repo.root, paths=["wiki/"], staged=True)
    # Also check for untracked files under wiki/.
    from glean.git import status as git_status

    raw_status = git_status(repo.root)
    untracked_under_wiki = [
        line for line in raw_status.splitlines() if line.startswith("??") and line[3:].startswith("wiki/")
    ]

    if wiki_diff or wiki_staged or untracked_under_wiki:
        raise GleanRepoError(
            "wiki/ has uncommitted changes. Gate 3 must run against a clean "
            "wiki/ so the resulting diff is reviewable in isolation per "
            "AGENTS.md v0.2 §5. Commit or stash your wiki/ changes first, then "
            "re-run with --resume."
        )


# =============================================================================
# LLM call + response parsing
# =============================================================================


def _extract_wiki_updates(
    *,
    source_id: str,
    repo: NotesRepo,
    backend: LLMBackend,
    approved_claims: list[Claim],
) -> tuple[str, LLMCallLog]:
    """Run the LLM to produce wiki updates. Returns (raw_response, call_log)."""
    template = load_prompt("wiki_update")
    today = date.today().isoformat()

    # Context tiers per D7.
    builder = ContextBuilder(budget_tokens=_WIKI_PROMPT_BUDGET)
    builder.add(
        repo.load_agents_md(),
        priority=ContextPriority.REQUIRED,
        label="agents_md",
    )
    builder.add(
        _render_claims_for_prompt(approved_claims),
        priority=ContextPriority.REQUIRED,
        label="approved_claims",
    )

    if repo.index_md_path.is_file():
        builder.add(
            repo.index_md_path.read_text(),
            priority=ContextPriority.PREFERRED,
            label="index_md",
        )

    existing_summary = _existing_pages_summary(repo)
    if existing_summary:
        builder.add(
            existing_summary,
            priority=ContextPriority.OPTIONAL,
            label="existing_pages",
        )

    rendered = builder.render()

    prompt = template.safe_substitute(
        today=today,
        agents_md=rendered.get("agents_md", ""),
        index_md=rendered.get("index_md", "(no index.md yet)"),
        existing_pages_summary=rendered.get("existing_pages", "(no existing wiki pages)"),
        approved_claims=rendered.get("approved_claims", ""),
    )

    response, call_log = backend.complete(prompt, tier=ModelTier.DEEP)
    return response, call_log


def _render_claims_for_prompt(claims: list[Claim]) -> str:
    """Render claims as compact YAML blocks for the LLM to reason over."""
    parts: list[str] = []
    for c in claims:
        block = yaml.safe_dump(
            c.model_dump(mode="json", exclude_defaults=False),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
        parts.append("---\n" + block.rstrip())
    parts.append("---")
    return "\n".join(parts)


def _existing_pages_summary(repo: NotesRepo) -> str:
    """Compact summary of current wiki pages: id | kind | title.

    Excludes index.md and log.md.
    """
    lines: list[str] = []
    for page_id in repo.list_wiki_pages():
        try:
            page, _body = repo.load_wiki_page(page_id)
        except GleanRepoError:
            continue
        lines.append(f"{page_id} | {page.kind.value} | {page.title}")
    return "\n".join(lines)


def parse_wiki_updates(text: str) -> dict[str, str]:
    """Parse the LLM output into {page_id: body} pairs.

    Format per D20:
        ===== page_id =====
        <body text>
        ===== end =====

    The `===== end =====` marker is optional; the next `===== <other_id> =====`
    also terminates the prior block. Blocks without a recognizable page-id
    delimiter are silently skipped. Malformed bodies (e.g. no frontmatter) are
    included in the return dict and deferred to `_apply_wiki_page`'s validator.
    """
    updates: dict[str, str] = {}
    current_id: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.strip() == _PAGE_END_MARKER:
            if current_id is not None:
                updates[current_id] = "\n".join(current_lines).strip() + "\n"
            current_id = None
            current_lines = []
            continue
        if line.startswith(_PAGE_DELIM_PREFIX) and line.endswith(_PAGE_DELIM_SUFFIX):
            # Close the prior block if still open.
            if current_id is not None:
                updates[current_id] = "\n".join(current_lines).strip() + "\n"
            inner = line[len(_PAGE_DELIM_PREFIX) : -len(_PAGE_DELIM_SUFFIX)].strip()
            if inner == "end":
                # Already handled above; defensively reset.
                current_id = None
                current_lines = []
                continue
            current_id = inner
            current_lines = []
            continue
        if current_id is not None:
            current_lines.append(line)
        # Lines outside any block are treated as LLM preamble/epilogue.

    # Close trailing block if the LLM forgot the end marker.
    if current_id is not None and current_lines:
        updates[current_id] = "\n".join(current_lines).strip() + "\n"

    return updates


# =============================================================================
# Apply to working tree
# =============================================================================


def _apply_wiki_page(page_id: str, body: str, *, repo: NotesRepo) -> None:
    """Write/overwrite a wiki page body to the working tree.

    Validates frontmatter via WikiPage before writing; raises on schema
    violations (with a clear pointer at which page).
    """
    # Split frontmatter from body for validation.
    if not body.startswith("---\n"):
        raise GleanLLMError(f"wiki page {page_id!r} body does not begin with '---' frontmatter marker")
    parts = body.split("---\n", 2)
    if len(parts) < 3:
        raise GleanLLMError(f"wiki page {page_id!r} has no closing '---' for frontmatter")

    try:
        frontmatter = yaml.safe_load(parts[1])
    except yaml.YAMLError as e:
        raise GleanLLMError(f"wiki page {page_id!r} frontmatter YAML parse failed: {e}") from e

    if not isinstance(frontmatter, dict):
        raise GleanLLMError(f"wiki page {page_id!r} frontmatter is not a YAML mapping")

    try:
        page = WikiPage.model_validate(frontmatter)
    except ValidationError as e:
        raise GleanLLMError(f"wiki page {page_id!r} frontmatter fails schema validation: {e}") from e

    # Sanity check: the ID in the delimiter should match the ID in frontmatter.
    if page.id != page_id:
        raise GleanLLMError(f"wiki page delimiter ID {page_id!r} disagrees with frontmatter id {page.id!r}")

    # Write via NotesRepo so the atomic-write discipline holds.
    markdown_body = parts[2]
    repo.save_wiki_page(page, markdown_body)


# =============================================================================
# Log entry (D22)
# =============================================================================


def _append_log_entry(
    *,
    repo: NotesRepo,
    source_id: str,
    gate1_sub_bullets: list[str],
    gate2_sub_bullets: list[str],
    gate3_sub_bullets: list[str],
) -> None:
    """Append a per-ingest log entry with per-gate sub-bullets per D22."""
    body_lines: list[str] = [
        f"- source: {source_id}",
        "- gate 1:",
        *[f"  {b}" for b in gate1_sub_bullets],
        "- gate 2:",
        *[f"  {b}" for b in gate2_sub_bullets],
        "- gate 3:",
        *[f"  {b}" for b in gate3_sub_bullets],
    ]
    entry = LogEntry(
        date=date.today(),
        op="ingest",
        subject=source_id,
        body_lines=body_lines,
    )
    repo.append_log(entry)

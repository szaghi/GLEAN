"""Gate-2 orchestration: claim extraction + batch review (M3c).

Gate 2 is the claims-approval step per AGENTS.md v0.2 §3.1. The flow:

    1. Assemble prompt context (AGENTS.md + source content + existing-claims
       summary) via ContextBuilder per D7.
    2. Call the LLM (deep tier) with the claim_extract template.
    3. Parse the LLM's YAML-blocks-separated-by-`---` output into Claim drafts.
       Slug (from LLM) → claim_id (deterministic) per D19.
    4. Write each draft as `claims/<claim_id>.claim.draft` (gitignored).
    5. Write a combined `claims/_pending_<source_id>.yaml` listing every
       draft. Open $EDITOR on this batch file per D18.
    6. On editor save: re-parse, validate, detect rejections (missing blocks)
       and deferrals (blocks marked `# DEFERRED`).
    7. Promote approved drafts `.claim.draft` → `.md`; retain deferred drafts
       as `.claim.draft`; delete rejected drafts.
    8. git-add + git-commit the approved claims per AGENTS.md §5.

Resume (D12, filesystem-is-state):
    - If `claims/<claim_id>.md` files already exist for this source and
      the approved commit is in git history → gate 2 complete.
    - If any `claim_*.claim.draft` files exist with source == source_id →
      gate 2 in progress; re-enter the editor loop with them.
    - Otherwise → fresh extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml
from pydantic import ValidationError

from glean.config import Config
from glean.enums import SourceType
from glean.errors import GleanLLMError, GleanRepoError
from glean.git import add as git_add
from glean.git import commit as git_commit
from glean.ids import claim_id_for, is_valid_claim_id, slugify
from glean.ingest_gate1 import _open_in_editor
from glean.llm import (
    ContextBuilder,
    ContextPriority,
    LLMBackend,
    LLMCallLog,
    ModelTier,
    load_prompt,
)
from glean.repo import NotesRepo
from glean.schema import Claim

# Token budget for the claim-extraction prompt. Deliberately conservative so
# the whole prompt fits a 16k-context model comfortably after the LLM also
# needs headroom for its response.
_CLAIM_PROMPT_BUDGET = 12000

# Max retries when the user's edits to the pending batch file fail validation.
_BATCH_EDITOR_RETRIES = 5


@dataclass
class Gate2Result:
    """What gate 2 produces on success."""

    source_id: str
    approved_claim_ids: list[str] = field(default_factory=list)
    rejected_count: int = 0
    deferred_count: int = 0
    commit_sha: str | None = None
    was_resumed: bool = False
    llm_call_log: LLMCallLog | None = None


# =============================================================================
# Main entry
# =============================================================================


def run_gate2(
    source_id: str,
    *,
    repo: NotesRepo,
    config: Config,
    backend: LLMBackend,
    resume: bool = False,
) -> Gate2Result:
    """Execute gate 2 for the given source ID.

    Parameters
    ----------
    source_id
        The source for which to extract claims. Gate 1 must have completed
        for this source (i.e. `sources/<id>/source.yaml` committed OR, for
        notebook sources, the entry exists in `notebook/`).
    repo
        Target rossum repo.
    config
        Loaded user config (unused at gate 2 directly but kept for
        parameter-shape consistency with run_gate1 and run_gate3).
    backend
        LLM backend to use (Ollama default, Anthropic if --cloud).
    resume
        If True, pick up existing drafts for the same source in the pending
        batch file. Otherwise, if pending state exists, refuse.
    """
    _ = config  # reserved for future gate-2 options (retry strategy, etc.)
    _load_source_or_raise(repo, source_id)

    # Short-circuit: if gate 2 is already complete, return.
    if _gate2_complete(repo, source_id):
        return Gate2Result(
            source_id=source_id,
            approved_claim_ids=list(_claims_for_source(repo, source_id)),
            was_resumed=True,
        )

    # Detect pending state (drafts or batch file exist for this source).
    pending_batch_path = _pending_batch_path(repo, source_id)
    existing_drafts = list(_drafts_for_source(repo, source_id))

    if (pending_batch_path.exists() or existing_drafts) and not resume:
        raise GleanRepoError(
            f"partial gate-2 state exists for {source_id!r}: "
            f"{len(existing_drafts)} pending drafts, batch file "
            f"{'present' if pending_batch_path.exists() else 'absent'}. "
            f"Pass --resume to continue or delete the .claim.draft files to start fresh."
        )

    if resume and existing_drafts:
        # Resume path: use existing drafts; no fresh LLM call.
        draft_claims = existing_drafts
        call_log = None
    else:
        # Fresh extraction path: prompt, LLM call, parse.
        draft_claims, call_log = _extract_claims(source_id, repo=repo, backend=backend)
        for c in draft_claims:
            repo.save_claim_draft(c)

    # Build and review the pending batch file until valid (or all empty).
    approved, rejected_count, deferred_count = _review_pending_batch(
        source_id, draft_claims, repo=repo, editor=config.editor
    )

    # Promote approved drafts to .md; delete rejected drafts; leave deferred.
    approved_ids: list[str] = []
    for claim in approved:
        # Ensure draft file exists before promoting (it may already exist from above).
        draft_path = repo.claims_dir / f"{claim.id}.claim.draft"
        if not draft_path.exists():
            repo.save_claim_draft(claim)
        repo.promote_claim_draft(claim.id)
        approved_ids.append(claim.id)

    # Commit approved claims per AGENTS.md §5 whitelisted exception.
    commit_sha: str | None = None
    if approved_ids:
        git_add(
            repo.root,
            [f"claims/{cid}.md" for cid in approved_ids],
        )
        commit_sha = git_commit(
            repo.root,
            f"claims: add {len(approved_ids)} claims from {source_id}",
        )

    return Gate2Result(
        source_id=source_id,
        approved_claim_ids=approved_ids,
        rejected_count=rejected_count,
        deferred_count=deferred_count,
        commit_sha=commit_sha,
        was_resumed=resume and bool(existing_drafts),
        llm_call_log=call_log,
    )


# =============================================================================
# Claim extraction — prompt assembly + LLM call + parsing
# =============================================================================


def _extract_claims(source_id: str, *, repo: NotesRepo, backend: LLMBackend) -> tuple[list[Claim], LLMCallLog]:
    """Run the LLM to extract claim drafts. Returns (claims, call_log)."""
    source = repo.load_source(source_id)

    # Read the source content per AGENTS.md v0.2 §3.1 substrate rules.
    content = _load_source_substrate(repo, source_id)

    # Assemble context with priority budget.
    builder = ContextBuilder(budget_tokens=_CLAIM_PROMPT_BUDGET)
    builder.add(repo.load_agents_md(), priority=ContextPriority.REQUIRED, label="agents_md")
    builder.add(content, priority=ContextPriority.REQUIRED, label="source_content")

    index_md_text = ""
    if repo.index_md_path.is_file():
        index_md_text = repo.index_md_path.read_text()
    if index_md_text:
        builder.add(index_md_text, priority=ContextPriority.PREFERRED, label="index")

    existing_summary = _existing_claims_summary(repo, exclude_source=source_id)
    if existing_summary:
        builder.add(
            existing_summary,
            priority=ContextPriority.OPTIONAL,
            label="existing_claims",
        )

    rendered = builder.render()

    # Render the prompt template.
    template = load_prompt("claim_extract")
    source_yaml_str = yaml.safe_dump(
        source.model_dump(mode="json", exclude_defaults=False),
        sort_keys=False,
        default_flow_style=False,
    )

    # Extract year from the source for claim-id templating (LLM sees it).
    year = _year_for_source(source)
    source_slug = _source_slug_for(source_id)

    prompt = template.safe_substitute(
        year=year,
        source_slug=source_slug,
        source_id=source_id,
        extracted_date=date.today().isoformat(),
        agents_md=rendered.get("agents_md", ""),
        source_yaml=source_yaml_str,
        source_content=rendered.get("source_content", ""),
        existing_claims_summary=rendered.get("existing_claims", "(no prior claims in this repo)"),
    )

    response, call_log = backend.complete(prompt, tier=ModelTier.DEEP)
    claims = parse_llm_response(response, source_id=source_id, year=year)
    return claims, call_log


def parse_llm_response(text: str, *, source_id: str, year: int) -> list[Claim]:
    """Parse LLM output as YAML-blocks-separated-by-`---` into Claim drafts.

    Per D19: the LLM emits a short `slug:` field; GLEAN builds the final
    claim ID deterministically via claim_id_for(). If the LLM emits an `id`
    field instead, we accept it if valid; otherwise we construct from slug.
    """
    blocks = _split_yaml_blocks(text)
    if not blocks:
        raise GleanLLMError(
            "LLM response contained no parseable YAML blocks. Expected blocks separated by `---` lines per the prompt."
        )

    claims: list[Claim] = []
    errors: list[str] = []
    source_slug = _source_slug_for(source_id)

    for i, raw_block in enumerate(blocks):
        try:
            data = yaml.safe_load(raw_block)
        except yaml.YAMLError as e:
            errors.append(f"block {i}: YAML parse failed: {e}")
            continue
        if not isinstance(data, dict):
            errors.append(f"block {i}: not a YAML mapping (got {type(data).__name__})")
            continue

        # Resolve the claim ID: prefer explicit id if valid; else build from slug.
        claim_id = data.pop("id", None)
        slug = data.pop("slug", None)
        if isinstance(claim_id, str) and is_valid_claim_id(claim_id):
            resolved_id = claim_id
        elif isinstance(slug, str) and slug.strip():
            try:
                resolved_id = claim_id_for(year, source_slug, slugify(slug))
            except ValueError as e:
                errors.append(f"block {i}: slug {slug!r} produced invalid id: {e}")
                continue
        else:
            errors.append(f"block {i}: missing both `id:` and `slug:` fields")
            continue

        data["id"] = resolved_id
        data.setdefault("source", source_id)
        data.setdefault("status", "active")

        try:
            claim = Claim.model_validate(data)
        except ValidationError as e:
            errors.append(f"block {i} (id={resolved_id}): schema validation failed: {e}")
            continue

        claims.append(claim)

    if not claims:
        msg = "no valid claims parsed from LLM response"
        if errors:
            msg += "; errors:\n" + "\n".join(f"  - {e}" for e in errors[:5])
        raise GleanLLMError(msg)

    return claims


def _split_yaml_blocks(text: str) -> list[str]:
    """Split text on `---` markers (on their own line)."""
    # Strip any surrounding prose (LLMs sometimes wrap output in explanations).
    # A block is whatever sits between two `---` lines. We match liberally:
    # any line equal to exactly `---` (optional whitespace).
    lines = text.splitlines()
    blocks: list[list[str]] = [[]]
    for line in lines:
        if line.strip() == "---":
            blocks.append([])
        else:
            blocks[-1].append(line)
    # Collect non-empty blocks as strings.
    return ["\n".join(b).strip() for b in blocks if "\n".join(b).strip()]


# =============================================================================
# Existing-claims summary for cross-source citation (PLAN.md v2 open-q #5)
# =============================================================================


def _existing_claims_summary(repo: NotesRepo, *, exclude_source: str) -> str:
    """Produce a compact summary of existing rossum claims.

    Format: one line per claim as `<claim_id>: <claim_paraphrase>`.
    Claims from `exclude_source` are omitted (we don't want the LLM to
    re-cite its own extraction).
    """
    lines: list[str] = []
    for cid in repo.list_claims():
        try:
            claim, _body = repo.load_claim(cid)
        except GleanRepoError:
            continue
        if claim.source == exclude_source:
            continue
        lines.append(f"{cid}: {claim.claim}")
    return "\n".join(lines)


# =============================================================================
# Batch-review editor loop (D18)
# =============================================================================


def _review_pending_batch(
    source_id: str,
    drafts: list[Claim],
    *,
    repo: NotesRepo,
    editor: str,
) -> tuple[list[Claim], int, int]:
    """Open the pending batch file in `$EDITOR`; return (approved, rejected, deferred).

    The batch file is a single YAML-block stream, one block per draft,
    separated by `---`. User actions:
        - approve: leave block untouched
        - reject: delete the entire block (and its `---` separator)
        - edit: modify the block in place
        - defer: move the block under a line `# DEFERRED` (everything
          after this marker is treated as deferred)

    On save: parse, validate each block, resolve approved/deferred/rejected.
    If parse/validation fails, retry up to _BATCH_EDITOR_RETRIES times with
    inline `# !! ERROR:` markers.
    """
    batch_path = _pending_batch_path(repo, source_id)
    batch_content = _render_batch_file(drafts)
    batch_path.write_text(batch_content)

    try:
        last_error: str | None = None
        current_text = batch_content
        for _ in range(_BATCH_EDITOR_RETRIES):
            # Prepend error markers for retries.
            display_text = current_text
            if last_error:
                marker = (
                    f"# !! ERROR: batch review failed validation\n"
                    f"# !! ERROR: {last_error}\n"
                    f"# !! ERROR: fix the affected blocks and save again\n\n"
                )
                display_text = marker + _strip_batch_markers(current_text)

            # Use a tmp file with same content; let _open_in_editor handle $EDITOR.
            edited = _open_in_editor(display_text, editor, suffix=".yaml")
            edited_clean = _strip_batch_markers(edited)

            approved_drafts, deferred_drafts, rejected_count, parse_errors = _parse_batch_content(
                edited_clean, original_ids={c.id for c in drafts}
            )

            if parse_errors:
                last_error = "; ".join(parse_errors[:3])
                current_text = edited_clean  # preserve user's partial edits
                continue

            # Success — clean up and return.
            _apply_deferral_persistence(deferred_drafts, repo)
            _apply_rejection_cleanup(
                [c.id for c in drafts if c not in approved_drafts and c not in deferred_drafts],
                repo,
            )
            return approved_drafts, rejected_count, len(deferred_drafts)

        raise GleanRepoError(
            f"batch review failed validation after {_BATCH_EDITOR_RETRIES} retries. "
            f"Last error: {last_error}. "
            f"Pending file left at {batch_path} for manual inspection."
        )
    finally:
        # Clean up the batch file only on successful exit; the finally-block
        # runs after the success return and after exceptions.
        if batch_path.exists():
            batch_path.unlink()


_BATCH_DEFER_MARKER = "# DEFERRED"


def _render_batch_file(drafts: list[Claim]) -> str:
    """Render drafts as a YAML-blocks-with-separators file for editor review.

    Header comments intentionally avoid backticks and other characters YAML
    might mis-parse if the comment-skip heuristic fails: all guidance is
    plain ASCII prose.
    """
    parts: list[str] = [
        "# Batch review: edit, delete, or defer each block below.\n"
        "# - Approve: leave the block unchanged.\n"
        "# - Reject: delete the entire block (and its --- separator).\n"
        "# - Edit: modify fields in place.\n"
        "# - Defer: move the block below the DEFERRED line near the bottom.\n"
        "#   Deferred blocks stay as .claim.draft for the next session.\n",
    ]
    for c in drafts:
        block = yaml.safe_dump(
            c.model_dump(mode="json", exclude_defaults=False),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
        parts.append("---\n" + block.rstrip() + "\n")
    # Terminator so appending a DEFERRED section is easy visually.
    parts.append("---\n\n" + _BATCH_DEFER_MARKER + "\n# (move blocks below this line to defer)\n")
    return "\n".join(parts)


def _strip_batch_markers(text: str) -> str:
    """Remove `# !! ERROR:` lines from the batch text."""
    return "\n".join(line for line in text.splitlines() if not line.startswith("# !! ERROR:"))


def _parse_batch_content(text: str, *, original_ids: set[str]) -> tuple[list[Claim], list[Claim], int, list[str]]:
    """Parse the edited batch content.

    Returns (approved, deferred, rejected_count, parse_errors).
    - approved: Claim objects for blocks that validated and were NOT below the
      `# DEFERRED` marker.
    - deferred: Claim objects below the `# DEFERRED` marker.
    - rejected_count: len(original_ids) - len(approved) - len(deferred).
    - parse_errors: list of block-level error strings; non-empty → caller should retry.
    """
    # Partition the file into above-deferred and below-deferred halves.
    above, _marker_seen, below = text.partition(_BATCH_DEFER_MARKER)

    approved, approved_errors = _parse_blocks_as_claims(above)
    deferred, deferred_errors = _parse_blocks_as_claims(below)

    errors = approved_errors + deferred_errors
    seen_ids = {c.id for c in approved} | {c.id for c in deferred}
    rejected_count = len(original_ids - seen_ids)

    return approved, deferred, rejected_count, errors


def _parse_blocks_as_claims(text: str) -> tuple[list[Claim], list[str]]:
    """Parse `---`-separated YAML blocks; return (claims, errors).

    Blocks that are entirely comment lines (lines starting with `#` or empty)
    are treated as structural filler and silently skipped. Blocks that parse
    but produce a non-mapping (list, scalar, None) are also skipped silently.
    Blocks that look like they were INTENDED to be claim data but fail to
    parse/validate produce errors that trigger the retry loop.
    """
    claims: list[Claim] = []
    errors: list[str] = []
    for i, raw_block in enumerate(_split_yaml_blocks(text)):
        # Skip pure-comment / empty blocks silently.
        stripped_lines = [line for line in raw_block.splitlines() if line.strip() and not line.strip().startswith("#")]
        if not stripped_lines:
            continue
        try:
            data = yaml.safe_load(raw_block)
        except yaml.YAMLError as e:
            errors.append(f"block {i}: YAML parse failed: {e}")
            continue
        if not isinstance(data, dict):
            continue  # non-mapping YAML (list/scalar) — structural filler
        try:
            claims.append(Claim.model_validate(data))
        except ValidationError as e:
            errors.append(f"block {i}: schema validation failed: {e}")
    return claims, errors


def _apply_deferral_persistence(deferred: list[Claim], repo: NotesRepo) -> None:
    """Ensure deferred drafts remain on disk as `.claim.draft`."""
    for claim in deferred:
        draft_path = repo.claims_dir / f"{claim.id}.claim.draft"
        if not draft_path.exists():
            # Save a fresh copy reflecting any edits the user made.
            repo.save_claim_draft(claim)


def _apply_rejection_cleanup(rejected_ids: list[str], repo: NotesRepo) -> None:
    """Delete `.claim.draft` files for claims the user rejected."""
    for cid in rejected_ids:
        draft_path = repo.claims_dir / f"{cid}.claim.draft"
        if draft_path.exists():
            draft_path.unlink()


# =============================================================================
# Resume / state checks (D12: filesystem-is-state)
# =============================================================================


def _pending_batch_path(repo: NotesRepo, source_id: str) -> Path:
    """Path to the pending batch file for a given source."""
    return repo.claims_dir / f"_pending_{source_id}.yaml"


def _drafts_for_source(repo: NotesRepo, source_id: str) -> list[Claim]:
    """Return the existing .claim.draft entries belonging to `source_id`."""
    out: list[Claim] = []
    for path in sorted(repo.claims_dir.glob("*.claim.draft")):
        try:
            text = path.read_text()
            # Parse frontmatter block.
            frontmatter_parts = text.split("---\n", 2)
            if len(frontmatter_parts) < 3:
                continue
            data = yaml.safe_load(frontmatter_parts[1])
            if not isinstance(data, dict):
                continue
            if data.get("source") != source_id:
                continue
            out.append(Claim.model_validate(data))
        except (yaml.YAMLError, ValidationError, OSError):
            continue
    return out


def _claims_for_source(repo: NotesRepo, source_id: str) -> list[str]:
    """Return approved claim IDs belonging to `source_id`."""
    out: list[str] = []
    for cid in repo.list_claims():
        try:
            claim, _body = repo.load_claim(cid)
        except GleanRepoError:
            continue
        if claim.source == source_id:
            out.append(cid)
    return out


def _gate2_complete(repo: NotesRepo, source_id: str) -> bool:
    """True if approved claims for this source already exist AND are committed.

    Relaxed check: if any approved claim for this source exists in `claims/`,
    we treat gate 2 as complete. Users who want to re-extract must remove the
    approved claims manually and re-invoke — consistent with D12's
    filesystem-is-state discipline.
    """
    return bool(_claims_for_source(repo, source_id))


# =============================================================================
# Helpers
# =============================================================================


def _load_source_or_raise(repo: NotesRepo, source_id: str) -> None:
    """Verify the source exists (gate 1 completed). Raises if not."""
    try:
        repo.load_source(source_id)
    except GleanRepoError as e:
        raise GleanRepoError(
            f"cannot run gate 2 for {source_id!r}: source not found. "
            f"Run gate 1 first via `glean ingest`. Underlying: {e}"
        ) from e


def _source_slug_for(source_id: str) -> str:
    """Extract the 'body' of a source ID for use in claim-id construction.

    Strips the type prefix (`paper_`, `sim_`, `note_`, `repo_`, `comm_`,
    `web_`, etc.) and any numeric segments — year (4 digits), month (2
    digits), day (2 digits) — that appear as part of the standardized
    date fields in ID patterns per AGENTS.md v0.2 §2.3.

    Examples:
        paper_zaghi_2023_amr_gpu_ibm  -> zaghi_amr_gpu_ibm
        sim_2026_04_prism_rmf_restart -> prism_rmf_restart
        repo_szaghi_adam_dbe47a44     -> szaghi_adam_dbe47a44  (hex preserved)
        note_2026_04_23_extending_amr -> extending_amr

    This is a heuristic — v0.2 may introduce an explicit `claim_slug_base`
    field per source.
    """
    parts = source_id.split("_")
    if len(parts) < 2:
        return source_id
    # Drop the type prefix (first segment).
    rest = parts[1:]
    # Drop all purely-numeric segments anywhere in the ID — these are date
    # components (year, month, day) per §2.3 patterns; they repeat in the
    # claim ID's own year prefix and would be noise.
    return "_".join(p for p in rest if not p.isdigit())


def _year_for_source(source: object) -> int:
    """Best-effort extraction of the year from a source object.

    Different source types carry year under different fields; we try the
    common ones and fall back to the current year.
    """
    for attr in ("year", "date", "run_date", "commit_date", "archived_at"):
        value = getattr(source, attr, None)
        if isinstance(value, int) and 1000 <= value <= 9999:
            return value
        if isinstance(value, date):
            return value.year
    return date.today().year


def _load_source_substrate(repo: NotesRepo, source_id: str) -> str:
    """Load the claim-extraction substrate for a source per AGENTS.md v0.2 §3.1.

    - Paper: `sources/<id>/paper.md` (extracted markdown)
    - Simulation: `sources/<id>/output_summary.md` (authored narrative)
    - Notebook: `notebook/<file>.md` body
    - Repository, dataset, etc.: not yet supported at v0.1 M3
    """
    source = repo.load_source(source_id)
    if source.type == SourceType.PAPER:
        path = repo.sources_dir / source_id / "paper.md"
    elif source.type == SourceType.SIMULATION:
        # SimulationSource has an output_summary field pointing at the filename.
        summary_name = getattr(source, "output_summary", "output_summary.md")
        path = repo.sources_dir / source_id / summary_name
    elif source.type == SourceType.NOTEBOOK:
        # Scan notebook/ for the file whose frontmatter id matches.
        for candidate in repo.notebook_dir.glob("*.md"):
            try:
                text = candidate.read_text()
                fm_parts = text.split("---\n", 2)
                if len(fm_parts) >= 3:
                    fm = yaml.safe_load(fm_parts[1])
                    if isinstance(fm, dict) and fm.get("id") == source_id:
                        # Return just the body for notebook entries.
                        return fm_parts[2]
            except (yaml.YAMLError, OSError):
                continue
        raise GleanRepoError(f"notebook entry for {source_id!r} not found under notebook/")
    else:
        raise GleanRepoError(f"substrate loading not implemented for source type {source.type.value!r} at v0.1")

    if not path.is_file():
        raise GleanRepoError(f"substrate file not found for {source_id!r}: {path}")
    return path.read_text()

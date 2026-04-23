"""Top-level ingest orchestrator (M3d).

Wires the three gates into a single `glean ingest` entry point, with
--resume, --abort, --type, --cloud, --pdf-extractor, --no-network flags.

The flow:

    1. Build IngestInput from CLI args.
    2. Run gate 1 (source metadata + $EDITOR confirm + commit source).
       If --resume and gate 1 already complete, skip.
    3. Run gate 2 (claim extraction + batch review + commit claims).
       If the source is a notebook, skip gate 2 at v0.1 (notebook claims
       come from a later `glean promote notebook/<id>` command, deferred
       to v0.2 per PLAN.md v2).
    4. Run gate 3 (wiki updates + log entry; leave diffs in working tree).
    5. Print review instructions pointing the user at `git diff wiki/`.

--abort <source_id> bypasses the full flow and clears uncommitted gate-1
state per D27.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from glean.adapters import IngestInput
from glean.config import load_config
from glean.enums import SourceType
from glean.errors import GleanError
from glean.ingest_gate1 import Gate1Result, abort_gate1, run_gate1
from glean.ingest_gate2 import Gate2Result, run_gate2
from glean.ingest_gate3 import Gate3Result, run_gate3
from glean.llm import get_backend
from glean.repo import NotesRepo

log = logging.getLogger(__name__)


def ingest_command(
    *,
    source: str,
    repo_path: Path,
    source_type: SourceType | None = None,
    cloud: bool = False,
    pdf_extractor: str = "marker",
    offline: bool = False,
    resume: bool = False,
    confirm_type: bool = True,
) -> None:
    """Run the full ingest flow against a rossum repo.

    Errors from any gate propagate as GleanError subclasses; callers should
    catch them at the CLI boundary and emit user-friendly messages.
    """
    repo = NotesRepo(repo_path)
    config = load_config()
    backend = get_backend(config, cloud=cloud)

    inp = IngestInput(
        input_spec=source,
        type_override=source_type,
        offline=offline,
        pdf_extractor=pdf_extractor,
    )

    # --- Gate 1 ---
    g1 = run_gate1(
        inp,
        repo=repo,
        config=config,
        resume=resume,
        confirm_type=confirm_type,
    )
    gate1_bullets = _gate1_bullets(g1)

    # Notebook sources don't flow through gate 2/3 in the same way: the user
    # authored the entry, self-critique ran in gate 1. Claim extraction from
    # a notebook is a separate `glean promote` command, deferred to v0.2.
    if g1.source_type == SourceType.NOTEBOOK:
        _print_notebook_ingest_summary(g1)
        return

    # --- Gate 2 ---
    g2 = run_gate2(
        g1.source_id,
        repo=repo,
        config=config,
        backend=backend,
        resume=resume,
    )
    gate2_bullets = _gate2_bullets(g2)

    # --- Gate 3 ---
    g3 = run_gate3(
        g1.source_id,
        repo=repo,
        config=config,
        backend=backend,
        approved_claim_ids=g2.approved_claim_ids,
        gate1_sub_bullets=gate1_bullets,
        gate2_sub_bullets=gate2_bullets,
    )

    _print_ingest_summary(g1, g2, g3, repo_path=repo.root)


def abort_command(*, source_id: str, repo_path: Path) -> None:
    """Clear uncommitted gate-1 state per D27."""
    repo = NotesRepo(repo_path)
    abort_gate1(repo, source_id)


# =============================================================================
# Log-bullet builders (for gate 3's per-ingest log entry)
# =============================================================================


def _gate1_bullets(g1: Gate1Result) -> list[str]:
    bullets: list[str] = [f"- source_id: {g1.source_id} (type={g1.source_type.value})"]
    if g1.was_resumed:
        bullets.append("- resumed: gate 1 was already complete")
    elif g1.commit_sha:
        bullets.append(f"- committed: {g1.commit_sha[:12]}")
    return bullets


def _gate2_bullets(g2: Gate2Result) -> list[str]:
    bullets: list[str] = [
        f"- claims: {len(g2.approved_claim_ids)} approved / "
        f"{g2.rejected_count} rejected / {g2.deferred_count} deferred",
    ]
    if g2.was_resumed:
        bullets.append("- resumed: picked up existing drafts")
    if g2.commit_sha:
        bullets.append(f"- committed: {g2.commit_sha[:12]}")
    if g2.llm_call_log is not None:
        bullets.extend(f"  {b}" for b in g2.llm_call_log.to_log_bullets())
    return bullets


# =============================================================================
# Terminal summary printing
# =============================================================================


def _print_notebook_ingest_summary(g1: Gate1Result) -> None:
    sys.stdout.write(
        f"\n=== Notebook ingest complete ===\n"
        f"Source: {g1.source_id}\n"
        f"Self-critique: run.\n"
        f"\nNotebook claim extraction is a separate command (v0.2+). "
        f"The notebook entry remains at its original path.\n"
    )


def _print_ingest_summary(g1: Gate1Result, g2: Gate2Result, g3: Gate3Result, *, repo_path: Path) -> None:
    created = ", ".join(g3.wiki_pages_created) if g3.wiki_pages_created else "(none)"
    updated = ", ".join(g3.wiki_pages_updated) if g3.wiki_pages_updated else "(none)"
    sys.stdout.write(
        f"\n=== Ingest summary ===\n"
        f"Source:   {g1.source_id} ({g1.source_type.value})\n"
        f"Gate 1:   committed {g1.commit_sha[:12] if g1.commit_sha else '(resumed)'}\n"
        f"Gate 2:   {len(g2.approved_claim_ids)} claims approved, "
        f"{g2.rejected_count} rejected, {g2.deferred_count} deferred\n"
        f"Gate 3:   wiki pages created: {created}\n"
        f"          wiki pages updated: {updated}\n"
        f"Log:      entry appended to wiki/log.md\n"
        f"\n"
        f"Gate-3 changes are in your working tree (uncommitted) per AGENTS.md §5.\n"
        f"Review them with:\n"
        f"    git -C {repo_path} diff wiki/\n"
        f"When you're satisfied, commit with:\n"
        f"    git -C {repo_path} add wiki/\n"
        f"    git -C {repo_path} commit -m 'wiki: ingest {g1.source_id}'\n"
    )


# =============================================================================
# Exception-to-CLI boundary helper
# =============================================================================


def run_cli_ingest(
    *,
    source: str,
    repo_path: Path,
    source_type: SourceType | None = None,
    cloud: bool = False,
    pdf_extractor: str = "marker",
    offline: bool = False,
    resume: bool = False,
) -> int:
    """CLI wrapper: run ingest, catch GleanError, emit human-readable message.

    Returns exit code: 0 success, 1 user/repo error, 2 LLM error.
    """
    from glean.errors import GleanLLMError

    try:
        ingest_command(
            source=source,
            repo_path=repo_path,
            source_type=source_type,
            cloud=cloud,
            pdf_extractor=pdf_extractor,
            offline=offline,
            resume=resume,
        )
        return 0
    except GleanLLMError as e:
        sys.stderr.write(f"LLM error: {e}\n")
        return 2
    except GleanError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted. Filesystem state preserved per D23 — re-run with --resume to continue.\n")
        return 130

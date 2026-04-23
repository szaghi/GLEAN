"""Query command: read the wiki + claims, synthesize a cited answer (M5).

The two-call flow per D31:

    1. Fast-tier call: given the question + page catalog, select relevant
       page IDs.
    2. Deep-tier call: given the question + those pages + the claims they
       cite, produce a cited answer.

File-back is deferred to v0.2 per D32. M5 only prints the answer.

Error surface: if Ollama is unreachable or the LLM returns nothing
parseable, raise GleanLLMError. Exit 2 at the CLI boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from glean.errors import GleanLLMError, GleanRepoError
from glean.llm import (
    ContextBuilder,
    ContextPriority,
    LLMBackend,
    LLMCallLog,
    ModelTier,
    load_prompt,
)
from glean.repo import NotesRepo
from glean.schema import Claim, WikiPage

# Token budget for the synthesis call. Larger than gate-3 because wiki pages
# and their claims can be hefty; the response itself is usually short.
_QUERY_PROMPT_BUDGET = 14000

# Sentinel the LLM uses when no page is relevant.
_NO_RELEVANT_PAGES = "NO_RELEVANT_PAGES"

# Pattern to extract claim citations from wiki page bodies (same as lint).
_CITATION_RE = re.compile(r"\[\[(claim_[a-z0-9_]+)\]\]")


@dataclass
class QueryResult:
    """What a successful query produces."""

    question: str
    selected_page_ids: list[str] = field(default_factory=list)
    answer: str = ""
    select_call_log: LLMCallLog | None = None
    synthesize_call_log: LLMCallLog | None = None


# =============================================================================
# Main entry
# =============================================================================


def run_query(
    question: str,
    *,
    repo: NotesRepo,
    backend: LLMBackend,
) -> QueryResult:
    """Answer `question` against the rossum wiki. Returns a QueryResult.

    Raises GleanLLMError if the LLM fails or produces unparseable output.
    Raises GleanRepoError if the repo is empty / malformed.
    """
    question = question.strip()
    if not question:
        raise GleanRepoError("query: empty question")

    # --- Step 1: page selection ---
    catalog = _build_page_catalog(repo)
    if not catalog:
        raise GleanRepoError("no wiki pages in this repo; `glean query` requires at least one page")

    selected_ids, select_log = _select_pages(
        question=question,
        page_catalog=catalog,
        backend=backend,
    )

    if not selected_ids:
        # Either the LLM returned NO_RELEVANT_PAGES, or the intersection with
        # the actual catalog was empty. Short-circuit with an honest answer.
        return QueryResult(
            question=question,
            selected_page_ids=[],
            answer=(
                "The wiki does not contain any pages relevant to this question. "
                "Consider ingesting a source that addresses it."
            ),
            select_call_log=select_log,
        )

    # --- Step 2: synthesize ---
    answer, synth_log = _synthesize_answer(
        question=question,
        selected_page_ids=selected_ids,
        repo=repo,
        backend=backend,
    )

    return QueryResult(
        question=question,
        selected_page_ids=selected_ids,
        answer=answer,
        select_call_log=select_log,
        synthesize_call_log=synth_log,
    )


# =============================================================================
# Step 1: page selection (fast tier)
# =============================================================================


def _build_page_catalog(repo: NotesRepo) -> list[tuple[str, WikiPage]]:
    """Return a list of (page_id, WikiPage) for every regular wiki page."""
    out: list[tuple[str, WikiPage]] = []
    for pid in repo.list_wiki_pages():
        try:
            page, _body = repo.load_wiki_page(pid)
        except GleanRepoError:
            continue
        out.append((pid, page))
    return out


def _render_catalog(catalog: list[tuple[str, WikiPage]]) -> str:
    """Format the page catalog as `id | kind | title | tag1,tag2` lines."""
    lines: list[str] = []
    for pid, page in catalog:
        tags = ",".join(page.tags) if page.tags else "-"
        lines.append(f"{pid} | {page.kind.value} | {page.title} | {tags}")
    return "\n".join(lines)


def _select_pages(
    *,
    question: str,
    page_catalog: list[tuple[str, WikiPage]],
    backend: LLMBackend,
) -> tuple[list[str], LLMCallLog]:
    """Fast-tier LLM call: pick relevant page IDs from the catalog."""
    template = load_prompt("query_select_pages")
    prompt = template.safe_substitute(
        question=question,
        page_catalog=_render_catalog(page_catalog),
    )
    response, call_log = backend.complete(prompt, tier=ModelTier.FAST)

    if response.strip() == _NO_RELEVANT_PAGES:
        return [], call_log

    # Parse: one page ID per line. Tolerate preamble / numbering / bullets.
    valid_ids = {pid for pid, _ in page_catalog}
    selected: list[str] = []
    seen: set[str] = set()
    for raw_line in response.splitlines():
        candidate = raw_line.strip()
        # Strip common list prefixes the LLM may add despite instructions.
        candidate = candidate.lstrip("-*0123456789.) \t")
        if not candidate or candidate.startswith("#"):
            continue
        if candidate in valid_ids and candidate not in seen:
            selected.append(candidate)
            seen.add(candidate)

    return selected, call_log


# =============================================================================
# Step 2: synthesis (deep tier)
# =============================================================================


def _synthesize_answer(
    *,
    question: str,
    selected_page_ids: list[str],
    repo: NotesRepo,
    backend: LLMBackend,
) -> tuple[str, LLMCallLog]:
    """Deep-tier LLM call: produce a cited answer from the selected pages."""
    # Load pages + their cited claims.
    pages_text, claim_ids = _load_pages_text(selected_page_ids, repo=repo)
    claims_text = _load_claims_text(claim_ids, repo=repo)

    builder = ContextBuilder(budget_tokens=_QUERY_PROMPT_BUDGET)
    builder.add(pages_text, priority=ContextPriority.REQUIRED, label="wiki_pages")
    if claims_text:
        builder.add(claims_text, priority=ContextPriority.REQUIRED, label="claims")
    rendered = builder.render()

    template = load_prompt("query_synthesize")
    prompt = template.safe_substitute(
        question=question,
        wiki_pages=rendered.get("wiki_pages", ""),
        claims=rendered.get("claims", "(no claims cited by selected pages)"),
    )
    response, call_log = backend.complete(prompt, tier=ModelTier.DEEP)
    if not response.strip():
        raise GleanLLMError("synthesis LLM call returned empty response")
    return response.strip(), call_log


def _load_pages_text(page_ids: list[str], *, repo: NotesRepo) -> tuple[str, list[str]]:
    """Return (rendered_pages, claim_ids_cited_by_those_pages)."""
    chunks: list[str] = []
    claim_ids: list[str] = []
    seen: set[str] = set()
    for pid in page_ids:
        try:
            page, body = repo.load_wiki_page(pid)
        except GleanRepoError:
            continue
        header = f"===== {pid} | {page.kind.value} | {page.title} =====\n"
        chunks.append(header + body.rstrip() + "\n")
        for m in _CITATION_RE.finditer(body):
            cid = m.group(1)
            if cid not in seen:
                claim_ids.append(cid)
                seen.add(cid)
    return "\n".join(chunks), claim_ids


def _load_claims_text(claim_ids: list[str], *, repo: NotesRepo) -> str:
    """Render claims as id | confidence | paraphrase lines."""
    lines: list[str] = []
    for cid in claim_ids:
        try:
            claim, _body = repo.load_claim(cid)
        except GleanRepoError:
            continue
        lines.append(_format_claim_line(cid, claim))
    return "\n".join(lines)


def _format_claim_line(cid: str, claim: Claim) -> str:
    """Compact one-line summary of a claim suitable for prompt context."""
    # claim.claim may be multi-line prose; collapse whitespace for the prompt.
    paraphrase = " ".join(claim.claim.split())
    if len(paraphrase) > 240:
        paraphrase = paraphrase[:237] + "..."
    return f"{cid} | {claim.confidence.value} | {paraphrase}"


# =============================================================================
# CLI wrapper (exit-code boundary)
# =============================================================================


def run_cli_query(
    question: str,
    *,
    repo: NotesRepo,
    backend: LLMBackend,
) -> int:
    """Run query + print; return CLI exit code. 0 success, 1 repo err, 2 LLM err."""
    import sys

    try:
        result = run_query(question, repo=repo, backend=backend)
    except GleanLLMError as e:
        sys.stderr.write(f"LLM error: {e}\n")
        return 2
    except GleanRepoError as e:
        sys.stderr.write(f"Repo error: {e}\n")
        return 1
    sys.stdout.write(result.answer + "\n")
    return 0

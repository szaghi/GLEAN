# GLEAN v0.1 — Implementation Plan

**Revision status:** plan v2, 2026-04-23. Revised after Phase 1 of `rossum/` bring-up produced AGENTS.md v0.2 and 13 schema findings. Previous plan version assumed AGENTS.md v0.1-draft; this revision reflects what we actually learned from manually ingesting a paper, a simulation, and a notebook entry.

This plan defines the work from the current skeleton to a v0.1 that can ingest a real source into `rossum/` end to end, under Stefano's three human gates, using a local Ollama backend — with the `rossum` schema as AGENTS.md v0.2 (or later), not the pre-Phase-1 draft.

The plan is **incremental and testable at every milestone**. Each milestone leaves GLEAN in a usable state for the scope it covers; nothing is merged half-finished.

---

## Guiding design principles

1. **Git is the database.** No SQLite, no vector store. Every persistent artifact is a markdown or YAML file. The working tree + commit history are the data model + audit log.
2. **Human gates are first-class.** The three gates (source metadata confirm, claim drafts approve, wiki diffs review) are the product. Build them properly before building anything clever.
3. **Schema before code.** `rossum/AGENTS.md` is the authoritative specification. GLEAN's code must validate against it; when they disagree, AGENTS.md wins and the code changes.
4. **LLM as a component, not the core.** The core is file I/O, schema validation, and diff presentation. The LLM is a callable that produces draft content for human review. Keep the LLM's failure surface small and detectable.
5. **No agentic loops at v0.1.** No tool use from the LLM side. Single-turn prompts. Parse the response; act deterministically. If it breaks, make it break loudly.
6. **Ship v0.1 when it works on one real paper end to end.** Not before, not after.
7. **The three gates as Phase 1 performed them are the reference behavior.** Phase 1 was manual LLM-as-agent execution of AGENTS.md by Claude Code. GLEAN's CLI must reproduce the *output* of that process (same source layout, same claims, same wiki diffs) at substantially lower wall-clock cost. If GLEAN produces different output than the manual reference, GLEAN is wrong — not the reference.

---

## v0.1 target behavior

After v0.1 Stefano can run:

```bash
cd ~/rossum
glean init                                # already-bootstrapped repo: idempotent no-op
glean ingest ~/Downloads/smith2024.pdf    # paper
glean ingest ~/runs/chimera_mach05/       # simulation dir
glean ingest notebook/roe_vs_hllc.md      # own notebook entry
glean lint                                # produces health report
glean query "what does the wiki say about Roe vs HLLC at high Mach?"
```

Each command respects the three human gates where applicable. Every LLM call is local (Ollama). `--cloud` flag opens the cloud backend per-invocation.

---

## Milestones

### M0 — Repo hygiene (0.5 day)

Already done by scaffolding and the first commit `373750e`.

To finish M0:
- [x] First commit
- [ ] `make dev` creates the .venv and installs successfully
- [ ] `make test` runs the smoke tests and passes
- [ ] `make lint` passes

**Deferred from M0 to M3:** CI workflow. Setting up GitHub Actions before the core modules are written slows the first iteration without catching anything `make test` doesn't catch. Add CI when M3 exit-criterion tests exist and the ingest pipeline is real.

**Exit criterion:** `make dev test lint` all green on a fresh clone.

---

### M1 — Schema models and config (3–4 days)

Implement the data model and configuration loader. **Effort revised upward from 1.5 days.** AGENTS.md v0.2 has substantially more schema surface than v0.1-draft: 11 source types with type-discriminated required fields, the `generation_spec` alternative for simulations, the `run:` and `external_refs:` claim frontmatter fields, the `meta` tag convention, and the citation-rule-by-page-kind distinction all need validators. The model surface is ~2× v0.1-draft.

**Files to implement:**

- `src/glean/schema.py` — Pydantic models:
  - `SourceManifest` as a **discriminated union** on the `type:` field, with one concrete class per source type (`PaperSource`, `RepositorySource`, `SimulationSource`, etc.). Type-specific required/optional fields per AGENTS.md v0.2 §2.2 table.
  - `SimulationSource` carries the `input_files` XOR `generation_spec` alternation (at least one must be present).
  - `RepositorySource` includes optional `commit_date`, `commit_subject`, `local_clone`.
  - `Claim` with frontmatter fields including optional `run:`, `external_refs:` (list of structured entries), and `tags:` (where `meta` is recognized as a special value).
  - `WikiPage` with `kind` enum (`entity | concept | method | synthesis | comparison`). The citation-rule-by-kind distinction is enforced at lint time (M4), not at schema time.
  - `NotebookEntry` — frontmatter only; body is free-form markdown per AGENTS.md v0.2 §2.4.
  - `LogEntry` — the grep-parseable log format.
  - Enum types for `SourceType`, `Confidence` (with v0.2's expanded meanings — `author_hypothesis` and `author_reasoning` are semantically distinct per §2.5), `ClaimStatus`, `WikiKind`.
  - Custom validators for ID format regex (one per type, per §2.3).

- `src/glean/config.py` — Typed config loader:
  - Reads `~/.config/glean/config.toml`.
  - Fields: Ollama endpoint URL, model tier mapping (fast/deep), optional Anthropic API key, default notes repo path, editor command.
  - Dumps a starter config on first run (`glean init` writes it if absent).

- `src/glean/ids.py` — ID construction and validation:
  - `source_id_for(type: SourceType, **kwargs) -> str` (deterministic).
  - `claim_id_for(source_id: str, assertion_slug: str) -> str`.
  - `notebook_id_for(date: date, slug: str) -> str`.
  - `is_valid_<kind>_id(s: str) -> bool` — one per ID namespace.

**Tests:**
- `test_schema_source.py` — one parametrized test per source type, happy path + at least 2 failure modes per type; includes the `input_files` XOR `generation_spec` case for simulations.
- `test_schema_claim.py` — happy path, invalid citation format, invalid confidence, meta-tag handling, `run:` and `external_refs:` fields.
- `test_schema_notebook.py` — frontmatter happy path, missing fields, free-form body accepted.
- `test_ids.py` — deterministic ID generation, collision detection, invalid input rejection.
- `test_config.py` — loads a sample config; falls back to defaults; rejects malformed.

**Exit criterion:** `glean version` still works; Pydantic models round-trip **every real artifact from rossum** (all 4 sources, all 31 claims, all 11 wiki pages as of rossum commit `eed0a80`). This is the strongest possible correctness test — if a v0.1-valid claim from Phase 1 doesn't round-trip through the v0.2 schema, one of the two is wrong and we investigate before moving on.

---

### M2 — LLM backend + repo I/O primitives (2–3 days)

Build the low-level plumbing: LLM calls, file I/O, git wrapper.

**Files to implement:**

- `src/glean/llm.py`:
  - `LLMBackend` (abstract protocol): `complete(prompt: str, tier: Literal["fast","deep"]) -> str`
  - `OllamaBackend` using `httpx` against `/api/generate` (or `/api/chat`). Streaming optional for v0.1; synchronous is fine.
  - `AnthropicBackend` using `anthropic` SDK (opt-in, behind `cloud` extra).
  - `get_backend(cloud: bool, config: Config) -> LLMBackend` — factory.
  - Prompt templates live in `src/glean/prompts/` as plain text files with `{placeholders}`. One per operation. Keep them short and testable.
  - Every call logs: `(timestamp, backend, model, prompt_hash, response_hash)` appended to `wiki/log.md` under the active operation's entry. Redaction happens here for cloud calls per AGENTS.md §7.

- `src/glean/repo.py` (new module) — Repo I/O primitives:
  - `class NotesRepo`: constructed with a path; validates it's a GLEAN repo (has AGENTS.md + four layer dirs).
  - `load_source(id) -> Source`, `save_source(src)`, `list_sources()`
  - `load_claim(id) -> Claim`, `save_claim_draft(c)`, `promote_claim_draft(id)`, `list_claims()`
  - `load_wiki_page(id) -> WikiPage`, `save_wiki_page(p)`, `list_wiki_pages()`
  - `append_log(entry: LogEntry)`
  - Every write method is atomic (write to temp, fsync, rename). No partial writes on crash.

- `src/glean/git.py` (new module) — thin wrapper around `subprocess` calls to `git`:
  - `status()`, `diff(paths)`, `add(paths)`, `commit(message)`, `is_clean()`
  - Never runs `commit` or `push` from inside ingest flows. Only `status` + `diff` for read, and explicit `add` only for confirmed source directories.

**Tests:**
- `test_llm_ollama.py` — mock `httpx` client; verify request shape and response parsing. No real network.
- `test_llm_anthropic.py` — mock the SDK; verify redaction happens.
- `test_repo_io.py` — against the `empty_notes_repo` fixture; round-trip sources, claims, wiki pages.
- `test_git_wrapper.py` — against a tmp git repo; verify read-only ops; verify no implicit commits.

**Exit criterion:** `glean` can read and write every layer of a rossum repo programmatically and make real Ollama calls. `make test` passes with both real and mocked Ollama (pick by env var).

---

### M3 — Ingest pipeline (the core of v0.1; 4–6 days)

This is where the three human gates are implemented. Everything up to now has been infrastructure. **Effort revised upward from 3–5 days.** AGENTS.md v0.2 added two requirements the original plan did not anticipate: the notebook-specific author-side self-critique pass before gate 1 (§3.1 v0.2), and the mandatory `output_summary.md` precondition for simulation gate 2 (§3.1 v0.2). Both are small in code but each is its own interactive flow.

**3a. Source adapters** — one per type in `src/glean/sources/`:

Each adapter implements:
```python
class SourceAdapter(Protocol):
    def can_handle(self, input: str) -> bool: ...  # path or URL
    def extract_metadata(self, input: str) -> dict: ...
    def extract_content(self, input: str, dest: Path) -> None: ...  # writes artifacts
    def default_id(self, metadata: dict) -> str: ...
```

Build in this order (reflects Phase 1's actual type frequencies: paper, simulation, notebook are the tested types; others deferred until exercised):
1. `PaperAdapter` (PDF → marker markdown when two-column layout, else pymupdf4llm; DOI lookup via crossref.org; manual confirm). **Phase 1 lesson:** pymupdf4llm fails on Elsevier two-column layouts; marker is the default for scientific papers even though it's heavier (~2 GB first-run download).
2. `NotebookAdapter` (already-written markdown file → source; no extraction needed, just validate frontmatter + file the entry at `notebook/<slug>.md`). **Does NOT create a `sources/note_<id>/` stub** per AGENTS.md v0.2 §2.1 and §2.4.
3. `SimulationAdapter` (directory with `input.ini` or equivalent; requires author-written `output_summary.md` at gate 1; requires either `input_files` list or `generation_spec` subobject per v0.2 §2.2). Also handles the companion `RepositoryAdapter` creation when `solver_repo_id` references a repo not yet in rossum.
4. `WebArticleAdapter` (URL → BeautifulSoup + markdownify → markdown; optional archived snapshot).
5. Others (`repository` standalone, `dataset`, `talk`, `book`, `standard`, `personal_comm`, `preprint`) — stubs that raise `NotImplementedError` with a pointer to the schema section. Don't speculatively implement what Phase 1 didn't exercise.

**3b. Human gate 1 — source.yaml confirm**

In `src/glean/ingest.py`:
```python
def ingest_gate_1_source(input: str, config: Config, repo: NotesRepo) -> Source:
    adapter = pick_adapter(input)
    meta = adapter.extract_metadata(input)
    draft_yaml = render_source_yaml(meta)  # string
    confirmed = human_confirm_editor(draft_yaml)  # opens $EDITOR; on save, reload + validate
    src = SourceManifest.model_validate_strings(confirmed)
    adapter.extract_content(input, repo.sources_dir / src.id)
    (repo.sources_dir / src.id / "source.yaml").write_text(src.model_dump_yaml())

    # Gate 1 preconditions per AGENTS.md v0.2 §3.1:
    if isinstance(src, SimulationSource):
        ensure_output_summary_md(repo.sources_dir / src.id)  # open in $EDITOR if missing
    if isinstance(src, NotebookSource):
        run_author_self_critique(repo.notebook_dir / f"{src.id}.md", config)

    return src
```

`human_confirm_editor` uses `$EDITOR` (or `vi` fallback). A TUI is explicitly out of scope for v0.1 — `$EDITOR` is the review surface, consistent with using `git diff` later.

**Two v0.2-mandated gate-1 extensions:**

- **`ensure_output_summary_md`** — if the simulation source directory has no `output_summary.md` after artifact extraction, write a template skeleton (mirroring the one used in Phase 1 source 2) and open it in `$EDITOR`. The ingest cannot proceed to gate 2 until the file exists and is non-trivially filled. Detect "non-trivially filled" by checking that no `<FILL>` markers remain and that the file is >200 words. Soft heuristic, not a hard rule.
- **`run_author_self_critique`** — for notebook sources, invoke a dedicated `deep`-tier LLM call that reads the notebook entry and produces a critique of overstatement, unsupported steel-manning, implicit conclusions the prose doesn't actually derive. Print the critique to the terminal. Prompt: "fix / acknowledge in frontmatter / proceed as-is." This replaces, for notebook sources, the external-author check that papers and simulations receive automatically.

**3c. Claim extraction + Human gate 2**

```python
def ingest_gate_2_claims(src: Source, backend: LLMBackend, repo: NotesRepo) -> list[Claim]:
    content = repo.load_source_content(src.id)
    prompt = prompts.load("claim_extract").format(
        agents_md=repo.load_agents_md(),  # feed the schema itself
        source_yaml=src.model_dump_yaml(),
        content=content,
    )
    raw = backend.complete(prompt, tier="deep")
    draft_claims = parse_claims_response(raw)  # expect JSON-lines output; validate each
    for c in draft_claims:
        repo.save_claim_draft(c)
    approved = human_approve_drafts(draft_claims, repo)
    for c in approved:
        repo.promote_claim_draft(c.id)
    return approved
```

`human_approve_drafts` — **decision locked: `$EDITOR`-on-YAML-batch** (Phase-1-style, option (a) from the original open question). After draft extraction, GLEAN writes all drafts as one combined YAML file `claims/_pending_<source_id>.yaml` containing every draft in sequence, opens it in `$EDITOR`, and waits for save-and-close. The human:

- **approves** a draft by leaving it in the file
- **rejects** a draft by deleting its block
- **edits** a draft in place
- **defers** a draft by moving its block under a `# DEFERRED` comment separator (stays in the pending file for next session)

On save, GLEAN parses, validates each surviving block, writes approved drafts as `claims/<claim_id>.md`, writes deferred drafts back as `.claim.draft` (gitignored), and deletes `_pending_<source_id>.yaml`.

**Why this over per-claim terminal prompts:** Phase 1 showed that a 20-claim paper has mostly approve-with-no-changes drafts plus 2–4 that need thinking. A per-claim prompt forces you to respond to every item; the batch-editor lets you scroll past the easy ones and invest attention on the hard ones. Matches the actual cognitive shape of the task.

**3d. Wiki updates + Human gate 3**

```python
def ingest_gate_3_wiki(approved: list[Claim], src: Source, backend: LLMBackend, repo: NotesRepo) -> None:
    existing_pages = repo.list_wiki_pages()
    index_md = (repo.wiki_dir / "index.md").read_text()
    prompt = prompts.load("wiki_update").format(
        agents_md=repo.load_agents_md(),
        claims=[c.model_dump_yaml() for c in approved],
        existing_index=index_md,
        existing_pages_summaries=summarize_pages(existing_pages),
    )
    raw = backend.complete(prompt, tier="deep")
    edits = parse_wiki_edits(raw)  # list of create/update page operations
    for e in edits:
        apply_edit_to_working_tree(e, repo)
    # Print summary; instruct human to review via git diff.
    typer.echo("Wiki edits applied to working tree. Review with: git -C {repo} diff wiki/")
    typer.echo(f"When satisfied: git -C {repo} add wiki/ && git commit -m 'wiki: ingest {src.id}'")
```

No `git add`, no `git commit`. Per AGENTS.md §5.

**3e. Log entry**

After all three gates succeed, append a log entry in the exact format from AGENTS.md §3.1.

**Tests:**
- `test_ingest_paper_happy_path.py` — fake PDF, mocked Ollama, mocked `$EDITOR` (auto-confirms), mocked approval (leaves all drafts untouched = approve-all). End-to-end to a populated tmp rossum.
- `test_ingest_gate_rejections.py` — verify each gate can abort cleanly without corrupting the repo (editor exits without save → no changes committed; draft YAML emptied → all drafts rejected; wiki diff rejected → working tree reset).
- `test_ingest_notebook_self_critique.py` — notebook path; mocked critique LLM; verify the prompt runs and the user can proceed.
- `test_ingest_simulation_requires_output_summary.py` — verify gate 1 blocks if `output_summary.md` is missing or still contains `<FILL>` markers.
- `test_ingest_simulation_generation_spec.py` — simulation with `generation_spec` instead of `input_files`; verify schema accepts, ingest proceeds.
- `test_ingest_idempotent_resume.py` — interrupt after gate 2 approval; re-run; verify resume skips gate 1 and re-presents deferred `.claim.draft` files.

**Exit criterion:** ingesting the three Phase-1 sources (paper, simulation, notebook) into a **fresh** rossum-clone repo via `glean ingest`, with real Ollama calls, produces commits that are **diff-equivalent** to the Phase-1 manual ingest commits — modulo timestamps, draft approval IDs, and whitespace normalization. This is the "GLEAN must reproduce Phase 1's reference output" test from principle #7. Any substantive divergence is a bug.

---

### M4 — Lint (2 days)

Implement the checks enumerated in AGENTS.md v0.2 §3.3 plus the v0.2-added checks.

**Files:**
- `src/glean/lint.py` — one function per check, each returning `list[LintFinding]`. Findings have severity (`error`/`warning`), location (`file:line` or `claim_id`), message.
- CLI: `glean lint [--strict] [--fix-index]`. `--strict` exits nonzero on warnings. `--fix-index` regenerates `index.md` (the one check that has a safe auto-fix).

**Checks** (§3.3 original 10 + v0.2 additions):

1. Uncited sentences in wiki — **per-page-kind rule** per AGENTS.md v0.2 §2.6. Strict on entity/concept/method; licensed on synthesis/comparison.
2. Dangling claim citations.
3. Orphan claims.
4. Dangling source references, including the notebook case where `source:` points into `notebook/` not `sources/`.
5. Contradictions both active.
6. Stale frontmatter.
7. Index freshness.
8. Log completeness.
9. ID collisions.
10. Draft leakage.
11. **v0.2 new:** `external_refs:` promotion candidates — report externals cited by ≥2 distinct rossum sources.
12. **v0.2 new:** `output_summary.md` present and non-trivially filled for every `simulation` source.
13. **v0.2 new:** `solver_commit` carries `+dirty` marker or is a clean git hash for every `simulation` source.
14. **v0.2 new:** meta-claims are co-cited — if a wiki page cites a content claim that has a governing meta-claim (same source, `tags: [meta]`), the wiki page should also cite the meta-claim. Warning, not error.

**Tests:** one test per check — construct a deliberately broken tmp repo, run the check, assert the finding. Plus an end-to-end test that runs `glean lint` against the real rossum repo and asserts zero findings (the positive control).

**Exit criterion:** lint catches every violation enumerated above on synthetic repos. Running lint on the current rossum returns clean.

---

### M5 — Query (1–2 days)

Implement the read-and-synthesize flow.

**Files:**
- `src/glean/query.py`:
  - Given a question, read `index.md` → select candidate pages via `fast` tier classification ("which of these pages are likely relevant to the question?").
  - Load candidate pages and the claims they cite.
  - Single `deep` tier call: "Here is the question, here are the relevant wiki pages and claims. Produce an answer; cite claims inline; refuse to answer if the evidence is insufficient."
  - Print answer.
  - Offer to file the answer as a synthesis wiki page via the same human gate 3 flow as ingest.

**Tests:** mocked Ollama; verify prompt shape, answer parsing, and file-back flow.

**Exit criterion:** a real question against a populated rossum produces an answer with valid citations.

---

### M6 — v0.1 polish and tag (0.25 day)

**Trimmed from 1 day.** The full FoBiS release workflow is premature for v0.1; ship by local tag, no PyPI, no CI publish job.

- [ ] `glean init` creates AGENTS.md from the current v0.2 bundled template, scaffolds all four layers, writes a starter `~/.config/glean/config.toml` if absent.
- [ ] All commands have `--help` that actually helps.
- [ ] Error messages point at AGENTS.md sections when schema validation fails.
- [ ] README.md updated with a "what works" section pointing at M1–M5 capabilities.
- [ ] CI (deferred from M0): GitHub Actions workflow running `make lint test` on push/PR. No publish job.
- [ ] First tag: `v0.1.0` — local annotated tag, no push to PyPI. Publishing is a separate decision after dogfooding for a month.

**Deferred to v0.2 or later:** `release.sh`, VitePress docs site, PyPI publish, trusted-publisher setup.

---

## CLI API (frozen for v0.1)

```
glean [--verbose] <command> [args]

Commands:
  version                   Print version
  init [PATH]               Scaffold a new notes repo at PATH (default: .)
  ingest SOURCE             Ingest a source (path or URL); --type overrides detection
         [--type TYPE]
         [--cloud]          Use cloud backend for synthesis pass
         [--yes]            Skip interactive confirmations (for scripting; still uses $EDITOR)
  lint                      Health-check the repo
       [--strict]           Exit nonzero on warnings
       [--fix-index]        Regenerate index.md (safe auto-fix)
  query QUESTION            Ask a question; returns cited answer
        [--cloud]
        [--file-back]       Prompt to file the answer as a synthesis page
```

Notes:
- All commands require CWD to be inside a valid rossum repo unless `--repo PATH` is passed (deferred; v0.1 uses CWD only).
- Exit codes: 0 success, 1 user error (invalid input, aborted gate), 2 repo error (invalid AGENTS.md, corrupt file), 3 LLM error (backend unreachable, unparseable response).

---

## Open design questions — resolutions after Phase 1

Phase 1 gave us data; most of these are now decided.

1. **Claim response format from the LLM — resolved as YAML-blocks-separated-by-`---`.** Phase 1 drafts were structured as YAML frontmatter + body, one per file. That's what gate 2 expects; the LLM should produce the same shape as one concatenated stream, parseable by splitting on `---` markers. JSON-lines rejected: claim bodies contain prose with embedded quotes, backslashes, and newlines that JSON-lines encoding makes brittle.
2. **Prompt length vs model context — implement context-budget check in M2.** Feeding full AGENTS.md + full paper + existing wiki summaries exceeds an 8k-context small model easily (the paper we ingested alone was ~39k tokens extracted). Budget rule: priority order is **AGENTS.md (required) > source content (required) > existing index.md (preferred) > existing wiki page summaries (optional, truncate first)**. Never truncate source content; fail the call and tell the user to switch to cloud if budget can't fit AGENTS.md + source.
3. **Interactive approval UX — resolved as `$EDITOR`-on-YAML-batch.** See M3 3c above. Phase 1 demonstrated the batch-review cognitive pattern empirically.
4. **Idempotency of ingest — resolved.** Resume strategy:
   - If `sources/<id>/source.yaml` exists, skip gate 1 (assume already-confirmed).
   - If `_pending_<source_id>.yaml` exists in `claims/`, re-open it in `$EDITOR` (resume gate 2 approval).
   - If any `.claim.draft` files match the source ID prefix, offer to re-present them in a fresh `_pending_<source_id>.yaml`.
   - If the wiki working tree has uncommitted edits tagged by source ID (stored in a hidden marker file `.glean-cache/<source_id>-wiki-edits`), offer to resume gate 3.
   - Never silently overwrite. Always ask.
5. **NEW — Context-feeding strategy for cross-source claim citation.** Phase 1 showed that notebook claims productively cite paper claims (F13). For GLEAN to reproduce this, the claim-extraction prompt for a new source must include summaries of existing claims that might be citable. Strategy: for every existing claim, include its `claim:` paraphrase + `id` in the prompt context. For 31 claims that's ~3K tokens — tractable. At 200+ claims we'd need retrieval; defer until actually needed.

---

## What is explicitly out of scope for v0.1

- TUI for review (gates use `$EDITOR` and terminal prompts)
- Local web view of the wiki (use VS Code / Obsidian / any markdown reader)
- Embedding-based retrieval (index.md only)
- Figure handling (PDF is stored; LLM reads text only)
- Multi-agent loops or tool use from the LLM
- Cloud-backed ingest (cloud is opt-in per invocation)
- Migration from any existing notes system
- Mobile, sync beyond git, multi-user
- Remote repository automation (pushing, PR creation)
- `glean promote notebook/foo.md` sugar command (manual path-based ingest works)

---

## Test strategy summary

- **Mock all external I/O by default.** No real Ollama, no real HTTP, no real git beyond tmp repos in every unit test.
- **Integration tests** live in `tests/integration/` and are marked `@pytest.mark.integration`. They run against real Ollama when `GLEAN_INTEGRATION=1` is set. Not part of the default `make test`.
- **Real-paper smoke test** — a single `tests/integration/test_real_paper.py` that ingests one checked-in tiny public-domain paper. Gated on `GLEAN_REAL=1`.
- **Golden-file tests for prompts** — prompt templates have golden outputs for deterministic input. Catches unintended prompt drift.

---

## Timeline estimate — revised

Assuming ~3–5 focused hours/day:

| Milestone | Work | Change from v1 plan | Calendar |
|---|---|---|---|
| M0 | 0.25 day | −0.25 (CI deferred to M6) | Day 1 |
| M1 | 3.5 days | +2.0 (v0.2 schema surface larger) | Days 1–5 |
| M2 | 2.5 days | unchanged | Days 6–8 |
| M3 | 5 days | +1.0 (notebook self-critique + output_summary gate) | Days 9–14 |
| M4 | 2 days | +0.5 (v0.2 adds 4 new checks) | Days 15–16 |
| M5 | 1.5 days | unchanged | Days 17–18 |
| M6 | 0.25 day | −0.75 (release.sh + VitePress deferred) | Day 19 |

≈ **4 calendar weeks to v0.1** (up from 3 in the original plan). Real-world drift factor: 1.5×. Target: v0.1 tag within 5–6 weeks.

The single highest-risk milestone is still M3 (the three gates). If it takes two weeks instead of five days, that is a normal outcome and not a planning failure — the gates are the product.

**Where the extra time went compared to v1:** +2 days on M1 because the v0.2 schema is larger; +1 day on M3 because v0.2 added two new gate-1 requirements; +0.5 day on M4 for new lint checks; offset by −1 day on M6 (release scope trimmed) and −0.25 on M0 (CI deferred). Net +2.25 days, ~75% of which is directly traceable to Phase 1 schema findings.

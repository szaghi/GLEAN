# GLEAN v0.1 — Implementation Plan

This plan defines the work from the current empty skeleton to a v0.1 that can ingest a real paper into `rossum/` end to end, under Stefano's three human gates, using a local Ollama backend.

The plan is **incremental and testable at every milestone**. Each milestone leaves GLEAN in a usable state for the scope it covers; nothing is merged half-finished.

---

## Guiding design principles

1. **Git is the database.** No SQLite, no vector store. Every persistent artifact is a markdown or YAML file. The working tree + commit history are the data model + audit log.
2. **Human gates are first-class.** The three gates (source metadata confirm, claim drafts approve, wiki diffs review) are the product. Build them properly before building anything clever.
3. **Schema before code.** `rossum/AGENTS.md` is the authoritative specification. GLEAN's code must validate against it; when they disagree, AGENTS.md wins and the code changes.
4. **LLM as a component, not the core.** The core is file I/O, schema validation, and diff presentation. The LLM is a callable that produces draft content for human review. Keep the LLM's failure surface small and detectable.
5. **No agentic loops at v0.1.** No tool use from the LLM side. Single-turn prompts. Parse the response; act deterministically. If it breaks, make it break loudly.
6. **Ship v0.1 when it works on one real paper end to end.** Not before, not after.

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

Already done by scaffolding:
- pyproject.toml, Makefile, .gitignore, README.md, src/glean/ package, tests/ directory
- Git initialized, files staged

To finish M0:
- [ ] First commit (human runs `git commit`, not GLEAN)
- [ ] `make dev` creates the .venv and installs successfully
- [ ] `make test` runs the smoke tests and passes
- [ ] `make lint` passes
- [ ] CI workflow (`.github/workflows/python-package.yml`) — mirror FoBiS's shape: lint + test jobs on push/PR to main. Defer `publish` job until we tag a first release.

**Exit criterion:** `make dev test lint` all green on a fresh clone.

---

### M1 — Schema models and config (1–2 days)

Implement the data model and configuration loader.

**Files to implement:**

- `src/glean/schema.py` — Pydantic models:
  - `SourceManifest` (validates `source.yaml` for every source type, with type-discriminated fields as AGENTS.md §2.2)
  - `Claim` (validates claim file frontmatter + body)
  - `WikiPage` (validates wiki page frontmatter)
  - `NotebookEntry` (validates notebook entry frontmatter)
  - Enum types for `SourceType`, `Confidence`, `ClaimStatus`, `WikiKind`
  - Custom validators for ID format regex (one per type)

- `src/glean/config.py` — Typed config loader:
  - Reads `~/.config/glean/config.toml`
  - Fields: Ollama endpoint URL, model tier mapping (fast/deep), optional Anthropic API key, default notes repo path, editor command
  - Dumps a starter config on first run (`glean init` writes it if absent)

- `src/glean/ids.py` (new module) — ID construction and validation:
  - `source_id_for(type: SourceType, **kwargs) -> str` (deterministic)
  - `claim_id_for(source_id: str, assertion_slug: str) -> str`
  - `is_valid_source_id(s: str) -> bool` etc.

**Tests:**
- `test_schema_source.py` — one parametrized test per source type, happy path + at least 2 failure modes per type
- `test_schema_claim.py` — happy path, invalid citation format, invalid confidence
- `test_ids.py` — deterministic ID generation, collision detection, invalid input rejection
- `test_config.py` — loads a sample config; falls back to defaults; rejects malformed

**Exit criterion:** `glean version` still works; Pydantic models round-trip every example from AGENTS.md.

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

### M3 — Ingest pipeline (the core of v0.1; 3–5 days)

This is where the three human gates are implemented. Everything up to now has been infrastructure.

**3a. Source adapters** — one per type in `src/glean/sources/`:

Each adapter implements:
```python
class SourceAdapter(Protocol):
    def can_handle(self, input: str) -> bool: ...  # path or URL
    def extract_metadata(self, input: str) -> dict: ...
    def extract_content(self, input: str, dest: Path) -> None: ...  # writes artifacts
    def default_id(self, metadata: dict) -> str: ...
```

Build in this order:
1. `PaperAdapter` (PDF → pymupdf4llm markdown; DOI lookup via crossref.org if possible, else manual entry)
2. `NotebookAdapter` (already-written markdown file → source; trivial)
3. `SimulationAdapter` (directory with an input file + output_summary.md convention)
4. `WebArticleAdapter` (URL → BeautifulSoup + markdownify → markdown)
5. Others (`repository`, `dataset`, `talk`, `book`, `standard`, `personal_comm`) — stubs that raise `NotImplementedError` until needed

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
    return src
```

`human_confirm_editor` uses `$EDITOR` (or `vi` fallback). A TUI is explicitly out of scope for v0.1 — `$EDITOR` is the review surface, consistent with using `git diff` later.

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

`human_approve_drafts` — v0.1 uses a simple loop: for each draft, print it to the terminal (rich-formatted), prompt `[a]pprove / [e]dit / [r]eject / [s]kip / [q]uit`. Edits open `$EDITOR` on the draft file. Skipped drafts remain on disk as drafts for the next run.

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
- `test_ingest_paper_happy_path.py` — fake PDF, mocked Ollama, mocked `$EDITOR` (auto-confirms), mocked approval loop (approves all). End-to-end to a populated tmp rossum.
- `test_ingest_gate_rejections.py` — verify each gate can abort cleanly without corrupting the repo.
- `test_ingest_notebook.py` — shorter path, fewer claims.
- `test_ingest_simulation.py` — directory input; verify input-deck + output-summary are correctly preserved.

**Exit criterion:** ingesting a real paper into a real rossum repo on Stefano's machine, with a real Ollama call, produces a valid, committed source + claims + working-tree wiki diff. Tested on 2–3 real papers before declaring M3 done.

---

### M4 — Lint (1–2 days)

Implement the checks enumerated in AGENTS.md §3.3.

**Files:**
- `src/glean/lint.py` — one function per check, each returning `list[LintFinding]`. Findings have severity (`error`/`warning`), location (`file:line` or `claim_id`), message.
- CLI: `glean lint [--strict] [--fix-index]`. `--strict` exits nonzero on warnings. `--fix-index` regenerates `index.md` (the one check that has a safe auto-fix).

**Tests:** one test per check — construct a deliberately broken tmp repo, run the check, assert the finding.

**Exit criterion:** lint catches every violation enumerated in AGENTS.md §3.3 on synthetic repos. Running lint on real rossum returns clean.

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

### M6 — v0.1 polish and release (1 day)

- [ ] `glean init` creates AGENTS.md from a template bundled in the package, scaffolds all four layers, writes a starter `~/.config/glean/config.toml` if absent.
- [ ] All six commands have `--help` that actually helps.
- [ ] Error messages point at AGENTS.md sections when schema validation fails.
- [ ] `docs/` has a short getting-started doc. VitePress setup deferred.
- [ ] `release.sh` mirrored from FoBiS (trunk-based; lint → test → bump version → tag → push).
- [ ] First tag: `v0.1.0`. No PyPI publish yet — deferred until Stefano has used v0.1 for a month.

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

## Open design questions to resolve during implementation

1. **Claim response format from the LLM.** JSON-lines seems simplest (one claim per line, easy to parse and truncate on error). But large models sometimes misformat. Alternative: YAML front-matter blocks separated by `---`. Decide during M3 by prompting with both on real Ollama models.
2. **Prompt length vs model context.** Feeding full AGENTS.md + full paper + existing wiki summaries may exceed an 8k-context small model. Build a context-budget check into `llm.py` and truncate the "existing pages summaries" first, then existing wiki content, never the source.
3. **Interactive approval UX.** Per-claim terminal prompt is functional but tedious for 20-claim papers. Consider batched approval via `$EDITOR` on a single YAML file listing all drafts — reject by deleting, approve by leaving, edit in place. Try both on real papers during M3.
4. **Idempotency of ingest.** If ingest is interrupted at gate 2, can the human re-run it? Design a resume strategy — probably: check for existing `sources/<id>/source.yaml`, if present skip gate 1; check for existing `.claim.draft` files, if present offer to resume approval.

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

## Timeline estimate

Assuming ~3–5 focused hours/day:

| Milestone | Work | Calendar |
|---|---|---|
| M0 | 0.5 day | Day 1 |
| M1 | 1.5 days | Days 2–3 |
| M2 | 2.5 days | Days 4–6 |
| M3 | 4 days | Days 7–10 |
| M4 | 1.5 days | Days 11–12 |
| M5 | 1.5 days | Days 13–14 |
| M6 | 1 day | Day 15 |

≈ **3 calendar weeks to v0.1.** Real-world drift factor: 1.5×. Target: v0.1 tag within 4–5 weeks.

The single highest-risk milestone is M3 (the three gates). If it takes two weeks instead of four days, that is a normal outcome and not a planning failure — the gates are the product.

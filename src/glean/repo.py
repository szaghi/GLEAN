"""Rossum repo I/O primitives (M2b).

Wraps a rossum repo on disk with a typed API backed by the schema models.
Every read validates via Pydantic; every write is atomic (temp + fsync +
same-filesystem rename) so a mid-operation crash can never leave a partially
written file that later confuses parsing.

Architecture decisions (per M2 design conversation):

    D3: atomic writes via _atomic_write()
    D4: constructor validates repo once; trusts thereafter
    D5: load_source(id) dispatches on ID prefix:
            note_*  -> notebook/<filename>.md  (free naming; scan by id)
            else    -> sources/<id>/source.yaml
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import yaml
from pydantic import TypeAdapter, ValidationError

from glean.enums import SourceType
from glean.errors import GleanRepoError
from glean.ids import (
    is_valid_claim_id,
    is_valid_source_id,
    is_valid_wiki_page_id,
)
from glean.schema import (
    Claim,
    LogEntry,
    NotebookSource,
    PaperSource,
    RepositorySource,
    SimulationSource,
    SourceManifest,
    WebArticleSource,
    WikiIndex,
    WikiPage,
)

# TypeAdapter resolved at module level for the discriminated union.
_SOURCE_ADAPTER = TypeAdapter(SourceManifest)

# Any SourceCommon-descended subclass — what load_source returns except for notebook.
AnyFiledSource = PaperSource | SimulationSource | RepositorySource | WebArticleSource

# The layer directories a valid rossum repo must have.
_LAYER_DIRS: tuple[str, ...] = ("sources", "notebook", "claims", "wiki")


# =============================================================================
# Frontmatter parsing helpers
# =============================================================================


def _split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Split a markdown file's `---`-delimited YAML frontmatter from its body.

    Returns (frontmatter_dict, body_str). The body is everything after the
    second `---\n` line, with no leading newline stripped. Raises
    `GleanRepoError` if the file does not begin with frontmatter.
    """
    if not text.startswith("---\n"):
        raise GleanRepoError("file does not begin with '---\\n' frontmatter marker")
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        raise GleanRepoError("file has no closing '---' for frontmatter block")
    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        raise GleanRepoError(f"frontmatter YAML parse failed: {e}") from e
    if not isinstance(frontmatter, dict):
        raise GleanRepoError(f"frontmatter must be a YAML mapping, got {type(frontmatter).__name__}")
    return frontmatter, parts[2]


def _atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` atomically.

    Writes to a temp file in the SAME DIRECTORY as the target (so the rename
    stays within one filesystem and is truly atomic), fsyncs the file, then
    renames onto the target. A mid-operation crash leaves either the old file
    or nothing — never a partial write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # delete=False so we can rename; the NamedTemporaryFile context cleans up
    # only if we exit without renaming (error path).
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    try:
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# =============================================================================
# NotesRepo
# =============================================================================


class NotesRepo:
    """Typed I/O handle for a rossum repo at a known path.

    Construction validates once: the path must exist, must be a directory,
    and must contain `AGENTS.md` plus the four layer directories (§2.1).
    Per D4, subsequent reads/writes trust the validation; they do not
    re-verify the layer dirs.
    """

    def __init__(self, path: Path | str) -> None:
        self.root = Path(path).expanduser().resolve()
        if not self.root.is_dir():
            raise GleanRepoError(f"not a directory: {self.root}")
        if not (self.root / "AGENTS.md").is_file():
            raise GleanRepoError(
                f"{self.root} is not a valid rossum repo: missing AGENTS.md. "
                f"Run `glean init <path>` to scaffold a new repo."
            )
        for layer in _LAYER_DIRS:
            if not (self.root / layer).is_dir():
                raise GleanRepoError(
                    f"{self.root} is missing layer directory {layer!r}; "
                    f"expected all of {_LAYER_DIRS} per AGENTS.md v0.2 §2.1"
                )

    # --- directory accessors -------------------------------------------------

    @property
    def sources_dir(self) -> Path:
        return self.root / "sources"

    @property
    def notebook_dir(self) -> Path:
        return self.root / "notebook"

    @property
    def claims_dir(self) -> Path:
        return self.root / "claims"

    @property
    def wiki_dir(self) -> Path:
        return self.root / "wiki"

    @property
    def agents_md_path(self) -> Path:
        return self.root / "AGENTS.md"

    @property
    def log_md_path(self) -> Path:
        return self.wiki_dir / "log.md"

    @property
    def index_md_path(self) -> Path:
        return self.wiki_dir / "index.md"

    # --- source I/O ----------------------------------------------------------

    def load_source(self, source_id: str) -> AnyFiledSource | NotebookSource:
        """Load a source by ID. Dispatches on prefix (D5).

        For notebook IDs (`note_*`), scans `notebook/*.md` for the matching
        frontmatter `id:` (free filename naming per AGENTS.md §2.4). For
        all other types, reads `sources/<id>/source.yaml` directly.
        """
        if not is_valid_source_id(source_id):
            raise GleanRepoError(f"not a valid source ID: {source_id!r}")

        if is_valid_source_id(source_id, SourceType.NOTEBOOK):
            return self._load_notebook(source_id)
        return self._load_filed_source(source_id)

    def _load_filed_source(self, source_id: str) -> AnyFiledSource:
        yaml_path = self.sources_dir / source_id / "source.yaml"
        if not yaml_path.is_file():
            raise GleanRepoError(f"source.yaml not found for {source_id} at {yaml_path}")
        try:
            raw = yaml.safe_load(yaml_path.read_text())
        except yaml.YAMLError as e:
            raise GleanRepoError(f"source.yaml YAML parse failed for {source_id}: {e}") from e
        if not isinstance(raw, dict):
            raise GleanRepoError(f"source.yaml for {source_id} must contain a YAML mapping")
        try:
            parsed = _SOURCE_ADAPTER.validate_python(raw)
        except ValidationError as e:
            raise GleanRepoError(f"source.yaml for {source_id} fails schema validation: {e}") from e
        if isinstance(parsed, NotebookSource):
            # Should not happen — notebook IDs are routed to _load_notebook.
            raise GleanRepoError(
                f"source.yaml at {yaml_path} declares type=notebook, but notebook entries "
                f"must live in notebook/ per AGENTS.md v0.2 §2.1"
            )
        return parsed

    def _load_notebook(self, note_id: str) -> NotebookSource:
        """Scan notebook/*.md and return the entry whose frontmatter id matches."""
        for path in sorted(self.notebook_dir.glob("*.md")):
            try:
                frontmatter, _ = _split_frontmatter(path.read_text())
            except GleanRepoError:
                continue  # skip files without well-formed frontmatter
            if frontmatter.get("id") != note_id:
                continue
            # Inject type: notebook so the discriminated union dispatches correctly.
            # AGENTS.md §2.4 frontmatter does not carry a `type:` field; it is
            # implicit from location (files in notebook/).
            frontmatter.setdefault("type", SourceType.NOTEBOOK.value)
            try:
                parsed = _SOURCE_ADAPTER.validate_python(frontmatter)
            except ValidationError as e:
                raise GleanRepoError(f"notebook {path.name} frontmatter fails schema validation: {e}") from e
            if not isinstance(parsed, NotebookSource):
                raise GleanRepoError(f"notebook {path.name} frontmatter has wrong type: {parsed.type}")
            return parsed
        raise GleanRepoError(f"notebook entry not found with id {note_id!r}")

    def list_sources(self) -> Iterator[str]:
        """Yield all source IDs present in the repo (filed + notebook)."""
        for child in sorted(self.sources_dir.iterdir()):
            if child.is_dir() and (child / "source.yaml").is_file():
                yield child.name
        for path in sorted(self.notebook_dir.glob("*.md")):
            try:
                frontmatter, _ = _split_frontmatter(path.read_text())
            except GleanRepoError:
                continue
            note_id = frontmatter.get("id")
            if isinstance(note_id, str) and is_valid_source_id(note_id, SourceType.NOTEBOOK):
                yield note_id

    def save_source(self, source: AnyFiledSource) -> Path:
        """Write a filed source (not notebook) to sources/<id>/source.yaml.

        Returns the path written. Atomic: a crash mid-write leaves either the
        prior file or nothing.
        """
        if isinstance(source, NotebookSource):
            raise GleanRepoError("use save_notebook() for notebook sources, not save_source()")
        source_dir = self.sources_dir / source.id
        source_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = source_dir / "source.yaml"
        content = yaml.safe_dump(
            source.model_dump(mode="json", exclude_defaults=False),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
        _atomic_write(yaml_path, content)
        return yaml_path

    # --- claim I/O -----------------------------------------------------------

    def load_claim(self, claim_id: str) -> tuple[Claim, str]:
        """Load a claim by ID. Returns (parsed_frontmatter, body_markdown)."""
        if not is_valid_claim_id(claim_id):
            raise GleanRepoError(f"not a valid claim ID: {claim_id!r}")
        path = self.claims_dir / f"{claim_id}.md"
        if not path.is_file():
            raise GleanRepoError(f"claim not found: {path}")
        frontmatter, body = _split_frontmatter(path.read_text())
        try:
            parsed = Claim.model_validate(frontmatter)
        except ValidationError as e:
            raise GleanRepoError(f"claim {claim_id} fails schema validation: {e}") from e
        return parsed, body

    def save_claim_draft(self, claim: Claim, body: str = "") -> Path:
        """Write a claim as a `.claim.draft` (gitignored) awaiting approval.

        Per AGENTS.md v0.2 §2.5: drafts live as `<claim_id>.claim.draft`
        until the human approves them in gate 2, at which point they are
        promoted via `promote_claim_draft()`.
        """
        path = self.claims_dir / f"{claim.id}.claim.draft"
        _atomic_write(path, _render_frontmatter_md(claim, body))
        return path

    def save_claim(self, claim: Claim, body: str = "") -> Path:
        """Write a claim directly as `<id>.md` (approved form).

        Used when restoring claims or in tests; the ingest flow uses
        `save_claim_draft` + `promote_claim_draft`.
        """
        path = self.claims_dir / f"{claim.id}.md"
        _atomic_write(path, _render_frontmatter_md(claim, body))
        return path

    def promote_claim_draft(self, claim_id: str) -> Path:
        """Rename `<claim_id>.claim.draft` to `<claim_id>.md`.

        Atomic rename; raises GleanRepoError if the draft does not exist or
        the target already exists (would silently overwrite an approved claim).
        """
        draft = self.claims_dir / f"{claim_id}.claim.draft"
        final = self.claims_dir / f"{claim_id}.md"
        if not draft.is_file():
            raise GleanRepoError(f"draft not found: {draft}")
        if final.exists():
            raise GleanRepoError(
                f"refusing to promote draft: target already exists at {final}. "
                f"Delete the existing claim first if overwrite is intentional."
            )
        draft.rename(final)
        return final

    def list_claims(self) -> Iterator[str]:
        """Yield all approved claim IDs (not drafts)."""
        for path in sorted(self.claims_dir.glob("claim_*.md")):
            yield path.stem

    def list_claim_drafts(self) -> Iterator[str]:
        """Yield all pending claim draft IDs."""
        for path in sorted(self.claims_dir.glob("claim_*.claim.draft")):
            # strip '.claim.draft' (13 chars) to recover the id
            yield path.name[: -len(".claim.draft")]

    # --- wiki I/O ------------------------------------------------------------

    def load_wiki_page(self, page_id: str) -> tuple[WikiPage, str]:
        """Load a wiki page by ID. Returns (parsed_frontmatter, body_markdown)."""
        if not is_valid_wiki_page_id(page_id):
            raise GleanRepoError(f"not a valid wiki page ID: {page_id!r}")
        if page_id in {"index", "log"}:
            raise GleanRepoError(f"{page_id}.md is not a regular wiki page; use load_wiki_index() or log-entry APIs")
        path = self.wiki_dir / f"{page_id}.md"
        if not path.is_file():
            raise GleanRepoError(f"wiki page not found: {path}")
        frontmatter, body = _split_frontmatter(path.read_text())
        try:
            parsed = WikiPage.model_validate(frontmatter)
        except ValidationError as e:
            raise GleanRepoError(f"wiki page {page_id} fails schema validation: {e}") from e
        return parsed, body

    def save_wiki_page(self, page: WikiPage, body: str) -> Path:
        """Write a wiki page atomically. Body is the markdown after the frontmatter block."""
        path = self.wiki_dir / f"{page.id}.md"
        _atomic_write(path, _render_frontmatter_md(page, body))
        return path

    def list_wiki_pages(self) -> Iterator[str]:
        """Yield all wiki page IDs (excluding index.md and log.md)."""
        for path in sorted(self.wiki_dir.glob("*.md")):
            if path.name in {"index.md", "log.md"}:
                continue
            yield path.stem

    def load_wiki_index(self) -> tuple[WikiIndex, str]:
        """Load wiki/index.md. Returns (parsed_frontmatter, body_markdown)."""
        if not self.index_md_path.is_file():
            raise GleanRepoError(f"index.md not found: {self.index_md_path}")
        frontmatter, body = _split_frontmatter(self.index_md_path.read_text())
        try:
            parsed = WikiIndex.model_validate(frontmatter)
        except ValidationError as e:
            raise GleanRepoError(f"wiki/index.md fails schema validation: {e}") from e
        return parsed, body

    # --- log I/O -------------------------------------------------------------

    def append_log(self, entry: LogEntry) -> None:
        """Append a log entry to wiki/log.md.

        Not atomic: `log.md` is append-only and ordered, so a partial write
        is harder to recover from. We write through a temp file anyway for
        consistency with other writes. The read-then-rewrite approach is used
        because log.md has free-form surrounding markdown that we must
        preserve (header, instructions, prior entries).
        """
        existing = self.log_md_path.read_text() if self.log_md_path.is_file() else ""
        # Ensure the existing content ends with exactly one newline before we append.
        if existing and not existing.endswith("\n"):
            existing += "\n"
        new_content = existing + "\n" + entry.to_markdown()
        _atomic_write(self.log_md_path, new_content)

    # --- agents.md -----------------------------------------------------------

    def load_agents_md(self) -> str:
        """Return AGENTS.md as a raw string (for feeding into LLM prompts)."""
        return self.agents_md_path.read_text()


# =============================================================================
# Frontmatter rendering
# =============================================================================


def _render_frontmatter_md(model: Claim | WikiPage, body: str) -> str:
    """Serialize a Pydantic model as YAML frontmatter + body.

    Output shape:
        ---
        <yaml>
        ---

        <body>

    The empty line after the closing `---` matches the convention used by
    the Phase 1 files.
    """
    frontmatter_yaml = yaml.safe_dump(
        model.model_dump(mode="json", exclude_defaults=False),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    body_section = body if body.endswith("\n") else body + "\n" if body else ""
    return f"---\n{frontmatter_yaml}---\n\n{body_section}"

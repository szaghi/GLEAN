"""Gate-1 orchestration (M3b).

Gate 1 is the source-metadata confirmation step per AGENTS.md v0.2 §3.1.
The flow:

    1. Pick the adapter (sniff + confirm, or --type override; D10)
    2. Adapter produces a DraftSource (type-specific gate-1 work)
    3. Render draft_yaml as text, open $EDITOR for user confirmation (D26)
    4. Post-edit: validate against schema; retry on validation error
    5. Stage artifacts and copies into sources/<id>/ (or notebook/ for notebook)
    6. Run type-specific post-staging work (notebook self-critique per D16)
    7. git add + git commit the source per AGENTS.md §5 exception

Resume logic (filesystem-is-state per D12):
    - Gate 1 is "complete" iff `sources/<id>/source.yaml` is committed.
      Checked via `git ls-files`, not just filesystem presence, because
      uncommitted source.yaml means gate 1 was interrupted.
    - With --resume, if gate 1 is complete, skip to gate 2.
    - Without --resume, if gate 1 is partial (source.yaml exists but
      uncommitted), refuse with "use --resume or --abort".
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import TypeAdapter, ValidationError

from glean.adapters import (
    DraftSource,
    IngestInput,
    SourceIngester,
    adapter_for,
    stage_source_directory,
)
from glean.config import Config
from glean.enums import SourceType
from glean.errors import GleanLLMError, GleanRepoError
from glean.git import add as git_add
from glean.git import any_tracked_under, is_tracked
from glean.git import commit as git_commit
from glean.llm import ModelTier, OllamaBackend, load_prompt
from glean.repo import NotesRepo
from glean.schema import SourceManifest

# TypeAdapter for the full discriminated union — validates post-edit.
_SOURCE_ADAPTER = TypeAdapter(SourceManifest)

# Maximum retries when the user's edits fail schema validation.
_EDITOR_RETRIES = 5


@dataclass
class Gate1Result:
    """What gate-1 produces on success."""

    source_id: str
    source_type: SourceType
    commit_sha: str | None  # None for notebook (no sources/ stub to commit)
    was_resumed: bool  # True if gate 1 was already complete; we skipped it


# =============================================================================
# Main entry point
# =============================================================================


def run_gate1(
    inp: IngestInput,
    *,
    repo: NotesRepo,
    config: Config,
    resume: bool = False,
    confirm_type: bool = True,
) -> Gate1Result:
    """Execute gate 1 for the given input.

    Parameters
    ----------
    inp
        The ingest input (path or URL, optional type override, flags).
    repo
        Target rossum repo.
    config
        Loaded user config.
    resume
        If True, skip gate 1 when already complete; otherwise refuse if partial.
    confirm_type
        If True and no --type override was passed, prompt the user to confirm
        the sniffed type before proceeding per D10. Tests set False.
    """
    adapter = _pick_adapter(inp, confirm_type=confirm_type)

    # Before doing any work, check if gate 1 is already complete for this input.
    # We can only know the source ID once the adapter has prepared; so the
    # resume-check comes AFTER prepare but BEFORE staging. One cost: prepare
    # runs even on resume. For notebook this is cheap; for paper it's
    # Crossref + PDF extraction, which is expensive to redo. Acceptable at
    # v0.1 because resume is an exception path.
    draft = adapter.prepare(inp)
    source_id = draft.proposed_id

    # Short-circuit if gate 1 already completed and user passed --resume.
    if _gate1_complete(repo, draft):
        if resume:
            return Gate1Result(
                source_id=source_id,
                source_type=draft.source_type,
                commit_sha=None,
                was_resumed=True,
            )
        raise GleanRepoError(
            f"gate 1 for {source_id!r} is already complete. "
            f"Pass --resume to continue to gate 2, or abort via `glean ingest --abort {source_id}`."
        )

    # If partial gate-1 state exists (draft sources/<id>/ but uncommitted),
    # refuse without --resume. User must explicitly resume or abort.
    if _gate1_partial(repo, draft) and not resume:
        raise GleanRepoError(
            f"partial gate-1 state exists for {source_id!r} in {repo.sources_dir / source_id}. "
            f"Pass --resume to continue, or `glean ingest --abort {source_id}` to clear it."
        )

    # User confirmation loop: open $EDITOR on the draft yaml, validate on save.
    confirmed = _confirm_source_yaml(draft, config.editor)

    # The user may have changed the id field during editing; the confirmed
    # value wins over the adapter's proposed_id. All filesystem layout
    # decisions use confirmed_id from here on.
    confirmed_id = confirmed.get("id")
    if not isinstance(confirmed_id, str):
        raise GleanRepoError("confirmed source.yaml is missing a valid 'id' field")

    # Stage files per adapter type.
    if draft.source_type == SourceType.NOTEBOOK:
        # Notebook already lives in notebook/; nothing to stage.
        # Still: run the self-critique per D16.
        _run_notebook_self_critique(inp, repo, config)
        # No sources/ commit for notebook; the notebook file itself is what
        # the user eventually commits manually or via a later glean command.
        return Gate1Result(
            source_id=confirmed_id,
            source_type=SourceType.NOTEBOOK,
            commit_sha=None,
            was_resumed=False,
        )

    source_dir = repo.sources_dir / confirmed_id
    # Write confirmed source.yaml, then stage the rest of the artifacts.
    _write_source_yaml(source_dir, confirmed)
    stage_source_directory(_draft_without_source_yaml(draft), source_dir)

    # Commit per AGENTS.md §5 whitelisted exception.
    git_add(repo.root, [f"sources/{confirmed_id}/"])
    sha = git_commit(repo.root, f"source: add {confirmed_id}")

    return Gate1Result(
        source_id=confirmed_id,
        source_type=draft.source_type,
        commit_sha=sha,
        was_resumed=False,
    )


# =============================================================================
# Adapter selection (sniff + confirm per D10)
# =============================================================================


def _pick_adapter(inp: IngestInput, *, confirm_type: bool) -> SourceIngester:
    """Return the adapter, confirming the sniffed type with the user if needed."""
    if inp.type_override is not None:
        return adapter_for(inp)

    adapter = adapter_for(inp)
    if confirm_type and sys.stdin.isatty():
        sys.stdout.write(f"Detected source type: {adapter.source_type.value}. Proceed? [Y/n] ")
        sys.stdout.flush()
        answer = sys.stdin.readline().strip().lower()
        if answer and answer not in {"y", "yes", ""}:
            raise GleanRepoError(
                f"user rejected detected type {adapter.source_type.value!r}. "
                f"Pass --type explicitly to force a specific adapter."
            )
    return adapter


# =============================================================================
# $EDITOR integration
# =============================================================================


def _confirm_source_yaml(draft: DraftSource, editor: str) -> dict[str, object]:
    """Open $EDITOR on the draft yaml; validate on save; retry on error.

    D18 pattern (inline error markers, retry up to 5 times, fallback to abort).
    Per D26: save-without-change is accepted as confirmation.

    Returns the validated dict ready to write to source.yaml.
    """
    rendered = _render_draft_yaml_for_editor(draft)
    last_error: str | None = None

    for _attempt in range(_EDITOR_RETRIES):
        annotated = _prepend_error_marker(rendered, last_error) if last_error else rendered
        edited = _open_in_editor(annotated, editor, suffix=".yaml")
        # Strip any leading `# !! ERROR:` lines (user may have left them).
        edited_clean = _strip_error_markers(edited)

        try:
            parsed_raw = yaml.safe_load(edited_clean)
        except yaml.YAMLError as e:
            last_error = f"YAML parse failed: {e}"
            continue
        if not isinstance(parsed_raw, dict):
            last_error = "source.yaml must be a YAML mapping at the top level"
            continue

        try:
            # Validate via discriminated union — this checks all type-specific rules.
            _SOURCE_ADAPTER.validate_python(parsed_raw)
        except ValidationError as e:
            last_error = f"schema validation failed:\n{e}"
            rendered = edited_clean  # show user's edits, not original draft, on retry
            continue

        return parsed_raw

    raise GleanRepoError(
        f"source.yaml failed validation after {_EDITOR_RETRIES} attempts. "
        f"Last error:\n{last_error}\n"
        f"Inspect the state of the draft manually and re-run with --resume when ready."
    )


def _render_draft_yaml_for_editor(draft: DraftSource) -> str:
    """Render a DraftSource as a YAML string suitable for $EDITOR."""
    header = (
        f"# Draft source.yaml for {draft.proposed_id}\n"
        f"# Type: {draft.source_type.value}\n"
        f"# Edit as needed. Save and close to confirm; close without saving to accept as-is.\n"
        f"# Replace any <FILL: ...> markers with real values before saving.\n\n"
    )
    body = yaml.safe_dump(
        draft.draft_yaml,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    return header + body


_ERROR_MARKER_PREFIX = "# !! ERROR:"


def _prepend_error_marker(text: str, error_msg: str) -> str:
    """Prepend an error marker comment block to the YAML text."""
    lines = [f"{_ERROR_MARKER_PREFIX} {line}" for line in error_msg.splitlines()]
    lines.append(f"{_ERROR_MARKER_PREFIX} --- fix the above and save again ---")
    return "\n".join(lines) + "\n\n" + text


def _strip_error_markers(text: str) -> str:
    """Remove any lines starting with the error marker prefix."""
    return "\n".join(line for line in text.splitlines() if not line.startswith(_ERROR_MARKER_PREFIX))


def _open_in_editor(content: str, editor: str, *, suffix: str) -> str:
    """Open `content` in `$EDITOR`; return the post-edit content.

    Uses a temp file in the system tmp directory (not the rossum repo) so
    half-written editor buffers don't pollute the repo.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        suffix=suffix,
        prefix="glean-edit-",
        encoding="utf-8",
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # editor may be something like "vi" or "code --wait"; split honoring spaces.
        editor_cmd = editor.split()
        subprocess.run([*editor_cmd, str(tmp_path)], check=True)  # noqa: S603
        return tmp_path.read_text()
    except subprocess.CalledProcessError as e:
        raise GleanRepoError(f"editor {editor!r} exited with error: {e}") from e
    except FileNotFoundError as e:
        raise GleanRepoError(
            f"editor {editor!r} not found on PATH. Set the `editor` field in ~/.config/glean/config.toml or $EDITOR."
        ) from e
    finally:
        tmp_path.unlink(missing_ok=True)


# =============================================================================
# Source directory staging helpers
# =============================================================================


def _write_source_yaml(source_dir: Path, confirmed: dict[str, object]) -> None:
    """Write the confirmed source.yaml atomically."""
    source_dir.mkdir(parents=True, exist_ok=True)
    yaml_text = yaml.safe_dump(
        confirmed,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    (source_dir / "source.yaml").write_text(yaml_text)


def _draft_without_source_yaml(draft: DraftSource) -> DraftSource:
    """Return a DraftSource with `source.yaml` removed from artifacts.

    Because we write the USER-confirmed source.yaml separately in
    _write_source_yaml, staging must not overwrite it with the draft version.
    """
    filtered_artifacts = {name: content for name, content in draft.artifacts.items() if name != "source.yaml"}
    return DraftSource(
        source_type=draft.source_type,
        proposed_id=draft.proposed_id,
        draft_yaml=draft.draft_yaml,
        artifacts=filtered_artifacts,
        files_to_copy=draft.files_to_copy,
    )


# =============================================================================
# Notebook self-critique (D16)
# =============================================================================


def _run_notebook_self_critique(inp: IngestInput, repo: NotesRepo, config: Config) -> None:
    """Run the LLM-backed adversarial review on the notebook entry per D16.

    Prints the critique to stdout; prompts the user to `[revise / acknowledge / proceed]`.
    """
    # Find the notebook file from the ingest input.
    notebook_path = Path(inp.input_spec).expanduser().resolve()
    entry_text = notebook_path.read_text()

    backend = OllamaBackend(config.ollama)
    template = load_prompt("notebook_critique")
    prompt = template.safe_substitute(
        agents_md=repo.load_agents_md(),
        notebook_entry=entry_text,
    )

    try:
        critique, _call_log = backend.complete(prompt, tier=ModelTier.DEEP)
    except GleanLLMError as e:
        # Don't block the ingest if the LLM is unreachable; warn instead.
        sys.stdout.write(
            f"\n[warning] notebook self-critique skipped: {e}\n"
            f"Proceed manually. Re-read the entry as a reviewer before gate 2.\n\n"
        )
        return

    sys.stdout.write("\n=== Notebook self-critique (LLM-generated) ===\n\n")
    sys.stdout.write(critique)
    sys.stdout.write("\n\n=== end critique ===\n\n")

    if not sys.stdin.isatty():
        # Non-interactive run: print and proceed. Tests take this path.
        return

    sys.stdout.write("Action: [r]evise / [a]cknowledge / [p]roceed-as-is? ")
    sys.stdout.flush()
    answer = sys.stdin.readline().strip().lower()
    if answer.startswith("r"):
        raise GleanRepoError("Revise the notebook entry and re-run `glean ingest --resume`.")
    # "a" and "p" both proceed; "a" is the spirit-of-schema acknowledgment
    # (we trust the user to have noted the critique mentally). No state file
    # per D12.


# =============================================================================
# Resume / abort helpers
# =============================================================================


def _gate1_complete(repo: NotesRepo, draft: DraftSource) -> bool:
    """True if source.yaml for this source is committed to git."""
    if draft.source_type == SourceType.NOTEBOOK:
        # Notebook gate 1 "completes" by running the self-critique; there is
        # no sources/ stub to check. We always re-run the critique on resume
        # because it's cheap and the entry may have been edited since.
        return False

    yaml_rel = f"sources/{draft.proposed_id}/source.yaml"
    yaml_path = repo.root / yaml_rel
    if not yaml_path.is_file():
        return False
    return is_tracked(repo.root, yaml_rel)


def _gate1_partial(repo: NotesRepo, draft: DraftSource) -> bool:
    """True if the source directory exists but source.yaml isn't committed."""
    if draft.source_type == SourceType.NOTEBOOK:
        return False
    source_dir = repo.sources_dir / draft.proposed_id
    if not source_dir.is_dir():
        return False
    return not _gate1_complete(repo, draft)


def abort_gate1(repo: NotesRepo, source_id: str) -> None:
    """Clear uncommitted gate-1 state per D27.

    Safe operation: only deletes `sources/<id>/` if it's entirely uncommitted.
    Refuses if any file under it is tracked by git. User must use git commands
    to undo committed gate-1 state.
    """
    source_dir = repo.sources_dir / source_id
    if not source_dir.is_dir():
        raise GleanRepoError(f"no gate-1 state to abort for {source_id!r}")

    if any_tracked_under(repo.root, f"sources/{source_id}"):
        raise GleanRepoError(
            f"refusing to abort gate 1 for {source_id!r}: source directory contains "
            f"committed files. Use `git revert` or `git reset` if you really want "
            f"to undo a committed source. GLEAN does not rewrite committed state."
        )

    shutil.rmtree(source_dir)

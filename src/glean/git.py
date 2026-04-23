"""Thin wrapper around git subprocess calls (M2a).

GLEAN uses git as its database: every rossum repo is a git repo, and the
audit trail is `git log`. This module exposes a minimal, read-heavy API
sufficient for the three-gate ingest flow and the lint pass. Write operations
(`add`, `commit`) are deliberately present but documented for narrow use per
AGENTS.md v0.2 §5.

Scope at v0.1 (decision D1):
    - Read: status, diff, is_clean, current_commit, git_root
    - Write: add (paths), commit (message) — callers must respect §5

Not included (never-at-v0.1):
    - push, pull, fetch, branch, checkout, reset, stash, restore, rebase, merge
    - Any command that rewrites history or affects remotes

If M3 needs an operation not listed here, add it then with an explicit
justification in the docstring — don't pre-emptively broaden the surface.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from glean.errors import GleanRepoError


def _resolve_git_executable() -> str:
    """Return the absolute path to the git executable, or raise if not found."""
    found = shutil.which("git")
    if not found:
        raise GleanRepoError("git executable not found on PATH; GLEAN requires git to operate on rossum repos")
    return found


# Resolved at import time so we fail fast if git is missing, and so every
# subprocess call uses an absolute path (S607-clean, portable, no shell lookup
# per call).
_GIT_EXE: str = _resolve_git_executable()


# Every call goes through _GIT_BASE to neutralize environment-sensitive git
# behavior that would break programmatic use:
#   --no-pager:              don't try to spawn a pager on long output
#   -c color.*=never:        no ANSI color codes in diff/status/log output
#   -c commit.gpgSign=false: don't attempt GPG signing (no TTY available to
#                            prompt for a passphrase); per the project
#                            convention, GLEAN commits are never signed
_GIT_BASE: tuple[str, ...] = (
    _GIT_EXE,
    "--no-pager",
    "-c",
    "color.ui=never",
    "-c",
    "color.diff=never",
    "-c",
    "color.status=never",
    "-c",
    "commit.gpgSign=false",
)


def _run(args: tuple[str, ...], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run `git <args>` in `cwd`, capturing stdout/stderr as text.

    Returns the CompletedProcess. Raises `GleanRepoError` if `check=True` and
    git exited non-zero, carrying stderr in the exception message.
    """
    result = subprocess.run(  # noqa: S603 — args are constructed in this module, not user input
        (*_GIT_BASE, *args),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )

    if check and result.returncode != 0:
        raise GleanRepoError(
            f"git {' '.join(args)} failed (exit {result.returncode}) in {cwd}: {result.stderr.strip()}"
        )
    return result


def git_root(path: Path) -> Path:
    """Return the top-level directory of the git repo containing `path`.

    Raises `GleanRepoError` if `path` is not inside any git repo.
    """
    result = _run(("rev-parse", "--show-toplevel"), cwd=path)
    return Path(result.stdout.strip())


def is_clean(repo_path: Path) -> bool:
    """Return True if the working tree has no uncommitted changes.

    "Clean" here means: no staged changes, no unstaged changes, no untracked
    files that are not covered by `.gitignore`. Matches what `git status
    --porcelain` reports as an empty output.
    """
    result = _run(("status", "--porcelain"), cwd=repo_path)
    return result.stdout.strip() == ""


def status(repo_path: Path) -> str:
    """Return the output of `git status --porcelain` (short, parseable form).

    Empty string means clean. Each non-empty line is one changed path in the
    format `XY <path>` where X/Y are one-char status codes.
    """
    result = _run(("status", "--porcelain"), cwd=repo_path)
    return result.stdout


def diff(repo_path: Path, paths: list[str] | None = None, *, staged: bool = False) -> str:
    """Return the unified diff for the given paths (or entire tree if omitted).

    Parameters
    ----------
    paths
        Paths relative to `repo_path`, or None for the full repo.
    staged
        If True, show staged changes (`git diff --cached`). If False, show
        unstaged changes in the working tree.
    """
    args: list[str] = ["diff"]
    if staged:
        args.append("--cached")
    if paths:
        args.append("--")
        args.extend(paths)
    result = _run(tuple(args), cwd=repo_path)
    return result.stdout


def current_commit(repo_path: Path) -> str:
    """Return the full SHA-1 hash of HEAD."""
    result = _run(("rev-parse", "HEAD"), cwd=repo_path)
    return result.stdout.strip()


def add(repo_path: Path, paths: list[str]) -> None:
    """Stage the given paths.

    Per AGENTS.md v0.2 §5: the only automatic `add` is for `sources/<id>/`
    after the human confirms `source.yaml` in gate 1. All other `add` calls
    must be initiated by the human (via their shell, not GLEAN). Callers that
    invoke this function from ingest code must document the §5 exception they
    are exercising.
    """
    if not paths:
        raise GleanRepoError("git.add() requires a non-empty list of paths")
    _run(("add", "--", *paths), cwd=repo_path)


def is_tracked(repo_path: Path, path: str) -> bool:
    """Return True if `path` (relative to repo_path) is tracked by git.

    Uses `git ls-files --error-unmatch`, which exits non-zero if the path is
    not tracked. Distinguishes "file exists on disk but not committed" (False)
    from "file is committed to HEAD" (True).
    """
    result = _run(("ls-files", "--error-unmatch", path), cwd=repo_path, check=False)
    return result.returncode == 0


def any_tracked_under(repo_path: Path, dir_path: str) -> bool:
    """Return True if any file under `dir_path` (relative) is tracked by git."""
    result = _run(("ls-files", dir_path), cwd=repo_path, check=False)
    return result.returncode == 0 and result.stdout.strip() != ""


def commit(repo_path: Path, message: str) -> str:
    """Create a commit with the given message. Returns the new commit SHA.

    Per AGENTS.md v0.2 §5: see `add()`. Commit operations from ingest code
    are restricted to the one whitelisted case (gate 1 source commit after
    human confirmation). All other commits must be run by the human.

    Raises `GleanRepoError` if the index is empty (nothing staged).
    """
    if not message.strip():
        raise GleanRepoError("commit message must not be empty or whitespace-only")
    # --quiet suppresses the default commit summary; we return the SHA instead.
    _run(("commit", "-m", message, "--quiet"), cwd=repo_path)
    return current_commit(repo_path)

"""Tests for `glean.git` (M2a).

These tests use real tmp git repos, not mocks. Git has enough edge-case
behavior that mocking would test our assumptions about git rather than git
itself — and Phase 1 principle #7 is that GLEAN must reproduce the actual
behavior of the reference workflow. Real git is the reference.

Test repos are built with minimal commit history in conftest.py fixtures.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from glean.errors import GleanRepoError
from glean.git import (
    add,
    commit,
    current_commit,
    diff,
    git_root,
    is_clean,
    status,
)

_GIT = shutil.which("git") or "git"  # resolved path silences S607; fallback for completeness


@pytest.fixture
def tmp_git_repo(tmp_path: Path) -> Path:
    """A tmp git repo with one initial commit.

    Configures local user.name / user.email so commits don't fail on CI or
    machines without a global git identity. Disables GPG signing explicitly
    so we don't depend on the developer's signing key being available.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([_GIT, "init", "-q", "-b", "main"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "user.email", "test@example.com"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "user.name", "Test User"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "config", "commit.gpgSign", "false"], cwd=repo, check=True)  # noqa: S603
    # Seed with one committed file so HEAD exists.
    (repo / "seed.md").write_text("seed\n")
    subprocess.run([_GIT, "add", "seed.md"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([_GIT, "commit", "-q", "-m", "init"], cwd=repo, check=True)  # noqa: S603
    return repo


class TestGitRoot:
    def test_returns_repo_root_from_root(self, tmp_git_repo: Path) -> None:
        assert git_root(tmp_git_repo) == tmp_git_repo.resolve()

    def test_returns_repo_root_from_subdir(self, tmp_git_repo: Path) -> None:
        subdir = tmp_git_repo / "sub"
        subdir.mkdir()
        assert git_root(subdir) == tmp_git_repo.resolve()

    def test_raises_outside_repo(self, tmp_path: Path) -> None:
        outside = tmp_path / "not_a_repo"
        outside.mkdir()
        with pytest.raises(GleanRepoError, match=r"rev-parse"):
            git_root(outside)


class TestIsClean:
    def test_clean_repo(self, tmp_git_repo: Path) -> None:
        assert is_clean(tmp_git_repo) is True

    def test_dirty_with_unstaged_modification(self, tmp_git_repo: Path) -> None:
        (tmp_git_repo / "seed.md").write_text("modified\n")
        assert is_clean(tmp_git_repo) is False

    def test_dirty_with_untracked_file(self, tmp_git_repo: Path) -> None:
        (tmp_git_repo / "new.md").write_text("untracked\n")
        assert is_clean(tmp_git_repo) is False

    def test_dirty_with_staged_change(self, tmp_git_repo: Path) -> None:
        (tmp_git_repo / "new.md").write_text("staged\n")
        subprocess.run([_GIT, "add", "new.md"], cwd=tmp_git_repo, check=True)  # noqa: S603
        assert is_clean(tmp_git_repo) is False


class TestStatus:
    def test_clean_returns_empty(self, tmp_git_repo: Path) -> None:
        assert status(tmp_git_repo) == ""

    def test_untracked_shows_with_qq(self, tmp_git_repo: Path) -> None:
        (tmp_git_repo / "new.md").write_text("x\n")
        out = status(tmp_git_repo)
        assert "?? new.md" in out

    def test_modified_shows_with_space_m(self, tmp_git_repo: Path) -> None:
        (tmp_git_repo / "seed.md").write_text("changed\n")
        out = status(tmp_git_repo)
        assert " M seed.md" in out


class TestDiff:
    def test_no_changes_empty_diff(self, tmp_git_repo: Path) -> None:
        assert diff(tmp_git_repo) == ""

    def test_unstaged_change_shows_in_diff(self, tmp_git_repo: Path) -> None:
        (tmp_git_repo / "seed.md").write_text("seed\nnew line\n")
        d = diff(tmp_git_repo)
        assert "seed.md" in d
        assert "+new line" in d

    def test_staged_diff_distinct_from_unstaged(self, tmp_git_repo: Path) -> None:
        (tmp_git_repo / "seed.md").write_text("staged change\n")
        subprocess.run([_GIT, "add", "seed.md"], cwd=tmp_git_repo, check=True)  # noqa: S603
        assert "staged change" not in diff(tmp_git_repo, staged=False)
        assert "staged change" in diff(tmp_git_repo, staged=True)

    def test_path_filter(self, tmp_git_repo: Path) -> None:
        (tmp_git_repo / "a.md").write_text("a\n")
        (tmp_git_repo / "b.md").write_text("b\n")
        subprocess.run([_GIT, "add", "a.md", "b.md"], cwd=tmp_git_repo, check=True)  # noqa: S603
        subprocess.run(  # noqa: S603
            [_GIT, "commit", "-q", "-m", "ab"],
            cwd=tmp_git_repo,
            check=True,
        )
        (tmp_git_repo / "a.md").write_text("a changed\n")
        (tmp_git_repo / "b.md").write_text("b changed\n")
        d = diff(tmp_git_repo, paths=["a.md"])
        assert "a.md" in d
        assert "b.md" not in d


class TestCurrentCommit:
    def test_returns_hash(self, tmp_git_repo: Path) -> None:
        sha = current_commit(tmp_git_repo)
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_changes_after_new_commit(self, tmp_git_repo: Path) -> None:
        sha1 = current_commit(tmp_git_repo)
        (tmp_git_repo / "new.md").write_text("x\n")
        subprocess.run([_GIT, "add", "new.md"], cwd=tmp_git_repo, check=True)  # noqa: S603
        subprocess.run(  # noqa: S603
            [_GIT, "commit", "-q", "-m", "add new"],
            cwd=tmp_git_repo,
            check=True,
        )
        sha2 = current_commit(tmp_git_repo)
        assert sha1 != sha2


class TestAdd:
    def test_stages_files(self, tmp_git_repo: Path) -> None:
        (tmp_git_repo / "new.md").write_text("content\n")
        add(tmp_git_repo, ["new.md"])
        out = status(tmp_git_repo)
        assert "A  new.md" in out

    def test_rejects_empty_paths(self, tmp_git_repo: Path) -> None:
        with pytest.raises(GleanRepoError, match=r"non-empty"):
            add(tmp_git_repo, [])

    def test_rejects_nonexistent_path(self, tmp_git_repo: Path) -> None:
        with pytest.raises(GleanRepoError, match=r"pathspec"):
            add(tmp_git_repo, ["does_not_exist.md"])


class TestCommit:
    def test_creates_commit(self, tmp_git_repo: Path) -> None:
        (tmp_git_repo / "new.md").write_text("content\n")
        add(tmp_git_repo, ["new.md"])
        sha = commit(tmp_git_repo, "test: add new.md")
        assert len(sha) == 40
        assert is_clean(tmp_git_repo)
        # Verify the commit message landed.
        result = subprocess.run(  # noqa: S603
            [_GIT, "log", "-1", "--format=%s"],
            cwd=tmp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "test: add new.md"

    def test_rejects_empty_message(self, tmp_git_repo: Path) -> None:
        with pytest.raises(GleanRepoError, match=r"empty or whitespace"):
            commit(tmp_git_repo, "")

    def test_rejects_whitespace_message(self, tmp_git_repo: Path) -> None:
        with pytest.raises(GleanRepoError, match=r"empty or whitespace"):
            commit(tmp_git_repo, "   \n  ")

    def test_rejects_commit_with_empty_index(self, tmp_git_repo: Path) -> None:
        # Nothing staged; git commit fails.
        with pytest.raises(GleanRepoError):
            commit(tmp_git_repo, "test: nothing to commit")


class TestErrorHandling:
    def test_raises_outside_git_repo(self, tmp_path: Path) -> None:
        outside = tmp_path / "not_a_repo"
        outside.mkdir()
        with pytest.raises(GleanRepoError):
            is_clean(outside)

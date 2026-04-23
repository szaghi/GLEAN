"""Command-line entry point for GLEAN."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from glean import __version__

log = logging.getLogger(__name__)

app = typer.Typer(
    name="glean",
    help="Grounded Linked Evidence And Notes — an LLM-maintained research wiki.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Root callback — configures logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def version() -> None:
    """Print the GLEAN version."""
    typer.echo(f"glean {__version__}")


@app.command()
def init(
    path: Path = typer.Argument(Path(), help="Directory to initialize as a GLEAN notes repo."),
) -> None:
    """Initialize a new GLEAN notes repo (scaffold AGENTS.md + layer directories)."""
    raise NotImplementedError("init: to be implemented in M2 (see docs/PLAN.md)")


@app.command()
def ingest(
    source: str = typer.Argument(..., help="Path or URL of the source to ingest."),
    source_type: str | None = typer.Option(None, "--type", "-t", help="Source type override."),
    cloud: bool = typer.Option(False, "--cloud", help="Use cloud LLM for synthesis pass."),
    pdf_extractor: str = typer.Option(
        "marker", "--pdf-extractor", help="PDF extractor: 'marker' (default) or 'pymupdf'."
    ),
    offline: bool = typer.Option(False, "--no-network", help="Disable network lookups (e.g. Crossref for DOIs)."),
    resume: bool = typer.Option(False, "--resume", help="Resume a partially-completed ingest from prior state."),
    abort: str | None = typer.Option(
        None,
        "--abort",
        help="Abort and clear uncommitted gate-1 state for the given source_id. Mutually exclusive with normal ingest.",
    ),
    repo: Path = typer.Option(
        Path(),
        "--repo",
        "-r",
        help="Path to the rossum repo. Defaults to the current directory.",
    ),
) -> None:
    """Ingest a source through the three gates: source metadata → claims → wiki."""
    from glean.enums import SourceType
    from glean.errors import GleanError
    from glean.ingest import abort_command, run_cli_ingest

    if abort:
        try:
            abort_command(source_id=abort, repo_path=repo)
            typer.echo(f"Aborted: cleared uncommitted gate-1 state for {abort!r}.")
        except GleanError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None
        return

    resolved_type: SourceType | None = None
    if source_type is not None:
        try:
            resolved_type = SourceType(source_type)
        except ValueError as e:
            valid = ", ".join(t.value for t in SourceType)
            typer.echo(f"Error: unknown --type {source_type!r}. Valid: {valid}", err=True)
            raise typer.Exit(1) from e

    exit_code = run_cli_ingest(
        source=source,
        repo_path=repo,
        source_type=resolved_type,
        cloud=cloud,
        pdf_extractor=pdf_extractor,
        offline=offline,
        resume=resume,
    )
    if exit_code != 0:
        raise typer.Exit(exit_code)


@app.command()
def lint(
    repo: Path = typer.Option(Path(), "--repo", "-r", help="Path to the rossum repo (default: current directory)."),
    strict: bool = typer.Option(False, "--strict", help="Exit nonzero on warnings too, not only errors."),
    json_output: bool = typer.Option(
        False, "--json", help="Emit structured JSON output instead of human-readable stream."
    ),
    only: list[str] = typer.Option(
        [],
        "--only",
        help="Run only the named check(s). Repeatable. Default: all checks.",
    ),
) -> None:
    """Health-check the wiki: citations, orphans, contradictions, and more (15 checks)."""
    from glean.errors import GleanError
    from glean.lint import format_human, format_json, run_lint
    from glean.repo import NotesRepo

    try:
        notes_repo = NotesRepo(repo)
    except GleanError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    only_set: set[str] | None = set(only) if only else None
    report = run_lint(notes_repo, only=only_set)

    typer.echo(format_json(report) if json_output else format_human(report))

    if report.errors():
        raise typer.Exit(1)
    if strict and report.warnings():
        raise typer.Exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural-language question."),
    cloud: bool = typer.Option(False, "--cloud", help="Use cloud LLM for synthesis."),
) -> None:
    """Ask a question against the wiki; answers cite claims."""
    raise NotImplementedError("query: to be implemented in M5 (see docs/PLAN.md)")


if __name__ == "__main__":
    app()

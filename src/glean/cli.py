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
    path: Path = typer.Argument(Path("."), help="Directory to initialize as a GLEAN notes repo."),
) -> None:
    """Initialize a new GLEAN notes repo (scaffold AGENTS.md + layer directories)."""
    raise NotImplementedError("init: to be implemented in M2 (see docs/PLAN.md)")


@app.command()
def ingest(
    source: str = typer.Argument(..., help="Path or URL of the source to ingest."),
    source_type: str | None = typer.Option(None, "--type", "-t", help="Source type override."),
    cloud: bool = typer.Option(False, "--cloud", help="Use cloud LLM for synthesis pass."),
) -> None:
    """Ingest a source: extract metadata, propose claims, propose wiki diffs."""
    raise NotImplementedError("ingest: to be implemented in M3 (see docs/PLAN.md)")


@app.command()
def lint() -> None:
    """Health-check the wiki: uncited sentences, orphan claims, broken links, contradictions."""
    raise NotImplementedError("lint: to be implemented in M4 (see docs/PLAN.md)")


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural-language question."),
    cloud: bool = typer.Option(False, "--cloud", help="Use cloud LLM for synthesis."),
) -> None:
    """Ask a question against the wiki; answers cite claims."""
    raise NotImplementedError("query: to be implemented in M5 (see docs/PLAN.md)")


if __name__ == "__main__":
    app()

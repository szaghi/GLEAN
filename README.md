# GLEAN

**Grounded Linked Evidence And Notes** — an LLM-maintained research wiki with provenance by construction.

> To glean: to gather patiently, piece by piece.

## The pattern

Inspired by Andrej Karpathy's [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) pattern, but extended with a verification layer Karpathy handwaves. The core idea: instead of a RAG system that re-derives knowledge on every query, an LLM **incrementally compiles** sources into a persistent, cross-linked wiki. The wiki is the compounding artifact; the LLM is the maintainer.

GLEAN adds a **claim layer** between sources and wiki. Every wiki sentence cites an atomic, source-anchored claim. Contradictions are never silently reconciled — both claims remain, explicitly marked. LLM edits arrive as reviewable diffs, not silent rewrites.

## Architecture

```
sources/    # external: papers, code, datasets, simulations, talks
notebook/   # internal: your dated rough thinking (first-class source type)
claims/     # atomic assertions citing sources or notebook entries
wiki/       # synthesized pages; every sentence cites a claim ID
AGENTS.md   # the schema — how any LLM agent should operate on this repo
```

A GLEAN-managed notes repo is **just a git repo of markdown and YAML**. Git is the database. Git is the audit trail. No SQLite, no vector store at v0.1.

## Status

Pre-alpha. See `docs/PLAN.md` for the v0.1 implementation roadmap.

## License

GPL-3.0

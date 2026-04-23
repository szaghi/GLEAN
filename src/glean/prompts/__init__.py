"""Prompt templates shipped as package data.

Templates are plain text with `$name`-style placeholders (string.Template syntax,
NOT format()-style). The `$` syntax avoids collisions with YAML's `{` characters
that appear verbatim in AGENTS.md, claim frontmatter, and other content we feed
to the LLM — see M2 decision D6.

Templates are loaded via `glean.llm.load_prompt(name)`.
"""

from __future__ import annotations

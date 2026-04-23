"""User configuration loading.

Reads ~/.config/glean/config.toml for:
  - Ollama endpoint and model tier mapping (fast / deep)
  - Anthropic API key and model selection
  - Default notes repo path
  - Editor command for approval workflows

To be implemented in M1. See docs/PLAN.md.
"""

from __future__ import annotations

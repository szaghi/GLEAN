"""Wiki consistency checks.

Checks:
  - wiki sentences without claim citations
  - orphan claims (extracted but never cited)
  - claims pointing to missing or moved sources
  - contradicting claims both marked active
  - dangling wikilinks
  - index.md and log.md freshness

To be implemented in M4. See docs/PLAN.md.
"""

from __future__ import annotations

"""Wiki page generation, citation validation, diff surfacing.

Wiki pages are markdown with YAML frontmatter. Every non-trivial sentence ends
with one or more claim citations. Generation is never silent — diffs are left
in the working tree for human review via `git diff`.

To be implemented in M3. See docs/PLAN.md.
"""

from __future__ import annotations

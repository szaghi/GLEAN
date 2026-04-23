"""Source ingestion — one module per source type.

The ingest flow has three human gates:
  1. source.yaml metadata confirm
  2. claim drafts approve (.claim.draft -> .md)
  3. wiki diffs review (git diff as review surface)

To be implemented in M3. See docs/PLAN.md.
"""

from __future__ import annotations

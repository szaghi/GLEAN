"""LLM backend abstraction.

Two backends behind one interface:
  - OllamaBackend  (default; local, free, two model tiers: fast / deep)
  - AnthropicBackend (opt-in via --cloud flag; one model, configurable)

Every call logs prompt + response to wiki/log.md under the relevant operation
for full auditability.

To be implemented in M2. See docs/PLAN.md.
"""

from __future__ import annotations

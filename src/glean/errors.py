"""Project-level exception hierarchy for GLEAN.

All GLEAN exceptions derive from `GleanError`. Downstream callers (CLI,
tests) should catch the base class for generic handling or the specific
subclasses for targeted recovery. Never raise bare `Exception` or
`ValueError` from GLEAN code — always pick a typed subclass.
"""

from __future__ import annotations


class GleanError(Exception):
    """Base class for all GLEAN errors."""


class GleanConfigError(GleanError):
    """Configuration file is malformed or missing a required field."""


class GleanRepoError(GleanError):
    """I/O on a rossum repo failed, or the repo is structurally invalid."""


class GleanLLMError(GleanError):
    """LLM backend call failed or produced unparseable output."""


class GleanCloudError(GleanLLMError):
    """Cloud backend (Anthropic) is requested but unavailable or not configured."""

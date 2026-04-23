"""LLM backend abstraction (M2c).

Two backends behind one Protocol:
    - OllamaBackend (default; local, free, two model tiers)
    - AnthropicBackend (opt-in via --cloud flag + GLEAN_CLOUD_ENABLED config)

Supporting machinery:
    - ContextBuilder: priority-tiered context assembly with token-budget truncation
    - LLMCallLog: structured record of one LLM invocation (for wiki/log.md)
    - load_prompt: reads a template from src/glean/prompts/<name>.txt

Design decisions locked in M2 design conversation:
    D6: string.Template ($-style placeholders), not .format()
    D7: context-budget with REQUIRED / PREFERRED / OPTIONAL tiers
    D8: log hashes by default; full prompts only under GLEAN_LOG_PROMPTS=1
    D9: AnthropicBackend defined unconditionally; SDK imported lazily inside
        __init__ with a clear error if not installed
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime
from enum import IntEnum
from importlib.resources import files
from string import Template
from typing import TYPE_CHECKING, Literal, Protocol

import httpx

from glean.config import CloudConfig, Config, OllamaConfig
from glean.errors import GleanCloudError, GleanLLMError

if TYPE_CHECKING:
    from anthropic import Anthropic  # only for type hints; not required at runtime

log = logging.getLogger(__name__)


# =============================================================================
# Prompt loading
# =============================================================================


def load_prompt(name: str) -> Template:
    """Load a prompt template by name from `src/glean/prompts/<name>.txt`.

    Returns a `string.Template` ready for `.substitute(...)` with the template's
    placeholders. Using string.Template (not .format()) avoids collisions with
    YAML's `{` characters that appear verbatim in AGENTS.md and other context
    we feed to the LLM.
    """
    if "/" in name or name.startswith("."):
        raise GleanLLMError(f"invalid prompt name: {name!r}")
    try:
        raw = (files("glean.prompts") / f"{name}.txt").read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise GleanLLMError(f"prompt template not found: {name}") from e
    return Template(raw)


# =============================================================================
# Model tier
# =============================================================================


class ModelTier(IntEnum):
    """LLM tier selector.

    `FAST` is for metadata extraction, ID generation, classification, lint —
    short, cheap calls. `DEEP` is for claim extraction and wiki synthesis —
    long, expensive calls that demand better reasoning.
    """

    FAST = 1
    DEEP = 2


# =============================================================================
# Context budget
# =============================================================================


class ContextPriority(IntEnum):
    """Priority tier for context chunks, per D7.

    REQUIRED chunks are never truncated; if they overflow the budget, the
    ContextBuilder raises rather than silently dropping them. PREFERRED and
    OPTIONAL are truncated from the least-important side of the budget first.
    """

    REQUIRED = 3
    PREFERRED = 2
    OPTIONAL = 1


# Heuristic token estimator. For Ollama (where the tokenizer is per-model and
# we don't want a heavy dep for every local backend), we use ~4 chars/token +
# 20% safety margin — a deliberate underestimate so we stay inside the budget.
# For Anthropic, callers may pass a more accurate tokenizer callable.
_CHARS_PER_TOKEN = 4.0
_SAFETY_MULTIPLIER = 1.2


def heuristic_token_count(text: str) -> int:
    """Estimate token count of `text` conservatively (overestimates).

    Cheap and dependency-free. Suitable for Ollama where per-model tokenizers
    would require `tokenizers` as a hard dep. Callers needing accuracy (e.g.
    for Anthropic, where a miss wastes cloud dollars) should use the SDK's
    `client.count_tokens()` instead.
    """
    if not text:
        return 0
    return int((len(text) / _CHARS_PER_TOKEN) * _SAFETY_MULTIPLIER)


class ContextBuilder:
    """Assemble prompt context while respecting a token budget.

    Usage:

        cb = ContextBuilder(budget_tokens=8192)
        cb.add(agents_md, priority=ContextPriority.REQUIRED, label="agents_md")
        cb.add(source_content, priority=ContextPriority.REQUIRED, label="source")
        cb.add(index_md, priority=ContextPriority.PREFERRED, label="index")
        cb.add(page_summaries, priority=ContextPriority.OPTIONAL, label="pages")
        rendered = cb.render()  # dict of label -> (possibly truncated) content

    If REQUIRED chunks alone exceed the budget, `render()` raises
    GleanLLMError — the caller must either enlarge the budget (use a bigger
    model) or slim down the required content.
    """

    def __init__(self, budget_tokens: int) -> None:
        if budget_tokens <= 0:
            raise GleanLLMError(f"budget_tokens must be positive; got {budget_tokens}")
        self.budget_tokens = budget_tokens
        self._chunks: list[tuple[str, ContextPriority, str]] = []

    def add(self, content: str, *, priority: ContextPriority, label: str) -> None:
        """Register a content chunk. Labels must be unique within a builder."""
        if any(existing_label == label for _, _, existing_label in self._chunks):
            raise GleanLLMError(f"duplicate context label: {label!r}")
        self._chunks.append((content, priority, label))

    def render(self) -> dict[str, str]:
        """Return {label: possibly-truncated-content}.

        Truncation algorithm:
            1. Compute REQUIRED total. If > budget, raise.
            2. Add REQUIRED to output as-is.
            3. For each remaining priority tier (PREFERRED, then OPTIONAL):
               if the tier fits entirely, include verbatim; otherwise truncate
               the LAST (lowest-priority) chunk first with a '... [truncated]'
               marker. Chunks earlier in the add() order are preserved.
        """
        required = [c for c in self._chunks if c[1] == ContextPriority.REQUIRED]
        remaining_budget = self.budget_tokens - sum(heuristic_token_count(c[0]) for c in required)
        if remaining_budget < 0:
            total = self.budget_tokens - remaining_budget
            raise GleanLLMError(
                f"REQUIRED context alone ({total} tokens) exceeds budget "
                f"({self.budget_tokens}); use a larger-context model"
            )

        out: dict[str, str] = {label: content for content, _, label in required}

        for tier in (ContextPriority.PREFERRED, ContextPriority.OPTIONAL):
            tier_chunks = [c for c in self._chunks if c[1] == tier]
            for content, _, label in tier_chunks:
                cost = heuristic_token_count(content)
                if cost <= remaining_budget:
                    out[label] = content
                    remaining_budget -= cost
                else:
                    truncated = self._truncate_to_budget(content, remaining_budget)
                    out[label] = truncated
                    remaining_budget = 0

        return out

    @staticmethod
    def _truncate_to_budget(content: str, budget_tokens: int) -> str:
        """Truncate `content` to fit within `budget_tokens`. Appends a marker."""
        marker = "\n\n... [truncated to fit context budget]"
        marker_cost = heuristic_token_count(marker)
        content_budget = max(budget_tokens - marker_cost, 0)
        # Each token is ~4 chars / 1.2 safety; back-compute chars.
        target_chars = int(content_budget * _CHARS_PER_TOKEN / _SAFETY_MULTIPLIER)
        if target_chars <= 0:
            return marker.lstrip()
        return content[:target_chars] + marker


# =============================================================================
# Call logging
# =============================================================================


class LLMCallLog:
    """Structured record of one LLM call for wiki/log.md sub-entries.

    Per D8: hashes by default; full prompt/response content only when
    `GLEAN_LOG_PROMPTS=1` is set in the environment. Cloud calls never log
    full prompt content regardless (redaction per AGENTS.md v0.2 §7).
    """

    def __init__(
        self,
        *,
        backend: str,
        model: str,
        tier: ModelTier,
        prompt: str,
        response: str,
        started_at: datetime,
        elapsed_seconds: float,
        is_cloud: bool,
    ) -> None:
        self.backend = backend
        self.model = model
        self.tier = tier
        self.prompt_hash = _sha256(prompt)
        self.response_hash = _sha256(response)
        self.started_at = started_at
        self.elapsed_seconds = elapsed_seconds
        self.is_cloud = is_cloud

        # Full content is optional and only for local backends.
        log_full = os.environ.get("GLEAN_LOG_PROMPTS") == "1"
        self.prompt_text: str | None = prompt if log_full and not is_cloud else None
        self.response_text: str | None = response if log_full and not is_cloud else None

    def to_log_bullets(self) -> list[str]:
        """Render as bullets suitable for appending to a LogEntry body."""
        bullets = [
            f"- backend: {self.backend} ({self.model}, tier={self.tier.name.lower()})",
            f"- prompt_hash: {self.prompt_hash[:16]}",
            f"- response_hash: {self.response_hash[:16]}",
            f"- elapsed: {self.elapsed_seconds:.1f}s",
        ]
        if self.is_cloud:
            bullets.append("- cloud: redacted per AGENTS.md §7")
        return bullets


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =============================================================================
# Backend protocol
# =============================================================================


class LLMBackend(Protocol):
    """Common interface for Ollama and Anthropic backends."""

    name: str  # "ollama" or "anthropic"
    is_cloud: bool

    def complete(self, prompt: str, *, tier: ModelTier, timeout_seconds: float) -> tuple[str, LLMCallLog]:
        """Send `prompt` to the backend at the given tier. Return (response, log).

        Raises GleanLLMError on transport failure or empty response.
        """
        ...


# =============================================================================
# Ollama backend
# =============================================================================


class OllamaBackend:
    """Local Ollama backend via HTTP against /api/generate.

    Synchronous — streaming is a UX nice-to-have deferred past v0.1. One call,
    one response, blocking until complete.
    """

    name = "ollama"
    is_cloud = False

    def __init__(self, config: OllamaConfig) -> None:
        self.endpoint = config.endpoint
        self.model_fast = config.model_fast
        self.model_deep = config.model_deep

    def _model_for(self, tier: ModelTier) -> str:
        return self.model_fast if tier == ModelTier.FAST else self.model_deep

    def complete(
        self,
        prompt: str,
        *,
        tier: ModelTier = ModelTier.DEEP,
        timeout_seconds: float = 300.0,
    ) -> tuple[str, LLMCallLog]:
        if not prompt.strip():
            raise GleanLLMError("prompt must not be empty or whitespace-only")
        model = self._model_for(tier)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # synchronous; see docstring
            "options": {
                # Deterministic-ish: low temperature for schema work.
                "temperature": 0.2,
            },
        }
        started_at = datetime.now().astimezone()
        t0 = time.monotonic()
        try:
            response = httpx.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            raise GleanLLMError(f"Ollama HTTP call failed: {e}") from e
        elapsed = time.monotonic() - t0

        text = data.get("response", "")
        if not isinstance(text, str) or not text.strip():
            raise GleanLLMError(f"Ollama returned empty or non-string response: {data!r}")

        call_log = LLMCallLog(
            backend=self.name,
            model=model,
            tier=tier,
            prompt=prompt,
            response=text,
            started_at=started_at,
            elapsed_seconds=elapsed,
            is_cloud=self.is_cloud,
        )
        return text, call_log


# =============================================================================
# Anthropic backend (optional)
# =============================================================================


class AnthropicBackend:
    """Cloud backend using the Anthropic SDK. Opt-in.

    The `anthropic` SDK is an optional install (`pip install glean[cloud]`).
    If the SDK is not installed, constructing an `AnthropicBackend` raises
    GleanCloudError with the install hint — the class is always importable
    from this module, but non-functional without the SDK.

    The API key is read from the `ANTHROPIC_API_KEY` env var (never from
    the TOML config — see glean.config). If missing, construction raises.
    """

    name = "anthropic"
    is_cloud = True

    def __init__(self, config: CloudConfig) -> None:
        if not config.enabled:
            raise GleanCloudError(
                "cloud backend is disabled in config; set `[cloud] enabled = true` "
                "in config.toml to allow cloud invocations"
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise GleanCloudError(
                "ANTHROPIC_API_KEY not set. Cloud backend requires the API key in the environment, not in config.toml."
            )
        try:
            from anthropic import Anthropic  # local import: optional dep
        except ImportError as e:
            raise GleanCloudError("anthropic SDK not installed. Install with: pip install glean[cloud]") from e
        self._client: Anthropic = Anthropic(api_key=api_key)
        self.model = config.anthropic_model

    def complete(
        self,
        prompt: str,
        *,
        tier: ModelTier = ModelTier.DEEP,
        timeout_seconds: float = 300.0,
    ) -> tuple[str, LLMCallLog]:
        if not prompt.strip():
            raise GleanLLMError("prompt must not be empty or whitespace-only")
        started_at = datetime.now().astimezone()
        t0 = time.monotonic()
        try:
            message = self._client.messages.create(  # type: ignore[attr-defined]
                model=self.model,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout_seconds,
            )
        except Exception as e:  # anthropic raises various APIError subclasses
            raise GleanLLMError(f"Anthropic API call failed: {e}") from e
        elapsed = time.monotonic() - t0

        # The SDK returns message.content as a list of content blocks.
        if not message.content or not hasattr(message.content[0], "text"):
            raise GleanLLMError(f"Anthropic returned unexpected content shape: {message.content!r}")
        text = message.content[0].text
        if not isinstance(text, str) or not text.strip():
            raise GleanLLMError("Anthropic returned empty or non-string response")

        call_log = LLMCallLog(
            backend=self.name,
            model=self.model,
            tier=tier,
            prompt=prompt,
            response=text,
            started_at=started_at,
            elapsed_seconds=elapsed,
            is_cloud=self.is_cloud,
        )
        return text, call_log


# =============================================================================
# Factory
# =============================================================================


def get_backend(config: Config, *, cloud: bool) -> LLMBackend:
    """Return the appropriate backend based on flag and config.

    Parameters
    ----------
    config
        The loaded user config.
    cloud
        If True, caller explicitly requested cloud (--cloud flag). This
        additionally requires config.cloud.enabled; if not, raises
        GleanCloudError.

    Returns
    -------
    An OllamaBackend when cloud=False, an AnthropicBackend otherwise.
    """
    if cloud:
        return AnthropicBackend(config.cloud)
    return OllamaBackend(config.ollama)


# Literal types re-exported for callers that want non-enum dispatch.
TierName = Literal["fast", "deep"]


def tier_from_name(name: TierName) -> ModelTier:
    """Map a string tier name to the `ModelTier` enum."""
    if name == "fast":
        return ModelTier.FAST
    if name == "deep":
        return ModelTier.DEEP
    raise GleanLLMError(f"unknown tier: {name!r}")

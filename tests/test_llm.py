"""Tests for `glean.llm` (M2c).

Network calls are mocked. Integration tests against real Ollama are out of
scope for the default `make test` — they belong in tests/integration/ later.

Covers:
    - load_prompt: template loading via importlib.resources
    - ContextBuilder: priority tiers, truncation, REQUIRED overflow
    - LLMCallLog: hash-by-default, full-content-only-under-env-var
    - OllamaBackend: success path + error paths + tier dispatch
    - AnthropicBackend: config gating, missing-SDK handling, API-key checks
    - get_backend factory
"""

from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from glean.config import CloudConfig, Config, OllamaConfig
from glean.errors import GleanCloudError, GleanLLMError
from glean.llm import (
    AnthropicBackend,
    ContextBuilder,
    ContextPriority,
    LLMCallLog,
    ModelTier,
    OllamaBackend,
    get_backend,
    heuristic_token_count,
    load_prompt,
    tier_from_name,
)

# =============================================================================
# load_prompt
# =============================================================================


class TestLoadPrompt:
    def test_loads_claim_extract(self) -> None:
        tmpl = load_prompt("claim_extract")
        rendered = tmpl.safe_substitute(
            year=2026,
            source_slug="x",
            source_id="paper_x",
            extracted_date="2026-04-23",
            agents_md="<agents>",
            source_yaml="<src>",
            source_content="<content>",
            existing_claims_summary="<claims>",
        )
        assert "claim-extraction agent" in rendered
        assert "<agents>" in rendered

    def test_missing_prompt_raises(self) -> None:
        with pytest.raises(GleanLLMError, match=r"not found"):
            load_prompt("nonexistent_template")

    def test_rejects_path_separator(self) -> None:
        with pytest.raises(GleanLLMError, match=r"invalid"):
            load_prompt("../etc/passwd")

    def test_rejects_leading_dot(self) -> None:
        with pytest.raises(GleanLLMError, match=r"invalid"):
            load_prompt(".hidden")


# =============================================================================
# heuristic_token_count
# =============================================================================


class TestHeuristicTokenCount:
    def test_empty_string(self) -> None:
        assert heuristic_token_count("") == 0

    def test_monotonic_with_length(self) -> None:
        a = "x" * 100
        b = "x" * 1000
        assert heuristic_token_count(a) < heuristic_token_count(b)

    def test_overestimates_rather_than_underestimates(self) -> None:
        """The heuristic should be conservative: real count <= heuristic count."""
        # A typical English sentence of ~40 chars has ~8-10 tokens by GPT-4
        # tokenizer; our heuristic should give at least that many.
        text = "The quick brown fox jumps over the lazy dog."
        assert heuristic_token_count(text) >= 10


# =============================================================================
# ContextBuilder
# =============================================================================


class TestContextBuilder:
    def test_empty_builder_produces_empty_dict(self) -> None:
        cb = ContextBuilder(budget_tokens=1000)
        assert cb.render() == {}

    def test_required_included_verbatim(self) -> None:
        cb = ContextBuilder(budget_tokens=10000)
        cb.add("agents content", priority=ContextPriority.REQUIRED, label="agents")
        cb.add("source content", priority=ContextPriority.REQUIRED, label="source")
        out = cb.render()
        assert out["agents"] == "agents content"
        assert out["source"] == "source content"

    def test_required_overflow_raises(self) -> None:
        cb = ContextBuilder(budget_tokens=10)
        cb.add("a" * 10000, priority=ContextPriority.REQUIRED, label="big")
        with pytest.raises(GleanLLMError, match=r"REQUIRED context alone"):
            cb.render()

    def test_optional_truncated_when_overflow(self) -> None:
        cb = ContextBuilder(budget_tokens=50)
        cb.add("required" * 3, priority=ContextPriority.REQUIRED, label="req")
        cb.add("a" * 1000, priority=ContextPriority.OPTIONAL, label="opt")
        out = cb.render()
        assert out["req"] == "required" * 3
        assert len(out["opt"]) < 1000
        assert "truncated" in out["opt"]

    def test_preferred_survives_when_optional_truncated(self) -> None:
        cb = ContextBuilder(budget_tokens=200)
        cb.add("REQ", priority=ContextPriority.REQUIRED, label="req")
        cb.add("PRF" * 30, priority=ContextPriority.PREFERRED, label="pref")
        cb.add("OPT" * 500, priority=ContextPriority.OPTIONAL, label="opt")
        out = cb.render()
        assert out["pref"] == "PRF" * 30
        assert "truncated" in out["opt"]

    def test_duplicate_labels_rejected(self) -> None:
        cb = ContextBuilder(budget_tokens=1000)
        cb.add("a", priority=ContextPriority.REQUIRED, label="x")
        with pytest.raises(GleanLLMError, match=r"duplicate"):
            cb.add("b", priority=ContextPriority.REQUIRED, label="x")

    def test_invalid_budget_rejected(self) -> None:
        with pytest.raises(GleanLLMError, match=r"positive"):
            ContextBuilder(budget_tokens=0)
        with pytest.raises(GleanLLMError, match=r"positive"):
            ContextBuilder(budget_tokens=-100)


# =============================================================================
# LLMCallLog
# =============================================================================


def _make_call_log(is_cloud: bool = False, prompt: str = "p", response: str = "r") -> LLMCallLog:
    return LLMCallLog(
        backend="test",
        model="m1",
        tier=ModelTier.FAST,
        prompt=prompt,
        response=response,
        started_at=datetime(2026, 4, 23, 12, 0, 0),
        elapsed_seconds=1.5,
        is_cloud=is_cloud,
    )


class TestLLMCallLog:
    def test_hashes_default_no_full_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GLEAN_LOG_PROMPTS", raising=False)
        log = _make_call_log(prompt="secret", response="sensitive")
        assert len(log.prompt_hash) == 64  # sha256 hex
        assert log.prompt_text is None
        assert log.response_text is None

    def test_full_content_under_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GLEAN_LOG_PROMPTS", "1")
        log = _make_call_log(prompt="full prompt", response="full resp")
        assert log.prompt_text == "full prompt"
        assert log.response_text == "full resp"

    def test_cloud_never_stores_full_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even with GLEAN_LOG_PROMPTS=1, cloud calls redact."""
        monkeypatch.setenv("GLEAN_LOG_PROMPTS", "1")
        log = _make_call_log(is_cloud=True)
        assert log.prompt_text is None
        assert log.response_text is None

    def test_to_log_bullets_shape(self) -> None:
        log = _make_call_log()
        bullets = log.to_log_bullets()
        assert any("backend: test" in b for b in bullets)
        assert any("elapsed: 1.5s" in b for b in bullets)
        assert any("prompt_hash:" in b for b in bullets)

    def test_cloud_bullet_adds_redacted_marker(self) -> None:
        log = _make_call_log(is_cloud=True)
        bullets = log.to_log_bullets()
        assert any("redacted per AGENTS.md" in b for b in bullets)


# =============================================================================
# OllamaBackend
# =============================================================================


def _ollama_config() -> OllamaConfig:
    return OllamaConfig(endpoint="http://localhost:11434")


class TestOllamaBackend:
    def test_instantiates(self) -> None:
        b = OllamaBackend(_ollama_config())
        assert b.name == "ollama"
        assert b.is_cloud is False

    def test_tier_dispatches_to_correct_model(self) -> None:
        b = OllamaBackend(_ollama_config())
        assert b._model_for(ModelTier.FAST) == "llama3.2:3b"
        assert b._model_for(ModelTier.DEEP) == "qwen2.5-coder:32b"

    def test_rejects_empty_prompt(self) -> None:
        b = OllamaBackend(_ollama_config())
        with pytest.raises(GleanLLMError, match=r"empty"):
            b.complete("   \n\n")

    def test_complete_success(self) -> None:
        b = OllamaBackend(_ollama_config())
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "the answer"}
        mock_response.raise_for_status = MagicMock()
        with patch("glean.llm.httpx.post", return_value=mock_response) as mock_post:
            text, log = b.complete("the question", tier=ModelTier.FAST)
        assert text == "the answer"
        assert log.backend == "ollama"
        assert log.model == "llama3.2:3b"
        assert log.is_cloud is False
        # Verify request was well-formed.
        call_args = mock_post.call_args
        assert call_args.args[0] == "http://localhost:11434/api/generate"
        assert call_args.kwargs["json"]["model"] == "llama3.2:3b"
        assert call_args.kwargs["json"]["stream"] is False

    def test_complete_raises_on_http_error(self) -> None:
        b = OllamaBackend(_ollama_config())
        with (
            patch("glean.llm.httpx.post", side_effect=httpx.ConnectError("refused")),
            pytest.raises(GleanLLMError, match=r"Ollama HTTP call failed"),
        ):
            b.complete("q")

    def test_complete_raises_on_empty_response_field(self) -> None:
        b = OllamaBackend(_ollama_config())
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": ""}
        mock_response.raise_for_status = MagicMock()
        with (
            patch("glean.llm.httpx.post", return_value=mock_response),
            pytest.raises(GleanLLMError, match=r"empty"),
        ):
            b.complete("q")


# =============================================================================
# AnthropicBackend
# =============================================================================


class TestAnthropicBackend:
    def test_rejects_when_cloud_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        with pytest.raises(GleanCloudError, match=r"disabled in config"):
            AnthropicBackend(CloudConfig(enabled=False))

    def test_rejects_when_api_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(GleanCloudError, match=r"ANTHROPIC_API_KEY not set"):
            AnthropicBackend(CloudConfig(enabled=True))

    def test_reports_missing_sdk_clearly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When `anthropic` is not installed, the error includes the install hint."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        # Simulate anthropic not being installed by patching __import__ to raise.
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr("builtins.__import__", fake_import)
        # The anthropic module may already be imported from an earlier test; purge.
        monkeypatch.delitem(sys.modules, "anthropic", raising=False)

        with pytest.raises(GleanCloudError, match=r"pip install glean\[cloud\]"):
            AnthropicBackend(CloudConfig(enabled=True))

    def test_happy_path_with_mocked_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Build the backend with a stub `anthropic` module and call complete()."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")

        # Stub the Anthropic class so AnthropicBackend can construct.
        fake_content = MagicMock()
        fake_content.text = "cloud answer"
        fake_message = MagicMock()
        fake_message.content = [fake_content]
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_message

        fake_anthropic_module = MagicMock()
        fake_anthropic_module.Anthropic = MagicMock(return_value=fake_client)
        monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)

        b = AnthropicBackend(CloudConfig(enabled=True))
        text, log = b.complete("prompt")
        assert text == "cloud answer"
        assert log.is_cloud is True
        assert log.backend == "anthropic"


# =============================================================================
# get_backend factory
# =============================================================================


class TestGetBackend:
    def test_non_cloud_returns_ollama(self) -> None:
        cfg = Config()
        b = get_backend(cfg, cloud=False)
        assert isinstance(b, OllamaBackend)

    def test_cloud_with_disabled_config_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
        cfg = Config()  # cloud.enabled defaults to False
        with pytest.raises(GleanCloudError):
            get_backend(cfg, cloud=True)


# =============================================================================
# tier_from_name
# =============================================================================


class TestTierFromName:
    def test_fast(self) -> None:
        assert tier_from_name("fast") == ModelTier.FAST

    def test_deep(self) -> None:
        assert tier_from_name("deep") == ModelTier.DEEP

    def test_unknown_raises(self) -> None:
        with pytest.raises(GleanLLMError, match=r"unknown tier"):
            tier_from_name("medium")  # type: ignore[arg-type]

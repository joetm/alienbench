"""Per-model ``provider`` field: config parsing, validation, and routing."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from alienbench.config import Config
from alienbench.provenance import config_hash

_PROMPTS = [{"id": "baseline", "label": "Baseline", "text": "t"}]


def _cfg(judge_provider=None, model_provider=None, override="openai") -> Config:
    data = {
        "models": [{"id": "openai/gpt-5.4", "provider": model_provider}],
        "judge_models": [{"id": "openai/gpt-5.4", "provider": judge_provider}],
        "prompt_variants": _PROMPTS,
    }
    if override:
        data["judge_overrides"] = {
            "openai/gpt-5.4": {"provider": override, "model_id": "gpt-5.4"},
        }
    return Config(**data)


def test_provider_lookup():
    cfg = _cfg(judge_provider="openai/flex", model_provider="openai")
    assert cfg.provider_for("openai/gpt-5.4") == "openai"
    assert cfg.judge_provider_for("openai/gpt-5.4") == "openai/flex"
    assert cfg.provider_for("unknown/model") is None


def test_allow_fallbacks_is_per_model_and_off_by_default():
    cfg = Config(
        models=[
            {"id": "m/pinned", "provider": "a"},
            {"id": "m/open", "provider": "b", "allow_fallbacks": True},
        ],
        judge_models=["j/x"],
        prompt_variants=_PROMPTS,
    )
    assert cfg.allow_fallbacks_for("m/pinned") is False
    assert cfg.allow_fallbacks_for("m/open") is True
    assert cfg.allow_fallbacks_for("unknown/model") is False


def test_allow_fallbacks_does_not_change_config_hash():
    def cfg(allow: bool) -> Config:
        return Config(
            models=[{"id": "m/a", "provider": "a", "allow_fallbacks": allow}],
            judge_models=["j/x"],
            prompt_variants=_PROMPTS,
        )

    assert config_hash(cfg(False)) == config_hash(cfg(True))


def test_native_judge_rejects_mismatched_provider():
    with pytest.raises(ValueError, match="routes to 'openai'"):
        _cfg(judge_provider="azure")


def test_variant_only_on_native_openai():
    with pytest.raises(ValueError, match="only supported on the native OpenAI"):
        Config(
            models=["m/a"],
            judge_models=[{"id": "anthropic/x", "provider": "anthropic/flex"}],
            judge_overrides={"anthropic/x": {"provider": "anthropic", "model_id": "x"}},
            prompt_variants=_PROMPTS,
        )


def test_openrouter_judge_accepts_any_slug():
    cfg = _cfg(judge_provider="azure", override=None)
    assert cfg.judge_provider_for("openai/gpt-5.4") == "azure"


def test_provider_does_not_change_config_hash():
    assert config_hash(_cfg()) == config_hash(
        _cfg(judge_provider="openai/flex", model_provider="openai/flex")
    )


def test_native_openai_judge_sends_service_tier(monkeypatch):
    from alienbench import judges

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    fake = MagicMock()
    monkeypatch.setattr("openai.OpenAI", lambda **_: fake)
    choice = MagicMock()
    choice.message.content = "{}"
    fake.chat.completions.create.return_value = MagicMock(choices=[choice], usage=None)

    judge = judges.make_judge("openai/gpt-5.4", _cfg(judge_provider="openai/flex"))
    judge.complete(prompt="p", temperature=0.0, max_tokens=10)

    assert fake.chat.completions.create.call_args.kwargs["service_tier"] == "flex"


def test_openrouter_judge_passes_provider(monkeypatch):
    from alienbench import client as client_module, judges

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(client_module, "OpenAI", lambda **_: MagicMock())
    judge = judges.make_judge("openai/gpt-5.4", _cfg(judge_provider="openai/flex", override=None))
    judge._client.complete = MagicMock()
    judge.complete(prompt="p", temperature=0.0, max_tokens=10)
    assert judge._client.complete.call_args.kwargs["provider"] == "openai/flex"

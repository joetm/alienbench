"""Connectivity smoke tests for the live judge panel.

Marked ``live``; skipped when the required API key is absent.  Run explicitly::

    pytest -m live alienbench/tests/test_judge_connectivity.py

One trivial request is sent to each judge listed in ``config.yaml``'s
``judge_overrides``, confirming the model is reachable and returns a
non-empty response.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_LIVE_CONFIG = Path(__file__).parent.parent / "config.yaml"

_PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def _judge_aliases() -> list[str]:
    from alienbench.config import load_config
    cfg = load_config(_LIVE_CONFIG)
    return list(cfg.judge_overrides.keys())


@pytest.mark.live
@pytest.mark.parametrize("alias", _judge_aliases())
def test_judge_reachable(alias: str) -> None:
    from alienbench.config import load_config
    from alienbench.judges import make_judge

    cfg = load_config(_LIVE_CONFIG)
    override = cfg.judge_overrides[alias]
    env_var = override.api_key_env or _PROVIDER_KEY_ENV[override.provider]
    if not os.environ.get(env_var):
        pytest.skip(f"{env_var} not set")

    judge = make_judge(alias, cfg)
    response = judge.complete(
        prompt="Reply with the single word: hello",
        temperature=0.0,
        max_tokens=10,
    )
    assert response.text.strip(), f"Judge {alias!r} returned an empty response"

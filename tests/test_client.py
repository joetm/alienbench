"""Unit tests for OpenRouterClient response-shape handling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from alienbench import client as client_module
from alienbench.client import MalformedOpenRouterResponse, OpenRouterClient


def _well_formed(content: str = "hello") -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2)
    return SimpleNamespace(
        choices=[choice], model="m-resolved", usage=usage, id="gen-123",
    )


def _null_choices() -> SimpleNamespace:
    return SimpleNamespace(choices=None, model="m-resolved", usage=None, id="gen-err")


@pytest.fixture
def _no_sleep(monkeypatch):
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)


@pytest.fixture
def _client(monkeypatch) -> OpenRouterClient:
    cfg = SimpleNamespace(
        api_key="k",
        openrouter_base_url="https://example",
        allowed_providers=None,
        allow_provider_fallbacks=False,
    )
    monkeypatch.setattr(client_module, "OpenAI", lambda **_: MagicMock())
    return OpenRouterClient(cfg)


def test_complete_retries_on_null_choices(_no_sleep, _client):
    create = _client._client.chat.completions.create
    create.side_effect = [_null_choices(), _well_formed("recovered")]

    resp = _client.complete(model="m", prompt="p", temperature=1.0, max_tokens=10)

    assert resp.text == "recovered"
    assert create.call_count == 2


def test_complete_gives_up_after_max_retries(_no_sleep, _client):
    create = _client._client.chat.completions.create
    create.side_effect = [_null_choices()] * (client_module._MAX_RETRIES + 2)

    with pytest.raises(RuntimeError, match="Failed after"):
        _client.complete(model="m", prompt="p", temperature=1.0, max_tokens=10)

    assert create.call_count == client_module._MAX_RETRIES


def test_malformed_response_is_retryable():
    exc = MalformedOpenRouterResponse("boom")
    assert client_module._openrouter_retryable(exc) is True

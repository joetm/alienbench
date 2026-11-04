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


def test_default_provider_block_allows_any_provider(monkeypatch):
    cfg = SimpleNamespace(
        api_key="k",
        openrouter_base_url="https://example",
        allowed_providers=None,
        allow_provider_fallbacks=False,
    )
    monkeypatch.setattr(client_module, "OpenAI", lambda **_: MagicMock())
    c = OpenRouterClient(cfg)
    provider = c._extra_body["provider"]
    assert provider["allow_fallbacks"] is True
    assert provider["require_parameters"] is False


def test_per_call_provider_pins_routing(_client):
    create = _client._client.chat.completions.create
    create.return_value = _well_formed()

    _client.complete(model="m", prompt="p", temperature=1.0, max_tokens=10,
                     provider="openai/flex")

    provider = create.call_args.kwargs["extra_body"]["provider"]
    assert provider["order"] == ["openai/flex"]
    assert provider["allow_fallbacks"] is False
    # The client-wide block is left untouched for other models.
    assert _client._extra_body["provider"]["allow_fallbacks"] is True


def test_per_call_provider_can_allow_fallbacks(_client):
    create = _client._client.chat.completions.create
    create.return_value = _well_formed()

    _client.complete(model="m", prompt="p", temperature=1.0, max_tokens=10,
                     provider="decart/fp4", allow_fallbacks=True)

    provider = create.call_args.kwargs["extra_body"]["provider"]
    assert provider["order"] == ["decart/fp4"]
    assert provider["allow_fallbacks"] is True


def test_response_records_the_provider_that_served_the_call(_client):
    create = _client._client.chat.completions.create
    create.return_value = SimpleNamespace(**vars(_well_formed()), provider="Relace")

    resp = _client.complete(model="m", prompt="p", temperature=1.0, max_tokens=10)

    assert resp.provider == "Relace"


def test_no_provider_keeps_default_routing(_client):
    create = _client._client.chat.completions.create
    create.return_value = _well_formed()

    _client.complete(model="m", prompt="p", temperature=1.0, max_tokens=10)

    assert create.call_args.kwargs["extra_body"] is _client._extra_body


def test_complete_records_finish_reason_and_reasoning_tokens(_client):
    result = _well_formed("cut off")
    result.choices[0].finish_reason = "length"
    result.usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=1500)
    _client._client.chat.completions.create.return_value = result

    resp = _client.complete(model="m", prompt="p", temperature=1.0, max_tokens=10)

    assert resp.finish_reason == "length"
    assert resp.reasoning_tokens == 1500


def test_complete_defaults_reasoning_tokens_when_unreported(_client):
    _client._client.chat.completions.create.return_value = _well_formed()

    resp = _client.complete(model="m", prompt="p", temperature=1.0, max_tokens=10)

    assert resp.reasoning_tokens == 0

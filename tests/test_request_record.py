"""Every generation and extraction record stores the request exactly as sent.

The ``request`` field holds the API name and every argument of the call:
messages, sampling parameters, routing, and provider settings. Extraction
records also keep the judge's raw response text.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from alienbench import extract, generate
from alienbench.client import OpenRouterClient, Response
from alienbench.paths import cut_offs_path, extractions_path, generations_path
from alienbench.tests.test_pipeline import FAKE_CREATURE, _make_extraction_response
from alienbench.tests.test_token_limits import JUDGE, MODEL, _config, _read


def _openrouter_result(content: str, finish: str = "stop"):
    return SimpleNamespace(
        id="or-1", model=MODEL,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                 finish_reason=finish, native_finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7,
                              completion_tokens_details=None))


class _FakeOpenRouter(OpenRouterClient):
    """The real OpenRouterClient over a fake SDK that is cut off below 2000 tokens."""

    sent: list[dict] = []

    def __init__(self, cfg):
        self._extra_body = {"provider": {"order": ["acme"], "allow_fallbacks": False}}

        def create(**kwargs):
            type(self).sent.append(kwargs)
            cut = kwargs["max_tokens"] < 2000
            return _openrouter_result("" if cut else FAKE_CREATURE,
                                      "length" if cut else "stop")

        self._client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))


def test_generation_applies_fallbacks_and_records_the_served_provider(tmp_path, monkeypatch):
    path = Path(_config(tmp_path))
    path.write_text(path.read_text().replace(
        f"  - {MODEL}\n",
        f"  - id: {MODEL}\n    provider: gone/fp4\n    allow_fallbacks: true\n",
    ))
    sent: list[dict] = []

    class FakeClient:
        def __init__(self, cfg):
            pass

        def complete(self, **kwargs):
            sent.append(kwargs)
            return Response(FAKE_CREATURE, MODEL, 5, 7, generation_id="g",
                            finish_reason="stop", provider="Relace")

    monkeypatch.setattr(generate, "OpenRouterClient", FakeClient)
    generate.run(str(path))

    assert sent[0]["provider"] == "gone/fp4"
    assert sent[0]["allow_fallbacks"] is True
    rec = _read(generations_path(tmp_path / "data", MODEL, "baseline"))[0]
    assert rec["served_provider"] == "Relace"


def test_generation_record_stores_the_request(tmp_path, monkeypatch):
    _FakeOpenRouter.sent = []
    monkeypatch.setattr(generate, "OpenRouterClient", _FakeOpenRouter)
    generate.run(_config(tmp_path))
    path = generations_path(tmp_path / "data", MODEL, "baseline")
    rec = _read(path)[0]
    req = rec["request"]
    assert req["api"] == "openrouter/chat.completions"
    assert req["model"] == MODEL
    assert req["messages"] == [{"role": "user", "content": rec["prompt_text"]}]
    assert req["temperature"] == 1.0
    assert req["max_tokens"] == 2000
    assert req["seed"] == rec["seed"]
    assert req["extra_body"] == {"provider": {"order": ["acme"], "allow_fallbacks": False}}
    # The discarded cut-off call keeps its own request.
    cut = [c for c in _read(cut_offs_path(path)) if c["sample_index"] == rec["sample_index"]]
    assert [c["request"]["max_tokens"] for c in cut] == [1000]


def test_extraction_record_stores_request_and_raw_text(tmp_path, monkeypatch):
    monkeypatch.setattr(generate, "OpenRouterClient", _FakeOpenRouter)
    cfg = _config(tmp_path)
    generate.run(cfg)

    def factory(alias, c):
        class Judge:
            def complete(self, prompt, temperature, max_tokens, system=None):
                return Response(
                    _make_extraction_response(), "judge-resolved", 1, 2,
                    generation_id="j", finish_reason="stop",
                    request={"api": "fake", "prompt": prompt, "system": system,
                             "temperature": temperature, "max_tokens": max_tokens})
        return Judge()

    monkeypatch.setattr(extract, "make_judge", factory)
    extract.run(cfg)
    rec = _read(extractions_path(tmp_path / "data", JUDGE, MODEL, "baseline"))[0]
    assert rec["raw_response"] == _make_extraction_response()
    assert rec["request"]["max_tokens"] == 500
    assert rec["request"]["temperature"] == 0.0
    assert FAKE_CREATURE in rec["request"]["prompt"]
    assert rec["request"]["system"] == extract._SYSTEM


def _anthropic_message():
    return SimpleNamespace(content=[SimpleNamespace(type="text", text="{}")],
                           stop_reason="end_turn", model="claude", id="a",
                           usage=SimpleNamespace(input_tokens=1, output_tokens=2))


def _anthropic_judge(bedrock: bool):
    from alienbench.judges import _AnthropicJudge

    judge = _AnthropicJudge.__new__(_AnthropicJudge)
    judge._model_id = "claude-opus-4-6"
    judge._bedrock_model_id = "us.anthropic.claude-opus-4-6-v1"
    judge._supports_temperature = True
    judge._sdk = SimpleNamespace(BadRequestError=type("B", (Exception,), {}),
                                 RateLimitError=type("R", (Exception,), {}),
                                 APIStatusError=type("S", (Exception,), {}))
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: _anthropic_message()))
    judge._client = client
    judge._bedrock_client = client if bedrock else None
    judge._bedrock_until = float("inf") if bedrock else 0.0
    judge._bedrock_cooldown_seconds = 30.0
    return judge


def test_anthropic_and_bedrock_requests():
    for bedrock, api, model in ((False, "anthropic/messages", "claude-opus-4-6"),
                                (True, "bedrock/messages", "us.anthropic.claude-opus-4-6-v1")):
        req = _anthropic_judge(bedrock).complete("p", 0.0, 100, system="s").request
        assert req == {"api": api, "model": model, "max_tokens": 100,
                       "messages": [{"role": "user", "content": "p"}],
                       "temperature": 0.0, "system": "s"}
        json.dumps(req)


def test_openai_request_includes_service_tier():
    from alienbench.judges import _OpenAIJudge

    r = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
        model="gpt", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2), id="o")
    judge = _OpenAIJudge.__new__(_OpenAIJudge)
    judge._model_id = "gpt-5.4"
    judge._service_tier = "flex"
    judge._client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: r)))
    judge._RateLimitError = type("R", (Exception,), {})
    judge._APIStatusError = type("S", (Exception,), {})
    req = judge.complete("p", 0.0, 100, system="s").request
    assert req["api"] == "openai/chat.completions"
    assert req["service_tier"] == "flex"
    assert req["max_completion_tokens"] == 100
    assert req["messages"] == [{"role": "system", "content": "s"},
                               {"role": "user", "content": "p"}]


def test_google_request_serializes_the_sdk_config():
    from google.genai import errors, types

    from alienbench.judges import _GoogleJudge

    r = SimpleNamespace(
        text="{}", response_id="g", model_version="gemini",
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=2))
    judge = _GoogleJudge.__new__(_GoogleJudge)
    judge._model_id = "gemini-3.1-pro-preview"
    judge._use_vertex = True
    judge._types = types
    judge._errors = errors
    judge._thinking_probe_done = True
    judge._supports_disabled_thinking = True
    judge._client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: r))
    req = judge.complete("p", 0.0, 100, system="s").request
    assert req["api"] == "google/vertex/generate_content"
    assert req["model"] == "gemini-3.1-pro-preview"
    assert req["contents"] == "p"
    assert req["config"]["max_output_tokens"] == 100
    assert req["config"]["temperature"] == 0.0
    assert req["config"]["system_instruction"] == "s"
    assert req["config"]["response_mime_type"] == "application/json"
    assert req["config"]["thinking_config"] == {"thinking_budget": 0}
    json.dumps(req)

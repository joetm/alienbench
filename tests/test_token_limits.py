"""Tests for output-token limits in generation and extraction.

A call starts at the configured limit. A response cut off at the limit is
discarded and logged in ``cut_offs.jsonl``, and the call is repeated with
the limit doubled, up to the ceiling. A response still cut off at the
ceiling is not recorded, so the next run retries it. The same rule holds
for subject models and judges.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alienbench import extract, generate
from alienbench.claims import limit_section
from alienbench.client import CutOffAtCeiling, Response, complete_within_limit
from alienbench.config import load_config
from alienbench.paths import cut_offs_path, extractions_path, generations_path
from alienbench.tests.test_pipeline import _make_extraction_response

MODEL = "acme/model-a"
JUDGE = "acme/judge"


def _config(tmp_path: Path, extra: str = "") -> str:
    text = f"""
models:
  - {MODEL}
judge_models:
  - {JUDGE}
samples_per_condition: 2
max_tokens: 1000
max_tokens_ceiling: 4000
judge_max_tokens: 500
judge_max_tokens_ceiling: 1500
prompt_variants:
  - id: baseline
    label: Baseline
    text: "Imagine a creature that lives on an alien planet. Describe it in detail."
data_dir: {tmp_path}/data
results_dir: {tmp_path}/results
{extra}
"""
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return str(path)


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _response(finish: str, text: str = "text") -> Response:
    return Response(text, MODEL, 10, 20, generation_id="g", finish_reason=finish)


class _Client:
    """Fake OpenRouter client that is cut off below ``needs`` tokens."""

    needs = 0
    calls: list[int] = []

    def __init__(self, cfg):
        pass

    def complete(self, model, prompt, temperature, max_tokens, system=None,
                 seed=None, provider=None, allow_fallbacks=False):
        type(self).calls.append(max_tokens)
        return _response("length" if max_tokens < self.needs else "stop")


@pytest.fixture()
def client(monkeypatch):
    _Client.calls = []
    _Client.needs = 0
    monkeypatch.setattr(generate, "OpenRouterClient", _Client)
    return _Client


# ---------------------------------------------------------------------------
# complete_within_limit
# ---------------------------------------------------------------------------

def test_limit_doubles_until_the_response_ends():
    limits = []

    def call(limit):
        limits.append(limit)
        return _response("length" if limit < 3000 else "stop")

    response, limit, cut_offs = complete_within_limit(
        call, max_tokens=1000, ceiling=8000, label="x")
    assert limits == [1000, 2000, 4000]
    assert limit == 4000
    assert response.finish_reason == "stop"
    assert [c["max_tokens"] for c in cut_offs] == [1000, 2000]


def test_last_step_is_clipped_to_the_ceiling():
    limits = []

    def call(limit):
        limits.append(limit)
        return _response("length")

    with pytest.raises(CutOffAtCeiling) as exc:
        complete_within_limit(call, max_tokens=1000, ceiling=3000, label="x")
    assert limits == [1000, 2000, 3000]
    assert len(exc.value.cut_offs) == 3


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def test_generation_records_the_raised_limit(tmp_path, client):
    client.needs = 2000
    generate.run(_config(tmp_path))
    path = generations_path(tmp_path / "data", MODEL, "baseline")
    records = _read(path)
    assert sorted(r["sample_index"] for r in records) == [0, 1]
    assert {r["max_tokens"] for r in records} == {2000}
    assert {r["n_cut_offs"] for r in records} == {1}
    assert {r["finish_reason"] for r in records} == {"stop"}
    cut_offs = _read(cut_offs_path(path))
    assert sorted(c["sample_index"] for c in cut_offs) == [0, 1]
    assert {c["max_tokens"] for c in cut_offs} == {1000}


def test_generation_without_cut_off_records_the_start_limit(tmp_path, client):
    generate.run(_config(tmp_path))
    path = generations_path(tmp_path / "data", MODEL, "baseline")
    assert {(r["max_tokens"], r["n_cut_offs"]) for r in _read(path)} == {(1000, 0)}
    assert not cut_offs_path(path).exists()


def test_generation_cut_off_at_ceiling_is_not_recorded(tmp_path, client):
    client.needs = 10 ** 9
    cfg = _config(tmp_path)
    generate.run(cfg)
    path = generations_path(tmp_path / "data", MODEL, "baseline")
    assert _read(path) == []
    assert len(_read(cut_offs_path(path))) == 2 * 3  # 1000, 2000, 4000 per sample

    # The slots are not done: the next run retries them.
    client.needs = 0
    generate.run(cfg)
    assert sorted(r["sample_index"] for r in _read(path)) == [0, 1]


def test_model_override_sets_start_and_ceiling(tmp_path, client):
    client.needs = 10 ** 9
    generate.run(_config(tmp_path, f"""
model_overrides:
  {MODEL}:
    max_tokens: 3000
    max_tokens_ceiling: 5000
"""))
    assert client.calls[:3] == [3000, 5000, 3000]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _judge_factory(needs: int, calls: list[int]):
    def factory(alias, cfg):
        class Judge:
            def complete(self, prompt, temperature, max_tokens, system=None):
                calls.append(max_tokens)
                finish = "length" if max_tokens < needs else "stop"
                return Response(_make_extraction_response(), "judge-resolved", 100, 200,
                                generation_id="j", finish_reason=finish)
        return Judge()
    return factory


def test_extraction_records_the_raised_limit(tmp_path, client, monkeypatch):
    cfg = _config(tmp_path)
    generate.run(cfg)
    calls: list[int] = []
    monkeypatch.setattr(extract, "make_judge", _judge_factory(1000, calls))
    extract.run(cfg)
    path = extractions_path(tmp_path / "data", JUDGE, MODEL, "baseline")
    records = _read(path)
    assert len(records) == 2
    assert {(r["max_tokens"], r["n_cut_offs"]) for r in records} == {(1000, 1)}
    cut_offs = _read(cut_offs_path(path))
    assert len(cut_offs) == 2
    assert {c["max_tokens"] for c in cut_offs} == {500}
    assert {c["generation_id"] for c in cut_offs} == {r["generation_id"] for r in records}


def test_extraction_cut_off_at_ceiling_is_not_recorded(tmp_path, client, monkeypatch):
    cfg = _config(tmp_path)
    generate.run(cfg)
    calls: list[int] = []
    monkeypatch.setattr(extract, "make_judge", _judge_factory(10 ** 9, calls))
    extract.run(cfg)
    path = extractions_path(tmp_path / "data", JUDGE, MODEL, "baseline")
    assert _read(path) == []
    assert calls == [500, 1000, 1500] * 2
    assert len(_read(cut_offs_path(path))) == 6

    monkeypatch.setattr(extract, "make_judge", _judge_factory(0, calls))
    extract.run(cfg)
    assert len(_read(path)) == 2


def _edit(path: str, old: str, new: str) -> str:
    text = Path(path).read_text()
    assert old in text
    Path(path).write_text(text.replace(old, new))
    return path


def test_judge_entry_sets_start_and_ceiling(tmp_path):
    cfg = load_config(_edit(
        _config(tmp_path), f"  - {JUDGE}\n",
        f"  - id: {JUDGE}\n    max_tokens: 700\n    max_tokens_ceiling: 900\n"))
    assert cfg.judge_max_tokens_for(JUDGE) == 700
    assert cfg.judge_max_tokens_ceiling_for(JUDGE) == 900


# ---------------------------------------------------------------------------
# Native judges report cut-offs
# ---------------------------------------------------------------------------

def test_anthropic_judge_reports_cut_off_without_text():
    from alienbench.judges import _AnthropicJudge

    judge = _AnthropicJudge.__new__(_AnthropicJudge)
    judge._model_id = "claude"
    r = SimpleNamespace(content=[], stop_reason="max_tokens", model="claude", id="a",
                        usage=SimpleNamespace(input_tokens=1, output_tokens=8))
    response = judge._format_response(r, {})
    assert response.finish_reason == "length"
    assert response.native_finish_reason == "max_tokens"

    r.stop_reason = "end_turn"
    r.content = [SimpleNamespace(type="text", text="{}")]
    assert judge._format_response(r, {}).finish_reason == "stop"


def test_openai_judge_reports_cut_off(monkeypatch):
    from alienbench.judges import _OpenAIJudge

    r = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="length")],
        model="gpt", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=8), id="o")
    judge = _OpenAIJudge.__new__(_OpenAIJudge)
    judge._model_id = "gpt"
    judge._service_tier = None
    judge._client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: r)))
    judge._RateLimitError = type("RateLimitError", (Exception,), {})
    judge._APIStatusError = type("APIStatusError", (Exception,), {})
    judge._APIConnectionError = type("APIConnectionError", (Exception,), {})
    response = judge.complete("p", 0.0, 10)
    assert response.finish_reason == "length"


def test_google_judge_reports_cut_off():
    from alienbench.judges import _GoogleJudge

    r = SimpleNamespace(
        text="", response_id="g", model_version="gemini",
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))],
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=8))
    judge = _GoogleJudge.__new__(_GoogleJudge)
    judge._model_id = "gemini"
    judge._use_vertex = True
    judge._thinking_probe_done = True
    judge._supports_disabled_thinking = True
    judge._errors = SimpleNamespace(APIError=type("APIError", (Exception,), {}))
    judge._build_config = lambda *a: None
    judge._client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: r))
    response = judge.complete("p", 0.0, 10)
    assert response.finish_reason == "length"
    assert response.native_finish_reason == "MAX_TOKENS"


# ---------------------------------------------------------------------------
# Config and report
# ---------------------------------------------------------------------------

def test_start_limit_above_ceiling_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at most its ceiling"):
        load_config(_edit(_config(tmp_path), "max_tokens_ceiling: 4000",
                          "max_tokens_ceiling: 900"))


def test_report_lists_unresolved_cut_offs(tmp_path, client):
    client.needs = 10 ** 9
    cfg_path = _config(tmp_path)
    generate.run(cfg_path)
    lines, problems = limit_section(load_config(cfg_path))
    assert problems == ["2 samples of model-a cut off at the ceiling of 4000 tokens"]
    assert any(l.startswith("| model-a | 1000 | 4000 | 0 | 0 | 0 | 6 | **2** |") for l in lines)


def test_openrouter_returns_empty_cut_off_response(monkeypatch):
    # Hidden reasoning can use the whole limit and leave no text. The client
    # must return the cut-off response so the limit is raised, not raise.
    from alienbench.client import OpenRouterClient

    result = SimpleNamespace(
        id="o", model=MODEL,
        choices=[SimpleNamespace(message=SimpleNamespace(content=""),
                                 finish_reason="length", native_finish_reason="length")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=8,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=8)))
    client = OpenRouterClient.__new__(OpenRouterClient)
    client._extra_body = {}
    client._client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: result)))
    response = client.complete(model=MODEL, prompt="p", temperature=1.0, max_tokens=8)
    assert response.finish_reason == "length"
    assert response.text == ""

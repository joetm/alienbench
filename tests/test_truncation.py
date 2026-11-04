"""Tests for tracking how each response ended.

Every subject model must end its description on its own stop token. The
harness reports every other ending: a response cut off at ``max_tokens``, a
response without a finish reason (the provider's stream ended early), and a
response stopped by a provider refusal. The claims report covers the main
conditions and the paraphrase ablation. The generation stage reports the
same counts when it finishes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from alienbench import generate
from alienbench.claims import truncation_rows, truncation_section
from alienbench.client import Response
from alienbench.config import Config, load_config
from alienbench.paths import ending, generations_path


def _config(tmp_path: Path, n: int = 2) -> Config:
    text = f"""
models:
  - acme/model-a
judge_models:
  - acme/judge
samples_per_condition: {n}
prompt_ablation_samples_per_condition: 2
max_tokens: 16000
prompt_variants:
  - id: baseline
    label: Baseline
    text: "Imagine a creature that lives on an alien planet. Describe it in detail."
prompt_paraphrases:
  - id: baseline
    label: Baseline
    text: "Imagine a creature that lives on an alien planet. Describe it in detail."
  - id: baseline_para_a
    label: Paraphrase A
    text: "Picture a creature from an alien world. Describe it in detail."
data_dir: {tmp_path}/data
results_dir: {tmp_path}/results
"""
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return load_config(str(path))


def _write(cfg: Config, prompt_id: str, reasons: list) -> None:
    """Write one record per entry. An entry is a finish_reason, or a
    (finish_reason, native_finish_reason) pair."""
    path = generations_path(Path(cfg.data_dir), "acme/model-a", prompt_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for i, reason in enumerate(reasons):
            finish, native = reason if isinstance(reason, tuple) else (reason, None)
            record = {"id": f"{prompt_id}-{i}", "sample_index": i, "response": "text"}
            if finish is not None:
                record["finish_reason"] = finish
            if native is not None:
                record["native_finish_reason"] = native
            fh.write(json.dumps(record) + "\n")


@pytest.mark.parametrize("record, expected", [
    ({"finish_reason": "stop"}, "stop"),
    ({"finish_reason": "stop", "native_finish_reason": "end_turn"}, "stop"),
    ({"finish_reason": "length"}, "truncated"),
    ({}, "no_finish_reason"),
    ({"finish_reason": ""}, "no_finish_reason"),
    ({"finish_reason": "stop", "native_finish_reason": "refusal"}, "refusal"),
    ({"finish_reason": "content_filter"}, "other"),
])
def test_ending(record, expected):
    assert ending(record) == expected


def test_counts_each_ending(tmp_path):
    cfg = _config(tmp_path)
    _write(cfg, "baseline", ["stop", "length"])
    _write(cfg, "baseline_para_a", [None, ("stop", "refusal"), "content_filter"])
    [row] = truncation_rows(cfg)
    assert row["responses"] == 5
    assert row["stop"] == 1
    assert row["truncated"] == 1
    assert row["no_finish_reason"] == 1
    assert row["refusal"] == 1
    assert row["other"] == 1


def test_baseline_shared_with_paraphrases_is_counted_once(tmp_path):
    cfg = _config(tmp_path)
    _write(cfg, "baseline", ["stop", "stop"])
    [row] = truncation_rows(cfg)
    assert row["responses"] == 2


def test_report_flags_each_problem(tmp_path):
    cfg = _config(tmp_path)
    _write(cfg, "baseline", ["stop", "length"])
    _write(cfg, "baseline_para_a", [None, ("stop", "refusal")])
    lines, problems = truncation_section(cfg)
    report = "\n".join(lines)
    assert len(problems) == 3
    assert "1 response ended at `max_tokens`" in report
    assert "1 response has no finish reason" in report
    assert "1 response was stopped by a provider refusal" in report


def test_report_passes_when_every_response_stops(tmp_path):
    cfg = _config(tmp_path)
    _write(cfg, "baseline", ["stop", "stop"])
    _write(cfg, "baseline_para_a", ["stop", "stop"])
    lines, problems = truncation_section(cfg)
    assert problems == []
    assert "Every response ended on the model's own stop token." in lines


def test_generation_reports_endings_when_it_finishes(tmp_path, monkeypatch, caplog):
    cfg = _config(tmp_path, n=3)
    endings = iter([("stop", "stop"), ("stop", "refusal"), ("", "")])

    class FakeClient:
        def __init__(self, cfg):
            pass

        def complete(self, model, prompt, temperature, max_tokens, system=None,
                     seed=None, provider=None, allow_fallbacks=False):
            finish, native = next(endings)
            return Response("text", model, 10, 20, generation_id="g",
                            finish_reason=finish, native_finish_reason=native)

    monkeypatch.setattr(generate, "OpenRouterClient", FakeClient)
    with caplog.at_level(logging.WARNING, logger="alienbench.generate"):
        generate.run(str(tmp_path / "config.yaml"))
    messages = [r.getMessage() for r in caplog.records]
    per_call = [m for m in messages if "sample_index=" in m]
    assert len(per_call) == 2
    assert "was stopped by a provider refusal" in per_call[0]
    assert "has no finish reason" in per_call[1]
    summary = [m for m in messages if m.startswith("2 of 3 responses")]
    assert summary == ["2 of 3 responses did not end on the model's own stop token: "
                       "1 has no finish reason, 1 was stopped by a provider refusal. "
                       "Run `python -m alienbench paper` for the per-model report "
                       "in results/claims_check.md."]

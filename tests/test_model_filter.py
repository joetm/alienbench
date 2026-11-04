"""Tests for ``--model``: run the API stages for selected subject models.

``--model`` restricts generation, extraction, and scoring to the named
subject models. Every judge in the panel still scores their responses.
Analysis stages always cover the full panel in the config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alienbench import __main__ as cli
from alienbench import generate
from alienbench.client import Response
from alienbench.config import load_config
from alienbench.paths import generations_path, load_completed_sample_indices


def _config_path(tmp_path: Path) -> str:
    text = f"""
models:
  - id: acme/model-a
    name: "Model A"
    key: ModelA
  - id: acme/model-b
    name: "Model B"
    key: ModelB
judge_models:
  - acme/judge-1
  - acme/judge-2
samples_per_condition: 2
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
    return str(path)


@pytest.mark.parametrize("selector", ["acme/model-b", "ModelB", "modelb", "Model B", "model b"])
def test_only_models_accepts_id_key_or_name(tmp_path, selector):
    cfg = load_config(_config_path(tmp_path)).only_models([selector])
    assert cfg.models == ["acme/model-b"]
    assert cfg.judge_models == ["acme/judge-1", "acme/judge-2"]


def test_only_models_keeps_config_order_and_drops_duplicates(tmp_path):
    cfg = load_config(_config_path(tmp_path)).only_models(["ModelB", "acme/model-a", "Model B"])
    assert cfg.models == ["acme/model-a", "acme/model-b"]


def test_only_models_rejects_unknown_model(tmp_path):
    cfg = load_config(_config_path(tmp_path))
    with pytest.raises(ValueError, match="acme/model-a, acme/model-b"):
        cfg.only_models(["acme/model-z"])


def test_generate_runs_only_the_selected_model(tmp_path, monkeypatch):
    class FakeClient:
        def __init__(self, cfg):
            pass

        def complete(self, model, prompt, temperature, max_tokens, system=None,
                     seed=None, provider=None, allow_fallbacks=False):
            return Response("text", model, 10, 20, generation_id="g",
                            finish_reason="stop", native_finish_reason="stop")

    monkeypatch.setattr(generate, "OpenRouterClient", FakeClient)
    path = _config_path(tmp_path)
    generate.run(path, models=["ModelB"])
    data_dir = Path(load_config(path).data_dir)
    assert load_completed_sample_indices(generations_path(data_dir, "acme/model-b", "baseline")) == {0, 1}
    assert not generations_path(data_dir, "acme/model-a", "baseline").exists()


@pytest.fixture()
def calls(monkeypatch):
    """Record every stage call the CLI makes instead of running it."""
    recorded = []

    def fake_runner(stage):
        def run(config_path, human_models=None, models=None):
            recorded.append((stage, models, human_models))
        return run

    monkeypatch.setattr(cli, "_stage_runner", fake_runner)
    monkeypatch.setattr(cli, "_record_manifest", lambda stage, config_path: None)
    return recorded


def test_run_passes_model_to_api_stages_only(tmp_path, calls):
    cli.main(["run", "--config", _config_path(tmp_path), "--model", "ModelB"])
    assert calls == [
        ("generate", ["ModelB"], None),
        ("extract", ["ModelB"], None),
        ("score", ["ModelB"], None),
        ("analyze", None, None),
        ("latex", None, None),
        ("paper", None, None),
    ]


def test_model_can_be_repeated(tmp_path, calls):
    cli.main(["generate", "--config", _config_path(tmp_path),
              "--model", "ModelA", "--model", "ModelB"])
    assert calls == [("generate", ["ModelA", "ModelB"], None)]


def test_model_combines_with_human(tmp_path, calls):
    cli.main(["extract", "--config", _config_path(tmp_path), "--model", "ModelA", "--human"])
    assert calls == [("extract", ["ModelA"], [cli._HUMAN_MODEL_KEY])]


def test_unknown_model_is_a_usage_error(tmp_path, calls):
    with pytest.raises(SystemExit):
        cli.main(["run", "--config", _config_path(tmp_path), "--model", "acme/model-z"])
    assert calls == []


def test_prompt_ablation_passes_model(tmp_path, monkeypatch):
    seen = {}
    import alienbench.ablation_prompt as abl
    monkeypatch.setattr(abl, "run", lambda config_path, models=None: seen.update(models=models))
    cli.main(["ablation", "prompt", "--config", _config_path(tmp_path), "--model", "ModelB"])
    assert seen == {"models": ["ModelB"]}


@pytest.mark.parametrize("name", ["dimensions", "judges"])
def test_offline_ablations_reject_model(tmp_path, name):
    with pytest.raises(SystemExit):
        cli.main(["ablation", name, "--config", _config_path(tmp_path), "--model", "ModelB"])

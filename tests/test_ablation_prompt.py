"""Unit tests for ablation_prompt (no API calls)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import yaml

# Match test_pipeline.py: mock krippendorff before any alienbench import.
_mock_krippendorff = types.ModuleType("krippendorff")
_mock_krippendorff.alpha = lambda reliability_data, level_of_measurement: 0.85
sys.modules.setdefault("krippendorff", _mock_krippendorff)

from alienbench.ablation_prompt import (  # noqa: E402
    MAD_STABLE_THRESHOLD,
    RHO_STABLE_THRESHOLD,
    _format_latex_table,
    _mad_matrix,
    _materialize_paraphrase_config,
    _per_model_variant_means,
    _permutation_null_spearman,
    _spearman_matrix,
)
from alienbench.config import Config, PromptVariant  # noqa: E402


def _make_cfg(tmp_path: Path) -> Config:
    return Config(
        models=["m1", "m2", "m3"],
        judge_models=["j1"],
        prompt_variants=[PromptVariant(id="baseline", label="B", text="t")],
        prompt_paraphrases=[
            PromptVariant(id="baseline", label="B", text="t"),
            PromptVariant(id="baseline_para_a", label="A", text="ta"),
        ],
        samples_per_condition=3,
        data_dir=str(tmp_path / "data"),
        results_dir=str(tmp_path / "results"),
    )


def test_materialize_paraphrase_config_swaps_prompt_variants(tmp_path):
    cfg = _make_cfg(tmp_path)
    out = tmp_path / "shadow.yaml"
    _materialize_paraphrase_config(cfg, out)
    loaded = yaml.safe_load(out.read_text())
    assert [p["id"] for p in loaded["prompt_variants"]] == [
        "baseline",
        "baseline_para_a",
    ]
    assert loaded["models"] == ["m1", "m2", "m3"]
    assert loaded["judge_models"] == ["j1"]


def test_per_model_variant_means_averages_across_judges_then_generations():
    # Two judges agree on m1/baseline=8, disagree on m2/baseline (6 vs 4 -> 5)
    rows = [
        {"generation_id": "g1", "judge_model": "j1", "subject_model": "m1",
         "prompt_variant": "baseline", "ward_score": 8},
        {"generation_id": "g1", "judge_model": "j2", "subject_model": "m1",
         "prompt_variant": "baseline", "ward_score": 8},
        {"generation_id": "g2", "judge_model": "j1", "subject_model": "m2",
         "prompt_variant": "baseline", "ward_score": 6},
        {"generation_id": "g2", "judge_model": "j2", "subject_model": "m2",
         "prompt_variant": "baseline", "ward_score": 4},
        {"generation_id": "g3", "judge_model": "j1", "subject_model": "m1",
         "prompt_variant": "para_a", "ward_score": 7},
        {"generation_id": "g4", "judge_model": "j1", "subject_model": "m2",
         "prompt_variant": "para_a", "ward_score": 3},
    ]
    wide = _per_model_variant_means(pd.DataFrame(rows))
    assert wide.loc["m1", "baseline"] == pytest.approx(8.0)
    assert wide.loc["m2", "baseline"] == pytest.approx(5.0)
    assert wide.loc["m1", "para_a"] == pytest.approx(7.0)
    assert wide.loc["m2", "para_a"] == pytest.approx(3.0)


def test_spearman_matrix_detects_stable_ranking():
    wide = pd.DataFrame(
        {
            "baseline": [8.0, 5.0, 2.0],
            "para_a":   [7.0, 4.5, 1.0],  # same order
            "para_b":   [1.0, 5.0, 8.0],  # reversed order
        },
        index=["m1", "m2", "m3"],
    )
    rho, n = _spearman_matrix(wide)
    assert rho.loc["baseline", "para_a"] == pytest.approx(1.0)
    assert rho.loc["baseline", "para_b"] == pytest.approx(-1.0)
    assert rho.loc["baseline", "baseline"] == 1.0
    assert (n == 3).all().all()


def test_spearman_matrix_nan_for_too_few_models():
    wide = pd.DataFrame(
        {"baseline": [8.0, 5.0], "para_a": [7.0, 4.5]},
        index=["m1", "m2"],
    )
    rho, n = _spearman_matrix(wide)
    assert pd.isna(rho.loc["baseline", "para_a"])
    assert n.loc["baseline", "para_a"] == 2


def test_permutation_null_mean_near_zero_for_independent_inputs():
    # Eight models with per-column means that carry no shared ranking.
    wide = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "b": [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0],
        },
        index=[f"m{i}" for i in range(8)],
    )
    rho, _ = _spearman_matrix(wide)
    null_mean, null_lo, null_hi, pval = _permutation_null_spearman(
        wide, rho, n_permutations=2000, seed=0,
    )
    # Under a proper permutation null, E[ρ] = 0.
    assert abs(null_mean.loc["a", "b"]) < 0.05
    # 95% interval is symmetric-ish around zero.
    assert null_lo.loc["a", "b"] < 0 < null_hi.loc["a", "b"]
    # Diagonal entries are undefined under permutation.
    assert pd.isna(null_mean.loc["a", "a"])
    assert pd.isna(null_hi.loc["a", "a"])
    # p-value lies in [0, 1].
    assert 0.0 <= pval.loc["a", "b"] <= 1.0


def test_permutation_null_small_pvalue_for_perfectly_ranked_inputs():
    # Six models with identical rank orders -> observed ρ = 1.0. No draw
    # from the null can match |ρ_obs| = 1 except the (1/n!)-probability
    # identity permutation, so the two-sided p-value should be tiny.
    wide = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        },
        index=[f"m{i}" for i in range(6)],
    )
    rho, _ = _spearman_matrix(wide)
    assert rho.loc["a", "b"] == pytest.approx(1.0)
    _, _, _, pval = _permutation_null_spearman(
        wide, rho, n_permutations=1000, seed=0,
    )
    assert pval.loc["a", "b"] < 0.05


def test_mad_matrix_captures_score_shift_that_spearman_misses():
    # para_a == baseline + 2 everywhere: ρ = 1 but MAD = 2.
    wide = pd.DataFrame(
        {
            "baseline": [3.0, 5.0, 7.0],
            "para_a":   [5.0, 7.0, 9.0],
            "para_b":   [3.0, 5.0, 7.0],  # identical to baseline
        },
        index=["m1", "m2", "m3"],
    )
    rho, _ = _spearman_matrix(wide)
    mad = _mad_matrix(wide)
    assert rho.loc["baseline", "para_a"] == pytest.approx(1.0)
    assert mad.loc["baseline", "para_a"] == pytest.approx(2.0)
    assert mad.loc["baseline", "para_b"] == pytest.approx(0.0)
    # Diagonal is zero.
    assert mad.loc["baseline", "baseline"] == 0.0


def test_mad_matrix_nan_for_no_overlap():
    wide = pd.DataFrame(
        {
            "a": [1.0, float("nan")],
            "b": [float("nan"), 4.0],
        },
        index=["m1", "m2"],
    )
    mad = _mad_matrix(wide)
    assert pd.isna(mad.loc["a", "b"])


def test_latex_table_marks_preregistered_stable_pairs():
    # baseline/para_pass: identical scores -> rho=1, MAD=0 -> pass both bars.
    # baseline/para_mad_fail: same ranks but shifted by +3 -> rho=1, MAD=3 ->
    # only rho bar passes.
    # baseline/para_rho_fail: reversed ranks -> rho=-1, MAD=4 -> fails both.
    wide = pd.DataFrame(
        {
            "baseline":       [3.0, 5.0, 7.0],
            "para_pass":      [3.0, 5.0, 7.0],
            "para_mad_fail":  [6.0, 8.0, 10.0],
            "para_rho_fail":  [7.0, 5.0, 3.0],
        },
        index=["m1", "m2", "m3"],
    )
    rho, _ = _spearman_matrix(wide)
    mad = _mad_matrix(wide)
    # Null hi at 0 so the \\dagger marker also fires for perfectly correlated
    # pairs; this keeps both markers observable in the rendered output.
    null_hi = pd.DataFrame(
        0.0, index=wide.columns, columns=wide.columns, dtype=float,
    )
    tex = _format_latex_table(
        wide, rho, {c: c for c in wide.columns}, null_hi=null_hi, mad=mad,
    )
    # Preregistered threshold marker is present on a passing pair.
    assert "\\ddagger" in tex
    # Caption reports the joint pass count (1 ordered pair each direction
    # for baseline/para_pass, so 2 off-diagonal pairs out of 12).
    assert "2/12" in tex
    # Constants match what the caption quotes.
    assert RHO_STABLE_THRESHOLD == pytest.approx(0.9)
    assert MAD_STABLE_THRESHOLD == pytest.approx(1.0)


def test_permutation_null_nan_for_too_few_models():
    wide = pd.DataFrame(
        {"a": [1.0, 2.0], "b": [3.0, 4.0]},
        index=["m1", "m2"],
    )
    rho, _ = _spearman_matrix(wide)
    null_mean, null_lo, null_hi, pval = _permutation_null_spearman(
        wide, rho, n_permutations=100, seed=0,
    )
    assert pd.isna(null_mean.loc["a", "b"])
    assert pd.isna(null_lo.loc["a", "b"])
    assert pd.isna(null_hi.loc["a", "b"])
    assert pd.isna(pval.loc["a", "b"])

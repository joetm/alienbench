"""Unit tests for ablation_dimensions (no API calls)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

_mock_krippendorff = types.ModuleType("krippendorff")
_mock_krippendorff.alpha = lambda reliability_data, level_of_measurement: 0.85
sys.modules.setdefault("krippendorff", _mock_krippendorff)

from alienbench.ablation_dimensions import (  # noqa: E402
    RHO_THRESHOLD,
    TAU_THRESHOLD,
    _kendall_tau,
    _leave_one_out_table,
    _mad_series,
    _min_k_from_summary,
    _minimum_subset_search,
    _per_model_mean_from_subset,
    _permutation_null_min_k,
    _permutation_null_rho,
    _ranks_equal,
    _subset_size_summary,
)
from alienbench.dimensions import DIMENSION_IDS  # noqa: E402


def _synthetic_ward_df(per_model_dims: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Build ward_df with one generation per model, one judge, deterministic dims.

    ``per_model_dims[model][dim]`` is the binary departure for that cell.
    """
    rows = []
    for model, dims in per_model_dims.items():
        row = {
            "generation_id": f"gen_{model}",
            "judge_model": "j1",
            "subject_model": model,
            "prompt_variant": "baseline",
            "ward_score": sum(dims.get(d, 0) for d in DIMENSION_IDS),
        }
        for d in DIMENSION_IDS:
            row[f"dim_{d}"] = int(dims.get(d, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def test_drop_zero_dimension_preserves_ranking():
    # All models have 0 on "cognition"; dropping it must leave ranking unchanged.
    per_model = {
        "m1": {d: 1 for d in DIMENSION_IDS if d != "cognition"},
        "m2": {d: 1 for d in DIMENSION_IDS[:5] if d != "cognition"},
        "m3": {d: 1 for d in DIMENSION_IDS[:2]},
    }
    df = _synthetic_ward_df(per_model)
    full = _per_model_mean_from_subset(df, tuple(DIMENSION_IDS))
    subset = tuple(d for d in DIMENSION_IDS if d != "cognition")
    reduced = _per_model_mean_from_subset(df, subset)
    assert _ranks_equal(reduced, full)


def test_single_dimension_that_replicates_ranking_gives_min_k_1():
    # Two models with full ordering m1 < m2. Any single dimension on which
    # m2 departs and m1 does not reproduces the ranking on its own.
    per_model = {
        "m1": {d: 0 for d in DIMENSION_IDS},
        "m2": {"symmetry": 1, "sensory_organs": 1},
    }
    df = _synthetic_ward_df(per_model)
    full = _per_model_mean_from_subset(df, tuple(DIMENSION_IDS))
    _, rank_preserving, min_k = _minimum_subset_search(df, full)
    assert min_k == 1
    assert ("symmetry",) in rank_preserving
    assert ("sensory_organs",) in rank_preserving


def test_loo_table_rows_cover_all_dimensions():
    per_model = {
        "m1": {d: 1 for d in DIMENSION_IDS[:5]},
        "m2": {d: 1 for d in DIMENSION_IDS[5:]},
        "m3": {d: 1 for d in DIMENSION_IDS[:3]},
    }
    df = _synthetic_ward_df(per_model)
    full = _per_model_mean_from_subset(df, tuple(DIMENSION_IDS))
    loo = _leave_one_out_table(df, full, n_null_permutations=200)
    assert set(loo["dropped_dimension"]) == set(DIMENSION_IDS)
    assert loo["departure_rate"].between(0.0, 1.0).all()
    assert {"null_rho_hi95", "mad_vs_full", "rho_above_null_hi95"}.issubset(
        loo.columns
    )


def test_per_model_mean_averages_judges_then_generations():
    # Two judges disagree on m1/dim_symmetry per generation; mean should be 0.5.
    rows = []
    for judge, val in [("j1", 1), ("j2", 0)]:
        for gen in ["g1", "g2"]:
            row = {
                "generation_id": gen,
                "judge_model": judge,
                "subject_model": "m1",
                "prompt_variant": "baseline",
                "ward_score": val,
            }
            for d in DIMENSION_IDS:
                row[f"dim_{d}"] = val if d == "symmetry" else 0
            rows.append(row)
    df = pd.DataFrame(rows)
    means = _per_model_mean_from_subset(df, ("symmetry",))
    assert means.loc["m1"] == pytest.approx(0.5)


def test_minimum_subset_requires_at_least_two_models():
    per_model = {"m1": {d: 1 for d in DIMENSION_IDS}}
    df = _synthetic_ward_df(per_model)
    full = _per_model_mean_from_subset(df, tuple(DIMENSION_IDS))
    _, rank_preserving, min_k = _minimum_subset_search(df, full)
    # With a single model, rank comparison is undefined, so no preservation.
    assert min_k == -1
    assert rank_preserving == []


def test_mad_series_captures_score_shift():
    a = pd.Series({"m1": 5.0, "m2": 6.0, "m3": 7.0})
    b = pd.Series({"m1": 6.0, "m2": 7.0, "m3": 8.0})  # uniform +1 shift
    assert _mad_series(a, b) == pytest.approx(1.0)
    # Empty overlap returns NaN.
    c = pd.Series({"m4": 1.0})
    assert pd.isna(_mad_series(a, c))


def test_permutation_null_rho_symmetric_around_zero():
    reduced = pd.Series({f"m{i}": float(i) for i in range(8)})
    full = pd.Series({f"m{i}": float(i) + 0.1 for i in range(8)})
    mean, lo, hi = _permutation_null_rho(
        reduced, full, n_permutations=500, seed=0,
    )
    assert abs(mean) < 0.1
    assert lo < 0 < hi


def test_permutation_null_rho_nan_for_too_few_models():
    reduced = pd.Series({"m1": 1.0, "m2": 2.0})
    full = pd.Series({"m1": 1.0, "m2": 2.0})
    mean, lo, hi = _permutation_null_rho(
        reduced, full, n_permutations=100, seed=0,
    )
    assert pd.isna(mean)
    assert pd.isna(lo)
    assert pd.isna(hi)


def test_kendall_tau_matches_spearman_sign():
    a = pd.Series({"m1": 1.0, "m2": 2.0, "m3": 3.0, "m4": 4.0})
    b = pd.Series({"m1": 4.0, "m2": 3.0, "m3": 2.0, "m4": 1.0})  # reversed
    assert _kendall_tau(a, b) == pytest.approx(-1.0)
    c = pd.Series({"m1": 10.0, "m2": 20.0, "m3": 30.0, "m4": 40.0})
    assert _kendall_tau(a, c) == pytest.approx(1.0)


def test_subset_size_summary_has_best_rho_and_tau_per_k():
    per_model = {
        "m1": {d: 0 for d in DIMENSION_IDS},
        "m2": {d: 1 for d in DIMENSION_IDS[:3]},
        "m3": {d: 1 for d in DIMENSION_IDS},
    }
    df = _synthetic_ward_df(per_model)
    full = _per_model_mean_from_subset(df, tuple(DIMENSION_IDS))
    summary, best_rho_by_k, best_tau_by_k = _subset_size_summary(df, full)
    assert list(summary["k"]) == list(range(1, len(DIMENSION_IDS) + 1))
    assert {"best_rho", "best_tau", "min_mad"}.issubset(summary.columns)
    # Best ρ at k=10 must hit 1.0 (the full set).
    assert summary.iloc[-1]["best_rho"] == pytest.approx(1.0)
    # Each k that produces a finite best-ρ has a matching subset recorded.
    for k in summary["k"]:
        if pd.notna(summary.loc[summary["k"] == k, "best_rho"].iloc[0]):
            assert k in best_rho_by_k


def test_min_k_from_summary_returns_smallest_qualifying_k():
    summary = pd.DataFrame(
        {"k": [1, 2, 3, 4], "best_rho": [0.5, 0.85, 0.92, 0.99]}
    )
    assert _min_k_from_summary(summary, "best_rho", 0.9) == 3
    assert _min_k_from_summary(summary, "best_rho", 0.99) == 4
    # No qualifying k.
    assert _min_k_from_summary(summary, "best_rho", 1.0) == -1


def test_permutation_null_min_k_returns_finite_draws():
    full = pd.Series({f"m{i}": float(i) for i in range(8)})
    draws = _permutation_null_min_k(
        full, n_permutations=20, seed=0,
    )
    assert len(draws) == 20
    assert {"min_k_rho_null", "min_k_tau_null"} == set(draws.columns)
    # Every draw is in [1, 11]; 11 means "never crossed threshold".
    assert draws["min_k_rho_null"].between(1, len(DIMENSION_IDS) + 1).all()
    assert draws["min_k_tau_null"].between(1, len(DIMENSION_IDS) + 1).all()


def test_preregistered_thresholds_are_exposed():
    # Guard the API contract: these are referenced in the paper prose.
    assert RHO_THRESHOLD == 0.9
    assert TAU_THRESHOLD == 0.8

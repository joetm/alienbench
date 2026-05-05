"""Unit tests for ablation_judges (no API calls)."""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

_mock_krippendorff = types.ModuleType("krippendorff")
_mock_krippendorff.alpha = lambda reliability_data, level_of_measurement: 0.9
sys.modules.setdefault("krippendorff", _mock_krippendorff)

from alienbench.ablation_judges import (  # noqa: E402
    ALPHA_DROP_THRESHOLD,
    RHO_STABLE_THRESHOLD,
    _heldout_reference_means,
    _loo_table,
    _panel_per_model_means,
    _permutation_null_heldout_rho,
    _single_judge_table,
    _size_sweep,
)
from alienbench.dimensions import DIMENSION_IDS  # noqa: E402


def _make_rows(cells: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """Cells: list of (judge, model, generation_id, ward_score)."""
    rows = []
    for judge, model, gen, score in cells:
        row = {
            "generation_id": gen,
            "judge_model": judge,
            "subject_model": model,
            "prompt_variant": "baseline",
            "ward_score": score,
        }
        for d in DIMENSION_IDS:
            row[f"dim_{d}"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def test_identical_judges_yield_rho_one_and_rank_preserved():
    # Two judges score identically across three models -> rho = 1 on both axes.
    cells = []
    for j in ["j1", "j2"]:
        cells += [
            (j, "m1", f"m1_{j}", 8),
            (j, "m2", f"m2_{j}", 5),
            (j, "m3", f"m3_{j}", 2),
        ]
    df = _make_rows(cells)
    full = _panel_per_model_means(df)
    single = _single_judge_table(
        df, full, ["j1", "j2"], n_null_permutations=50,
    )
    assert single["rho_vs_full"].tolist() == pytest.approx([1.0, 1.0])
    assert single["rho_vs_heldout"].tolist() == pytest.approx([1.0, 1.0])
    assert all(single["rank_preserved_vs_heldout"])


def test_adversarial_judge_heldout_rho_is_minus_one():
    # j1 ranks m1>m2>m3; j2 inverts it. With only two judges, the held-out
    # reference for j1 IS j2 (and vice versa) -> rho_vs_heldout == -1.
    # The full-panel reference averages to a tie, so rho_vs_full is NaN.
    df = _make_rows([
        ("j1", "m1", "g1", 9), ("j1", "m2", "g2", 5), ("j1", "m3", "g3", 1),
        ("j2", "m1", "g4", 1), ("j2", "m2", "g5", 5), ("j2", "m3", "g6", 9),
    ])
    full = _panel_per_model_means(df)
    single = _single_judge_table(
        df, full, ["j1", "j2"], n_null_permutations=50,
    )
    # rho_vs_full is NaN (full panel has zero variance).
    assert single["rho_vs_full"].isna().all()
    # rho_vs_heldout cleanly reveals the disagreement.
    assert single["rho_vs_heldout"].tolist() == pytest.approx([-1.0, -1.0])


def test_loo_alpha_nan_when_only_one_judge_remains():
    df = _make_rows([
        ("j1", "m1", "g1", 8), ("j1", "m2", "g2", 3),
        ("j2", "m1", "g3", 7), ("j2", "m2", "g4", 4),
    ])
    full = _panel_per_model_means(df)
    loo = _loo_table(df, full, ["j1", "j2"], n_null_permutations=50)
    assert loo["alpha_cross_model"].isna().all()
    assert (loo["n_judges_remaining"] == 1).all()


def test_size_sweep_enumerates_c_n_k_subsets():
    df = _make_rows([
        ("j1", "m1", "g1", 8), ("j1", "m2", "g2", 3), ("j1", "m3", "g3", 6),
        ("j2", "m1", "g4", 7), ("j2", "m2", "g5", 4), ("j2", "m3", "g6", 5),
        ("j3", "m1", "g7", 6), ("j3", "m2", "g8", 5), ("j3", "m3", "g9", 4),
    ])
    full = _panel_per_model_means(df)
    sweep = _size_sweep(
        df, full, ["j1", "j2", "j3"], n_null_permutations=50,
    )
    assert list(sweep["k"]) == [1, 2, 3]
    assert list(sweep["n_subsets"]) == [3, 3, 1]
    # k = n: no disjoint held-out reference -> held-out columns are NaN.
    full_row = sweep.loc[sweep["k"] == 3].iloc[0]
    assert pd.isna(full_row["mean_rho_vs_heldout"])
    assert pd.isna(full_row["min_rho_vs_heldout"])
    assert pd.isna(full_row["max_null_rho_hi95"])
    # k < n: held-out rho is finite.
    for k in [1, 2]:
        row = sweep.loc[sweep["k"] == k].iloc[0]
        assert pd.notna(row["mean_rho_vs_heldout"])


def test_heldout_reference_excludes_subset_judges():
    # Four judges. j1/j2/j3 weakly prefer m1>m2>m3>m4. j4 strongly
    # reverses the ranking and dominates the full-panel aggregate. This
    # produces rho_vs_full(j4, full) = +1 (j4's ranking matches the
    # full panel because j4 drives the full panel) but
    # rho_vs_heldout(j4, {j1,j2,j3}) = -1 (j4 is the opposite of the
    # disjoint reference). The gap is exactly the self-reference
    # inflation that the held-out design removes.
    cells = []
    # Weak agreement on m1>m2>m3>m4.
    for j in ["j1", "j2", "j3"]:
        cells += [
            (j, "m1", f"m1_{j}", 6),
            (j, "m2", f"m2_{j}", 5),
            (j, "m3", f"m3_{j}", 4),
            (j, "m4", f"m4_{j}", 3),
        ]
    # j4 reverses strongly: its magnitude dominates the full average.
    cells += [
        ("j4", "m1", "m1_j4", 0),
        ("j4", "m2", "m2_j4", 10),
        ("j4", "m3", "m3_j4", 20),
        ("j4", "m4", "m4_j4", 30),
    ]
    df = _make_rows(cells)
    judges = ["j1", "j2", "j3", "j4"]
    full = _panel_per_model_means(df)
    single = _single_judge_table(
        df, full, judges, n_null_permutations=50,
    )
    j4_row = single.loc[single["judge_model"] == "j4"].iloc[0]
    # Disjoint reference: j4 is exactly anti-correlated with j1/j2/j3.
    assert j4_row["rho_vs_heldout"] == pytest.approx(-1.0)
    # Self-referential reference: j4 drives the full aggregate, so the
    # naive rho reports near-perfect agreement.
    assert j4_row["rho_vs_full"] == pytest.approx(1.0)
    # The inflation gap is the entire point of the fix.
    assert j4_row["rho_vs_full"] > j4_row["rho_vs_heldout"]
    # Held-out helper drops the subject judge from the reference.
    heldout = _heldout_reference_means(df, ("j4",), judges)
    assert set(heldout.index) == {"m1", "m2", "m3", "m4"}


def test_alpha_column_is_nan_at_k1_explicitly():
    # At k=1, the size-sweep mean-alpha column is NaN by design, since
    # alpha requires at least two raters. Single-judge rows carry NaN
    # alpha as well.
    df = _make_rows([
        ("j1", "m1", "g1", 8), ("j1", "m2", "g2", 3), ("j1", "m3", "g3", 6),
        ("j2", "m1", "g4", 7), ("j2", "m2", "g5", 4), ("j2", "m3", "g6", 5),
        ("j3", "m1", "g7", 6), ("j3", "m2", "g8", 5), ("j3", "m3", "g9", 4),
    ])
    full = _panel_per_model_means(df)
    sweep = _size_sweep(
        df, full, ["j1", "j2", "j3"], n_null_permutations=50,
    )
    k1 = sweep.loc[sweep["k"] == 1].iloc[0]
    assert pd.isna(k1["mean_alpha_cross_model"])
    assert pd.isna(k1["min_alpha_cross_model"])
    assert int(k1["n_alpha_above_tentative"]) == -1
    single = _single_judge_table(
        df, full, ["j1", "j2", "j3"], n_null_permutations=50,
    )
    assert single["alpha_cross_model"].isna().all()


def test_preregistered_rho_bar_tracks_heldout_correlation():
    # j1, j2, j3 share the ranking m1 > m2 > m3; j4 inverts it. For j1,
    # the held-out reference is the mean of {j2, j3, j4} which has
    # variance and a ranking close to j1's, so rho_vs_heldout clears the
    # preregistered bar. For j4, the held-out reference is the mean of
    # {j1, j2, j3}, whose ranking is the opposite of j4's, so j4 does
    # not clear the bar.
    cells = []
    scores = {"j1": (9, 5, 1), "j2": (8, 4, 2), "j3": (10, 6, 0), "j4": (0, 5, 10)}
    for j, (s1, s2, s3) in scores.items():
        cells += [
            (j, "m1", f"m1_{j}", s1),
            (j, "m2", f"m2_{j}", s2),
            (j, "m3", f"m3_{j}", s3),
        ]
    df = _make_rows(cells)
    judges = ["j1", "j2", "j3", "j4"]
    full = _panel_per_model_means(df)
    single = _single_judge_table(df, full, judges, n_null_permutations=50)
    j1_row = single.loc[single["judge_model"] == "j1"].iloc[0]
    j4_row = single.loc[single["judge_model"] == "j4"].iloc[0]
    assert j1_row["rho_vs_heldout"] >= RHO_STABLE_THRESHOLD
    assert bool(j1_row["rho_heldout_above_stable_bar"]) is True
    assert j4_row["rho_vs_heldout"] < RHO_STABLE_THRESHOLD
    assert bool(j4_row["rho_heldout_above_stable_bar"]) is False


def test_alpha_drop_within_bar_threshold_triggers_correctly():
    # Three judges rate the SAME three generations (shared generation_id
    # across judges) so Krippendorff's alpha is defined on the reduced
    # k=2 panels. The mocked krippendorff.alpha returns a fixed constant
    # regardless of scores; we read it at test time so the assertions are
    # robust to which test module registered the mock first via
    # sys.modules.setdefault.
    df = _make_rows([
        ("j1", "m1", "g1", 8), ("j1", "m2", "g2", 3), ("j1", "m3", "g3", 6),
        ("j2", "m1", "g1", 7), ("j2", "m2", "g2", 4), ("j2", "m3", "g3", 5),
        ("j3", "m1", "g1", 6), ("j3", "m2", "g2", 5), ("j3", "m3", "g3", 4),
    ])
    judges = ["j1", "j2", "j3"]
    full = _panel_per_model_means(df)
    mock_alpha = sys.modules["krippendorff"].alpha([[0, 0], [0, 0]], level_of_measurement="interval")
    # full_alpha chosen so the drop is 0.05 (within bar).
    loo_within = _loo_table(
        df, full, judges, n_null_permutations=50,
        full_alpha=mock_alpha + 0.05,
    )
    assert loo_within["alpha_drop_from_full"].iloc[0] == pytest.approx(0.05)
    assert bool(loo_within["alpha_drop_within_bar"].iloc[0]) is True
    # full_alpha chosen so the drop is 0.15 (outside bar).
    loo_outside = _loo_table(
        df, full, judges, n_null_permutations=50,
        full_alpha=mock_alpha + 0.15,
    )
    assert loo_outside["alpha_drop_from_full"].iloc[0] == pytest.approx(0.15)
    assert bool(loo_outside["alpha_drop_within_bar"].iloc[0]) is False
    # Constants match what the paper quotes.
    assert ALPHA_DROP_THRESHOLD == pytest.approx(0.1)
    assert RHO_STABLE_THRESHOLD == pytest.approx(0.9)


def test_size_sweep_reports_stable_bar_counts():
    # Four judges: j1, j2, j3 share the m1>m2>m3 ranking on disjoint
    # generations; j4 inverts it. At k=1, each single-judge subset's
    # held-out reference is the mean of the other three judges. Leaving
    # out j1, j2, or j3 leaves a held-out ordering that still matches the
    # single judge (rho = 1); leaving out j4 leaves a held-out ordering
    # that matches j4's inverted scores (rho = -1). So 3 of 4 single-judge
    # subsets clear the rho >= 0.9 bar.
    df = _make_rows([
        ("j1", "m1", "g1", 9),  ("j1", "m2", "g2", 5),  ("j1", "m3", "g3", 1),
        ("j2", "m1", "g4", 8),  ("j2", "m2", "g5", 4),  ("j2", "m3", "g6", 2),
        ("j3", "m1", "g7", 10), ("j3", "m2", "g8", 6),  ("j3", "m3", "g9", 0),
        ("j4", "m1", "g10", 0), ("j4", "m2", "g11", 5), ("j4", "m3", "g12", 10),
    ])
    full = _panel_per_model_means(df)
    sweep = _size_sweep(
        df, full, ["j1", "j2", "j3", "j4"], n_null_permutations=50,
        full_alpha=0.9,
    )
    k1 = sweep.loc[sweep["k"] == 1].iloc[0]
    assert int(k1["n_rho_heldout_above_stable_bar"]) == 3
    # alpha drop columns are NaN / -1 at k=1 (alpha undefined).
    assert int(k1["n_alpha_drop_within_bar"]) == -1
    # k=n: held-out undefined -> -1.
    k_full = sweep.loc[sweep["k"] == 4].iloc[0]
    assert int(k_full["n_rho_heldout_above_stable_bar"]) == -1


def test_permutation_null_heldout_rho_mean_near_zero():
    # Eight models with paired values that carry no shared ranking;
    # permuting one side yields a null distribution centred near zero.
    reduced = pd.Series({f"m{i}": float(i) for i in range(8)})
    heldout = pd.Series(
        {f"m{i}": v for i, v in enumerate([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0])}
    )
    mean, lo, hi = _permutation_null_heldout_rho(
        reduced, heldout, n_permutations=1000, seed=0,
    )
    assert abs(mean) < 0.1
    assert lo < 0 < hi


def test_permutation_null_heldout_rho_nan_for_too_few_models():
    reduced = pd.Series({"m1": 1.0, "m2": 2.0})
    heldout = pd.Series({"m1": 1.0, "m2": 2.0})
    mean, lo, hi = _permutation_null_heldout_rho(
        reduced, heldout, n_permutations=100, seed=0,
    )
    assert pd.isna(mean)
    assert pd.isna(lo)
    assert pd.isna(hi)

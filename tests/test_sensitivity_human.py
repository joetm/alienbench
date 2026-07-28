"""Tests for the human-baseline sensitivity analysis.

The MDE helpers are checked against values a standard power calculator
returns, against the self-consistency requirement that the power at the
returned MDE equals the target, and against monotonicity in the sample size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alienbench.sensitivity_human import (
    DIMENSION_IDS,
    dimension_sensitivity,
    load_generation_scores,
    mde_cohens_d,
    mde_proportion_difference,
    mde_rank_biserial,
    mean_ci_half_width,
    power_mannwhitney,
    power_two_proportions,
    power_two_sample_t,
    precision_curve,
    rank_biserial_to_shift,
    total_score_sensitivity,
)


# ---------------------------------------------------------------------------
# Parametric MDE
# ---------------------------------------------------------------------------

def test_power_two_sample_t_matches_reference():
    # G*Power / pwr.t.test: n = 64 per group, d = 0.5, alpha = .05 -> power ~= .80
    assert power_two_sample_t(0.5, 64, 64, 0.05) == pytest.approx(0.80, abs=0.01)


def test_mde_equal_groups_matches_reference():
    # pwr.t.test(n = 100, sig.level = .05, power = .80) -> d = 0.3981
    assert mde_cohens_d(100, 100, 0.05, 0.80) == pytest.approx(0.398, abs=0.005)


def test_mde_is_self_consistent():
    for n1, n2 in [(100, 30), (100, 100), (50, 20)]:
        d = mde_cohens_d(n1, n2, 0.05, 0.80)
        assert power_two_sample_t(d, n1, n2, 0.05) == pytest.approx(0.80, abs=1e-3)


def test_mde_decreases_with_sample_size():
    ds = [mde_cohens_d(n, 30, 0.05, 0.80) for n in (25, 50, 100, 200, 400)]
    assert all(a > b for a, b in zip(ds, ds[1:]))


def test_mde_increases_under_bonferroni():
    plain = mde_cohens_d(100, 30, 0.05, 0.80)
    corrected = mde_cohens_d(100, 30, 0.005, 0.80)
    assert corrected > plain


def test_mde_saturates_when_one_group_is_fixed():
    """With n_model fixed, growing the human sample cannot drive the MDE to zero."""
    limit = mde_cohens_d(10_000_000, 30, 0.05, 0.80)
    assert limit > 0.5
    assert mde_cohens_d(100, 30, 0.05, 0.80) - limit < 0.25


def test_mean_ci_half_width_matches_t_interval():
    sd, n = 1.888, 100
    half = mean_ci_half_width(sd, n, 0.95)
    assert half == pytest.approx(1.984 * sd / np.sqrt(n), rel=1e-3)


# ---------------------------------------------------------------------------
# Nonparametric MDE
# ---------------------------------------------------------------------------

def test_rank_biserial_shift_roundtrip():
    from scipy.stats import norm
    for r in (0.1, 0.3, 0.6):
        mu = rank_biserial_to_shift(r)
        assert 2 * norm.cdf(mu / np.sqrt(2)) - 1 == pytest.approx(r, abs=1e-9)


@pytest.mark.parametrize("r, tol", [(0.05, 0.01), (0.25, 0.02), (0.35, 0.04)])
def test_rank_biserial_shift_agrees_with_linearization(r, tol):
    """The stats.retrospective_power convention mu = r*sqrt(pi) holds for small r."""
    exact = rank_biserial_to_shift(r)
    approx = r * np.sqrt(np.pi)
    assert abs(exact - approx) / exact < tol


def test_power_mannwhitney_is_monotone():
    powers = [power_mannwhitney(r, 100, 30, 0.05, n_sims=600, seed=7)
              for r in (0.1, 0.3, 0.5, 0.7)]
    assert all(a <= b for a, b in zip(powers, powers[1:]))


def test_mde_rank_biserial_is_plausible():
    r = mde_rank_biserial(100, 30, 0.05, n_sims=600, seed=7)
    assert 0.2 < r < 0.6
    assert power_mannwhitney(r, 100, 30, 0.05, n_sims=1200, seed=11) == pytest.approx(
        0.80, abs=0.06
    )


# ---------------------------------------------------------------------------
# Proportion MDE
# ---------------------------------------------------------------------------

def test_mde_proportion_is_self_consistent():
    p_ref = 0.20
    delta = mde_proportion_difference(p_ref, 100, 30, 0.05, 0.80)
    assert power_two_proportions(p_ref, p_ref + delta, 100, 30, 0.05) == pytest.approx(
        0.80, abs=1e-3
    )


def test_mde_proportion_is_positive_and_bounded():
    for p in (0.05, 0.25, 0.50, 0.80):
        delta = mde_proportion_difference(p, 100, 30, 0.05, 0.80)
        assert 0 < delta < 1 - p


def test_mde_proportion_grows_under_correction():
    plain = mde_proportion_difference(0.25, 100, 30, 0.05, 0.80)
    corrected = mde_proportion_difference(0.25, 100, 30, 0.005, 0.80)
    assert corrected > plain


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------

def _write_scores(tmp_path, judge, subject, variant, rows):
    from alienbench.paths import ward_scores_path
    path = ward_scores_path(tmp_path, judge, subject, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _record(gen_id, judge, score):
    return {
        "generation_id": gen_id,
        "judge_model": judge,
        "subject_model": "human/prolific-baseline",
        "prompt_variant": "baseline",
        "ward_score": score,
        "per_dimension": {d: (1 if i < score else 0) for i, d in enumerate(DIMENSION_IDS)},
    }


def test_load_generation_scores_averages_across_judges(tmp_path):
    judges = ["j/a", "j/b"]
    _write_scores(tmp_path, "j/a", "human/prolific-baseline", "baseline",
                  [_record("g1", "j/a", 2), _record("g2", "j/a", 4)])
    _write_scores(tmp_path, "j/b", "human/prolific-baseline", "baseline",
                  [_record("g1", "j/b", 4), _record("g2", "j/b", 4)])

    df = load_generation_scores(tmp_path, judges, "human/prolific-baseline", "baseline")

    assert len(df) == 2
    assert set(df["n_judges"]) == {2}
    assert df.set_index("generation_id").loc["g1", "ward_score"] == 3.0
    assert df.set_index("generation_id").loc["g2", "ward_score"] == 4.0


def test_load_generation_scores_returns_empty_frame_when_absent(tmp_path):
    df = load_generation_scores(tmp_path, ["j/a"], "human/prolific-baseline", "baseline")
    assert df.empty
    assert "ward_score" in df.columns


# ---------------------------------------------------------------------------
# Assembled analysis
# ---------------------------------------------------------------------------

def test_total_score_sensitivity_flags_resolvable_gaps():
    rng = np.random.default_rng(0)
    human = rng.normal(2.0, 1.9, 100)
    models = {
        "big/gap": rng.normal(6.0, 1.5, 30),     # ~2x the MDE
        "no/gap": rng.normal(2.02, 1.5, 30),     # far below the MDE
    }
    df = total_score_sensitivity(human, models, n_sims=200)

    assert set(df["model"]) == {"big/gap", "no/gap"}
    assert bool(df.set_index("model").loc["big/gap", "detected"]) is True
    assert bool(df.set_index("model").loc["no/gap", "detected"]) is False
    # The Bonferroni-corrected threshold is the stricter one.
    assert (df["mde_cohens_d_bonferroni"] > df["mde_cohens_d"]).all()


def test_dimension_sensitivity_covers_every_dimension():
    rng = np.random.default_rng(1)
    human_df = pd.DataFrame(
        {f"dim_{d}": rng.random(100) < 0.2 for d in DIMENSION_IDS}
    ).astype(float)
    df = dimension_sensitivity(human_df, 30)

    assert list(df["dimension"]) == DIMENSION_IDS
    assert (df["mde_pp"] > 0).all()
    assert (df["mde_pp_bonferroni"] >= df["mde_pp"]).all()


def test_dimension_sensitivity_scores_observed_gaps():
    rng = np.random.default_rng(2)
    human_df = pd.DataFrame(
        {f"dim_{d}": rng.random(100) < 0.10 for d in DIMENSION_IDS}
    ).astype(float)
    rates = {
        "far/above": {d: 0.90 for d in DIMENSION_IDS},   # gap ~80pp, resolvable
        "at/human": {d: 0.10 for d in DIMENSION_IDS},    # gap ~0pp, not resolvable
    }
    df = dimension_sensitivity(human_df, 30, n_model_pooled=180,
                               model_dim_rates=rates).set_index("dimension")

    # A larger model sample lowers the detectable difference.
    assert (df["mde_pp_pooled_bonferroni"] < df["mde_pp_bonferroni"]).all()
    assert (df["n_models"] == 2).all()
    assert (df["n_models_resolved"] == 1).all()
    assert (df["max_observed_gap_pp"] > df["median_observed_gap_pp"]).all()


def test_precision_curve_is_monotone_and_carries_the_limit():
    df = precision_curve(1.9, 30, [25, 50, 100, 200, 400])
    assert df["mde_cohens_d"].is_monotonic_decreasing
    assert df["ci_half_width_ward_points"].is_monotonic_decreasing
    limit = df.attrs["mde_cohens_d_limit"]
    assert 0 < limit < float(df["mde_cohens_d"].iloc[-1])

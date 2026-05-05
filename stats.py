"""Shared statistical helpers.

Used by the main analysis stage (``alienbench.analyze``) and by the ablation
modules. Kept dependency-light: NumPy, pandas, SciPy, and ``krippendorff``.
"""

from __future__ import annotations

from itertools import combinations

import krippendorff
import numpy as np
import pandas as pd
import scipy.stats as stats


def mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    """Returns (mean, lower_ci, upper_ci)."""
    n = len(values)
    m = values.mean()
    if n < 2:
        return m, m, m
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return m, m - h, m + h


def kruskal_posthoc(df: pd.DataFrame, group_col: str, score_col: str,
                    n_boot: int = 10_000):
    """Kruskal-Wallis test + pairwise Mann-Whitney U with Bonferroni correction.

    Each pair reports the rank-biserial effect size with a bootstrap 95%
    percentile CI. Returns (kruskal_result, posthoc_df).
    """
    grouped = {name: grp[score_col].dropna().values for name, grp in df.groupby(group_col)}
    groups = list(grouped.values())
    stat, p = stats.kruskal(*groups)
    kruskal_result = {"kruskal_H": stat, "kruskal_p": p}

    group_names = list(grouped.keys())
    pairs = list(combinations(group_names, 2))
    n_comparisons = len(pairs)
    posthoc_rows = []
    for a, b in pairs:
        u_stat, p_val = stats.mannwhitneyu(grouped[a], grouped[b], alternative="two-sided")
        n_a, n_b = len(grouped[a]), len(grouped[b])
        r_rb, r_lo, r_hi = rank_biserial(grouped[a], grouped[b], n_boot=n_boot)
        posthoc_rows.append({
            "group_a": a,
            "group_b": b,
            "U": u_stat,
            "p_raw": p_val,
            "p_corrected": min(p_val * n_comparisons, 1.0),
            "rank_biserial_r": r_rb,
            "r_ci_lo": r_lo,
            "r_ci_hi": r_hi,
            "n_a": n_a,
            "n_b": n_b,
        })

    return kruskal_result, pd.DataFrame(posthoc_rows)


def krippendorff_alpha(df: pd.DataFrame, rater_col: str, item_col: str, score_col: str,
                       level: str = "interval") -> float:
    """Compute Krippendorff's alpha across raters.

    Args:
        level: Level of measurement — "nominal" for binary Ward dimensions,
               "interval" for Ward total score.

    Returns 1.0 when all observed values are identical (perfect agreement by definition).
    Returns NaN when there is insufficient data to compute alpha.
    """
    pivot = df.pivot_table(index=item_col, columns=rater_col, values=score_col)
    matrix = pivot.values.T  # shape: (raters, items)
    flat = matrix[~np.isnan(matrix)]
    if len(flat) == 0 or np.unique(flat).size < 2:
        return 1.0  # all raters agreed on every item
    # Krippendorff requires at least one item with values from ≥2 raters.
    coders_per_item = (~np.isnan(matrix)).sum(axis=0)
    if (coders_per_item >= 2).sum() == 0:
        return float("nan")
    return krippendorff.alpha(reliability_data=matrix, level_of_measurement=level)


def krippendorff_alpha_stratified(df: pd.DataFrame, subject_col: str, judge_col: str,
                                  item_col: str, score_col: str,
                                  level: str = "interval") -> dict:
    """Compute Krippendorff's alpha stratified by judge-subject model overlap.

    Self-judging (judge model equals subject model) is a documented source of
    bias in LLM-as-judge benchmarks (Panickssery et al., 2024). This helper
    reports three alphas — across all pairs, cross-model pairs only, and
    same-model pairs only — plus the mean-score delta between same-model and
    cross-model judgments on matched items, so self-preference inflation can
    be quantified directly.

    Exclusion rule for a future primary model-comparison analysis: if
    (same-model alpha exceeds cross-model alpha by more than 0.10) OR
    (self-judge mean exceeds cross-judge mean on matched items by more than
    0.5 score points), the primary analysis should exclude self-judge pairs.

    Returns a dict with keys: all, cross_model, same_model, mean_delta_self,
    n_all, n_cross, n_same.
    """
    cross_df = df[df[judge_col] != df[subject_col]]
    same_df = df[df[judge_col] == df[subject_col]]

    all_alpha = krippendorff_alpha(df, judge_col, item_col, score_col, level=level)
    cross_alpha = (
        krippendorff_alpha(cross_df, judge_col, item_col, score_col, level=level)
        if len(cross_df) > 0 else float("nan")
    )
    same_alpha = (
        krippendorff_alpha(same_df, judge_col, item_col, score_col, level=level)
        if len(same_df) > 0 else float("nan")
    )

    # Mean-score delta on matched items (items judged under both regimes)
    if len(same_df) > 0 and len(cross_df) > 0:
        same_mean = same_df.groupby(item_col)[score_col].mean()
        cross_mean = cross_df.groupby(item_col)[score_col].mean()
        matched = same_mean.index.intersection(cross_mean.index)
        mean_delta = (
            float((same_mean.loc[matched] - cross_mean.loc[matched]).mean())
            if len(matched) > 0 else float("nan")
        )
    else:
        mean_delta = float("nan")

    return {
        "all": all_alpha,
        "cross_model": cross_alpha,
        "same_model": same_alpha,
        "mean_delta_self": mean_delta,
        "n_all": len(df),
        "n_cross": len(cross_df),
        "n_same": len(same_df),
    }


def bootstrap_ci(values, stat_fn=np.mean, n_boot: int = 10_000,
                 level: float = 0.95, seed: int = 1234) -> tuple[float, float, float]:
    """Bootstrap percentile CI for an arbitrary scalar statistic.

    Returns (point_estimate, lower, upper). NaNs are dropped before
    resampling. Returns (nan, nan, nan) if no data remain.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat_fn(arr))
    rng = np.random.default_rng(seed=seed)
    n = len(arr)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = stat_fn(arr[idx])
    q_lo = (1 - level) / 2
    q_hi = 1 - q_lo
    return point, float(np.quantile(boot, q_lo)), float(np.quantile(boot, q_hi))


def rank_biserial(group_a, group_b, n_boot: int = 10_000,
                  level: float = 0.95,
                  seed: int = 1234) -> tuple[float, float, float]:
    """Rank-biserial correlation with bootstrap percentile CI.

    r = 1 − (2 U) / (n1 n2), where U is the Mann-Whitney U statistic of
    group_a against group_b. Returns (r, lower, upper).
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")

    def _r(aa: np.ndarray, bb: np.ndarray) -> float:
        u, _ = stats.mannwhitneyu(aa, bb, alternative="two-sided")
        return 1.0 - (2.0 * u) / (len(aa) * len(bb))

    point = _r(a, b)
    rng = np.random.default_rng(seed=seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        boot[i] = _r(aa, bb)
    q_lo = (1 - level) / 2
    q_hi = 1 - q_lo
    return point, float(np.quantile(boot, q_lo)), float(np.quantile(boot, q_hi))


def retrospective_power(effect_sizes, n_per_group: int, n_comparisons: int,
                        alpha: float = 0.05, n_sims: int = 2000,
                        target_power: float = 0.80,
                        seed: int = 1234) -> dict:
    """Retrospective power for Mann-Whitney U under Bonferroni correction.

    Given a set of rank-biserial effect sizes (e.g., those observed in pilot
    pairwise comparisons), simulate power to detect each effect with the
    given n_per_group and a Bonferroni-corrected alpha (alpha / n_comparisons).
    Also scans a grid of effect sizes to find the minimum detectable
    rank-biserial at the target power level.

    Simulation model: two independent samples of size n_per_group drawn from
    unit-variance normals, with a location shift mu = r * sqrt(pi) that
    corresponds (approximately) to a population rank-biserial of r. This
    convention gives a conservative power estimate relative to heavier-tailed
    distributions and is standard in NHST power calculators.

    Returns a dict with: per_effect_power (mapping r -> power), grid_scan
    (list of (r, power) pairs), min_detectable_r, bonferroni_alpha,
    n_per_group, n_comparisons.
    """
    from math import pi as _pi, sqrt as _sqrt

    bonf_alpha = alpha / max(int(n_comparisons), 1)
    rng = np.random.default_rng(seed=seed)

    def _power(r: float) -> float:
        mu = r * _sqrt(_pi)
        hits = 0
        for _ in range(n_sims):
            a = rng.normal(0.0, 1.0, n_per_group)
            b = rng.normal(mu, 1.0, n_per_group)
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            if p < bonf_alpha:
                hits += 1
        return hits / n_sims

    per_effect = {float(r): _power(float(abs(r))) for r in effect_sizes}

    grid = np.linspace(0.05, 0.95, 19)
    grid_scan = [(float(r), _power(float(r))) for r in grid]
    min_detect = float("nan")
    for r, p in grid_scan:
        if p >= target_power:
            min_detect = r
            break

    return {
        "per_effect_power": per_effect,
        "grid_scan": grid_scan,
        "min_detectable_r": min_detect,
        "target_power": target_power,
        "bonferroni_alpha": bonf_alpha,
        "n_per_group": n_per_group,
        "n_comparisons": n_comparisons,
    }

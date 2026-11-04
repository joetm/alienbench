"""Sensitivity analysis for the human baseline sample size (paper sec:human).

The human reference for the Ward Departure Score is collected from
``N = 100`` Prolific participants under the baseline prompt. This module
answers the design question that follows from that number: given the
realized sample sizes, what is the smallest human-model difference the
comparison can detect?

A sensitivity analysis fixes the sample size, the significance level, and
the target power, and solves for the effect size (Lakens, 2022). It is not
observed (post-hoc) power, which is a deterministic function of the p-value
and carries no information beyond it. Every quantity here is computed before
looking at whether a given comparison was significant.

Three comparisons are covered, matching the three ways the human sample
enters the paper:

1. **Ward total score, human vs. one model.** Welch-style two-sample t
   (parametric) and Mann-Whitney U (the test the paper reports). The
   parametric minimum detectable effect (MDE) is returned as Cohen's d and
   converted to Ward points using the observed human standard deviation.
   The nonparametric MDE is returned as a rank-biserial correlation and is
   estimated by Monte Carlo, so it also covers the unequal-n case that the
   equal-n helper in :mod:`alienbench.stats` does not.

2. **Per-dimension departure rate, human vs. one model.** Two-proportion
   z-test, giving an MDE in percentage points at each dimension's observed
   human rate.

3. **Precision of the human anchor itself.** Half-width of the 95%
   confidence interval on the human mean, as a function of N. The human
   sample is a reference point, not a treatment arm, so its precision is
   the quantity a reader needs in order to place the model means against
   it.

Each comparison is reported both uncorrected and under the Bonferroni
correction the paper applies (10 models, 10 dimensions).

The analysis re-aggregates existing Ward score records. No regeneration and
no additional data collection is required.
"""

from __future__ import annotations

import argparse
import logging
from math import sqrt
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.stats as stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from alienbench.config import load_config  # noqa: E402
from alienbench.dimensions import WARD_DIMENSIONS  # noqa: E402
from alienbench.paths import iter_jsonl, ward_scores_path  # noqa: E402

logger = logging.getLogger(__name__)

# Subject-model key under which the Prolific responses are stored.
HUMAN_MODEL = "human/prolific-baseline"

# The human baseline was collected under the baseline prompt only, so the
# like-for-like model comparison is restricted to that condition.
HUMAN_PROMPT_VARIANT = "baseline"

ALPHA = 0.05
POWER_TARGET = 0.80
N_SIMS = 4000
SEED = 1234

# Sample sizes swept for the precision curve. Brackets the realized N = 100
# and extends far enough to show the shape of the returns to more data.
N_GRID = [25, 50, 75, 100, 150, 200, 300, 400, 800]

DIMENSION_IDS = [d["id"] for d in WARD_DIMENSIONS]
DIMENSION_LABELS = {d["id"]: d["label"] for d in WARD_DIMENSIONS}

_WARD_MAX = 10.0  # Ward total is the count of departed dimensions, 0-10.

plt.rcParams.update({
    # Times matches the NeurIPS body font. The fallbacks cover systems
    # without Times New Roman. Type 42 embeds TrueType instead of Type 3.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Parametric MDE: two independent samples, difference in means
# ---------------------------------------------------------------------------

# Above this many degrees of freedom the noncentral t is numerically
# unstable at large noncentrality and is indistinguishable from a normal.
_NORMAL_DF = 1e5


def power_two_sample_t(d: float, n1: int, n2: int, alpha: float = ALPHA) -> float:
    """Power of a two-sided two-sample t-test at Cohen's ``d``.

    Exact under the noncentral t distribution with equal variances. The
    noncentrality parameter is ``d * sqrt(n1 n2 / (n1 + n2))``. For very
    large degrees of freedom the noncentral t evaluation loses precision, so
    the limiting normal is used instead.
    """
    if n1 < 2 or n2 < 2:
        return float("nan")
    df = n1 + n2 - 2
    ncp = abs(d) * sqrt((n1 * n2) / (n1 + n2))

    if df > _NORMAL_DF:
        crit = stats.norm.ppf(1 - alpha / 2)
        return float(stats.norm.sf(crit - ncp) + stats.norm.cdf(-crit - ncp))

    crit = stats.t.ppf(1 - alpha / 2, df)
    upper = stats.nct.sf(crit, df, ncp)
    lower = stats.nct.cdf(-crit, df, ncp)
    total = float(upper + lower)
    if np.isnan(total):
        # nct underflows only once the alternative is far beyond the
        # critical value, where power is saturated at 1.
        return 1.0
    return total


def mde_cohens_d(n1: int, n2: int, alpha: float = ALPHA,
                 power: float = POWER_TARGET) -> float:
    """Smallest Cohen's d detectable at ``power`` with a two-sided t-test.

    Solved by root-finding on :func:`power_two_sample_t`, which is strictly
    increasing in ``d``.
    """
    if n1 < 2 or n2 < 2:
        return float("nan")

    def _gap(d: float) -> float:
        return power_two_sample_t(d, n1, n2, alpha) - power

    hi = 1.0
    while _gap(hi) < 0 and hi < 1e4:
        hi *= 2.0
    if _gap(hi) < 0:
        return float("nan")
    return float(optimize.brentq(_gap, 1e-9, hi, xtol=1e-10))


def mean_ci_half_width(sd: float, n: int, confidence: float = 0.95) -> float:
    """Half-width of the t-based confidence interval on a mean."""
    if n < 2:
        return float("nan")
    return float(stats.t.ppf((1 + confidence) / 2, df=n - 1) * sd / sqrt(n))


# ---------------------------------------------------------------------------
# Nonparametric MDE: Mann-Whitney U, rank-biserial correlation
# ---------------------------------------------------------------------------

def rank_biserial_to_shift(r: float) -> float:
    """Location shift between two unit normals giving population ``r``.

    For X ~ N(0,1) and Y ~ N(mu,1), the rank-biserial correlation is
    ``r = 2 P(X < Y) - 1 = 2 Phi(mu / sqrt(2)) - 1``. Inverting gives
    ``mu = sqrt(2) Phi^-1((r + 1) / 2)``. The small-effect linearization
    ``mu = r sqrt(pi)`` used by
    :func:`alienbench.stats.retrospective_power` agrees with this exact form
    to within 2% for ``|r| <= 0.25`` and 4% for ``|r| <= 0.35``, and
    understates the shift beyond that.
    """
    r = float(np.clip(abs(r), 0.0, 1 - 1e-9))
    return float(sqrt(2.0) * stats.norm.ppf((r + 1.0) / 2.0))


def power_mannwhitney(r: float, n1: int, n2: int, alpha: float = ALPHA,
                      n_sims: int = N_SIMS, seed: int = SEED) -> float:
    """Monte-Carlo power of a two-sided Mann-Whitney U test at ``r``.

    Two independent unit-variance normal samples of size ``n1`` and ``n2``
    are drawn with the location shift implied by ``r``. Normal margins are
    the conservative choice relative to the heavier-tailed, discrete Ward
    distribution, which the rank test handles more efficiently.
    """
    if n1 < 2 or n2 < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    mu = rank_biserial_to_shift(r)
    a = rng.normal(0.0, 1.0, size=(n_sims, n1))
    b = rng.normal(mu, 1.0, size=(n_sims, n2))
    hits = 0
    for i in range(n_sims):
        _, p = stats.mannwhitneyu(a[i], b[i], alternative="two-sided")
        if p < alpha:
            hits += 1
    return hits / n_sims


def mde_rank_biserial(n1: int, n2: int, alpha: float = ALPHA,
                      power: float = POWER_TARGET, n_sims: int = N_SIMS,
                      seed: int = SEED, step: float = 0.02) -> float:
    """Smallest rank-biserial ``r`` detectable at ``power``, by Monte Carlo.

    Scans a grid of ``r`` and linearly interpolates the first crossing of
    the target power. Returns NaN if the grid never reaches ``power``.
    Unlike :func:`alienbench.stats.retrospective_power` this accepts unequal
    group sizes, which the human-model comparison requires.
    """
    grid = np.arange(step, 0.95 + step, step)
    prev_r, prev_p = 0.0, alpha  # power at r = 0 equals the nominal alpha
    for r in grid:
        p = power_mannwhitney(float(r), n1, n2, alpha, n_sims=n_sims, seed=seed)
        if p >= power:
            if p == prev_p:
                return float(r)
            frac = (power - prev_p) / (p - prev_p)
            return float(prev_r + frac * (r - prev_r))
        prev_r, prev_p = float(r), p
    return float("nan")


# ---------------------------------------------------------------------------
# Proportion MDE: per-dimension departure rates
# ---------------------------------------------------------------------------

def power_two_proportions(p1: float, p2: float, n1: int, n2: int,
                          alpha: float = ALPHA) -> float:
    """Power of a two-sided two-proportion z-test (normal approximation)."""
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled = (n1 * p1 + n2 * p2) / (n1 + n2)
    se_null = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    se_alt = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se_null <= 0 or se_alt <= 0:
        return float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf((z * se_null - abs(p2 - p1)) / se_alt))


def mde_proportion_difference(p_ref: float, n1: int, n2: int,
                              alpha: float = ALPHA,
                              power: float = POWER_TARGET) -> float:
    """Smallest upward departure-rate difference detectable at ``power``.

    Returns ``p_alt - p_ref`` in proportion units, where ``p_alt > p_ref`` is
    the model rate. The upward direction matches the paper's finding that
    models depart more than humans on every dimension. Returns NaN when no
    ``p_alt < 1`` reaches the target power.

    Treating the judge-averaged rate as a binomial proportion is
    conservative. Averaging three judges per generation reduces the variance
    of the per-generation rate below the binomial variance assumed here, so
    the true MDE is no larger than the value returned.
    """
    p_ref = float(np.clip(p_ref, 1e-6, 1 - 1e-6))

    def _gap(p_alt: float) -> float:
        return power_two_proportions(p_ref, p_alt, n1, n2, alpha) - power

    hi = 1 - 1e-6
    if _gap(hi) < 0:
        return float("nan")
    lo = p_ref + 1e-6
    if _gap(lo) > 0:
        return float(lo - p_ref)
    return float(optimize.brentq(_gap, lo, hi, xtol=1e-8) - p_ref)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_generation_scores(data_dir: Path, judge_models: list[str],
                           subject_model: str, prompt_variant: str) -> pd.DataFrame:
    """Judge-averaged Ward scores, one row per generation.

    Columns: ``generation_id``, ``n_judges``, ``ward_score``, and one
    ``dim_<id>`` per Ward dimension. Averaging across judges before
    aggregating across generations matches the convention used by
    :mod:`alienbench.latex_tables` and :mod:`alienbench.analyze`.
    """
    rows: list[dict] = []
    for judge in judge_models:
        path = ward_scores_path(data_dir, judge, subject_model, prompt_variant)
        if not path.exists():
            continue
        for rec in iter_jsonl(path):
            row = {
                "generation_id": rec["generation_id"],
                "judge_model": rec["judge_model"],
                "ward_score": float(rec["ward_score"]),
            }
            per_dim = rec.get("per_dimension", {})
            row.update({f"dim_{d}": float(per_dim.get(d, 0)) for d in DIMENSION_IDS})
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["generation_id", "n_judges", "ward_score"]
            + [f"dim_{d}" for d in DIMENSION_IDS]
        )

    df = pd.DataFrame(rows)
    value_cols = ["ward_score"] + [f"dim_{d}" for d in DIMENSION_IDS]
    agg = df.groupby("generation_id")[value_cols].mean().reset_index()
    agg["n_judges"] = df.groupby("generation_id")["judge_model"].nunique().values
    return agg


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _corrected_alpha(alpha: float, n_comparisons: int) -> float:
    return alpha / max(int(n_comparisons), 1)


def total_score_sensitivity(human: np.ndarray, model_scores: dict[str, np.ndarray],
                            alpha: float = ALPHA, power: float = POWER_TARGET,
                            n_sims: int = N_SIMS, seed: int = SEED) -> pd.DataFrame:
    """Per-model observed human-model gap against the detectable gap.

    One row per model. ``detected`` marks whether the observed effect
    exceeds the Bonferroni-corrected MDE, that is, whether the design can
    resolve the gap the data show. The comparison is made on Cohen's d,
    which depends only on the sample sizes and so is one threshold for every
    model.

    The MDE is also expressed in Ward points. That conversion needs a
    standard deviation, and the pooled SD differs slightly across models, so
    two columns are returned: ``mde_ward_points`` uses each model's own
    pooled SD, and ``mde_ward_points_ref`` uses a single reference SD built
    from the human sample and the median model SD. The reference column is
    the one to quote, because it is a single number rather than ten.
    """
    n_h = len(human)
    n_comparisons = len(model_scores)
    alpha_bonf = _corrected_alpha(alpha, n_comparisons)
    sd_h = float(np.std(human, ddof=1))
    mean_h = float(np.mean(human))

    n_med = int(np.median([len(s) for s in model_scores.values()]))
    sd_med = float(np.median([np.std(s, ddof=1) for s in model_scores.values()]))
    sd_ref = sqrt(((n_h - 1) * sd_h ** 2 + (n_med - 1) * sd_med ** 2)
                  / (n_h + n_med - 2))

    rows = []
    for model, scores in sorted(model_scores.items()):
        n_m = len(scores)
        mean_m = float(np.mean(scores))
        sd_m = float(np.std(scores, ddof=1))
        sd_pooled = sqrt(((n_h - 1) * sd_h ** 2 + (n_m - 1) * sd_m ** 2) / (n_h + n_m - 2))
        gap = mean_m - mean_h
        d_obs = gap / sd_pooled if sd_pooled > 0 else float("nan")

        d_mde = mde_cohens_d(n_h, n_m, alpha, power)
        d_mde_bonf = mde_cohens_d(n_h, n_m, alpha_bonf, power)
        r_obs, _ = stats.mannwhitneyu(scores, human, alternative="two-sided")
        r_obs = 1.0 - (2.0 * r_obs) / (n_m * n_h)

        rows.append({
            "model": model,
            "n_human": n_h,
            "n_model": n_m,
            "human_mean": mean_h,
            "model_mean": mean_m,
            "observed_gap_ward_points": gap,
            "observed_cohens_d": d_obs,
            "observed_rank_biserial": -r_obs,
            "mde_cohens_d": d_mde,
            "mde_ward_points": d_mde * sd_pooled,
            "mde_cohens_d_bonferroni": d_mde_bonf,
            "mde_ward_points_bonferroni": d_mde_bonf * sd_pooled,
            "mde_ward_points_ref": d_mde * sd_ref,
            "mde_ward_points_ref_bonferroni": d_mde_bonf * sd_ref,
            "detected": bool(abs(d_obs) >= d_mde_bonf),
        })
    df = pd.DataFrame(rows)
    df.attrs["sd_reference"] = sd_ref
    return df


def dimension_sensitivity(human_df: pd.DataFrame, n_model: int,
                          alpha: float = ALPHA, power: float = POWER_TARGET,
                          n_model_pooled: int | None = None,
                          model_dim_rates: dict[str, dict[str, float]] | None = None
                          ) -> pd.DataFrame:
    """Per-dimension detectable departure-rate difference in percentage points.

    ``n_model`` is the baseline-matched model sample. ``n_model_pooled``, when
    given, adds the same MDE at the larger sample the paper's per-dimension
    table is built from, which pools every prompt condition.

    ``model_dim_rates`` maps each model to its per-dimension departure rate.
    When supplied, the returned frame carries the median and maximum observed
    model-human gap and the number of models whose gap clears the pooled
    corrected MDE. The count is the quantity of interest: it states how much
    of the reported per-dimension separation the design can resolve, rather
    than collapsing to the single weakest model.
    """
    n_h = len(human_df)
    alpha_bonf = _corrected_alpha(alpha, len(DIMENSION_IDS))
    rows = []
    for dim in DIMENSION_IDS:
        p_h = float(human_df[f"dim_{dim}"].mean())
        mde = mde_proportion_difference(p_h, n_h, n_model, alpha, power)
        mde_bonf = mde_proportion_difference(p_h, n_h, n_model, alpha_bonf, power)
        row = {
            "dimension": dim,
            "label": DIMENSION_LABELS[dim],
            "human_rate": p_h,
            "n_human": n_h,
            "n_model": n_model,
            "mde_pp": mde * 100,
            "mde_pp_bonferroni": mde_bonf * 100,
            "detectable_model_rate": (p_h + mde) * 100,
            "detectable_model_rate_bonferroni": (p_h + mde_bonf) * 100,
        }
        threshold = mde_bonf
        if n_model_pooled is not None:
            row["n_model_pooled"] = n_model_pooled
            row["mde_pp_pooled"] = mde_proportion_difference(
                p_h, n_h, n_model_pooled, alpha, power) * 100
            threshold = mde_proportion_difference(
                p_h, n_h, n_model_pooled, alpha_bonf, power)
            row["mde_pp_pooled_bonferroni"] = threshold * 100
        if model_dim_rates:
            gaps = np.array([rates[dim] - p_h for rates in model_dim_rates.values()])
            row["median_observed_gap_pp"] = float(np.median(gaps)) * 100
            row["max_observed_gap_pp"] = float(gaps.max()) * 100
            row["n_models"] = len(gaps)
            row["n_models_resolved"] = int((np.abs(gaps) >= threshold).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# Nominal comparison size standing in for "many more generations per model".
# The MDE at this size is within 0.01 of the n -> infinity limit, so it
# reports the floor imposed by the human sample alone.
_EFFECTIVELY_INFINITE = 10_000_000


def design_variants(variants: list[tuple[str, int]], n_human: int,
                    sd_scale: float, alpha: float = ALPHA,
                    power: float = POWER_TARGET,
                    n_comparisons: int = 10) -> pd.DataFrame:
    """MDE under each way the human sample can be compared against models.

    ``variants`` pairs a label with the model-side sample size. Including a
    very large size isolates the floor imposed by the human sample alone:
    below that floor, no amount of model generation helps, and above it the
    model side is the binding constraint. This separation is what determines
    whether ``N`` or ``n`` is worth increasing.

    ``sd_scale`` converts Cohen's d to Ward points.
    """
    alpha_bonf = _corrected_alpha(alpha, n_comparisons)
    rows = []
    for label, n_model in variants:
        d = mde_cohens_d(n_human, n_model, alpha, power)
        d_bonf = mde_cohens_d(n_human, n_model, alpha_bonf, power)
        rows.append({
            "design": label,
            "n_human": n_human,
            "n_model": n_model,
            "mde_cohens_d": d,
            "mde_ward_points": d * sd_scale,
            "mde_cohens_d_bonferroni": d_bonf,
            "mde_ward_points_bonferroni": d_bonf * sd_scale,
            "binding_constraint": "human" if n_model >= n_human else "model",
        })
    return pd.DataFrame(rows)


def precision_curve(sd_human: float, n_model: int, n_grid: list[int],
                    alpha: float = ALPHA, power: float = POWER_TARGET,
                    sd_pooled: float | None = None) -> pd.DataFrame:
    """Detectable gap and anchor precision as a function of human N.

    ``n_model`` is held fixed at its realized value, which is what makes the
    curve informative: it separates the contribution of the human sample
    size from the contribution of the model sample size.
    """
    scale = sd_pooled if sd_pooled is not None else sd_human
    rows = []
    for n in n_grid:
        d_mde = mde_cohens_d(n, n_model, alpha, power)
        rows.append({
            "n_human": n,
            "mde_cohens_d": d_mde,
            "mde_ward_points": d_mde * scale,
            "ci_half_width_ward_points": mean_ci_half_width(sd_human, n),
        })
    df = pd.DataFrame(rows)
    # Limit of an infinitely large human sample with n_model held fixed.
    d_limit = mde_cohens_d(10_000_000, n_model, alpha, power)
    df.attrs["mde_cohens_d_limit"] = d_limit
    df.attrs["mde_ward_points_limit"] = d_limit * scale
    return df


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _write_latex_table(total_df: pd.DataFrame, curve_df: pd.DataFrame,
                       results_dir: Path) -> Path:
    """Emit the rebuttal table of observed gaps against the detectable gap."""
    n_h = int(total_df["n_human"].iloc[0])
    n_m = int(total_df["n_model"].iloc[0])
    mde_pts = float(total_df["mde_ward_points_ref_bonferroni"].iloc[0])
    mde_d = float(total_df["mde_cohens_d_bonferroni"].iloc[0])

    ordered = total_df.sort_values("observed_gap_ward_points", ascending=False)
    body = []
    for _, row in ordered.iterrows():
        mark = "yes" if row["detected"] else "no"
        body.append(
            f"    {row['model'].replace('_', ' ')} & "
            f"{row['model_mean']:.2f} & "
            f"{row['observed_gap_ward_points']:+.2f} & "
            f"{row['observed_cohens_d']:+.2f} & {mark} \\\\"
        )

    lines = [
        "\\begin{table}[htb]",
        "  \\centering",
        "  \\caption{Sensitivity of the human-model comparison at the realized "
        f"sample sizes ($N={n_h}$ human responses, $n={n_m}$ generations per "
        "model, baseline prompt). At 80\\% power and a Bonferroni-corrected "
        f"$\\alpha' = {ALPHA / len(total_df):.4f}$ over "
        f"{len(total_df)} model comparisons, the design detects any gap of "
        f"{mde_pts:.2f} Ward points ($d = {mde_d:.2f}$) or larger. "
        "The rightmost column reports whether the observed gap exceeds that "
        "threshold.}",
        "  \\small",
        "  \\label{tab:sensitivity_human}",
        "  \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    Model & Ward mean & Gap vs. human & Cohen's $d$ & Detectable \\\\",
        "    \\midrule",
        *body,
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    out_path = results_dir / "tab_sensitivity_human.tex"
    out_path.write_text("\n".join(lines))
    return out_path


def _write_designs_latex_table(variants_df: pd.DataFrame, results_dir: Path) -> Path:
    """Emit the MDE-by-design table separating the human and model constraints."""
    n_h = int(variants_df["n_human"].iloc[0])
    body = []
    for _, row in variants_df.iterrows():
        n_m = int(row["n_model"])
        n_str = "$\\to\\infty$" if n_m >= _EFFECTIVELY_INFINITE else f"{n_m}"
        label = row["design"].split(" (n")[0]
        body.append(
            f"    {label} & {n_str} & "
            f"{row['mde_cohens_d']:.2f} & {row['mde_ward_points']:.2f} & "
            f"{row['mde_cohens_d_bonferroni']:.2f} & "
            f"{row['mde_ward_points_bonferroni']:.2f} \\\\"
        )

    lines = [
        "\\begin{table}[htb]",
        "  \\centering",
        "  \\caption{Minimum detectable effect for the human-model comparison at "
        f"80\\% power and $N = {n_h}$ human responses, under each pairing of the "
        "human sample with the model side. The last row holds the human sample "
        "fixed and lets the model sample grow without bound, isolating the floor "
        "that $N$ alone imposes. Ward points use a single reference pooled "
        "standard deviation, built from the human sample and the median model "
        "sample, so the column is comparable across rows.}",
        "  \\small",
        "  \\label{tab:sensitivity_human_designs}",
        "  \\begin{tabular}{lccccc}",
        "    \\toprule",
        "    & & \\multicolumn{2}{c}{$\\alpha = 0.05$} & "
        "\\multicolumn{2}{c}{$\\alpha' = 0.005$} \\\\",
        "    \\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        "    Comparison & $n$ & $d$ & Ward pts & $d$ & Ward pts \\\\",
        "    \\midrule",
        *body,
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    out_path = results_dir / "tab_sensitivity_human_designs.tex"
    out_path.write_text("\n".join(lines))
    return out_path


def _write_figure(total_df: pd.DataFrame, curve_df: pd.DataFrame,
                  results_dir: Path) -> Path:
    """Two-panel figure: returns to human N, and observed gaps against the MDE."""
    accent = "#1f4e79"
    muted = "#8c8c8c"

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # Panel A: detectable gap as a function of human N, n_model fixed.
    n_h_real = int(total_df["n_human"].iloc[0])
    n_m = int(total_df["n_model"].iloc[0])
    limit = curve_df.attrs["mde_ward_points_limit"]
    ax_a.plot(curve_df["n_human"], curve_df["mde_ward_points"],
              color=accent, linewidth=2, marker="o", markersize=4)
    ax_a.axhline(limit, color=muted, linewidth=1, linestyle="--")
    ax_a.annotate(f"floor imposed by $n={n_m}$: {limit:.2f}",
                  xy=(curve_df["n_human"].iloc[0], limit), xytext=(2, 6),
                  textcoords="offset points", ha="left", fontsize=8, color=muted)
    realized = curve_df.loc[curve_df["n_human"] == n_h_real, "mde_ward_points"]
    if not realized.empty:
        y = float(realized.iloc[0])
        ax_a.plot([n_h_real], [y], marker="o", markersize=8,
                  markerfacecolor="white", markeredgecolor=accent,
                  markeredgewidth=2, zorder=5)
        ax_a.annotate(f"$N={n_h_real}$: {y:.2f}", xy=(n_h_real, y), xytext=(8, 8),
                      textcoords="offset points", fontsize=9, color=accent)
    ax_a.set_xscale("log")
    ax_a.set_xticks(curve_df["n_human"])
    ax_a.set_xticklabels([str(int(n)) for n in curve_df["n_human"]], fontsize=8)
    ax_a.set_xlabel("Human sample size $N$")
    ax_a.set_ylabel("Detectable gap (Ward points)")
    ax_a.set_title(f"(a) Returns to human $N$ (models fixed at $n={n_m}$)")
    ax_a.grid(True, alpha=0.25, linewidth=0.5)
    ax_a.set_axisbelow(True)

    # Panel B: observed gaps against the corrected MDE band.
    ordered = total_df.sort_values("observed_gap_ward_points")
    mde = float(ordered["mde_ward_points_ref_bonferroni"].iloc[0])
    labels = [m.split("/")[-1] for m in ordered["model"]]
    ypos = np.arange(len(ordered))
    ax_b.axvspan(-mde, mde, color=muted, alpha=0.18, linewidth=0)
    ax_b.axvline(0, color="black", linewidth=0.8)
    colors = [accent if d else muted for d in ordered["detected"]]
    ax_b.scatter(ordered["observed_gap_ward_points"], ypos, s=42,
                 color=colors, zorder=3)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels(labels, fontsize=8)
    ax_b.set_xlabel("Observed gap vs. human (Ward points)")
    ax_b.set_title(f"(b) Observed gaps vs. detectable band ($\\pm${mde:.2f})")
    ax_b.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax_b.set_axisbelow(True)
    # The shaded band is named in the panel title, so it needs no inline label.
    ax_b.set_ylim(-0.6, len(ordered) - 0.4)

    fig.tight_layout()
    out_path = results_dir / "fig_sensitivity_human.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out_path


def _write_summary(human: np.ndarray, total_df: pd.DataFrame,
                   dim_df: pd.DataFrame, curve_df: pd.DataFrame,
                   variants_df: pd.DataFrame,
                   r_mde: float, r_mde_bonf: float,
                   data_dir: Path, results_dir: Path) -> Path:
    n_h = len(human)
    n_m = int(total_df["n_model"].iloc[0])
    sd_h = float(np.std(human, ddof=1))
    alpha_bonf = _corrected_alpha(ALPHA, len(total_df))
    half = mean_ci_half_width(sd_h, n_h)
    half_double = mean_ci_half_width(sd_h, 2 * n_h)
    d_mde = float(total_df["mde_cohens_d"].iloc[0])
    d_mde_bonf = float(total_df["mde_cohens_d_bonferroni"].iloc[0])
    pts_bonf = float(total_df["mde_ward_points_ref_bonferroni"].iloc[0])
    limit_pts = curve_df.attrs["mde_ward_points_limit"]

    undetected = total_df.loc[~total_df["detected"], "model"].tolist()
    detected = total_df.loc[total_df["detected"], "model"].tolist()
    smallest = total_df.loc[total_df["detected"]] \
        .assign(abs_gap=lambda d: d["observed_gap_ward_points"].abs()) \
        .sort_values("abs_gap")

    lines = [
        "# Sensitivity analysis: human baseline sample size",
        "",
        f"Data directory: {data_dir}",
        f"Human sample: {HUMAN_MODEL}, prompt variant '{HUMAN_PROMPT_VARIANT}'.",
        "",
        "## Realized design",
        f"  Human responses          N = {n_h}",
        f"  Generations per model    n = {n_m}",
        f"  Model comparisons            {len(total_df)}",
        f"  Human Ward mean (0-10)       {np.mean(human):.2f} (SD {sd_h:.2f})",
        f"  95% CI half-width            {half:.3f} Ward points",
        "",
        "## Minimum detectable effect, Ward total score (80% power, two-sided)",
        f"  alpha = {ALPHA:.2f}            d = {d_mde:.3f} "
        f"({float(total_df['mde_ward_points_ref'].iloc[0]):.2f} Ward points)",
        f"  alpha' = {alpha_bonf:.4f}        d = {d_mde_bonf:.3f} "
        f"({pts_bonf:.2f} Ward points)",
        f"  Mann-Whitney, alpha = {ALPHA:.2f}   rank-biserial r = {r_mde:.3f}",
        f"  Mann-Whitney, alpha' = {alpha_bonf:.4f}  rank-biserial r = {r_mde_bonf:.3f}",
        "",
        "## Minimum detectable effect by comparison design (N human held at "
        f"{n_h})",
        f"  {'Comparison':<48} {'n':>10} {'d':>7} {'pts':>7} "
        f"{'d(Bonf)':>9} {'pts(Bonf)':>10}",
        "  " + "-" * 94,
    ]
    for _, row in variants_df.iterrows():
        n_variant = int(row["n_model"])
        n_str = "inf" if n_variant >= _EFFECTIVELY_INFINITE else str(n_variant)
        label = row["design"].split(" (n")[0]
        lines.append(
            f"  {label:<48} {n_str:>10} {row['mde_cohens_d']:7.3f} "
            f"{row['mde_ward_points']:7.2f} "
            f"{row['mde_cohens_d_bonferroni']:9.3f} "
            f"{row['mde_ward_points_bonferroni']:10.2f}"
        )
    lines += [
        "",
        "## Where the constraint sits",
        f"  Detectable gap at N = {n_h}, n = {n_m}:        {pts_bonf:.2f} Ward points",
        f"  Detectable gap as N -> infinity, n = {n_m}:  {limit_pts:.2f} Ward points",
        f"  Reduction available from more humans:   "
        f"{pts_bonf - limit_pts:.2f} Ward points "
        f"({100 * (pts_bonf - limit_pts) / pts_bonf:.0f}%)",
        f"  Anchor precision at N = {n_h}:  +/- {half:.3f} Ward points",
        f"  Anchor precision at N = {2 * n_h}:  +/- {half_double:.3f} Ward points "
        f"(gain {half - half_double:.3f})",
        "",
        "## Observed gaps against the detectable threshold",
        f"  Resolved (|d| >= MDE):   {len(detected)}/{len(total_df)} models",
        f"  Unresolved:              {len(undetected)}/{len(total_df)} models"
        + (f" [{', '.join(undetected)}]" if undetected else ""),
    ]
    if not smallest.empty:
        row = smallest.iloc[0]
        lines.append(
            f"  Smallest resolved gap:   {row['model']} "
            f"{row['observed_gap_ward_points']:+.2f} Ward points (d = "
            f"{row['observed_cohens_d']:+.2f}), "
            f"{abs(row['observed_cohens_d']) / d_mde_bonf:.2f}x the MDE"
        )
    lines += [
        "",
        "## Per-dimension detectable difference (percentage points, Bonferroni "
        "over 10 dimensions)",
        "  'baseline' pairs the human sample with the n=%d baseline generations."
        % int(dim_df["n_model"].iloc[0]),
        "  'pooled' pairs it with the n=%d generations the paper's dimension "
        "table aggregates." % int(dim_df.get("n_model_pooled",
                                             pd.Series([0])).iloc[0]),
        "  'median gap' and 'max gap' are the observed model-human differences; "
        "'resolved' counts",
        "  the models whose gap clears the pooled corrected MDE.",
        "",
        f"  {'Dimension':<28} {'Human':>7} {'baseline':>10} {'pooled':>9} "
        f"{'median gap':>12} {'max gap':>9} {'resolved':>9}",
        "  " + "-" * 89,
    ]
    for _, row in dim_df.iterrows():
        pooled = row.get("mde_pp_pooled_bonferroni", float("nan"))
        med = row.get("median_observed_gap_pp", float("nan"))
        mx = row.get("max_observed_gap_pp", float("nan"))
        n_res = row.get("n_models_resolved", -1)
        n_tot = row.get("n_models", -1)
        res_str = "n/a" if n_res < 0 else f"{int(n_res)}/{int(n_tot)}"
        lines.append(
            f"  {row['label']:<28} {row['human_rate'] * 100:6.1f}% "
            f"{row['mde_pp_bonferroni']:9.1f}pp {pooled:8.1f}pp "
            f"{med:11.1f}pp {mx:8.1f}pp {res_str:>9}"
        )
    if "n_models_resolved" in dim_df:
        total_res = int(dim_df["n_models_resolved"].sum())
        total_cmp = int(dim_df["n_models"].sum())
        lines.append(
            f"  Resolved model-dimension comparisons: {total_res}/{total_cmp} "
            f"({100 * total_res / total_cmp:.0f}%)"
        )
    lines += [
        "",
        "## Sample-size curve (n_model fixed)",
        f"  {'N':>6} {'MDE d':>8} {'MDE pts':>9} {'CI half-width':>14}",
        "  " + "-" * 40,
    ]
    for _, row in curve_df.iterrows():
        lines.append(
            f"  {int(row['n_human']):6d} {row['mde_cohens_d']:8.3f} "
            f"{row['mde_ward_points']:9.2f} {row['ci_half_width_ward_points']:14.3f}"
        )
    lines.append("")

    out_path = results_dir / "summary_sensitivity_human.txt"
    out_path.write_text("\n".join(lines))
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config_path: str = "config.yaml", human_data_dir: str | None = None,
        n_sims: int = N_SIMS) -> dict:
    """Run the human-baseline sensitivity analysis and write results.

    ``human_data_dir`` overrides the location of the human Ward score
    records when they are archived apart from the model records. It
    defaults to the configured ``data_dir``.
    """
    cfg = load_config(config_path)
    data_dir = Path(cfg.data_dir)
    h_dir = Path(human_data_dir) if human_data_dir else data_dir
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    human_df = load_generation_scores(
        h_dir, cfg.judge_models, HUMAN_MODEL, HUMAN_PROMPT_VARIANT
    )
    if human_df.empty:
        raise SystemExit(
            f"No Ward scores for {HUMAN_MODEL} under '{HUMAN_PROMPT_VARIANT}' in "
            f"{h_dir / 'scores'}. Pass --human-data-dir to point at the "
            "directory holding the human records."
        )

    model_scores: dict[str, np.ndarray] = {}
    n_all_conditions: list[int] = []
    pooled_dim_rates: dict[str, dict[str, float]] = {}
    for model in cfg.models:
        df = load_generation_scores(
            data_dir, cfg.judge_models, model, HUMAN_PROMPT_VARIANT
        )
        if df.empty:
            logger.warning("No baseline Ward scores for %s; skipping", model)
            continue
        model_scores[model] = df["ward_score"].to_numpy()

        # The paper's per-dimension table pools every prompt condition.
        pooled = pd.concat(
            [load_generation_scores(data_dir, cfg.judge_models, model, v.id)
             for v in cfg.prompt_variants],
            ignore_index=True,
        )
        n_all_conditions.append(len(pooled))
        pooled_dim_rates[model] = {
            d: float(pooled[f"dim_{d}"].mean()) for d in DIMENSION_IDS
        }

    if not model_scores:
        raise SystemExit(f"No model Ward scores found in {data_dir / 'scores'}.")

    human = human_df["ward_score"].to_numpy()
    n_h, n_m = len(human), int(np.median([len(v) for v in model_scores.values()]))
    logger.info("Human N = %d, model n = %d, %d models", n_h, n_m, len(model_scores))

    total_df = total_score_sensitivity(human, model_scores, n_sims=n_sims)

    n_pooled = int(np.median(n_all_conditions)) if n_all_conditions else n_m
    dim_df = dimension_sensitivity(
        human_df, n_m, n_model_pooled=n_pooled, model_dim_rates=pooled_dim_rates
    )

    sd_h = float(np.std(human, ddof=1))
    sd_ref = float(total_df.attrs["sd_reference"])
    curve_df = precision_curve(sd_h, n_m, N_GRID, sd_pooled=sd_ref)

    n_panel = n_m * len(model_scores)
    variants_df = design_variants(
        [
            (f"Single model, baseline prompt (n={n_m})", n_m),
            (f"Balanced reference (n={n_h})", n_h),
            (f"Single model, pooled over conditions (n={n_pooled})", n_pooled),
            (f"Full model panel, baseline prompt (n={n_panel})", n_panel),
            ("Human sample as the only constraint (n -> inf)", _EFFECTIVELY_INFINITE),
        ],
        n_h, sd_ref, n_comparisons=len(model_scores),
    )

    alpha_bonf = _corrected_alpha(ALPHA, len(model_scores))
    r_mde = mde_rank_biserial(n_h, n_m, ALPHA, n_sims=n_sims)
    r_mde_bonf = mde_rank_biserial(n_h, n_m, alpha_bonf, n_sims=n_sims)
    total_df["mde_rank_biserial"] = r_mde
    total_df["mde_rank_biserial_bonferroni"] = r_mde_bonf

    total_df.to_csv(results_dir / "table_sensitivity_human.csv", index=False)
    dim_df.to_csv(results_dir / "table_sensitivity_human_dimensions.csv", index=False)
    curve_df.to_csv(results_dir / "table_sensitivity_human_curve.csv", index=False)
    variants_df.to_csv(results_dir / "table_sensitivity_human_designs.csv", index=False)
    tex_path = _write_latex_table(total_df, curve_df, results_dir)
    tex_designs = _write_designs_latex_table(variants_df, results_dir)
    fig_path = _write_figure(total_df, curve_df, results_dir)
    sum_path = _write_summary(human, total_df, dim_df, curve_df, variants_df,
                              r_mde, r_mde_bonf, h_dir, results_dir)

    for p in (results_dir / "table_sensitivity_human.csv", tex_path, tex_designs,
              fig_path, sum_path):
        logger.info("Saved %s", p)

    return {
        "total": total_df,
        "dimensions": dim_df,
        "curve": curve_df,
        "designs": variants_df,
        "mde_rank_biserial": r_mde,
        "mde_rank_biserial_bonferroni": r_mde_bonf,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)  # silence per-glyph PDF font-subsetting logs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument(
        "--human-data-dir",
        default=None,
        help="Directory holding the human Ward score records "
             "(defaults to data_dir from the config).",
    )
    parser.add_argument("--n-sims", type=int, default=N_SIMS,
                        help="Monte-Carlo replicates for the Mann-Whitney MDE.")
    args = parser.parse_args()
    run(args.config, human_data_dir=args.human_data_dir, n_sims=args.n_sims)


if __name__ == "__main__":
    main()

"""Ablation: Judge Panel Composition (paper sec:ablation_judges).

Quantifies the effect of judge panel size along two distinct axes:

* **Rank stability.** For each size-``k`` subset ``S`` of judges, compare
  the per-model Ward means of ``S`` against a disjoint held-out reference
  computed from the complement ``\\bar S``. Reported via Spearman ``rho``
  against a permutation null obtained by shuffling model labels on one
  side. Comparing ``S`` against the full-panel aggregate (which contains
  ``S``) would inflate ``rho`` by construction because ``k/n`` of the
  reference is the subset itself; the held-out design removes this
  self-reference.

* **Inter-rater reliability.** For each reduced panel with ``k >= 2``,
  report cross-model stratified Krippendorff ``alpha`` on Ward total.
  ``alpha`` is undefined for ``k = 1`` and is not reported there.

Pure re-aggregation of existing Ward score records; no regeneration needed.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

from alienbench.ablation_utils import n_rank_changes, ranks_equal, spearman
from alienbench.config import load_config
from alienbench.paths import load_ward_scores
from alienbench.stats import krippendorff_alpha_stratified

logger = logging.getLogger(__name__)

NULL_SEED = 0
NULL_N = 10_000


def _same_family_map(
    judge_models: list[str], subject_models: list[str]
) -> dict[str, list[str]]:
    """Map each judge alias to subject models that share its provider prefix.

    Provider family is the first component of the OpenRouter model ID
    (e.g. ``anthropic`` from ``anthropic/claude-opus-4.6``).
    """
    def _provider(m: str) -> str:
        return m.split("/")[0]

    return {
        j: [m for m in subject_models if _provider(m) == _provider(j)]
        for j in judge_models
    }


def _rank_of(series: pd.Series, subject: str) -> float:
    """1-indexed rank of *subject* in *series* (rank 1 = highest value).

    Returns NaN when *subject* is empty or absent from *series*.
    """
    if not subject or subject not in series.index:
        return float("nan")
    ranked = series.rank(ascending=False, method="min")
    return float(ranked[subject])


ALPHA_TENTATIVE = 0.667  # Krippendorff's "tentative conclusions" threshold.

# Preregistered thresholds. RHO_STABLE_THRESHOLD is the rank-stability
# bar applied to held-out Spearman rho (same value used by the
# dimension and paraphrase ablations). ALPHA_DROP_THRESHOLD is the
# allowed drop in cross-model stratified alpha from the full panel to
# a reduced panel (at k >= 2); it sits alongside ALPHA_TENTATIVE on
# the reliability axis rather than replacing it.
RHO_STABLE_THRESHOLD = 0.9
ALPHA_DROP_THRESHOLD = 0.1


def _panel_per_model_means(ward_df: pd.DataFrame) -> pd.Series:
    """Average judges per generation, then generations per model."""
    per_gen = (
        ward_df.groupby(["generation_id", "subject_model"])["ward_score"]
        .mean()
        .reset_index()
    )
    return per_gen.groupby("subject_model")["ward_score"].mean()


def _heldout_reference_means(
    ward_df: pd.DataFrame,
    subset: tuple[str, ...],
    all_judges: list[str],
) -> pd.Series:
    """Per-model means computed from the judges NOT in ``subset``.

    Returns an empty Series when the complement is empty (``k = n``).
    """
    complement = [j for j in all_judges if j not in subset]
    if not complement:
        return pd.Series(dtype=float)
    return _panel_per_model_means(
        ward_df[ward_df["judge_model"].isin(complement)]
    )


def _alpha_cross_model(ward_df: pd.DataFrame) -> float:
    """Cross-model stratified alpha for Ward total on a given panel.

    Returns NaN when the panel has fewer than two judges (alpha undefined).
    """
    if ward_df["judge_model"].nunique() < 2:
        return float("nan")
    strat = krippendorff_alpha_stratified(
        ward_df, "subject_model", "judge_model", "generation_id", "ward_score",
        level="interval",
    )
    return float(strat["cross_model"])


def _permutation_null_heldout_rho(
    reduced: pd.Series,
    heldout: pd.Series,
    n_permutations: int = NULL_N,
    seed: int = NULL_SEED,
) -> tuple[float, float, float]:
    """Empirical null of Spearman rho under random relabeling of models.

    For each of ``n_permutations`` draws, shuffle the model labels on the
    held-out side and recompute rho. Returns (mean, 2.5%, 97.5%) of the
    null distribution, or (NaN, NaN, NaN) when fewer than three models
    overlap or either side has zero variance.
    """
    common = reduced.index.intersection(heldout.index)
    if len(common) < 3:
        return (float("nan"), float("nan"), float("nan"))
    a = reduced.loc[common].to_numpy(dtype=float)
    b = heldout.loc[common].to_numpy(dtype=float)
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(b)
        r = stats.spearmanr(a, perm).statistic
        draws[i] = r if r == r else np.nan
    finite = draws[~np.isnan(draws)]
    if finite.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return (
        float(finite.mean()),
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
    )


def _single_judge_table(
    ward_df: pd.DataFrame,
    full_means: pd.Series,
    judges: list[str],
    n_null_permutations: int = NULL_N,
    null_seed: int = NULL_SEED,
    full_alpha: float = float("nan"),
) -> pd.DataFrame:
    """Per-judge row reporting rank stability on two axes.

    * ``rho_vs_full``: Spearman rho against the full-panel aggregate
      (self-referential; kept for diagnostic comparison).
    * ``rho_vs_heldout``: Spearman rho against the mean of the other
      ``n-1`` judges (disjoint reference).
    * ``null_rho_hi95``: 97.5th percentile of the permutation null for
      ``rho_vs_heldout``.

    ``alpha_cross_model`` is always NaN (alpha undefined for ``k=1``).
    """
    rows = []
    for j in judges:
        sub = ward_df[ward_df["judge_model"] == j]
        if sub.empty:
            rows.append({
                "judge_model": j,
                "n_generations": 0,
                "rho_vs_full": float("nan"),
                "rho_vs_heldout": float("nan"),
                "null_rho_mean": float("nan"),
                "null_rho_lo95": float("nan"),
                "null_rho_hi95": float("nan"),
                "rho_above_null_hi95": False,
                "rho_heldout_above_stable_bar": False,
                "rank_changes_vs_heldout": -1,
                "rank_preserved_vs_heldout": False,
                "alpha_cross_model": float("nan"),
                "alpha_drop_from_full": float("nan"),
                "alpha_drop_within_bar": False,
            })
            continue
        means = _panel_per_model_means(sub)
        heldout = _heldout_reference_means(ward_df, (j,), judges)
        rho_h = spearman(means, heldout)
        null_mean, null_lo, null_hi = _permutation_null_heldout_rho(
            means, heldout,
            n_permutations=n_null_permutations, seed=null_seed,
        )
        rows.append({
            "judge_model": j,
            "n_generations": int(sub["generation_id"].nunique()),
            "rho_vs_full": spearman(means, full_means),
            "rho_vs_heldout": rho_h,
            "null_rho_mean": null_mean,
            "null_rho_lo95": null_lo,
            "null_rho_hi95": null_hi,
            "rho_above_null_hi95": bool(
                pd.notna(rho_h) and pd.notna(null_hi) and rho_h > null_hi
            ),
            "rho_heldout_above_stable_bar": bool(
                pd.notna(rho_h) and float(rho_h) >= RHO_STABLE_THRESHOLD
            ),
            "rank_changes_vs_heldout": n_rank_changes(means, heldout),
            "rank_preserved_vs_heldout": ranks_equal(means, heldout),
            "alpha_cross_model": float("nan"),
            "alpha_drop_from_full": float("nan"),
            "alpha_drop_within_bar": False,
        })
    return pd.DataFrame(rows)


def _loo_table(
    ward_df: pd.DataFrame,
    full_means: pd.Series,
    judges: list[str],
    n_null_permutations: int = NULL_N,
    null_seed: int = NULL_SEED,
    full_alpha: float = float("nan"),
    same_family_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Leave-one-judge-out rank stability and reliability.

    ``rho_vs_heldout`` compares the remaining ``n-1`` judges to the single
    excluded judge (a disjoint reference). ``alpha_cross_model`` reports
    the inter-rater reliability *of* the reduced ``n-1``-judge panel.
    """
    rows = []
    for j in judges:
        sub = ward_df[ward_df["judge_model"] != j]
        if sub.empty or sub["judge_model"].nunique() == 0:
            rows.append({
                "excluded_judge": j,
                "n_judges_remaining": 0,
                "rho_vs_full": float("nan"),
                "rho_vs_heldout": float("nan"),
                "null_rho_mean": float("nan"),
                "null_rho_lo95": float("nan"),
                "null_rho_hi95": float("nan"),
                "rho_above_null_hi95": False,
                "rho_heldout_above_stable_bar": False,
                "rank_changes_vs_heldout": -1,
                "rank_preserved_vs_heldout": False,
                "alpha_cross_model": float("nan"),
                "alpha_drop_from_full": float("nan"),
                "alpha_drop_within_bar": False,
                "sf_subject": "",
                "sf_rank_in_full": float("nan"),
                "sf_rank_in_loo": float("nan"),
                "sf_rank_delta": float("nan"),
            })
            continue
        means = _panel_per_model_means(sub)
        # Complement of "drop j" is the single judge j.
        heldout = _panel_per_model_means(
            ward_df[ward_df["judge_model"] == j]
        )
        rho_h = spearman(means, heldout)
        null_mean, null_lo, null_hi = _permutation_null_heldout_rho(
            means, heldout,
            n_permutations=n_null_permutations, seed=null_seed,
        )
        alpha_reduced = _alpha_cross_model(sub)
        if pd.notna(alpha_reduced) and pd.notna(full_alpha):
            alpha_drop = float(full_alpha) - float(alpha_reduced)
            alpha_drop_within_bar = bool(alpha_drop <= ALPHA_DROP_THRESHOLD)
        else:
            alpha_drop = float("nan")
            alpha_drop_within_bar = False
        sf_subjects = (same_family_map or {}).get(j, [])
        sf_subject = sf_subjects[0] if sf_subjects else ""
        sf_rank_full = _rank_of(full_means, sf_subject)
        sf_rank_loo = _rank_of(means, sf_subject)
        sf_rank_delta = (
            float(sf_rank_loo - sf_rank_full)
            if pd.notna(sf_rank_full) and pd.notna(sf_rank_loo)
            else float("nan")
        )
        rows.append({
            "excluded_judge": j,
            "n_judges_remaining": int(sub["judge_model"].nunique()),
            "rho_vs_full": spearman(means, full_means),
            "rho_vs_heldout": rho_h,
            "null_rho_mean": null_mean,
            "null_rho_lo95": null_lo,
            "null_rho_hi95": null_hi,
            "rho_above_null_hi95": bool(
                pd.notna(rho_h) and pd.notna(null_hi) and rho_h > null_hi
            ),
            "rho_heldout_above_stable_bar": bool(
                pd.notna(rho_h) and float(rho_h) >= RHO_STABLE_THRESHOLD
            ),
            "rank_changes_vs_heldout": n_rank_changes(means, heldout),
            "rank_preserved_vs_heldout": ranks_equal(means, heldout),
            "alpha_cross_model": alpha_reduced,
            "alpha_drop_from_full": alpha_drop,
            "alpha_drop_within_bar": alpha_drop_within_bar,
            "sf_subject": sf_subject,
            "sf_rank_in_full": sf_rank_full,
            "sf_rank_in_loo": sf_rank_loo,
            "sf_rank_delta": sf_rank_delta,
        })
    return pd.DataFrame(rows)


def _size_sweep(
    ward_df: pd.DataFrame,
    full_means: pd.Series,
    judges: list[str],
    n_null_permutations: int = NULL_N,
    null_seed: int = NULL_SEED,
    full_alpha: float = float("nan"),
) -> pd.DataFrame:
    """Summarise rho (held-out reference) and alpha (within panel) per k.

    For ``k = n`` no disjoint reference exists, so the held-out columns
    are NaN. For ``k = 1`` alpha is undefined and ``mean_alpha_cross_model``
    is NaN.
    """
    n = len(judges)
    rows = []
    for k in range(1, n + 1):
        rhos_full: list[float] = []
        rhos_held: list[float] = []
        null_his: list[float] = []
        alphas: list[float] = []
        alpha_drops: list[float] = []
        n_preserving_held = 0
        n_above_null = 0
        n_rho_above_bar = 0
        n_alpha_drop_within_bar = 0
        subsets = list(combinations(judges, k))
        for subset in subsets:
            sub = ward_df[ward_df["judge_model"].isin(subset)]
            if sub.empty:
                continue
            means = _panel_per_model_means(sub)
            rho_full = spearman(means, full_means)
            if rho_full == rho_full:
                rhos_full.append(rho_full)
            if k < n:
                heldout = _heldout_reference_means(ward_df, subset, judges)
                rho_h = spearman(means, heldout)
                if rho_h == rho_h:
                    rhos_held.append(rho_h)
                    if float(rho_h) >= RHO_STABLE_THRESHOLD:
                        n_rho_above_bar += 1
                if ranks_equal(means, heldout):
                    n_preserving_held += 1
                _, _, null_hi = _permutation_null_heldout_rho(
                    means, heldout,
                    n_permutations=n_null_permutations, seed=null_seed,
                )
                if null_hi == null_hi:
                    null_his.append(null_hi)
                    if rho_h == rho_h and rho_h > null_hi:
                        n_above_null += 1
            if k >= 2:
                alpha = _alpha_cross_model(sub)
                if alpha == alpha:
                    alphas.append(alpha)
                    if pd.notna(full_alpha):
                        drop = float(full_alpha) - float(alpha)
                        alpha_drops.append(drop)
                        if drop <= ALPHA_DROP_THRESHOLD:
                            n_alpha_drop_within_bar += 1

        def _mean(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else float("nan")

        def _min(xs: list[float]) -> float:
            return float(min(xs)) if xs else float("nan")

        def _max(xs: list[float]) -> float:
            return float(max(xs)) if xs else float("nan")

        rows.append({
            "k": k,
            "n_subsets": len(subsets),
            "mean_rho_vs_full": _mean(rhos_full),
            "mean_rho_vs_heldout": _mean(rhos_held) if k < n else float("nan"),
            "min_rho_vs_heldout": _min(rhos_held) if k < n else float("nan"),
            "max_null_rho_hi95": _max(null_his) if k < n else float("nan"),
            "n_rank_preserving_heldout": n_preserving_held if k < n else -1,
            "n_rho_above_null_hi95": n_above_null if k < n else -1,
            "n_rho_heldout_above_stable_bar": (
                n_rho_above_bar if k < n else -1
            ),
            "mean_alpha_cross_model": _mean(alphas) if k >= 2 else float("nan"),
            "min_alpha_cross_model": _min(alphas) if k >= 2 else float("nan"),
            "n_alpha_above_tentative": (
                sum(a >= ALPHA_TENTATIVE for a in alphas) if k >= 2 else -1
            ),
            "mean_alpha_drop_from_full": (
                _mean(alpha_drops) if k >= 2 and alpha_drops else float("nan")
            ),
            "max_alpha_drop_from_full": (
                _max(alpha_drops) if k >= 2 and alpha_drops else float("nan")
            ),
            "n_alpha_drop_within_bar": (
                n_alpha_drop_within_bar if k >= 2 else -1
            ),
        })
    return pd.DataFrame(rows)


def _format_latex_table(
    single: pd.DataFrame,
    loo: pd.DataFrame,
    sweep: pd.DataFrame,
    sf_data: pd.DataFrame | None = None,
) -> str:
    """Two- or three-panel table: Panel A rank stability, Panel B reliability, Panel C same-family."""

    def _judge_label(m: str) -> str:
        parts = m.split("/", 1)
        return parts[-1].replace("_", " ")

    def _fmt(x: float, nd: int = 2) -> str:
        return f"{x:.{nd}f}" if pd.notna(x) else "--"

    def _rho_marker(row: pd.Series) -> str:
        marks = ""
        if row.get("rho_above_null_hi95", False):
            marks += "\\dag"
        if row.get("rho_heldout_above_stable_bar", False):
            marks += "\\ddag"
        return marks

    # Panel A: single-judge rows under the held-out reference.
    single_lines = []
    for _, row in single.iterrows():
        rho_h = row["rho_vs_heldout"]
        null_hi = row["null_rho_hi95"]
        marker = _rho_marker(row)
        rc = row["rank_changes_vs_heldout"]
        rc_str = f"{int(rc)}" if rc >= 0 else "--"
        single_lines.append(
            f"{_judge_label(row['judge_model'])} & "
            f"{int(row['n_generations'])} & {_fmt(rho_h)}{marker} & "
            f"{_fmt(null_hi)} & {rc_str} \\\\"
        )

    # Panel A: LOO rows.
    loo_lines = []
    for _, row in loo.iterrows():
        rho_h = row["rho_vs_heldout"]
        null_hi = row["null_rho_hi95"]
        marker = _rho_marker(row)
        rc = row["rank_changes_vs_heldout"]
        rc_str = f"{int(rc)}" if rc >= 0 else "--"
        loo_lines.append(
            f"drop {_judge_label(row['excluded_judge'])} & "
            f"{int(row['n_judges_remaining'])} & "
            f"{_fmt(rho_h)}{marker} & {_fmt(null_hi)} & {rc_str} \\\\"
        )

    # Panel A: size-sweep rows.
    sweep_rs_lines = []
    for _, row in sweep.iterrows():
        k = int(row["k"])
        mean_rho = row["mean_rho_vs_heldout"]
        min_rho = row["min_rho_vs_heldout"]
        max_null = row["max_null_rho_hi95"]
        n_above = row["n_rho_above_null_hi95"]
        n_sub = int(row["n_subsets"])
        if k == len(sweep):  # k = n: held-out undefined.
            sweep_rs_lines.append(
                f"{k} & {n_sub} & \\textsc{{nd}} & \\textsc{{nd}} & "
                f"\\textsc{{nd}} & \\textsc{{nd}} \\\\"
            )
        else:
            n_above_str = f"{int(n_above)}/{n_sub}" if n_above >= 0 else "--"
            sweep_rs_lines.append(
                f"{k} & {n_sub} & {_fmt(mean_rho)} & {_fmt(min_rho)} & "
                f"{_fmt(max_null)} & {n_above_str} \\\\"
            )

    # Panel B: reliability rows (LOO + size sweep for k >= 2).
    loo_rel_lines = []
    for _, row in loo.iterrows():
        alpha = row["alpha_cross_model"]
        drop = row.get("alpha_drop_from_full", float("nan"))
        drop_mark = "\\ddag" if row.get("alpha_drop_within_bar", False) else ""
        loo_rel_lines.append(
            f"drop {_judge_label(row['excluded_judge'])} & "
            f"{int(row['n_judges_remaining'])} & {_fmt(alpha)} & "
            f"{_fmt(drop)}{drop_mark} \\\\"
        )

    sweep_rel_lines = []
    for _, row in sweep.iterrows():
        k = int(row["k"])
        n_sub = int(row["n_subsets"])
        if k == 1:
            sweep_rel_lines.append(
                f"{k} & {n_sub} & \\textsc{{nd}} & \\textsc{{nd}} & "
                f"\\textsc{{nd}} & \\textsc{{nd}} \\\\"
            )
        else:
            mean_a = row["mean_alpha_cross_model"]
            min_a = row["min_alpha_cross_model"]
            n_above = row["n_alpha_above_tentative"]
            n_above_str = f"{int(n_above)}/{n_sub}" if n_above >= 0 else "--"
            n_drop = row.get("n_alpha_drop_within_bar", -1)
            n_drop_str = (
                f"{int(n_drop)}/{n_sub}"
                if pd.notna(n_drop) and int(n_drop) >= 0
                else "--"
            )
            sweep_rel_lines.append(
                f"{k} & {n_sub} & {_fmt(mean_a)} & {_fmt(min_a)} & "
                f"{n_above_str} & {n_drop_str} \\\\"
            )

    sf_lines: list[str] = []
    if sf_data is not None and not sf_data.empty:
        for _, row in sf_data.iterrows():
            full_r = row["sf_rank_in_full"]
            loo_r = row["sf_rank_in_loo"]
            delta = row["sf_rank_delta"]
            sign = "+" if pd.notna(delta) and delta > 0 else ""
            sf_lines.append(
                f"{_judge_label(row['excluded_judge'])} & "
                f"{_judge_label(row['sf_subject'])} & "
                f"{_fmt(full_r, 0)} & {_fmt(loo_r, 0)} & "
                f"{sign}{_fmt(delta, 0)} \\\\"
            )

    panel_c: list[str] = []
    if sf_lines:
        panel_c = [
            "",
            "\\vspace{0.75em}",
            "\\textbf{Panel C. Same-family rank sensitivity.}",
            "\\vspace{0.25em}",
            "",
            "\\begin{tabular}{llccc}",
            "\\toprule",
            "Excluded judge & Same-family subject & Rank (full) & "
            "Rank (LOO) & $\\Delta$ (LOO$-$full) \\\\",
            "\\midrule",
            *sf_lines,
            "\\bottomrule",
            "\\end{tabular}",
        ]

    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Judge panel composition. Panel A reports rank stability "
        "under a disjoint held-out reference: each subset $S$ of $k$ judges "
        "is compared against the aggregate of the complementary $n{-}k$ "
        "judges via Spearman $\\rho$. Comparing $S$ against the full-panel "
        "aggregate would be self-referential because $k/n$ of the reference "
        "is the subset itself. The null $95\\%$ upper bound is the $97.5$th "
        "percentile of $\\rho$ under random relabeling of models "
        "($B=10{,}000$); cells marked $\\dag$ exceed this bound. Cells "
        "marked $\\ddag$ clear the preregistered rank-stability bar "
        "$\\rho_{\\text{heldout}} \\geq 0.9$. \\textsc{nd} "
        "(not defined) marks $k=n$, where no disjoint reference exists. "
        "Panel B reports inter-rater reliability within the reduced panel "
        "via cross-model stratified Krippendorff $\\alpha$ on Ward total. "
        "$\\alpha$ requires at least two raters and is \\textsc{nd} for "
        "$k=1$; the tentative-conclusions bar is $\\alpha \\geq 0.667$. The "
        "$\\alpha$-drop column reports $\\alpha_{\\text{full}} - "
        "\\alpha_{\\text{reduced}}$, and $\\ddag$ marks sub-panels that "
        "stay within the preregistered drop bar of $0.1$. The two axes "
        "measure different quantities (model-level rank stability vs.\\ "
        "judge-level reliability) and are reported separately. "
        "Panel C reports same-family rank sensitivity: for each subject-judge "
        "pair that shares a provider family, the subject model's rank under "
        "the full three-judge panel is compared against its rank under the "
        "two-judge panel that excludes the same-family judge. A positive "
        "$\\Delta$ (LOO$-$full) indicates the subject ranked lower when its "
        "same-family judge was removed, consistent with upward inflation by "
        "self-preference.}",
        "\\label{tab:ablation_judges}",
        "\\vspace{0.25em}",
        "\\textbf{Panel A. Rank stability (disjoint held-out reference).}",
        "\\vspace{0.25em}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Subset & $n$ gens & $\\rho_{\\text{heldout}}$ & "
        "Null $95\\%$ hi & Rank changes \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\emph{Single judge ($k=1$)}} \\\\",
        *single_lines,
        "\\midrule",
        "\\multicolumn{5}{l}{\\emph{Leave-one-judge-out ($k=n{-}1$)}} \\\\",
        *loo_lines,
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "\\vspace{0.5em}",
        "",
        "\\begin{tabular}{cccccc}",
        "\\toprule",
        "$k$ & \\#subsets & Mean $\\rho_{\\text{heldout}}$ & "
        "Min $\\rho_{\\text{heldout}}$ & Max null $95\\%$ hi & "
        "\\#above null \\\\",
        "\\midrule",
        *sweep_rs_lines,
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "\\vspace{0.75em}",
        "\\textbf{Panel B. Inter-rater reliability (within reduced panel).}",
        "\\vspace{0.25em}",
        "",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Subset & $n$ judges & $\\alpha_{\\text{cross-model}}$ & "
        "$\\alpha_{\\text{full}} - \\alpha_{\\text{reduced}}$ \\\\",
        "\\midrule",
        "\\multicolumn{4}{l}{\\emph{Leave-one-judge-out}} \\\\",
        *loo_rel_lines,
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "\\vspace{0.5em}",
        "",
        "\\begin{tabular}{cccccc}",
        "\\toprule",
        "$k$ & \\#subsets & Mean $\\alpha$ & Min $\\alpha$ & "
        "\\#$\\alpha \\geq 0.667$ & \\#drop $\\leq 0.1$ \\\\",
        "\\midrule",
        *sweep_rel_lines,
        "\\bottomrule",
        "\\end{tabular}",
        *panel_c,
        "\\end{table}",
        "",
    ])


def run(config_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = load_config(config_path)
    data_dir = Path(cfg.data_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ward_df = load_ward_scores(
        data_dir, cfg.judge_models, cfg.models, cfg.prompt_variants
    )
    if ward_df.empty:
        raise RuntimeError(
            "No Ward scores found. Run `alienbench score` before this ablation."
        )

    judges = sorted(ward_df["judge_model"].unique())
    if len(judges) < 2:
        raise RuntimeError(
            "Judge Panel Composition ablation requires at least two judges in "
            f"the scored data (found {len(judges)}: {judges}). Add judges to "
            "the config and re-run the extract+score stages."
        )

    full_means = _panel_per_model_means(ward_df)
    full_alpha = _alpha_cross_model(ward_df)
    logger.info(
        "Full panel: %d judges, %d models, alpha=%.3f",
        len(judges), len(full_means), full_alpha,
    )

    sf_map = _same_family_map(cfg.judge_models, cfg.models)
    logger.info(
        "Same-family pairs detected: %s",
        {j: subs for j, subs in sf_map.items() if subs},
    )

    single = _single_judge_table(
        ward_df, full_means, judges, full_alpha=full_alpha,
    )
    loo = _loo_table(
        ward_df, full_means, judges, full_alpha=full_alpha,
        same_family_map=sf_map,
    )
    sweep = _size_sweep(
        ward_df, full_means, judges, full_alpha=full_alpha,
    )

    if "sf_subject" in loo.columns:
        sf_rows = loo[loo["sf_subject"].astype(str).str.len() > 0].copy()
    else:
        sf_rows = pd.DataFrame()

    single.to_csv(results_dir / "table_ablation_judges_single.csv", index=False)
    loo.to_csv(results_dir / "table_ablation_judges_loo.csv", index=False)
    sweep.to_csv(results_dir / "table_ablation_judges_sizesweep.csv", index=False)
    logger.info("Saved per-judge, LOO, and size-sweep CSVs")

    tex = _format_latex_table(
        single, loo, sweep,
        sf_data=sf_rows if not sf_rows.empty else None,
    )
    (results_dir / "tab_ablation_judges.tex").write_text(tex)
    logger.info("Saved tab_ablation_judges.tex")

    lines: list[str] = ["# Judge Panel Composition\n"]
    lines.append("## Full-panel per-model means\n")
    lines.append(full_means.sort_values(ascending=False).round(3).to_string())

    lines.append("\n## Rank stability (disjoint held-out reference)\n")
    lines.append(
        "Rho is computed between a subset of k judges and the aggregate of "
        "the complementary n-k judges (disjoint reference). Comparing a "
        "subset against the full-panel aggregate would be self-referential "
        "because k/n of the reference is the subset itself. The null 95%% "
        f"upper bound is the 97.5th percentile of rho under random model "
        f"relabeling (B={NULL_N})."
    )
    lines.append("\n### Single-judge vs held-out\n")
    lines.append(single.round(3).to_string(index=False))
    lines.append("\n### Leave-one-judge-out vs held-out\n")
    lines.append(loo.round(3).to_string(index=False))
    lines.append("\n### Size sweep vs held-out\n")
    lines.append(sweep.round(3).to_string(index=False))

    rho_h_min = single["rho_vs_heldout"].min()
    rho_h_max = single["rho_vs_heldout"].max()
    n_preserving_single = int(single["rank_preserved_vs_heldout"].sum())
    n_above_null_single = int(single["rho_above_null_hi95"].sum())
    lines.append(
        f"\nSingle-judge rho (vs held-out) range: "
        f"[{rho_h_min:.3f}, {rho_h_max:.3f}]; "
        f"{n_preserving_single}/{len(single)} single judges preserve the "
        f"held-out ranking exactly; "
        f"{n_above_null_single}/{len(single)} single-judge rhos exceed "
        f"their null 95%% upper bound."
    )
    rho_full_min = single["rho_vs_full"].min()
    rho_full_max = single["rho_vs_full"].max()
    lines.append(
        f"Inflation diagnostic: single-judge rho-vs-full range "
        f"[{rho_full_min:.3f}, {rho_full_max:.3f}]; rho-vs-full is "
        f"mechanically inflated relative to rho-vs-heldout because the "
        f"single judge is part of the full-panel reference."
    )

    lines.append("\n## Inter-rater reliability (within reduced panel)\n")
    lines.append(
        "Cross-model stratified Krippendorff alpha on Ward total, computed "
        "on the judges INSIDE each reduced panel. Alpha requires at least "
        "two raters and is therefore NOT DEFINED for k=1; single-judge "
        "panels cannot be assessed on this axis. This is a structural "
        "asymmetry of the reliability axis, not a missing measurement."
    )
    loo_alpha_min = loo["alpha_cross_model"].min()
    loo_alpha_max = loo["alpha_cross_model"].max()
    lines.append(
        f"\nLeave-one-judge-out alpha range: "
        f"[{loo_alpha_min:.3f}, {loo_alpha_max:.3f}]."
    )
    mask_k2plus = sweep["k"] >= 2
    if mask_k2plus.any():
        a_min = sweep.loc[mask_k2plus, "min_alpha_cross_model"].min()
        a_max = sweep.loc[mask_k2plus, "mean_alpha_cross_model"].max()
        lines.append(
            f"Size-sweep alpha (k>=2) spans min={a_min:.3f} to "
            f"mean-at-best-k={a_max:.3f}; tentative-conclusions bar is "
            f"alpha >= {ALPHA_TENTATIVE}."
        )

    # Preregistered bars: fixed before computing the statistics.
    lines.append(
        f"\n## Preregistered thresholds "
        f"(rho_heldout >= {RHO_STABLE_THRESHOLD:.2f}; "
        f"alpha drop from full panel <= {ALPHA_DROP_THRESHOLD:.2f})\n"
    )
    lines.append(f"Full-panel cross-model alpha: {full_alpha:.3f}.")
    n_rho_bar_single = int(single["rho_heldout_above_stable_bar"].sum())
    lines.append(
        f"Single judges clearing rho_heldout >= "
        f"{RHO_STABLE_THRESHOLD:.2f}: {n_rho_bar_single}/{len(single)}."
    )
    n_rho_bar_loo = int(loo["rho_heldout_above_stable_bar"].sum())
    lines.append(
        f"Leave-one-judge-out sub-panels clearing rho_heldout >= "
        f"{RHO_STABLE_THRESHOLD:.2f}: {n_rho_bar_loo}/{len(loo)}."
    )
    n_alpha_bar_loo = int(loo["alpha_drop_within_bar"].sum())
    lines.append(
        f"Leave-one-judge-out sub-panels with alpha drop <= "
        f"{ALPHA_DROP_THRESHOLD:.2f}: {n_alpha_bar_loo}/{len(loo)}."
    )
    for _, row in sweep.iterrows():
        k = int(row["k"])
        n_sub = int(row["n_subsets"])
        parts = [f"k={k} ({n_sub} subsets):"]
        n_bar = row.get("n_rho_heldout_above_stable_bar", -1)
        if pd.notna(n_bar) and int(n_bar) >= 0:
            parts.append(f"rho_heldout bar {int(n_bar)}/{n_sub}")
        else:
            parts.append("rho_heldout bar nd")
        n_ad = row.get("n_alpha_drop_within_bar", -1)
        if pd.notna(n_ad) and int(n_ad) >= 0:
            parts.append(f"alpha-drop bar {int(n_ad)}/{n_sub}")
        else:
            parts.append("alpha-drop bar nd")
        lines.append("  " + "; ".join(parts) + ".")

    lines.append("\n## Same-family rank sensitivity\n")
    _fmt_rank = lambda x: str(int(x)) if pd.notna(x) else "N/A"  # noqa: E731
    if not sf_rows.empty:
        for _, row in sf_rows.iterrows():
            lines.append(
                f"  {row['excluded_judge']} -> {row['sf_subject']}: "
                f"full rank {_fmt_rank(row['sf_rank_in_full'])}, "
                f"LOO rank {_fmt_rank(row['sf_rank_in_loo'])}, "
                f"delta {_fmt_rank(row['sf_rank_delta'])} "
                f"(positive = subject dropped when SF judge excluded)."
            )
    else:
        lines.append("  No same-family pairs detected.")

    (results_dir / "summary_ablation_judges.txt").write_text(
        "\n".join(lines) + "\n"
    )
    logger.info("Saved summary_ablation_judges.txt")

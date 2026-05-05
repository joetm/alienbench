"""Ablation: Dimension Sensitivity (paper sec:ablation_dimensions).

Quantifies how sensitive model rank orderings and absolute Ward scores
are to subsetting the ten Ward dimensions. The ablation does not claim
to measure dominance by any single dimension; it reports stability of
rank and absolute score under reduction of the dimension set.

Reports:

1. Leave-one-out analysis. For each dimension d, compute the
   nine-dimension reduced per-model mean, and report (a) Spearman
   rank correlation ρ between the reduced and full per-model means,
   (b) an empirical null for ρ built by permuting model labels on the
   reduced score (because the full score mechanically equals the
   reduced score plus the dropped dimension, raw ρ is inflated and
   the meaningful comparison is against the null), and (c) pairwise
   mean absolute difference (MAD) between the reduced and full means
   to capture the absolute-score loss that ρ is invariant to.

2. Continuous subset search. For each subset size k, report the best
   Spearman ρ, best Kendall τ, and minimum MAD across all C(10, k)
   subsets. The ablation uses preregistered continuous thresholds
   (ρ ≥ 0.9 and τ ≥ 0.8) rather than exact rank preservation, which
   is brittle to single-tie swaps. The reported min-k is accompanied
   by an empirical null distribution of min-k under the hypothesis
   that reduced rankings are independent of the full ranking,
   correcting for the multiple-comparisons inflation introduced by
   searching over C(10, k) subsets at each size.

The ablation is purely a re-aggregation of existing Ward score records
(``dim_<id>`` columns); no regeneration is required.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

from alienbench.ablation_utils import RANK_ATOL, ranks_equal, spearman
from alienbench.config import load_config
from alienbench.dimensions import DIMENSION_IDS
from alienbench.paths import load_ward_scores

logger = logging.getLogger(__name__)

# Local aliases preserve the previous module-level names for tests.
_RANK_ATOL = RANK_ATOL
_ranks_equal = ranks_equal
_spearman = spearman

# Fixed seeds so the permutation nulls are bit-stable across re-runs.
LOO_NULL_SEED = 0
LOO_NULL_N = 10_000
SUBSET_NULL_SEED = 1
SUBSET_NULL_N = 1_000

# Preregistered continuous thresholds for the subset search. ρ ≥ 0.9
# corresponds to a strong rank correlation; τ ≥ 0.8 corresponds to at
# most one non-adjacent pair swap on 10 models. We fix these before
# computing the statistics to avoid post-hoc threshold selection.
RHO_THRESHOLD = 0.9
TAU_THRESHOLD = 0.8


def _per_model_mean_from_subset(
    ward_df: pd.DataFrame, subset: tuple[str, ...]
) -> pd.Series:
    """Sum ``dim_<d>`` across ``subset``, average judges, then models.

    Mirrors the main pipeline's score/aggregation convention: judges are
    averaged per generation before generations are averaged per model.
    """
    dim_cols = [f"dim_{d}" for d in subset]
    reduced = ward_df[dim_cols].sum(axis=1)
    working = ward_df.assign(_subset_score=reduced)
    per_gen = (
        working.groupby(["generation_id", "subject_model"])["_subset_score"]
        .mean()
        .reset_index()
    )
    return per_gen.groupby("subject_model")["_subset_score"].mean()


def _mad_series(a: pd.Series, b: pd.Series) -> float:
    """Mean absolute difference between ``a`` and ``b`` on their shared index.

    Returns ``NaN`` when the two series share no models.
    """
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return float("nan")
    return float((a.loc[common] - b.loc[common]).abs().mean())


def _permutation_null_rho(
    reduced: pd.Series,
    full_means: pd.Series,
    n_permutations: int = LOO_NULL_N,
    seed: int = LOO_NULL_SEED,
) -> tuple[float, float, float]:
    """Null distribution of Spearman ρ under random relabeling of models.

    For each of ``n_permutations`` draws, the reduced-score vector is
    permuted across models and Spearman ρ is recomputed against
    ``full_means`` on the shared index. Returns
    ``(null_mean, null_lo95, null_hi95)``; all three are ``NaN`` when
    fewer than three models are shared. The permutation decouples the
    reduced and full rankings while preserving the marginal distribution
    of reduced scores, so the null is the correct reference for ρ when
    the reduced score is a strict subset of the full.
    """
    common = reduced.index.intersection(full_means.index)
    if len(common) < 3:
        return (float("nan"), float("nan"), float("nan"))
    x = reduced.loc[common].to_numpy()
    y = full_means.loc[common].to_numpy()
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        x_perm = rng.permutation(x)
        draws[i] = float(stats.spearmanr(x_perm, y).statistic)
    return (
        float(draws.mean()),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def _leave_one_out_table(
    ward_df: pd.DataFrame,
    full_means: pd.Series,
    n_null_permutations: int = LOO_NULL_N,
    null_seed: int = LOO_NULL_SEED,
) -> pd.DataFrame:
    """One row per dropped dimension with departure rate, ρ, null, and MAD.

    ``spearman_rho`` is biased upward because the reduced score is a
    strict subset of the full (they share nine of ten dimensions), so
    raw ρ close to 1 is not by itself evidence of rank stability. The
    meaningful reference is ``null_rho_hi95``: observed ρ that exceeds
    the null 95% upper bound indicates rank preservation above chance
    on the reduced scale. ``mad_vs_full`` captures the absolute-score
    loss on the 0--10 Ward scale, which ρ is invariant to.
    """
    dim_cols = [f"dim_{d}" for d in DIMENSION_IDS]
    per_gen = (
        ward_df.groupby(["generation_id", "subject_model"])[dim_cols]
        .mean()
        .reset_index()
    )
    dep_rates = per_gen[dim_cols].mean()
    dep_rates.index = [c.replace("dim_", "") for c in dep_rates.index]

    rows = []
    for d in DIMENSION_IDS:
        subset = tuple(x for x in DIMENSION_IDS if x != d)
        reduced = _per_model_mean_from_subset(ward_df, subset)
        rho = _spearman(reduced, full_means)
        mad = _mad_series(reduced, full_means)
        null_mean, null_lo, null_hi = _permutation_null_rho(
            reduced, full_means,
            n_permutations=n_null_permutations, seed=null_seed,
        )
        common = reduced.index.intersection(full_means.index)
        if len(common) >= 2:
            r_sub = stats.rankdata(reduced.loc[common].values, method="average")
            r_full = stats.rankdata(full_means.loc[common].values, method="average")
            rank_changes = int(np.sum(r_sub != r_full))
            preserved = bool(np.allclose(r_sub, r_full, atol=_RANK_ATOL))
        else:
            rank_changes = -1
            preserved = False
        rho_above_null = (
            bool(pd.notna(rho) and pd.notna(null_hi) and float(rho) > float(null_hi))
        )
        rows.append({
            "dropped_dimension": d,
            "departure_rate": float(dep_rates.get(d, float("nan"))),
            "spearman_rho": rho,
            "null_rho_mean": null_mean,
            "null_rho_lo95": null_lo,
            "null_rho_hi95": null_hi,
            "rho_above_null_hi95": rho_above_null,
            "mad_vs_full": mad,
            "rank_changes": rank_changes,
            "rank_preserved": preserved,
        })
    return pd.DataFrame(rows)


def _kendall_tau(a: pd.Series, b: pd.Series) -> float:
    """Kendall τ between ``a`` and ``b`` on their shared index.

    Returns ``NaN`` when fewer than three models overlap or when either
    vector has zero variance on the shared index.
    """
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return float("nan")
    x = a.loc[common].to_numpy()
    y = b.loc[common].to_numpy()
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.kendalltau(x, y).statistic)


def _subset_size_summary(
    ward_df: pd.DataFrame, full_means: pd.Series
) -> tuple[pd.DataFrame, dict[int, tuple[str, ...]], dict[int, tuple[str, ...]]]:
    """For each subset size k, the best ρ, best τ, and minimum MAD.

    Returns ``(size_summary, best_rho_subset, best_tau_subset)``. Each
    entry in ``best_rho_subset`` / ``best_tau_subset`` is the subset
    that achieved the best ρ / τ at that size; ties are broken by
    minimum MAD, then by lexicographic order.
    """
    summary_rows: list[dict[str, object]] = []
    best_rho_subset: dict[int, tuple[str, ...]] = {}
    best_tau_subset: dict[int, tuple[str, ...]] = {}
    for k in range(1, len(DIMENSION_IDS) + 1):
        best_rho = float("-inf")
        best_tau = float("-inf")
        min_mad = float("inf")
        n_rho_above = 0
        n_tau_above = 0
        best_rho_sub: tuple[str, ...] | None = None
        best_tau_sub: tuple[str, ...] | None = None
        for subset in combinations(DIMENSION_IDS, k):
            reduced = _per_model_mean_from_subset(ward_df, subset)
            rho = _spearman(reduced, full_means)
            tau = _kendall_tau(reduced, full_means)
            mad = _mad_series(reduced, full_means)
            if pd.notna(rho):
                if rho > best_rho or (
                    rho == best_rho and pd.notna(mad) and mad < min_mad
                ):
                    best_rho = rho
                    best_rho_sub = subset
                    if pd.notna(mad):
                        min_mad = min(min_mad, mad)
                if rho >= RHO_THRESHOLD:
                    n_rho_above += 1
            if pd.notna(tau):
                if tau > best_tau:
                    best_tau = tau
                    best_tau_sub = subset
                if tau >= TAU_THRESHOLD:
                    n_tau_above += 1
            if pd.notna(mad) and mad < min_mad:
                min_mad = mad
        summary_rows.append({
            "k": k,
            "n_subsets": len(list(combinations(DIMENSION_IDS, k))),
            "best_rho": best_rho if best_rho != float("-inf") else float("nan"),
            "best_tau": best_tau if best_tau != float("-inf") else float("nan"),
            "min_mad": min_mad if min_mad != float("inf") else float("nan"),
            "n_rho_above_thresh": n_rho_above,
            "n_tau_above_thresh": n_tau_above,
        })
        if best_rho_sub is not None:
            best_rho_subset[k] = best_rho_sub
        if best_tau_sub is not None:
            best_tau_subset[k] = best_tau_sub
    return pd.DataFrame(summary_rows), best_rho_subset, best_tau_subset


def _min_k_from_summary(
    size_summary: pd.DataFrame, column: str, threshold: float
) -> int:
    """Smallest k whose ``column`` is ≥ ``threshold``; -1 if none."""
    for _, row in size_summary.iterrows():
        val = row[column]
        if pd.notna(val) and float(val) >= threshold:
            return int(row["k"])
    return -1


def _permutation_null_min_k(
    full_means: pd.Series,
    rho_threshold: float = RHO_THRESHOLD,
    tau_threshold: float = TAU_THRESHOLD,
    n_permutations: int = SUBSET_NULL_N,
    seed: int = SUBSET_NULL_SEED,
) -> pd.DataFrame:
    """Null distribution of min-k under independent reduced rankings.

    At subset size k there are C(10, k) subsets; under the hypothesis
    that each reduced ranking is independent of the full ranking, the
    best-ρ across C(10, k) draws has an inflated distribution that
    accounts for multiple comparisons. Each permutation draw samples
    ``C(10, k)`` random rerankings of ``full_means``, computes the max
    Spearman ρ and Kendall τ at each k, and records the smallest k
    whose best-ρ (respectively best-τ) crosses the preregistered
    threshold. The returned DataFrame has columns ``min_k_rho_null``
    and ``min_k_tau_null`` with ``n_permutations`` rows.
    """
    values = full_means.values
    n = len(values)
    if n < 3:
        return pd.DataFrame(
            {"min_k_rho_null": [], "min_k_tau_null": []}
        )
    rng = np.random.default_rng(seed)
    ks = list(range(1, len(DIMENSION_IDS) + 1))
    n_subsets_per_k = {k: len(list(combinations(DIMENSION_IDS, k))) for k in ks}
    records = []
    for _ in range(n_permutations):
        min_k_rho = -1
        min_k_tau = -1
        for k in ks:
            best_rho = float("-inf")
            best_tau = float("-inf")
            for _ in range(n_subsets_per_k[k]):
                perm = rng.permutation(values)
                if np.ptp(perm) == 0:
                    continue
                rho = float(stats.spearmanr(perm, values).statistic)
                tau = float(stats.kendalltau(perm, values).statistic)
                if rho > best_rho:
                    best_rho = rho
                if tau > best_tau:
                    best_tau = tau
            if min_k_rho < 0 and best_rho >= rho_threshold:
                min_k_rho = k
            if min_k_tau < 0 and best_tau >= tau_threshold:
                min_k_tau = k
            if min_k_rho > 0 and min_k_tau > 0:
                break
        if min_k_rho < 0:
            min_k_rho = len(DIMENSION_IDS) + 1
        if min_k_tau < 0:
            min_k_tau = len(DIMENSION_IDS) + 1
        records.append({"min_k_rho_null": min_k_rho, "min_k_tau_null": min_k_tau})
    return pd.DataFrame(records)


def _minimum_subset_search(
    ward_df: pd.DataFrame, full_means: pd.Series
) -> tuple[pd.DataFrame, list[tuple[str, ...]], int]:
    """Backwards-compatible shim retained for tests.

    Runs the new continuous-criterion search, then exposes
    ``(size_summary, rank_preserving_subsets_at_min_k, min_k)`` with the
    same contract as the previous exact-match search. The exact-match
    column is still computed per-subset for tests and downstream
    reporting, but the paper and the new summary table rely on the
    continuous criterion produced by :func:`_subset_size_summary`.
    """
    rank_preserving: dict[int, list[tuple[str, ...]]] = {}
    rows: list[dict[str, object]] = []
    for k in range(1, len(DIMENSION_IDS) + 1):
        subsets = list(combinations(DIMENSION_IDS, k))
        preserving: list[tuple[str, ...]] = []
        rhos: list[float] = []
        for subset in subsets:
            reduced = _per_model_mean_from_subset(ward_df, subset)
            rho = _spearman(reduced, full_means)
            if pd.notna(rho):
                rhos.append(float(rho))
            if _ranks_equal(reduced, full_means):
                preserving.append(subset)
        best_rho = max(rhos) if rhos else float("nan")
        rows.append({
            "k": k,
            "n_subsets": len(subsets),
            "n_rank_preserving": len(preserving),
            "best_rho": best_rho,
        })
        rank_preserving[k] = preserving
    min_k = -1
    for k, preserving in rank_preserving.items():
        if preserving:
            min_k = k
            break
    subsets_at_min = rank_preserving[min_k] if min_k > 0 else []
    return pd.DataFrame(rows), subsets_at_min, min_k


def _format_latex_table(
    loo: pd.DataFrame,
    size_summary: pd.DataFrame,
    min_k_rho: int,
    min_k_tau: int,
    min_k_rho_null_p5: float,
    min_k_tau_null_p5: float,
    best_rho_example: tuple[str, ...] | None,
    best_tau_example: tuple[str, ...] | None,
) -> str:
    """Two-panel LaTeX table: LOO + null-adjusted subset search summary.

    The LOO panel reports raw ρ alongside the permutation null 95% upper
    bound; cells whose ρ exceeds the null bound are marked ``$^\\dagger$``
    and MAD is shown to quantify absolute-score loss. The subset-search
    panel uses continuous criteria (Spearman ρ, Kendall τ) rather than
    exact rank preservation, and the caption reports the observed
    min-k alongside the null 5th percentile of min-k under independent
    reduced rankings.
    """
    loo_sorted = loo.sort_values("departure_rate", ascending=False)
    loo_lines = []
    for _, row in loo_sorted.iterrows():
        rho = row["spearman_rho"]
        hi = row["null_rho_hi95"]
        rho_str = "--"
        if pd.notna(rho):
            mark = (
                "$^\\dagger$"
                if (pd.notna(hi) and float(rho) > float(hi))
                else ""
            )
            rho_str = f"{float(rho):.2f}{mark}"
        hi_str = f"{float(hi):.2f}" if pd.notna(hi) else "--"
        mad = row["mad_vs_full"]
        mad_str = f"{float(mad):.2f}" if pd.notna(mad) else "--"
        loo_lines.append(
            f"{row['dropped_dimension'].replace('_', ' ')} & "
            f"{row['departure_rate']:.2f} & {rho_str} & {hi_str} & {mad_str} \\\\"
        )

    size_lines = []
    for _, row in size_summary.iterrows():
        best_rho = row["best_rho"]
        best_tau = row["best_tau"]
        min_mad = row["min_mad"]
        rho_s = f"{float(best_rho):.2f}" if pd.notna(best_rho) else "--"
        tau_s = f"{float(best_tau):.2f}" if pd.notna(best_tau) else "--"
        mad_s = f"{float(min_mad):.2f}" if pd.notna(min_mad) else "--"
        size_lines.append(
            f"{int(row['k'])} & {int(row['n_subsets'])} & "
            f"{rho_s} & {tau_s} & {mad_s} \\\\"
        )

    def _fmt_min_k(obs: int, null_p5: float, label_subset: tuple[str, ...] | None,
                   threshold_name: str) -> str:
        if obs < 0:
            return (
                f"No subset at any $k < 10$ reaches the "
                f"{threshold_name} threshold."
            )
        null_str = (
            f" null 5th percentile $k={null_p5:.1f}$"
            if pd.notna(null_p5) else ""
        )
        if label_subset is not None:
            ex = ", ".join(d.replace("_", " ") for d in label_subset)
            return (
                f"{threshold_name}: observed min $k={obs}$"
                f"{null_str} (example subset: \\{{{ex}\\}})."
            )
        return f"{threshold_name}: observed min $k={obs}${null_str}."

    min_rho_line = _fmt_min_k(
        min_k_rho, min_k_rho_null_p5, best_rho_example,
        f"$\\rho \\geq {RHO_THRESHOLD}$",
    )
    min_tau_line = _fmt_min_k(
        min_k_tau, min_k_tau_null_p5, best_tau_example,
        f"$\\tau \\geq {TAU_THRESHOLD}$",
    )

    caption = (
        "Dimension sensitivity. Top: leave-one-out per-dimension effect "
        "on rank and score. ``Departure rate'' is the per-dimension "
        "share of generations coded as departing from the Earth-typical "
        "default; $\\rho$ is the Spearman rank correlation between the "
        "nine-dimension per-model mean and the full ten-dimension "
        "per-model mean. Because the reduced score is a strict subset "
        "of the full score, raw $\\rho$ is inflated; ``$\\rho$ null "
        "$95\\%$ hi'' is the upper bound of a permutation null on the "
        "reduced score, and cells marked $^\\dagger$ exceed it. MAD is "
        "the mean absolute difference between reduced and full per-model "
        "means (Ward points), capturing absolute-score loss that $\\rho$ "
        "is invariant to. Bottom: continuous subset-search summary over "
        "all $\\binom{10}{k}$ subsets. ``Best $\\rho$'' / ``Best "
        "$\\tau$'' are the maxima across size-$k$ subsets; ``Min MAD'' "
        "is the minimum absolute-score loss at size $k$. The continuous "
        "thresholds $\\rho \\geq " f"{RHO_THRESHOLD}" "$ and $\\tau "
        "\\geq " f"{TAU_THRESHOLD}" "$ are preregistered."
    )

    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:ablation_dimensions}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dropped dim & Departure rate & $\\rho$ vs full & $\\rho$ null $95\\%$ hi & MAD \\\\",
        "\\midrule",
        *loo_lines,
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "\\vspace{0.5em}",
        "",
        "\\begin{tabular}{ccccc}",
        "\\toprule",
        "$k$ & \\#subsets & Best $\\rho$ & Best $\\tau$ & Min MAD \\\\",
        "\\midrule",
        *size_lines,
        "\\bottomrule",
        "\\end{tabular}",
        "",
        f"\\vspace{{0.3em}} \\small {min_rho_line} {min_tau_line}",
        "\\end{table}",
        "",
    ])


def run(config_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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

    missing = [d for d in DIMENSION_IDS if f"dim_{d}" not in ward_df.columns]
    if missing:
        raise RuntimeError(
            f"Ward score records are missing dimension columns: {missing}. "
            "Re-run the score stage with the current dimension taxonomy."
        )

    full_means = _per_model_mean_from_subset(ward_df, tuple(DIMENSION_IDS))
    full_means = full_means.sort_values(ascending=False)
    logger.info("Computed full-score per-model means for %d models", len(full_means))

    loo = _leave_one_out_table(ward_df, full_means)
    loo.to_csv(results_dir / "table_ablation_dimensions_loo.csv", index=False)
    logger.info("Saved leave-one-out table with permutation null and MAD")

    size_summary, best_rho_by_k, best_tau_by_k = _subset_size_summary(
        ward_df, full_means,
    )
    size_summary.to_csv(
        results_dir / "table_ablation_dimensions_minsubset.csv", index=False,
    )
    logger.info("Saved subset-size summary (continuous criterion)")

    min_k_rho = _min_k_from_summary(size_summary, "best_rho", RHO_THRESHOLD)
    min_k_tau = _min_k_from_summary(size_summary, "best_tau", TAU_THRESHOLD)

    logger.info(
        "Computing min-k permutation null (B=%d, seed=%d)...",
        SUBSET_NULL_N, SUBSET_NULL_SEED,
    )
    null_min_k = _permutation_null_min_k(full_means)
    null_min_k.to_csv(
        results_dir / "table_ablation_dimensions_null_min_k.csv", index=False,
    )

    if not null_min_k.empty:
        min_k_rho_null_p5 = float(np.percentile(null_min_k["min_k_rho_null"], 5))
        min_k_rho_null_p50 = float(np.percentile(null_min_k["min_k_rho_null"], 50))
        min_k_tau_null_p5 = float(np.percentile(null_min_k["min_k_tau_null"], 5))
        min_k_tau_null_p50 = float(np.percentile(null_min_k["min_k_tau_null"], 50))
        if min_k_rho > 0:
            p_rho = float(
                np.mean(null_min_k["min_k_rho_null"] <= min_k_rho)
            )
        else:
            p_rho = float("nan")
        if min_k_tau > 0:
            p_tau = float(
                np.mean(null_min_k["min_k_tau_null"] <= min_k_tau)
            )
        else:
            p_tau = float("nan")
    else:
        min_k_rho_null_p5 = float("nan")
        min_k_rho_null_p50 = float("nan")
        min_k_tau_null_p5 = float("nan")
        min_k_tau_null_p50 = float("nan")
        p_rho = float("nan")
        p_tau = float("nan")

    # Backwards-compatible exact-match artefact; kept for downstream scripts
    # and tests that expect the old file layout. The paper no longer reports
    # exact rank preservation.
    _, rank_preserving, exact_min_k = _minimum_subset_search(ward_df, full_means)
    if rank_preserving:
        pd.DataFrame({
            "subset": [",".join(s) for s in rank_preserving],
            "k": [exact_min_k] * len(rank_preserving),
        }).to_csv(
            results_dir / "table_ablation_dimensions_rank_preserving.csv",
            index=False,
        )
        logger.info(
            "Saved %d exact-match rank-preserving subsets at k=%d "
            "(legacy artefact; the paper uses continuous criteria)",
            len(rank_preserving), exact_min_k,
        )

    best_rho_example = best_rho_by_k.get(min_k_rho) if min_k_rho > 0 else None
    best_tau_example = best_tau_by_k.get(min_k_tau) if min_k_tau > 0 else None
    tex = _format_latex_table(
        loo, size_summary, min_k_rho, min_k_tau,
        min_k_rho_null_p5, min_k_tau_null_p5,
        best_rho_example, best_tau_example,
    )
    (results_dir / "tab_ablation_dimensions.tex").write_text(tex)
    logger.info("Saved tab_ablation_dimensions.tex")

    lines = ["# Dimension Sensitivity\n"]
    lines.append("## Full-score per-model means\n")
    lines.append(full_means.round(3).to_string())
    lines.append("\n## Leave-one-out (sorted by departure rate)\n")
    lines.append(
        loo.sort_values("departure_rate", ascending=False)
        .round(3).to_string(index=False)
    )
    lines.append(
        "\nLOO note: raw ρ is inflated because the reduced score is a "
        "strict subset of the full score. The meaningful comparison is "
        "whether ρ exceeds the permutation null 95% upper bound "
        "(column ``null_rho_hi95``); MAD reports the absolute-score "
        "loss on the 0--10 Ward scale, which ρ is invariant to."
    )
    lines.append(
        f"\n## Subset-size summary (preregistered thresholds: "
        f"ρ ≥ {RHO_THRESHOLD}, τ ≥ {TAU_THRESHOLD})\n"
    )
    lines.append(size_summary.round(3).to_string(index=False))
    lines.append(
        f"\nObserved min-k (best ρ ≥ {RHO_THRESHOLD}): "
        f"{min_k_rho if min_k_rho > 0 else 'not reached'}"
    )
    lines.append(
        f"Observed min-k (best τ ≥ {TAU_THRESHOLD}): "
        f"{min_k_tau if min_k_tau > 0 else 'not reached'}"
    )
    lines.append(
        f"\n## Permutation null on min-k (B={SUBSET_NULL_N}, "
        f"seed={SUBSET_NULL_SEED})\n"
    )
    lines.append(
        "Under H0 (reduced rankings independent of full), min-k is "
        "inflated by the multiple-comparisons burden of C(10, k) "
        "subsets per size."
    )
    lines.append(
        f"Null min-k (ρ): 5th percentile = {min_k_rho_null_p5:.1f}, "
        f"median = {min_k_rho_null_p50:.1f}; p(null min-k ≤ observed) "
        f"= {p_rho:.3f}"
    )
    lines.append(
        f"Null min-k (τ): 5th percentile = {min_k_tau_null_p5:.1f}, "
        f"median = {min_k_tau_null_p50:.1f}; p(null min-k ≤ observed) "
        f"= {p_tau:.3f}"
    )
    (results_dir / "summary_ablation_dimensions.txt").write_text(
        "\n".join(lines) + "\n"
    )
    logger.info("Saved summary_ablation_dimensions.txt")

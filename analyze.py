"""Stage 4: Statistical analysis and figure generation."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from alienbench.config import load_config
from alienbench.dimensions import DIMENSION_IDS
from alienbench.paths import (
    load_extraction_status,
    load_generation_tokens,
    load_ward_scores,
)
from alienbench.stats import (
    bootstrap_ci,
    krippendorff_alpha_stratified,
    kruskal_posthoc,
    retrospective_power,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def _save_figure(fig, path: Path) -> None:
    """Save ``fig`` as PDF (vector for the paper) and PNG (preview/screen)."""
    fig.savefig(path, bbox_inches="tight")
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    logger.info("Saved %s", path)
    logger.info("Saved %s", png_path)

# ---------------------------------------------------------------------------
# Figure 1: Heatmap of Ward feature departures per model
# ---------------------------------------------------------------------------

def fig_ward_heatmap(ward_df: pd.DataFrame, results_dir: Path) -> None:
    dim_cols = [f"dim_{d}" for d in DIMENSION_IDS]
    # Average across judges, then per model
    mean_df = (
        ward_df.groupby(["subject_model", "generation_id"])[dim_cols]
        .mean()  # average judges per generation
        .reset_index()
        .groupby("subject_model")[dim_cols]
        .mean()
    )
    mean_df.columns = [c.replace("dim_", "") for c in mean_df.columns]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(mean_df))))
    sns.heatmap(
        mean_df,
        ax=ax,
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Departure rate"},
    )
    ax.set_title("Ward Feature Departure Rates by Model")
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Model")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    _save_figure(fig, results_dir / "fig1_ward_heatmap.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Violin plots — Ward scores by model × prompt variant
# ---------------------------------------------------------------------------

def fig_violin_scores(ward_df: pd.DataFrame, results_dir: Path) -> None:
    w = ward_df.groupby(["generation_id", "subject_model", "prompt_variant"])["ward_score"].mean().reset_index()

    n_models = w["subject_model"].nunique()
    n_variants = w["prompt_variant"].nunique()
    height = max(4.0, 0.6 * n_models * n_variants)

    fig, ax = plt.subplots(figsize=(8, height))
    sns.violinplot(
        data=w,
        x="ward_score", y="subject_model", hue="prompt_variant",
        ax=ax, cut=0, inner="quartile", linewidth=0.8,
        orient="h",
    )
    ax.set_title("Ward Departure Score by Model and Prompt Variant")
    ax.set_xlabel("Ward Departure Score (0–10)")
    ax.set_ylabel("")
    ax.set_xlim(0, 10)
    ax.legend(title="Prompt variant", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    _save_figure(fig, results_dir / "fig2_violin_scores.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Overview — per-model point cloud + mean (95% bootstrap CI)
# ---------------------------------------------------------------------------

def fig_overview_scores(ward_df: pd.DataFrame, results_dir: Path) -> None:
    """Single-panel overview: one point per generation, per-model mean + CI.

    Ward scores are averaged across judges per generation, then rendered as
    a horizontal strip plot with models on the y-axis ordered by descending
    per-model mean. A black diamond at the per-model mean, with horizontal
    whiskers spanning the 95% bootstrap CI, overlays the strip.
    """
    w = (
        ward_df.groupby(["generation_id", "subject_model", "prompt_variant"])["ward_score"]
        .mean()
        .reset_index()
    )
    if w.empty:
        logger.warning("fig_overview_scores: no Ward records to plot")
        return

    model_order = (
        w.groupby("subject_model")["ward_score"].mean().sort_values(ascending=False).index.tolist()
    )

    variants = sorted(w["prompt_variant"].unique().tolist())
    palette = dict(zip(variants, sns.color_palette("tab10", n_colors=len(variants))))

    n_models = len(model_order)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * n_models)))

    sns.stripplot(
        data=w,
        x="ward_score", y="subject_model", hue="prompt_variant",
        order=model_order, hue_order=variants, palette=palette,
        jitter=0.25, alpha=0.6, size=3.5, dodge=False,
        ax=ax,
    )

    means = []
    los = []
    his = []
    for m in model_order:
        vals = w.loc[w["subject_model"] == m, "ward_score"].to_numpy()
        mean, lo, hi = bootstrap_ci(vals, stat_fn=np.mean, n_boot=10_000, level=0.95)
        means.append(mean)
        los.append(mean - lo)
        his.append(hi - mean)
    y_positions = np.arange(n_models)
    ax.errorbar(
        means, y_positions,
        xerr=[los, his],
        fmt="D", color="black", ecolor="black",
        markersize=6, elinewidth=1.2, capsize=3,
        label="Mean (95% CI)",
        zorder=10,
    )

    ax.set_xlim(0, 10)
    ax.set_xlabel("Ward Departure Score (0–10)")
    ax.set_ylabel("")
    ax.set_title(
        "Ward Departure Score: per-model distribution (points) and mean (95% bootstrap CI)"
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        title="Prompt variant",
        bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
        frameon=False,
    )

    plt.tight_layout()
    _save_figure(fig, results_dir / "fig4_overview.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Inter-rater reliability table
# ---------------------------------------------------------------------------

def fig_reliability_table(ward_df: pd.DataFrame, results_dir: Path) -> None:
    rows = []

    if ward_df["judge_model"].nunique() < 2:
        logger.warning("Not enough judges for inter-rater reliability (need >= 2)")
        return

    def _fmt(x: float) -> str:
        return f"{x:.3f}" if not np.isnan(x) else ""

    # Ward total (interval-level): stratified by judge-subject overlap
    strat = krippendorff_alpha_stratified(
        ward_df, "subject_model", "judge_model", "generation_id", "ward_score",
        level="interval",
    )
    rows.append({
        "Measure": "Ward Total (0–10)",
        "α (all)": _fmt(strat["all"]),
        "α (cross-model)": _fmt(strat["cross_model"]),
        "α (same-model)": _fmt(strat["same_model"]),
        "Δ self−cross": _fmt(strat["mean_delta_self"]),
    })

    # Per Ward dimension (nominal-level, binary): stratified for consistency
    for dim in DIMENSION_IDS:
        col = f"dim_{dim}"
        if col not in ward_df.columns:
            continue
        strat_d = krippendorff_alpha_stratified(
            ward_df, "subject_model", "judge_model", "generation_id", col,
            level="nominal",
        )
        rows.append({
            "Measure": f"Ward: {dim}",
            "α (all)": _fmt(strat_d["all"]),
            "α (cross-model)": _fmt(strat_d["cross_model"]),
            "α (same-model)": _fmt(strat_d["same_model"]),
            "Δ self−cross": _fmt(strat_d["mean_delta_self"]),
        })

    table_df = pd.DataFrame(rows)
    table_df.to_csv(results_dir / "table_reliability.csv", index=False)
    logger.info("Saved reliability table (%d rows)", len(table_df))

    # Also render as figure
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(table_df) + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax.set_title(
        "Inter-Rater Reliability (Krippendorff's α) — stratified by judge-subject overlap",
        pad=12,
    )
    plt.tight_layout()
    _save_figure(fig, results_dir / "fig3_reliability_table.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Token-count covariate analysis (Issue 4)
# ---------------------------------------------------------------------------

def write_token_covariate_analysis(ward_df: pd.DataFrame, tokens_df: pd.DataFrame,
                                   results_dir: Path) -> list[str]:
    """Spearman correlation between completion_tokens and Ward score.

    Computed overall and separately for each prompt variant to flag any
    residual instruction-following or response-length confounds. Conditions
    whose per-condition rho exceeds the cross-condition median by more than
    0.15 are flagged as token-confounded.

    Returns lines to append to the summary report.
    """
    if tokens_df.empty:
        return []

    # Average Ward score across judges per generation
    w = ward_df.groupby(["generation_id", "subject_model", "prompt_variant"])["ward_score"].mean().reset_index()
    merged = w.merge(tokens_df, on=["generation_id", "subject_model", "prompt_variant"], how="inner")

    if merged.empty:
        return []

    lines = ["\n## Token Count vs. Ward Score (Instruction-Following Covariate)\n"]

    def _rho_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 5_000) -> tuple[float, float, float, float]:
        if len(x) < 3:
            return float("nan"), float("nan"), float("nan"), float("nan")
        rho, p = stats.spearmanr(x, y)
        rng = np.random.default_rng(seed=1234)
        n = len(x)
        boot = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot[i] = stats.spearmanr(x[idx], y[idx]).statistic
        lo = float(np.nanquantile(boot, 0.025))
        hi = float(np.nanquantile(boot, 0.975))
        return float(rho), float(p), lo, hi

    x = merged["completion_tokens"].values
    y = merged["ward_score"].values
    rho_all, p_all, lo_all, hi_all = _rho_ci(x, y)
    lines.append(
        f"  Overall: ρ={rho_all:.3f} [{lo_all:.3f}, {hi_all:.3f}], p={p_all:.4f} (n={len(merged)})"
    )

    # Per condition
    per_cond_rows = []
    for variant, grp in merged.groupby("prompt_variant"):
        if len(grp) < 3:
            continue
        rho_v, p_v, lo_v, hi_v = _rho_ci(grp["completion_tokens"].values, grp["ward_score"].values)
        per_cond_rows.append({
            "prompt_variant": variant,
            "n": len(grp),
            "rho": rho_v,
            "rho_ci_lo": lo_v,
            "rho_ci_hi": hi_v,
            "p": p_v,
        })

    # Flag conditions whose rho exceeds the cross-condition median by >0.15
    flagged = []
    if per_cond_rows:
        rhos = np.array([r["rho"] for r in per_cond_rows], dtype=float)
        median_rho = float(np.nanmedian(rhos))
        threshold = median_rho + 0.15
        for r in per_cond_rows:
            flag = r["rho"] > threshold
            r["token_confounded"] = bool(flag)
            if flag:
                flagged.append(r["prompt_variant"])

        for r in per_cond_rows:
            mark = "  [TOKEN-CONFOUNDED]" if r["token_confounded"] else ""
            lines.append(
                f"  {r['prompt_variant']}: ρ={r['rho']:.3f} "
                f"[{r['rho_ci_lo']:.3f}, {r['rho_ci_hi']:.3f}], p={r['p']:.4f} "
                f"(n={r['n']}){mark}"
            )
        lines.append(
            f"  Cross-condition median ρ = {median_rho:.3f}; "
            f"flag threshold (median + 0.15) = {threshold:.3f}"
        )
        if flagged:
            lines.append(
                "  Flagged conditions should be excluded from primary "
                "model-comparison analyses: " + ", ".join(flagged)
            )

    # Save per-condition summary + raw merged data
    pd.DataFrame(per_cond_rows).to_csv(
        results_dir / "table_token_covariate_per_condition.csv", index=False
    )
    merged[["generation_id", "subject_model", "prompt_variant",
            "completion_tokens", "ward_score"]].to_csv(
        results_dir / "table_token_covariate.csv", index=False
    )
    logger.info("Saved token covariate tables")
    return lines


# ---------------------------------------------------------------------------
# Extraction reliability (parse-failure rate)
# ---------------------------------------------------------------------------

def write_parse_failure_analysis(
    status_df: pd.DataFrame, results_dir: Path
) -> list[str]:
    """Per-judge extraction success and failure rates.

    A failure on the extraction stage is itself a benchmark result: a judge
    that cannot reliably emit the requested JSON for a given subject model
    or prompt condition lowers the effective sample size for that cell and
    biases the reliability analysis. The breakdown distinguishes parse
    failures (the judge replied but its reply could not be parsed as the
    expected JSON) from API failures (the judge's API call was exhausted by
    the retry loop in :mod:`alienbench.client` without a usable response).

    A per-cell breakdown is written to ``table_extraction_status.csv``.
    Returns lines to append to the summary report.
    """
    if status_df.empty:
        return ["\n## Extraction Reliability\n", "  (no extraction records)"]

    statuses = ["success", "parse_error", "api_error"]
    cell = (
        status_df.assign(_one=1)
        .pivot_table(
            index=["judge_model", "subject_model", "prompt_variant"],
            columns="status",
            values="_one",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=statuses, fill_value=0)
        .reset_index()
    )
    cell["n_generations"] = cell[statuses].sum(axis=1)
    cell["success_rate"] = cell["success"] / cell["n_generations"].where(
        cell["n_generations"] > 0
    )
    cell["parse_error_rate"] = cell["parse_error"] / cell["n_generations"].where(
        cell["n_generations"] > 0
    )
    cell["api_error_rate"] = cell["api_error"] / cell["n_generations"].where(
        cell["n_generations"] > 0
    )
    cell.to_csv(results_dir / "table_extraction_status.csv", index=False)
    logger.info("Saved extraction status table (%d cells)", len(cell))

    lines = ["\n## Extraction Reliability (per-judge success rates)\n"]
    per_judge = (
        status_df.assign(_one=1)
        .pivot_table(
            index="judge_model",
            columns="status",
            values="_one",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=statuses, fill_value=0)
    )
    per_judge["n_generations"] = per_judge[statuses].sum(axis=1)
    for judge, row in per_judge.iterrows():
        n = int(row["n_generations"])
        if n == 0:
            continue
        n_succ = int(row["success"])
        n_parse = int(row["parse_error"])
        n_api = int(row["api_error"])
        rate = n_succ / n if n > 0 else float("nan")
        lines.append(
            f"  {judge}: {rate:.1%} successful "
            f"({n_succ}/{n}; {n_parse} parse_error, {n_api} api_error)"
        )

    flagged = cell[(cell["success_rate"] < 0.95) & (cell["n_generations"] > 0)]
    if not flagged.empty:
        lines.append(
            f"\n  {len(flagged)} (judge × model × prompt) cells below 95% extraction "
            f"success — see table_extraction_status.csv for the breakdown."
        )

    return lines


# ---------------------------------------------------------------------------
# Departure frequency analysis (Issue 5)
# ---------------------------------------------------------------------------

def write_departure_frequencies(ward_df: pd.DataFrame, results_dir: Path) -> list[str]:
    """Per-dimension departure rates to identify recurrent tropes.

    Dimensions with consistently high departure rates across all models and
    conditions may reflect training-data tropes rather than novel generation.

    Returns lines to append to the summary report.
    """
    dim_cols = [f"dim_{d}" for d in DIMENSION_IDS]

    # Average across judges per generation, then compute global mean per dimension
    per_gen = (
        ward_df.groupby(["generation_id", "subject_model", "prompt_variant"])[dim_cols]
        .mean()
        .reset_index()
    )
    freq = per_gen[dim_cols].mean().rename(lambda c: c.replace("dim_", ""))
    freq_df = freq.sort_values(ascending=False).reset_index()
    freq_df.columns = ["dimension", "departure_rate"]

    freq_df.to_csv(results_dir / "table_departure_freq.csv", index=False)
    logger.info("Saved departure frequency table")

    lines = ["\n## Per-Dimension Departure Rates (Training-Data Trope Analysis)\n"]
    for _, row in freq_df.iterrows():
        lines.append(f"  {row['dimension']}: {row['departure_rate']:.3f}")
    return lines


# ---------------------------------------------------------------------------
# Length-adjusted Ward analysis (Ward per 100 completion tokens)
# ---------------------------------------------------------------------------

def write_length_adjusted_analysis(
    ward_df: pd.DataFrame,
    tokens_df: pd.DataFrame,
    results_dir: Path,
    posthoc_raw: pd.DataFrame | None = None,
) -> list[str]:
    """Repeat the primary comparisons on Ward per 100 completion tokens.

    Longer responses admit more opportunities for departure, so the paper
    reports a length-adjusted sensitivity analysis alongside the raw score.
    This emits per-model bootstrap CIs, Kruskal-Wallis + Mann-Whitney U
    post-hoc on prompt variants, Spearman agreement between raw and
    length-adjusted per-model means, and a list of post-hoc pairs whose
    effect sign or significance differs from the raw analysis.

    Returns lines to append to the summary report.
    """
    if tokens_df.empty:
        return []

    w = (
        ward_df.groupby(["generation_id", "subject_model", "prompt_variant"])["ward_score"]
        .mean()
        .reset_index()
    )
    merged = w.merge(
        tokens_df, on=["generation_id", "subject_model", "prompt_variant"], how="inner"
    )
    if merged.empty:
        return []

    n_before = len(merged)
    merged = merged[merged["completion_tokens"] > 0].copy()
    n_dropped = n_before - len(merged)
    if n_dropped > 0:
        logger.warning(
            "Dropped %d rows with completion_tokens<=0 from length-adjusted analysis",
            n_dropped,
        )
    if merged.empty:
        return []

    merged["ward_per_100"] = merged["ward_score"] / merged["completion_tokens"] * 100.0

    merged[["generation_id", "subject_model", "prompt_variant",
            "ward_score", "completion_tokens", "ward_per_100"]].to_csv(
        results_dir / "table_length_adjusted.csv", index=False
    )

    lines = ["\n## Length-Adjusted Ward Score (per 100 completion tokens)\n"]
    if n_dropped > 0:
        lines.append(f"  Dropped {n_dropped} rows with completion_tokens<=0")

    lines.append("\n  Per-model mean (bootstrap 95% CI):")
    per_model_rows = []
    for model, grp in merged.groupby("subject_model"):
        m_adj, lo_adj, hi_adj = bootstrap_ci(grp["ward_per_100"].values, stat_fn=np.mean)
        m_raw, lo_raw, hi_raw = bootstrap_ci(grp["ward_score"].values, stat_fn=np.mean)
        per_model_rows.append({
            "subject_model": model,
            "n": len(grp),
            "mean_ward": m_raw, "ward_ci_lo": lo_raw, "ward_ci_hi": hi_raw,
            "mean_ward_per_100": m_adj,
            "ward_per_100_ci_lo": lo_adj, "ward_per_100_ci_hi": hi_adj,
        })
        lines.append(
            f"    {model}: ward_per_100={m_adj:.3f} "
            f"[{lo_adj:.3f}, {hi_adj:.3f}] (n={len(grp)})"
        )
    per_model_df = pd.DataFrame(per_model_rows)
    per_model_df.to_csv(
        results_dir / "table_length_adjusted_per_model.csv", index=False
    )

    if len(per_model_df) >= 2:
        rho_agree, p_agree = stats.spearmanr(
            per_model_df["mean_ward"].values,
            per_model_df["mean_ward_per_100"].values,
        )
        lines.append(
            f"\n  Agreement with raw Ward (Spearman ρ on per-model means): "
            f"ρ={float(rho_agree):.3f}, p={float(p_agree):.4f}"
        )

    lines.append("\n  Prompt Variant Effects on ward_per_100 (Kruskal-Wallis + Post-Hoc):")
    posthoc_adj: pd.DataFrame | None = None
    if merged["prompt_variant"].nunique() >= 2:
        kw_adj, posthoc_adj = kruskal_posthoc(merged, "prompt_variant", "ward_per_100")
        lines.append(
            f"    Kruskal-Wallis: H={kw_adj['kruskal_H']:.2f}, p={kw_adj['kruskal_p']:.4f}"
        )
        if kw_adj["kruskal_p"] < 0.05 and not posthoc_adj.empty:
            lines.append("\n    Pairwise Mann-Whitney U (Bonferroni-corrected):")
            for _, row in posthoc_adj.iterrows():
                sig = "*" if row["p_corrected"] < 0.05 else ""
                lines.append(
                    f"      {row['group_a']} vs {row['group_b']}: "
                    f"U={row['U']:.0f}, p={row['p_corrected']:.4f}{sig}, "
                    f"r={row['rank_biserial_r']:.3f} "
                    f"[{row['r_ci_lo']:.3f}, {row['r_ci_hi']:.3f}]"
                )
        if posthoc_adj is not None and not posthoc_adj.empty:
            posthoc_adj.to_csv(
                results_dir / "table_length_adjusted_posthoc.csv", index=False
            )
    else:
        lines.append("    Skipped (fewer than 2 prompt variants in data)")

    if (posthoc_raw is not None and posthoc_adj is not None
            and not posthoc_raw.empty and not posthoc_adj.empty):
        raw_key = posthoc_raw[[
            "group_a", "group_b", "p_corrected", "rank_biserial_r"
        ]].rename(columns={
            "p_corrected": "p_raw_metric", "rank_biserial_r": "r_raw_metric",
        })
        adj_key = posthoc_adj[[
            "group_a", "group_b", "p_corrected", "rank_biserial_r"
        ]].rename(columns={
            "p_corrected": "p_adj_metric", "rank_biserial_r": "r_adj_metric",
        })
        joined = raw_key.merge(adj_key, on=["group_a", "group_b"], how="inner")
        joined["sign_flip"] = (
            np.sign(joined["r_raw_metric"]) != np.sign(joined["r_adj_metric"])
        )
        joined["sig_flip"] = (
            (joined["p_raw_metric"] < 0.05) != (joined["p_adj_metric"] < 0.05)
        )
        joined["flipped"] = joined["sign_flip"] | joined["sig_flip"]
        divergent = joined[joined["flipped"]].copy()
        divergent.to_csv(
            results_dir / "table_length_adjusted_divergence.csv", index=False
        )
        lines.append(
            f"\n  Post-hoc divergence: {len(divergent)} of {len(joined)} pairs differ "
            f"between raw and length-adjusted analyses "
            f"(sign flip or α'=0.05 significance flip)."
        )
        for _, row in divergent.iterrows():
            lines.append(
                f"    {row['group_a']} vs {row['group_b']}: "
                f"raw r={row['r_raw_metric']:.3f} (p={row['p_raw_metric']:.4f}), "
                f"adj r={row['r_adj_metric']:.3f} (p={row['p_adj_metric']:.4f})"
            )

    logger.info("Saved length-adjusted tables")
    return lines


# ---------------------------------------------------------------------------
# Summary statistics report
# ---------------------------------------------------------------------------

def write_summary(ward_df: pd.DataFrame, results_dir: Path,
                  tokens_df: pd.DataFrame | None = None,
                  status_df: pd.DataFrame | None = None) -> None:
    lines = ["# AlienBench Results Summary\n"]

    w = ward_df.groupby(["generation_id", "subject_model"])["ward_score"].mean().reset_index()

    lines.append("## Ward Departure Score by Model (bootstrap 95% CI)\n")
    for model, grp in w.groupby("subject_model"):
        m, lo, hi = bootstrap_ci(grp["ward_score"].values, stat_fn=np.mean)
        lines.append(f"  {model}: {m:.2f} [{lo:.2f}, {hi:.2f}] (n={len(grp)})")

    if ward_df["judge_model"].nunique() >= 2:
        lines.append("\n## Inter-Rater Reliability (Ward Total, stratified)\n")
        strat = krippendorff_alpha_stratified(
            ward_df, "subject_model", "judge_model", "generation_id", "ward_score",
            level="interval",
        )
        lines.append(
            f"  α all-pairs     = {strat['all']:.3f}  (n={strat['n_all']})"
        )
        lines.append(
            f"  α cross-model   = {strat['cross_model']:.3f}  (n={strat['n_cross']})"
        )
        lines.append(
            f"  α same-model    = {strat['same_model']:.3f}  (n={strat['n_same']})"
        )
        lines.append(
            f"  Δ̄ self−cross    = {strat['mean_delta_self']:.3f} score points"
        )
        # Pre-specified exclusion rule from §3.5
        alpha_gap = (
            strat['same_model'] - strat['cross_model']
            if not (np.isnan(strat['same_model']) or np.isnan(strat['cross_model']))
            else float('nan')
        )
        delta = strat['mean_delta_self']
        flag = (not np.isnan(alpha_gap) and alpha_gap > 0.10) or (
            not np.isnan(delta) and delta > 0.5
        )
        lines.append(
            "  Self-judge exclusion rule triggered: "
            + ("YES — primary analysis should exclude self-judge pairs." if flag
               else "no — self-judge pairs may remain in the primary analysis.")
        )

    lines.append("\n## Prompt Variant Effects (Kruskal-Wallis + Post-Hoc)\n")
    posthoc: pd.DataFrame | None = None
    if ward_df["prompt_variant"].nunique() >= 2:
        w_by_prompt = ward_df.groupby(["generation_id", "prompt_variant"])["ward_score"].mean().reset_index()
        kw, posthoc = kruskal_posthoc(w_by_prompt, "prompt_variant", "ward_score")
        lines.append(f"  Kruskal-Wallis: H={kw['kruskal_H']:.2f}, p={kw['kruskal_p']:.4f}")

        if kw["kruskal_p"] < 0.05:
            lines.append("\n  Pairwise Mann-Whitney U (Bonferroni-corrected, rank-biserial 95% CI):\n")
            for _, row in posthoc.iterrows():
                sig = "*" if row["p_corrected"] < 0.05 else ""
                lines.append(
                    f"    {row['group_a']} vs {row['group_b']}: "
                    f"U={row['U']:.0f}, p={row['p_corrected']:.4f}{sig}, "
                    f"r={row['rank_biserial_r']:.3f} "
                    f"[{row['r_ci_lo']:.3f}, {row['r_ci_hi']:.3f}]"
                )
            posthoc.to_csv(results_dir / "table_posthoc.csv", index=False)
            logger.info("Saved post-hoc table")

            # Retrospective power on the observed pairwise effects (C6)
            observed_effects = posthoc["rank_biserial_r"].abs().tolist()
            # Use the smallest group size as the per-group n for the power calc
            n_per_group = int(min(posthoc["n_a"].min(), posthoc["n_b"].min()))
            power = retrospective_power(
                observed_effects,
                n_per_group=n_per_group,
                n_comparisons=len(posthoc),
            )
            lines.append("\n  Retrospective power (Mann-Whitney U, Bonferroni α'={:.4f}):".format(
                power["bonferroni_alpha"]))
            lines.append(
                f"    Minimum detectable |r| at {int(power['target_power']*100)}% power: "
                + (f"{power['min_detectable_r']:.2f}"
                   if not np.isnan(power['min_detectable_r'])
                   else "not reached in grid")
            )
            for r_obs, pwr in power["per_effect_power"].items():
                lines.append(f"    observed |r|={r_obs:.3f} -> simulated power={pwr:.2f}")
    else:
        lines.append("  Skipped (fewer than 2 prompt variants in data)")

    if status_df is not None:
        lines.extend(write_parse_failure_analysis(status_df, results_dir))

    if tokens_df is not None and not tokens_df.empty:
        lines.extend(write_token_covariate_analysis(ward_df, tokens_df, results_dir))
        lines.extend(
            write_length_adjusted_analysis(ward_df, tokens_df, results_dir, posthoc)
        )

    lines.extend(write_departure_frequencies(ward_df, results_dir))

    path = results_dir / "summary.txt"
    path.write_text("\n".join(lines) + "\n")
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str = "config.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(config_path)
    data_dir = Path(cfg.data_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading scores...")
    ward_df = load_ward_scores(data_dir, cfg.judge_models, cfg.models, cfg.prompt_variants)

    if ward_df.empty:
        raise RuntimeError("No Ward scores found. Run 'alienbench score' first.")

    logger.info("Ward records: %d", len(ward_df))

    logger.info("Loading generation tokens...")
    tokens_df = load_generation_tokens(data_dir, cfg.models, cfg.prompt_variants)
    logger.info("Token records: %d", len(tokens_df))

    logger.info("Loading extraction status...")
    status_df = load_extraction_status(
        data_dir, cfg.judge_models, cfg.models, cfg.prompt_variants
    )
    logger.info("Extraction attempts: %d", len(status_df))

    logger.info("Generating figures...")
    fig_ward_heatmap(ward_df, results_dir)
    fig_violin_scores(ward_df, results_dir)
    fig_overview_scores(ward_df, results_dir)
    fig_reliability_table(ward_df, results_dir)

    from alienbench.radar import (
        fig_ward_radar_overlay,
        fig_ward_radar_small_multiples,
    )
    fig_ward_radar_small_multiples(ward_df, results_dir)
    fig_ward_radar_overlay(ward_df, results_dir)

    write_summary(ward_df, results_dir, tokens_df=tokens_df, status_df=status_df)

    logger.info("Analysis complete. Results saved to %s/", results_dir)

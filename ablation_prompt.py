"""Ablation: Prompt Paraphrase Sensitivity (paper sec:ablation_prompt).

Quantifies how sensitive model rank orderings and absolute Ward scores
are to surface rewordings of the baseline prompt. The paraphrase set is
a collection of authored rewordings (one per entry in
``prompt_paraphrases``); we do not assert semantic equivalence between
them. For each ordered pair of paraphrases the ablation reports two
quantities:

1. Spearman rank correlation ρ between per-model mean Ward scores.
   Because Spearman ρ is invariant to uniform shifts in the underlying
   scores, a high ρ indicates rank stability only, not absolute-score
   stability.

2. Pairwise mean absolute difference (MAD) of per-model mean Ward
   scores. MAD captures the absolute-score shift that ρ ignores; large
   MAD with high ρ means the ranking survives the rewording but the
   absolute Ward scores do not.

To make ρ interpretable the ablation also computes an empirical null
for ρ by permuting model labels on one side of each pair and
recomputing ρ; an observed ρ that exceeds the null 95% upper bound
indicates that the ranking is preserved beyond chance.

The paraphrase set is defined in config under ``prompt_paraphrases``.
This module drives generate/extract/score on that set by writing a
shadow config whose ``prompt_variants`` equals the paraphrase list,
then aggregates the resulting Ward scores. Output paths are keyed by
prompt id (see ``alienbench.paths``), so paraphrase runs coexist with
the main pipeline in the same ``data_dir`` and benefit from the
existing resume-on-restart logic.

The ablation runs at a reduced per-cell N
(``Config.prompt_ablation_samples_per_condition``, default 10) rather
than ``samples_per_condition``, because each new paraphrase costs a
full generate/extract/score pass over every (subject_model, judge)
cell. The ``baseline`` paraphrase id collides with the main
``prompt_variants`` baseline by design (see
``Config._check_paraphrase_id_collision``); the baseline column of
the ablation therefore reuses the main-pipeline records and inherits
the main-pipeline N at that cell only, anchoring the ablation to the
ranking reported in the main results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml

from alienbench.config import Config, load_config
from alienbench.paths import load_ward_scores

logger = logging.getLogger(__name__)


SHADOW_CONFIG_NAME = "_ablation_prompt_config.yaml"

# Fixed seed so the permutation null distribution is bit-stable across
# re-runs of the ablation.
PERMUTATION_SEED = 0
N_PERMUTATIONS = 10_000

# Preregistered stability thresholds. A paraphrase pair is declared
# stable when the per-model Spearman ρ is at least RHO_STABLE_THRESHOLD
# (rank axis) AND the pairwise MAD in Ward points is at most
# MAD_STABLE_THRESHOLD (absolute-score axis). These bars are fixed
# before the statistics are computed, so off-diagonal pass counts are
# pre-specified rather than post-hoc. The ρ bar matches the dimension
# ablation (ablation_dimensions.RHO_THRESHOLD). The MAD bar is 1.0 Ward
# point on the 0--10 scale (10% of scale).
RHO_STABLE_THRESHOLD = 0.9
MAD_STABLE_THRESHOLD = 1.0


def _materialize_paraphrase_config(cfg: Config, out_path: Path) -> Path:
    """Write a YAML that mirrors ``cfg`` with paraphrases as prompt_variants.

    The shadow config is consumed by the main pipeline stages unchanged.
    """
    # The paraphrase ablation runs at a reduced per-cell N controlled by
    # ``prompt_ablation_samples_per_condition``. The shadow config sets
    # ``samples_per_condition`` to that value, which drives every pipeline
    # stage (generate creates this many; extract / score / analyze clip
    # to the same window). The baseline paraphrase shares on-disk paths
    # with the main pipeline; ``load_ward_scores`` reads whatever Ward
    # records are present for that cell.
    paraphrase_n = cfg.prompt_ablation_samples_per_condition
    data = {
        "models": list(cfg.models),
        "judge_models": list(cfg.judge_models),
        "prompt_variants": [p.model_dump() for p in cfg.prompt_paraphrases],
        "samples_per_condition": paraphrase_n,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "data_dir": cfg.data_dir,
        "results_dir": cfg.results_dir,
        "api_key_env": cfg.api_key_env,
        "openrouter_base_url": cfg.openrouter_base_url,
        "allow_provider_fallbacks": cfg.allow_provider_fallbacks,
        "primary_metric": cfg.primary_metric,
        "judge_overrides": {k: v.model_dump() for k, v in cfg.judge_overrides.items()},
    }
    if cfg.allowed_providers is not None:
        data["allowed_providers"] = list(cfg.allowed_providers)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out_path


def _per_model_variant_means(ward_df: pd.DataFrame) -> pd.DataFrame:
    """Average judges per generation, then generations per (model, variant).

    Matches the aggregation convention used in ``analyze.py`` and
    ``latex_tables.py``. Returns a wide DataFrame indexed by subject_model,
    with one column per prompt_variant.
    """
    per_gen = (
        ward_df.groupby(["generation_id", "subject_model", "prompt_variant"])["ward_score"]
        .mean()
        .reset_index()
    )
    per_cell = (
        per_gen.groupby(["subject_model", "prompt_variant"])["ward_score"]
        .mean()
        .reset_index()
    )
    return per_cell.pivot(index="subject_model", columns="prompt_variant", values="ward_score")


def _spearman_matrix(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Spearman ρ across columns (prompt variants).

    Returns ``(rho_matrix, n_matrix)``. Pairs with fewer than three models in
    common receive ``NaN``.
    """
    variants = list(wide.columns)
    rho = pd.DataFrame(index=variants, columns=variants, dtype=float)
    n = pd.DataFrame(index=variants, columns=variants, dtype=int)
    for a in variants:
        for b in variants:
            paired = wide[[a, b]].dropna()
            n.loc[a, b] = len(paired)
            if a == b:
                rho.loc[a, b] = 1.0
            elif len(paired) < 3:
                rho.loc[a, b] = float("nan")
            else:
                rho.loc[a, b] = float(stats.spearmanr(paired[a], paired[b]).statistic)
    return rho, n


def _mad_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    """Pairwise mean absolute difference (MAD) of per-model Ward means.

    For each ordered pair ``(a, b)`` of columns, ``MAD(a, b)`` is the
    mean over models of ``|wide[a] - wide[b]|``, restricted to models
    with a Ward mean in both variants. The diagonal is zero by
    definition; pairs with no overlapping models receive ``NaN``.

    MAD complements Spearman ρ: ρ is invariant to uniform shifts, so a
    paraphrase that translates every per-model mean by a constant still
    reports ρ = 1. MAD captures the absolute-score shift that ρ
    ignores.
    """
    variants = list(wide.columns)
    mad = pd.DataFrame(index=variants, columns=variants, dtype=float)
    for a in variants:
        for b in variants:
            if a == b:
                mad.loc[a, b] = 0.0
                continue
            paired = wide[[a, b]].dropna()
            if paired.empty:
                mad.loc[a, b] = float("nan")
            else:
                mad.loc[a, b] = float(
                    (paired[a] - paired[b]).abs().mean()
                )
    return mad


def _permutation_null_spearman(
    wide: pd.DataFrame,
    rho_obs: pd.DataFrame,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Empirical null distribution of pairwise Spearman ρ under random
    relabeling of models on one side of each paraphrase pair.

    For each ordered pair ``(a, b)`` of columns in ``wide`` with at least
    three models in common, the null is built by drawing
    ``n_permutations`` permutations of the model index on column ``b``
    and recomputing Spearman ρ against column ``a``. The null tests the
    hypothesis that the two paraphrases induce independent model
    rankings. Diagonal entries and pairs with fewer than three shared
    models receive ``NaN``.

    Returns four DataFrames (``null_mean``, ``null_lo95``, ``null_hi95``,
    ``p_value``) indexed and keyed the same way as ``rho_obs``. The
    p-value is two-sided: the fraction of permutation draws with
    ``|ρ_perm| ≥ |ρ_obs|``.
    """
    rng = np.random.default_rng(seed)
    variants = list(wide.columns)
    null_mean = pd.DataFrame(index=variants, columns=variants, dtype=float)
    null_lo = pd.DataFrame(index=variants, columns=variants, dtype=float)
    null_hi = pd.DataFrame(index=variants, columns=variants, dtype=float)
    pval = pd.DataFrame(index=variants, columns=variants, dtype=float)
    for a in variants:
        for b in variants:
            if a == b:
                null_mean.loc[a, b] = float("nan")
                null_lo.loc[a, b] = float("nan")
                null_hi.loc[a, b] = float("nan")
                pval.loc[a, b] = float("nan")
                continue
            paired = wide[[a, b]].dropna()
            if len(paired) < 3:
                null_mean.loc[a, b] = float("nan")
                null_lo.loc[a, b] = float("nan")
                null_hi.loc[a, b] = float("nan")
                pval.loc[a, b] = float("nan")
                continue
            x = paired[a].to_numpy()
            y = paired[b].to_numpy()
            n = len(x)
            draws = np.empty(n_permutations, dtype=float)
            for i in range(n_permutations):
                y_perm = rng.permutation(y)
                draws[i] = float(stats.spearmanr(x, y_perm).statistic)
            null_mean.loc[a, b] = float(draws.mean())
            null_lo.loc[a, b] = float(np.percentile(draws, 2.5))
            null_hi.loc[a, b] = float(np.percentile(draws, 97.5))
            obs = rho_obs.loc[a, b]
            if pd.isna(obs):
                pval.loc[a, b] = float("nan")
            else:
                pval.loc[a, b] = float(
                    np.mean(np.abs(draws) >= abs(float(obs)))
                )
    return null_mean, null_lo, null_hi, pval


def _format_latex_table(
    wide: pd.DataFrame,
    rho: pd.DataFrame,
    labels: dict[str, str],
    null_hi: pd.DataFrame | None = None,
    mad: pd.DataFrame | None = None,
    rho_stable_threshold: float = RHO_STABLE_THRESHOLD,
    mad_stable_threshold: float = MAD_STABLE_THRESHOLD,
) -> str:
    """Render the per-model means and the ρ matrix as a single LaTeX table.

    When ``null_hi`` is provided, off-diagonal cells whose observed
    ``rho`` exceeds the null 95% upper bound are marked with ``$^\\dagger$``
    and the caption quotes the range of null upper bounds across
    off-diagonal cells. When ``mad`` is provided, the caption also quotes
    the off-diagonal range of the pairwise mean absolute difference of
    per-model Ward means, which captures the absolute-score shift that
    ρ is invariant to, and off-diagonal cells that jointly satisfy the
    preregistered bars (``rho >= rho_stable_threshold`` AND
    ``mad <= mad_stable_threshold``) are marked with ``$^\\ddagger$``.
    """
    variants = list(wide.columns)
    header_cols = " & ".join(labels.get(v, v) for v in variants)

    # Per-model means block
    mean_lines = []
    for model, row in wide.iterrows():
        vals = " & ".join(
            f"{row[v]:.2f}" if pd.notna(row[v]) else "--" for v in variants
        )
        mean_lines.append(f"{model} & {vals} \\\\")

    # ρ matrix block; mark cells that exceed the null 95% upper bound
    # (\dagger) and cells that jointly satisfy the preregistered
    # stability bars (\ddagger).
    rho_lines = []
    for a in variants:
        cells = []
        for b in variants:
            r = rho.loc[a, b]
            if pd.isna(r):
                cells.append("--")
                continue
            marks = ""
            if null_hi is not None and a != b:
                u = null_hi.loc[a, b]
                if pd.notna(u) and float(r) > float(u):
                    marks += "$^\\dagger$"
            if mad is not None and a != b:
                m = mad.loc[a, b]
                if (
                    pd.notna(m)
                    and float(r) >= float(rho_stable_threshold)
                    and float(m) <= float(mad_stable_threshold)
                ):
                    marks += "$^\\ddagger$"
            cells.append(f"{float(r):.2f}{marks}")
        rho_lines.append(f"{labels.get(a, a)} & {' & '.join(cells)} \\\\")

    ncols = len(variants)
    col_spec = "l" + "c" * ncols

    caption = (
        "Prompt paraphrase sensitivity. Top: per-model mean Ward "
        "Departure Score (0--10) under each surface rewording of the "
        "baseline prompt. Bottom: pairwise Spearman rank correlation "
        "$\\rho$ of per-model means across paraphrases. High off-diagonal "
        "$\\rho$ indicates that the paraphrase wording does not change "
        "the induced model ranking, but $\\rho$ is invariant to uniform "
        "shifts in score and does not characterize absolute-score "
        "stability."
    )
    if null_hi is not None:
        off_diag_mask = ~np.eye(len(null_hi), dtype=bool)
        off_diag_hi = null_hi.where(off_diag_mask).stack().astype(float)
        if not off_diag_hi.empty:
            caption += (
                " An empirical null built by permuting model labels "
                f"({N_PERMUTATIONS:,} draws) gives off-diagonal $95\\%$ "
                f"upper bounds in $[{off_diag_hi.min():.2f},\\ "
                f"{off_diag_hi.max():.2f}]$; cells marked "
                "$^\\dagger$ exceed their null $95\\%$ upper bound."
            )
    if mad is not None:
        off_diag_mask_m = ~np.eye(len(mad), dtype=bool)
        off_diag_mad = mad.where(off_diag_mask_m).stack().astype(float)
        if not off_diag_mad.empty:
            caption += (
                " Pairwise mean absolute difference of per-model Ward "
                f"means across paraphrases lies in "
                f"$[{off_diag_mad.min():.2f},\\ {off_diag_mad.max():.2f}]$ "
                "points (on the 0--10 Ward scale), indicating the "
                "absolute-score shift that $\\rho$ does not capture."
            )

    # Preregistered joint-bar caption clause.
    if mad is not None:
        off_mask = ~np.eye(len(rho), dtype=bool)
        rho_vals = rho.where(off_mask).stack().astype(float)
        mad_vals = mad.where(off_mask).stack().astype(float)
        common_idx = rho_vals.index.intersection(mad_vals.index)
        if len(common_idx) > 0:
            both = (
                (rho_vals.loc[common_idx] >= float(rho_stable_threshold))
                & (mad_vals.loc[common_idx] <= float(mad_stable_threshold))
            )
            n_both = int(both.sum())
            n_total = int(len(common_idx))
            caption += (
                f" Cells marked $^\\ddagger$ satisfy the preregistered "
                f"stability bars $\\rho \\geq {rho_stable_threshold:.2f}$ "
                f"and $\\mathrm{{MAD}} \\leq {mad_stable_threshold:.2f}$ "
                f"Ward points jointly; {n_both}/{n_total} off-diagonal "
                "pairs clear both bars."
            )

    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:ablation_prompt}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"Model & {header_cols} \\\\",
        "\\midrule",
        *mean_lines,
        "\\midrule",
        f"Paraphrase & {header_cols} \\\\",
        "\\midrule",
        *rho_lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])


def run(config_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(config_path)

    if len(cfg.prompt_paraphrases) < 2:
        raise ValueError(
            "Prompt Paraphrase Sensitivity ablation requires at least two "
            "entries in `prompt_paraphrases` (got "
            f"{len(cfg.prompt_paraphrases)}). Add paraphrases of the baseline "
            "prompt to the config."
        )

    data_dir = Path(cfg.data_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    shadow_path = results_dir / SHADOW_CONFIG_NAME
    _materialize_paraphrase_config(cfg, shadow_path)
    logger.info("Wrote shadow config: %s", shadow_path)

    from alienbench.generate import run as run_generate
    from alienbench.extract import run as run_extract
    from alienbench.score import run as run_score

    logger.info("Stage: generate (paraphrases)")
    run_generate(str(shadow_path))
    logger.info("Stage: extract (paraphrases)")
    run_extract(str(shadow_path))
    logger.info("Stage: score (paraphrases)")
    run_score(str(shadow_path))

    logger.info("Aggregating Ward scores across paraphrases...")
    ward_df = load_ward_scores(
        data_dir, cfg.judge_models, cfg.models, cfg.prompt_paraphrases
    )
    if ward_df.empty:
        raise RuntimeError(
            "No Ward scores found for paraphrase variants after running the "
            "pipeline. Inspect data_dir and the shadow config."
        )

    wide = _per_model_variant_means(ward_df)
    # Preserve paraphrase order from config rather than alphabetical
    ordered_variants = [p.id for p in cfg.prompt_paraphrases if p.id in wide.columns]
    wide = wide[ordered_variants]

    rho, n = _spearman_matrix(wide)
    mad = _mad_matrix(wide)

    wide.to_csv(results_dir / "table_ablation_prompt.csv")
    rho.to_csv(results_dir / "table_ablation_prompt_corr.csv")
    n.to_csv(results_dir / "table_ablation_prompt_corr_n.csv")
    mad.to_csv(results_dir / "table_ablation_prompt_mad.csv")
    logger.info("Saved per-model means, ρ matrix, and MAD matrix CSVs")

    logger.info(
        "Computing permutation null (B=%d, seed=%d)...",
        N_PERMUTATIONS, PERMUTATION_SEED,
    )
    null_mean, null_lo, null_hi, pval = _permutation_null_spearman(wide, rho)

    # Long-form null table: one row per ordered pair.
    null_rows = []
    variants = list(wide.columns)
    for a in variants:
        for b in variants:
            r_val = rho.loc[a, b]
            m_val = mad.loc[a, b]
            rho_pass = (
                bool(pd.notna(r_val) and float(r_val) >= RHO_STABLE_THRESHOLD)
                if a != b else False
            )
            mad_pass = (
                bool(pd.notna(m_val) and float(m_val) <= MAD_STABLE_THRESHOLD)
                if a != b else False
            )
            null_rows.append({
                "pair_a": a,
                "pair_b": b,
                "rho_obs": r_val,
                "mad": m_val,
                "null_mean": null_mean.loc[a, b],
                "null_lo95": null_lo.loc[a, b],
                "null_hi95": null_hi.loc[a, b],
                "p_value": pval.loc[a, b],
                "n_models": n.loc[a, b],
                "rho_above_stable_bar": rho_pass,
                "mad_below_stable_bar": mad_pass,
                "both_bars_pass": rho_pass and mad_pass,
            })
    pd.DataFrame(null_rows).to_csv(
        results_dir / "table_ablation_prompt_null.csv", index=False,
    )
    logger.info("Saved table_ablation_prompt_null.csv")

    labels = {p.id: p.label for p in cfg.prompt_paraphrases}
    tex = _format_latex_table(wide, rho, labels, null_hi=null_hi, mad=mad)
    (results_dir / "tab_ablation_prompt.tex").write_text(tex)
    logger.info("Saved tab_ablation_prompt.tex")

    # Standalone summary so we do not collide with analyze.run's summary.txt
    lines = ["# Prompt Paraphrase Sensitivity\n"]
    lines.append("## Per-model mean Ward scores by paraphrase\n")
    lines.append(wide.round(3).to_string())
    lines.append("\n## Pairwise Spearman ρ (per-model means)\n")
    lines.append(rho.round(3).to_string())
    lines.append("\n## Pair n (models with means in both variants)\n")
    lines.append(n.to_string())
    off_mask = ~np.eye(len(rho), dtype=bool)
    off_diag = rho.where(off_mask)
    values = off_diag.stack().astype(float)
    if not values.empty:
        lines.append(
            f"\nOff-diagonal ρ: min={values.min():.3f}, "
            f"median={values.median():.3f}, max={values.max():.3f}"
        )

    lines.append("\n## Pairwise mean absolute difference of per-model means\n")
    lines.append(
        "ρ is invariant to uniform shifts in score, so a paraphrase that "
        "translates every per-model mean by a constant still reports "
        "ρ = 1. The MAD matrix captures the absolute-score shift that ρ "
        "ignores; values are in Ward points on the 0--10 scale.\n"
    )
    lines.append(mad.round(3).to_string())
    off_mad = mad.where(off_mask).stack().astype(float)
    if not off_mad.empty:
        lines.append(
            f"\nOff-diagonal MAD: min={off_mad.min():.3f}, "
            f"median={off_mad.median():.3f}, max={off_mad.max():.3f}"
        )

    lines.append(f"\n## Permutation null (B={N_PERMUTATIONS}, seed={PERMUTATION_SEED})\n")
    lines.append("Null mean ρ (should be ≈ 0 under H0):\n")
    lines.append(null_mean.round(3).to_string())
    lines.append("\nNull 95% upper bound:\n")
    lines.append(null_hi.round(3).to_string())
    lines.append("\nTwo-sided p-value (fraction of draws with |ρ_perm| ≥ |ρ_obs|):\n")
    lines.append(pval.round(4).to_string())

    off_null_mean = null_mean.where(off_mask).stack().astype(float)
    off_null_hi = null_hi.where(off_mask).stack().astype(float)
    off_pval = pval.where(off_mask).stack().astype(float)
    if not off_null_hi.empty:
        exceed_mask = values > off_null_hi.reindex(values.index)
        n_exceed = int(exceed_mask.sum())
        n_total = int(exceed_mask.notna().sum())
        lines.append(
            f"\nOff-diagonal null mean: min={off_null_mean.min():.3f}, "
            f"median={off_null_mean.median():.3f}, max={off_null_mean.max():.3f}"
        )
        lines.append(
            f"Off-diagonal null 95% upper bound: min={off_null_hi.min():.3f}, "
            f"median={off_null_hi.median():.3f}, max={off_null_hi.max():.3f}"
        )
        lines.append(
            f"Pairs with ρ_obs > null 95% upper bound: "
            f"{n_exceed}/{n_total}"
        )
        if not off_pval.empty:
            lines.append(
                f"Off-diagonal p-value: min={off_pval.min():.4f}, "
                f"median={off_pval.median():.4f}, max={off_pval.max():.4f}"
            )

    # Preregistered stability bars: pre-specified before computing the
    # statistics, so off-diagonal pass counts are not post-hoc.
    lines.append(
        f"\n## Preregistered stability bars "
        f"(ρ ≥ {RHO_STABLE_THRESHOLD:.2f}, "
        f"MAD ≤ {MAD_STABLE_THRESHOLD:.2f} Ward points)\n"
    )
    off_rho = rho.where(off_mask).stack().astype(float)
    off_mad_vals = mad.where(off_mask).stack().astype(float)
    common_idx = off_rho.index.intersection(off_mad_vals.index)
    if len(common_idx) > 0:
        rho_pass = off_rho.loc[common_idx] >= RHO_STABLE_THRESHOLD
        mad_pass = off_mad_vals.loc[common_idx] <= MAD_STABLE_THRESHOLD
        both_pass = rho_pass & mad_pass
        n_total = int(len(common_idx))
        lines.append(
            f"Off-diagonal pairs clearing ρ bar only: "
            f"{int(rho_pass.sum())}/{n_total}"
        )
        lines.append(
            f"Off-diagonal pairs clearing MAD bar only: "
            f"{int(mad_pass.sum())}/{n_total}"
        )
        lines.append(
            f"Off-diagonal pairs clearing both bars jointly: "
            f"{int(both_pass.sum())}/{n_total}"
        )

    (results_dir / "summary_ablation_prompt.txt").write_text("\n".join(lines) + "\n")
    logger.info("Saved summary_ablation_prompt.txt")

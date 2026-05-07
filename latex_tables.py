"""Stage 5: Generate LaTeX tables for direct inclusion in the paper."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from alienbench.config import load_config
from alienbench.dimensions import DIMENSION_IDS, WARD_DIMENSIONS  # noqa: F401
from alienbench.paths import load_ward_scores, model_dir_name  # noqa: F401
from alienbench.stats import bootstrap_ci, krippendorff_alpha

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reliability bands (Landis & Koch 1977)
# ---------------------------------------------------------------------------

_ALPHA_BANDS = [
    (0.80, "Excellent"),
    (0.60, "Good"),
    (0.40, "Moderate"),
    (0.20, "Fair"),
    (float("-inf"), "Poor"),
]


def _alpha_band(alpha: float) -> str:
    if np.isnan(alpha):
        return "N/A"
    for threshold, label in _ALPHA_BANDS:
        if alpha >= threshold:
            return label
    return "Poor"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _model_label(model_id: str) -> str:
    """Convert a provider/model-id string to a short display name."""
    mapping = {
        "openai/gpt-4o": "GPT-4o",
        "google/gemini-2.0-flash-001": "Gemini 2.0 Flash",
        "anthropic/claude-3.5-sonnet": "Claude 3.5 Sonnet",
    }
    if model_id in mapping:
        return mapping[model_id]
    # Fallback: strip provider prefix and capitalise
    parts = model_id.split("/", 1)
    return parts[-1] if len(parts) > 1 else model_id


_PROMPT_SHORT_LABELS = {
    "baseline": "Base",
    "departure_primed": "Dep-P",
    "constrained_no_light": "NoLight",
    "constrained_high_gravity": "HighG",
    "constrained_ammonia": "NH$_3$",
    "detailed_description": "Detail",
}


def _short_prompt_label(variant) -> str:
    """Compact column header for a prompt variant."""
    return _PROMPT_SHORT_LABELS.get(variant.id, variant.label)


def _fmt_mean_sd(values: pd.Series) -> str:
    v = values.dropna()
    if len(v) < 2:
        return "---"
    return f"{v.mean():.2f} ({v.std(ddof=1):.2f})"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}\\%"


def _bold(s: str) -> str:
    return f"\\textbf{{{s}}}"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _judge_avg(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Average across judges per (generation_id, subject_model, prompt_variant)."""
    return (
        df.groupby(["generation_id", "subject_model", "prompt_variant"])[score_col]
        .mean()
        .reset_index()
    )


# ---------------------------------------------------------------------------
# LaTeX table builders
# ---------------------------------------------------------------------------

def _make_score_table(
    df: pd.DataFrame,
    score_col: str,
    cfg,
    caption: str,
    label: str,
    score_range: str,
) -> str:
    models = [m for m in cfg.models if not df[df["subject_model"] == m].empty]
    variants = cfg.prompt_variants

    avg = _judge_avg(df, score_col)

    # One column per prompt variant plus a trailing Overall column.
    col_spec = "l" + "c" * len(variants) + "c"
    col_headers = " & ".join(
        [_short_prompt_label(v) for v in variants] + ["Overall"]
    )

    # Per-(model, variant) mean of per-generation judge-averaged scores.
    cell_means: dict[tuple[str, str], float] = {}
    cell_strs: dict[tuple[str, str], str] = {}
    for m in models:
        for v in variants:
            vals = avg.loc[
                (avg["subject_model"] == m) & (avg["prompt_variant"] == v.id),
                score_col,
            ]
            cell_means[(m, v.id)] = vals.mean() if len(vals) > 0 else float("nan")
            cell_strs[(m, v.id)] = _fmt_mean_sd(vals)

    # Per-model overall mean across all of that model's generations.
    overall_means: dict[str, float] = {
        m: (
            avg.loc[avg["subject_model"] == m, score_col].mean()
            if not avg.loc[avg["subject_model"] == m, score_col].empty
            else float("nan")
        )
        for m in models
    }

    # Bootstrap percentile 95% CI on each per-model overall mean. The
    # resampling unit is the per-generation judge-averaged score, matching
    # the bootstrap convention used elsewhere in §3.5 of the paper.
    overall_ci: dict[str, tuple[float, float]] = {}
    for m in models:
        vals = avg.loc[avg["subject_model"] == m, score_col].values
        if len(vals) > 0:
            _, lo, hi = bootstrap_ci(vals, n_boot=10_000)
            overall_ci[m] = (lo, hi)
        else:
            overall_ci[m] = (float("nan"), float("nan"))

    # For each column, find the max across models so we can bold it.
    col_max: dict[str, float] = {}
    for v in variants:
        col_vals = [cell_means[(m, v.id)] for m in models if not np.isnan(cell_means[(m, v.id)])]
        col_max[v.id] = max(col_vals) if col_vals else float("nan")
    overall_vals = [overall_means[m] for m in models if not np.isnan(overall_means[m])]
    overall_col_max = max(overall_vals) if overall_vals else float("nan")

    def _maybe_bold(value: float, col_best: float, text: str) -> str:
        if not np.isnan(value) and not np.isnan(col_best) and abs(value - col_best) < 1e-9:
            return _bold(text)
        return text

    rows = []
    for m in models:
        row_cells = [_model_label(m)]
        for v in variants:
            row_cells.append(
                _maybe_bold(cell_means[(m, v.id)], col_max[v.id], cell_strs[(m, v.id)])
            )
        if not np.isnan(overall_means[m]):
            lo, hi = overall_ci[m]
            if not np.isnan(lo) and not np.isnan(hi):
                overall_str = f"{overall_means[m]:.2f}\\,[{lo:.2f}, {hi:.2f}]"
            else:
                overall_str = f"{overall_means[m]:.2f}"
        else:
            overall_str = "---"
        row_cells.append(_maybe_bold(overall_means[m], overall_col_max, overall_str))
        rows.append(f"    {' & '.join(row_cells)} \\\\")

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        "  \\small",
        f"  \\label{{{label}}}",
        "  \\resizebox{\\textwidth}{!}{%",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    Model & {col_headers} \\\\",
        "    \\midrule",
    ]
    lines += rows
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}%",
        "  }",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _make_ward_scores_table(ward_df: pd.DataFrame, cfg) -> str:
    return _make_score_table(
        ward_df,
        "ward_score",
        cfg,
        caption=(
            "Ward Departure Scores (0--10) by model and prompt condition. "
            "Rows are subject models; columns are prompt conditions with a "
            "final Overall column averaged across all conditions. "
            "Per-condition cells report mean (SD) across generations after "
            "averaging judges per generation. The Overall column reports the "
            "per-model mean with a bootstrap percentile 95\\% CI in brackets "
            "($B=10{,}000$) over per-generation judge-averaged Ward scores. "
            "Bold marks the top model in each column. "
            "Column headers: Base = baseline, Dep-P = departure-primed, "
            "NoLight = constrained (no light), HighG = constrained "
            "(10$\\times$ Earth gravity), NH$_3$ = constrained (ammonia "
            "oceans), Detail = detailed description."
        ),
        label="tab:ward_scores",
        score_range="0--10",
    )


def _make_reliability_table(ward_df: pd.DataFrame) -> str:
    rows = []

    def _add(measure: str, alpha: float) -> None:
        rows.append((measure, f"{alpha:.3f}", _alpha_band(alpha)))

    # Ward total (interval-level measurement)
    if ward_df["judge_model"].nunique() >= 2:
        _add("Ward Total (0--10)", krippendorff_alpha(ward_df, "judge_model", "generation_id", "ward_score", level="interval"))

    # Per Ward dimension (nominal-level measurement for binary scores)
    for dim_info in WARD_DIMENSIONS:
        col = f"dim_{dim_info['id']}"
        if col in ward_df.columns and ward_df["judge_model"].nunique() >= 2:
            alpha = krippendorff_alpha(ward_df, "judge_model", "generation_id", col, level="nominal")
            _add(dim_info["label"], alpha)

    def _row(measure: str, alpha_str: str, band: str) -> str:
        return f"    {measure} & {alpha_str} & {band} \\\\"

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Inter-rater reliability (Krippendorff's $\\alpha$) between LLM judges "
        "across all generations. Ward Total uses interval-level $\\alpha$; individual "
        "dimensions use nominal-level $\\alpha$ (binary scores). "
        "Bands follow Landis \\& Koch (1977): Poor ($<$0.20), "
        "Fair (0.20--0.40), Moderate (0.40--0.60), Good (0.60--0.80), Excellent ($\\geq$0.80).}",
        "  \\small",
        "  \\label{tab:reliability_full}",
        "  \\begin{tabular}{lcc}",
        "    \\toprule",
        "    Measure & $\\alpha$ & Interpretation \\\\",
        "    \\midrule",
    ]
    for measure, alpha_str, band in rows:
        lines.append(_row(measure, alpha_str, band))

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _make_ward_dimensions_table(ward_df: pd.DataFrame, cfg) -> str:
    models = [m for m in cfg.models if not ward_df[ward_df["subject_model"] == m].empty]

    dim_cols = [f"dim_{d}" for d in DIMENSION_IDS]

    # Average across judges per (generation_id, subject_model)
    avg = (
        ward_df.groupby(["generation_id", "subject_model"])[dim_cols]
        .mean()
        .reset_index()
    )

    # Then average across generations per subject_model
    model_means = avg.groupby("subject_model")[dim_cols].mean()

    col_spec = "l" + "c" * len(models)
    col_headers = " & ".join([_model_label(m) for m in models])

    rows = []
    for dim_info, col in zip(WARD_DIMENSIONS, dim_cols):
        cells = []
        vals = []
        for m in models:
            if m in model_means.index:
                v = model_means.loc[m, col]
                cells.append((v, _fmt_pct(v)))
                vals.append(v)
            else:
                cells.append((float("nan"), "---"))
                vals.append(float("nan"))

        max_val = max((v for v in vals if not np.isnan(v)), default=float("nan"))
        formatted = []
        for (v, cell_str) in cells:
            if not np.isnan(v) and abs(v - max_val) < 1e-9:
                formatted.append(_bold(cell_str))
            else:
                formatted.append(cell_str)

        row_cells = " & ".join([dim_info["label"]] + formatted)
        rows.append(f"    {row_cells} \\\\")

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Ward feature departure rates (\\%) per dimension and model, "
        "averaged across all prompt conditions and judges. "
        "Bold marks the highest departure rate per dimension.}",
        "  \\small",
        "  \\label{tab:ward_dimensions}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    Dimension & {col_headers} \\\\",
        "    \\midrule",
    ]
    lines += rows
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
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
        logger.error("No Ward scores found. Run 'alienbench score' first.")
        return

    logger.info("Ward records: %d", len(ward_df))

    tables: dict[str, str] = {
        "tab_ward_scores.tex": _make_ward_scores_table(ward_df, cfg),
        "tab_ward_dimensions.tex": _make_ward_dimensions_table(ward_df, cfg),
        "tab_reliability.tex": _make_reliability_table(ward_df),
    }

    for filename, content in tables.items():
        path = results_dir / filename
        path.write_text(content)
        logger.info("Saved %s", path)

    logger.info("LaTeX tables written to %s/", results_dir)

"""Stage 5: Generate the paper's LaTeX tables.

Every file holds a ``tabular`` alone. The float, ``\\resizebox``, caption, and
label are authored in ``paper/latex/paper.tex``, so regenerating a table cannot
overwrite an edit to the wording of its caption. Model, judge, tier, and
prompt names come from ``config.yaml``, and the numbers a caption or sentence
quotes come from the macros of :mod:`alienbench.paper`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from alienbench.analyze import TOKENS_PER_RATE_UNIT, length_adjusted_scores
from alienbench.config import load_config
from alienbench.dimensions import DIMENSION_IDS, PAPER_TEXT, WARD_DIMENSIONS  # noqa: F401
from alienbench.paths import load_generation_tokens, load_ward_scores, model_dir_name  # noqa: F401
from alienbench.stats import BOOTSTRAP_N, bootstrap_ci, krippendorff_alpha

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

def _short_prompt_label(variant) -> str:
    """Compact column header for a prompt variant."""
    return variant.short


def _fmt_mean_sd(values: pd.Series) -> str:
    v = values.dropna()
    if len(v) < 2:
        return "--"
    return f"{v.mean():.2f} ({v.std(ddof=1):.2f})"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}\\%"


def _bold(s: str) -> str:
    return f"\\textbf{{{s}}}"


def _is_winner(value: float, best: float, decimals: int) -> bool:
    """True when ``value`` ties the column maximum at the displayed precision.

    Comparing rounded values bolds every model a reader sees as tied for the
    top, so the highlighting always matches the printed numbers.
    """
    if np.isnan(value) or np.isnan(best):
        return False
    return f"{value:.{decimals}f}" == f"{best:.{decimals}f}"


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
    tokens_df: pd.DataFrame | None = None,
) -> str:
    models = [m for m in cfg.models if not df[df["subject_model"] == m].empty]
    variants = cfg.prompt_variants

    avg = _judge_avg(df, score_col)

    # Per-model mean of the per-generation rate, with its bootstrap CI.
    rates: dict[str, tuple[float, float, float]] = {}
    if tokens_df is not None and not tokens_df.empty:
        adjusted, _ = length_adjusted_scores(df, tokens_df)
        for m in models:
            vals = adjusted.loc[adjusted["subject_model"] == m, "ward_per_100"].values
            rates[m] = bootstrap_ci(vals, n_boot=BOOTSTRAP_N)

    def _rate_rank(rate: tuple[float, float, float]) -> tuple[float, ...]:
        # The printed mean, then the printed interval, breaks a printed tie.
        return tuple(float(f"{x:.2f}") for x in rate)

    rate_best = max(
        (_rate_rank(r) for r in rates.values() if not np.isnan(r[0])), default=None
    )

    # An Overall column, the optional rate column, then one column per prompt
    # variant.
    rate_headers = [f"Ward / {TOKENS_PER_RATE_UNIT} tokens"] if rates else []
    col_spec = "l" + "c" + "c" * len(rate_headers) + "c" * len(variants)
    col_headers = " & ".join(
        ["Overall"] + rate_headers + [_short_prompt_label(v) for v in variants]
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
            _, lo, hi = bootstrap_ci(vals, n_boot=BOOTSTRAP_N)
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
        # Cells print means with two decimals.
        return _bold(text) if _is_winner(value, col_best, 2) else text

    def _open_weights(model: str) -> bool | None:
        info = cfg.model_info.get(model)
        return info.open_weights if info is not None else None

    rows = []
    for i, m in enumerate(models):
        # A rule separates proprietary and open-weight models.
        if i > 0 and None not in (_open_weights(models[i - 1]), _open_weights(m)):
            if _open_weights(models[i - 1]) != _open_weights(m):
                rows.append("  \\midrule")
        if not np.isnan(overall_means[m]):
            lo, hi = overall_ci[m]
            if not np.isnan(lo) and not np.isnan(hi):
                overall_str = f"{overall_means[m]:.2f}\\,[{lo:.2f}, {hi:.2f}]"
            else:
                overall_str = f"{overall_means[m]:.2f}"
        else:
            overall_str = "--"
        row_cells = [
            cfg.model_name(m),
            _maybe_bold(overall_means[m], overall_col_max, overall_str),
        ]
        if rates:
            mean, lo, hi = rates[m]
            rate_str = "--" if np.isnan(mean) else f"{mean:.2f}\\,[{lo:.2f}, {hi:.2f}]"
            is_rate_best = not np.isnan(mean) and _rate_rank(rates[m]) == rate_best
            row_cells.append(_bold(rate_str) if is_rate_best else rate_str)
        for v in variants:
            row_cells.append(
                _maybe_bold(cell_means[(m, v.id)], col_max[v.id], cell_strs[(m, v.id)])
            )
        rows.append(f"    {' & '.join(row_cells)} \\\\")

    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "  \\toprule",
        f"  Model & {col_headers} \\\\",
        "  \\midrule",
    ]
    lines += rows
    lines += [
        "  \\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def _make_ward_scores_table(
    ward_df: pd.DataFrame, cfg, tokens_df: pd.DataFrame | None = None
) -> str:
    return _make_score_table(ward_df, "ward_score", cfg, tokens_df)


def dimension_alphas(ward_df: pd.DataFrame) -> dict[str, float]:
    """Nominal-level Krippendorff alpha per Ward dimension id, over all generations."""
    out: dict[str, float] = {}
    if ward_df["judge_model"].nunique() < 2:
        return out
    for dim_info in WARD_DIMENSIONS:
        col = f"dim_{dim_info['id']}"
        if col in ward_df.columns:
            out[dim_info["id"]] = krippendorff_alpha(
                ward_df, "judge_model", "generation_id", col, level="nominal"
            )
    return out


def _make_reliability_table(ward_df: pd.DataFrame) -> str:
    """Per-dimension alpha, two dimensions per row (the composite alpha is a macro)."""
    alphas = dimension_alphas(ward_df)
    cells = []
    for dim_info in WARD_DIMENSIONS:
        if dim_info["id"] in alphas:
            a = alphas[dim_info["id"]]
            cells.append((dim_info["label"], f"{a:.3f}", _alpha_band(a)))

    lines = [
        "\\begin{tabular}{lcclcc}",
        "  \\toprule",
        "  Feature & $\\alpha$ & Interpretation & Feature & $\\alpha$ & Interpretation \\\\",
        "  \\midrule",
    ]
    for i in range(0, len(cells), 2):
        left = " & ".join(cells[i])
        right = " & ".join(cells[i + 1]) if i + 1 < len(cells) else " & &"
        lines.append(f"  {left} & {right} \\\\")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _human_dimension_means(human_df: pd.DataFrame) -> dict[str, float]:
    """Per-dimension departure rate of the human baseline (judges, then generations)."""
    dim_cols = [f"dim_{d}" for d in DIMENSION_IDS]
    per_gen = human_df.groupby("generation_id")[dim_cols].mean()
    return per_gen.mean().to_dict()


def _make_ward_dimensions_table(
    ward_df: pd.DataFrame, cfg, human_df: pd.DataFrame | None = None
) -> str:
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

    human_means = (
        _human_dimension_means(human_df)
        if human_df is not None and not human_df.empty else None
    )

    headers = [cfg.model_name(m) for m in models]
    if human_means is not None:
        headers.insert(0, cfg.human_baseline.name)
    col_spec = "l" + "c" * len(headers)
    col_headers = " & ".join(f"\\rotatebox{{90}}{{{h}}}" for h in headers)

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
                cells.append((float("nan"), "--"))
                vals.append(float("nan"))

        max_val = max((v for v in vals if not np.isnan(v)), default=float("nan"))
        formatted = []
        for (v, cell_str) in cells:
            # Rates print as percentages with one decimal.
            if _is_winner(v * 100, max_val * 100, 1):
                formatted.append(_bold(cell_str))
            else:
                formatted.append(cell_str)
        if human_means is not None:
            formatted.insert(0, _fmt_pct(human_means[col]))

        row_cells = " & ".join([dim_info["label"]] + formatted)
        rows.append(f"  {row_cells} \\\\")

    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "  \\toprule",
        f"  Feature & {col_headers} \\\\",
        "  \\midrule",
    ]
    lines += rows
    lines += ["  \\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _make_models_table(cfg) -> str:
    """Appendix table of the subject models: name, slug, resolved snapshot."""
    lines = [
        "\\begin{tabular}{lll}",
        "  \\toprule",
        "  Model & OpenRouter slug & Resolved version \\\\",
        "  \\midrule",
    ]
    for m in cfg.models:
        resolved = cfg.model_info[m].resolved_version or "--"
        lines.append(
            f"  {cfg.model_name(m)} & \\texttt{{{m}}} & \\texttt{{{resolved}}} \\\\"
        )
    lines += ["  \\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _make_prompts_table(cfg) -> str:
    """Prompt-condition table: acronym, condition name, verbatim prompt text."""
    lines = [
        "\\begin{tabular}{llp{8.5cm}}",
        "  \\toprule",
        "  Acronym & Condition & Prompt text \\\\",
        "  \\midrule",
    ]
    for v in cfg.prompt_variants:
        lines.append(f"  {v.short} & {v.display} & {v.text} \\\\")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _make_paraphrases_table(cfg) -> str:
    """Prompt-paraphrase table: wording label and verbatim prompt text.

    The first row is the baseline prompt of the main study, so the table
    shows each rewording next to the wording it replaces.
    """
    lines = [
        "\\begin{tabular}{lp{10cm}}",
        "  \\toprule",
        "  Wording & Prompt text \\\\",
        "  \\midrule",
    ]
    for v in cfg.prompt_paraphrases:
        lines.append(f"  {v.display} & {v.text} \\\\")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _make_dimension_definitions_table() -> str:
    """Feature, Earth-typical default, and example departures."""
    lines = [
        "\\begin{tabular}{llll}",
        "\\toprule",
        "\\# & Feature & Earth-typical default & Example departures \\\\",
        "\\midrule",
    ]
    for i, d in enumerate(WARD_DIMENSIONS, start=1):
        t = PAPER_TEXT[d["id"]]
        lines.append(
            f"{i} & {t['short']} & {t['summary_default']} & {t['summary_examples']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _make_boundary_table() -> str:
    """Boundary criteria of the extraction prompt, one row per dimension."""
    lines = [
        "\\begin{tabular}{lp{1.8cm}p{2.4cm}p{6.0cm}}",
        "\\toprule",
        "Feature & Earth default & Departure examples & Boundary rule \\\\",
        "\\midrule",
    ]
    for d in WARD_DIMENSIONS:
        t = PAPER_TEXT[d["id"]]
        lines.append(
            f"{t['short']} & {t['boundary_default']} & {t['boundary_examples']} & "
            f"{t['boundary_rule']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_human_scores(cfg) -> pd.DataFrame:
    """Ward scores of the human baseline, judged by the same panel (may be empty)."""
    baseline = [v for v in cfg.prompt_variants if v.id == "baseline"]
    return load_ward_scores(
        Path(cfg.data_dir), cfg.judge_models, [cfg.human_baseline.id], baseline
    )


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
    human_df = load_human_scores(cfg)
    tokens_df = load_generation_tokens(data_dir, cfg.models, cfg.prompt_variants)

    tables: dict[str, str] = {
        "tab_ward_scores.tex": _make_ward_scores_table(ward_df, cfg, tokens_df),
        "tab_ward_dimensions.tex": _make_ward_dimensions_table(ward_df, cfg, human_df),
        "tab_reliability.tex": _make_reliability_table(ward_df),
        "tab_models.tex": _make_models_table(cfg),
        "tab_prompts.tex": _make_prompts_table(cfg),
        "tab_paraphrases.tex": _make_paraphrases_table(cfg),
        "tab_ward_definitions.tex": _make_dimension_definitions_table(),
        "tab_boundary.tex": _make_boundary_table(),
    }

    for filename, content in tables.items():
        path = results_dir / filename
        path.write_text(content)
        logger.info("Saved %s", path)

    logger.info("LaTeX tables written to %s/", results_dir)

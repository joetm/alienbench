"""Ward dimension radar chart (spider plot).

Emits two figures that share the per-model, per-dimension feature departure
rate already reported in ``fig1_ward_heatmap`` and ``tab_ward_dimensions``:

* ``fig5_ward_radar.pdf`` — small multiples, one radar per subject model,
  ordered by descending overall Ward score. Main-text figure.
* ``fig5b_ward_radar_overlay.pdf`` — single radar overlaying all models.

Both figures visualise the same quantity as the heatmap but make the
per-model *shape* (which dimensions a model departs on) easier to read.

The module is part of Stage 4 (``analyze``) but is also runnable standalone
via ``python -m alienbench.radar`` so that figure iteration does not require
re-running the full analysis.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alienbench.config import load_config
from alienbench.dimensions import DIMENSION_IDS, WARD_DIMENSIONS
from alienbench.paths import load_ward_scores

logger = logging.getLogger(__name__)


_DIM_LABELS = {d["id"]: d["label"] for d in WARD_DIMENSIONS}

# One-word radar tick labels. Full labels (`_DIM_LABELS`) are too long for
# polar ticks at ten axes and collide at the diagonals.
_RADAR_TICK_LABELS = {
    "symmetry":       "Symmetry",
    "sensory_organs": "Sensing",
    "locomotion":     "Locomotion",
    "body_plan":      "Body plan",
    "skin_covering":  "Covering",
    "reproduction":   "Reproduction",
    "metabolism":     "Metabolism",
    "communication":  "Communication",
    "habitat":        "Habitat",
    "cognition":      "Cognition",
}


def _short_model_name(model: str) -> str:
    """Display name for a panel title: drop the provider prefix."""
    return model.split("/", 1)[-1]


def _per_model_departure_rates(ward_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model, per-dimension feature departure rate (0-1).

    Judges are averaged per generation first, then generations are averaged
    per subject model. The returned DataFrame is indexed by ``subject_model``
    with one column per dimension in :data:`DIMENSION_IDS` order.
    """
    dim_cols = [f"dim_{d}" for d in DIMENSION_IDS]
    rates = (
        ward_df.groupby(["subject_model", "generation_id"])[dim_cols]
        .mean()
        .reset_index()
        .groupby("subject_model")[dim_cols]
        .mean()
    )
    rates.columns = [c.replace("dim_", "") for c in rates.columns]
    return rates[DIMENSION_IDS]


def _per_model_overall_ward(ward_df: pd.DataFrame) -> pd.Series:
    """Per-model overall Ward score (0-10) used to order panels."""
    return (
        ward_df.groupby(["subject_model", "generation_id"])["ward_score"]
        .mean()
        .reset_index()
        .groupby("subject_model")["ward_score"]
        .mean()
    )


def _closed_theta(n: int) -> np.ndarray:
    """Evenly spaced angles around a polar plot, closed back to the start."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.concatenate([theta, theta[:1]])


def _style_polar_axis(ax, theta: np.ndarray, labels: list[str]) -> None:
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_rlim(0.0, 1.0)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis="x", pad=4)
    ax.grid(True, color="#cccccc", linewidth=0.5)
    ax.spines["polar"].set_color("#888888")
    ax.spines["polar"].set_linewidth(0.5)


_MODEL_COLOR = "#d97a43"
_HUMAN_COLOR = "#3b6ea8"


# Maximum panels per grid row right of the lead column. More models add rows
# rather than columns, so the panels keep a readable size at text width.
_MAX_GRID_COLS = 6


def _balanced_rows(items: list[str], max_cols: int = _MAX_GRID_COLS) -> list[list[str]]:
    """Split panels into the fewest rows of at most ``max_cols``, as evenly as possible."""
    nrows = int(np.ceil(len(items) / max_cols))
    ncols = int(np.ceil(len(items) / nrows))
    return [items[i:i + ncols] for i in range(0, len(items), ncols)]


def _radar_layout(
    order: list[str], human_id: str | None
) -> tuple[list[str], list[list[str]]]:
    """Place the panels: a lead column and a grid of the remaining models.

    ``order`` lists the models by descending overall score. The lead column
    holds the human baseline (when present) on top and the best model below
    it. The other models fill the grid in score order, left to right and top
    to bottom, distributed as evenly as possible over the rows.
    """
    lead = ([human_id] if human_id else []) + order[:1]
    rows = _balanced_rows(order[1:]) if len(order) > 1 else []
    return lead, rows


def fig_ward_radar_small_multiples(
    ward_df: pd.DataFrame,
    results_dir: Path,
    cfg=None,
    human_df: pd.DataFrame | None = None,
) -> None:
    """Radar panels with the human baseline and the top model pulled out on the left.

    The left column holds two larger, labeled panels: the human baseline (in
    its own color) above the model with the highest overall Ward score. The
    remaining models follow in descending score order on a grid to the right,
    distributed as evenly as possible over the rows, with a shorter row centered.
    """
    from alienbench.analyze import _save_figure

    rates = _per_model_departure_rates(ward_df)
    order = _per_model_overall_ward(ward_df).sort_values(ascending=False).index.tolist()
    order = [m for m in order if m in rates.index]
    if not order:
        logger.warning("No subject models found; skipping fig5_ward_radar.")
        return

    human_id = cfg.human_baseline.id if cfg is not None else None
    has_human = human_id is not None and human_df is not None and not human_df.empty
    if has_human:
        rates = pd.concat([rates, _per_model_departure_rates(human_df)])

    labels = [_RADAR_TICK_LABELS[d] for d in DIMENSION_IDS]
    theta = _closed_theta(len(DIMENSION_IDS))

    lead, rows = _radar_layout(order, human_id if has_human else None)
    nrows = max(len(lead), len(rows))
    ncols = max((len(row) for row in rows), default=0)

    # Panels are placed by hand (inches) so that the lead column can be larger
    # and a shorter grid row can be centered. Only the lead panels carry
    # dimension labels. All panels share their axis order.
    lead_w, lead_d = 1.85, 0.95    # lead column width, lead radar diameter
    cell_w, cell_d = 1.0, 0.88    # grid cell width, grid radar diameter
    row_h = 1.65
    fig_w = lead_w + ncols * cell_w + 0.1
    fig_h = nrows * row_h
    fig = plt.figure(figsize=(fig_w, fig_h))

    def _draw(model: str, cx: float, cy: float, d: float, labeled: bool) -> None:
        ax = fig.add_axes(
            [(cx - d / 2) / fig_w, (cy - d / 2) / fig_h, d / fig_w, d / fig_h],
            projection="polar",
        )
        is_human = has_human and model == human_id
        color = _HUMAN_COLOR if is_human else _MODEL_COLOR
        values = np.asarray([rates.loc[model, dim] for dim in DIMENSION_IDS], dtype=float)
        values = np.concatenate([values, values[:1]])
        ax.plot(theta, values, color=color, linewidth=1.0)
        ax.fill(theta, values, color=color, alpha=0.25)
        _style_polar_axis(ax, theta, labels)
        if labeled:
            ax.set_xticklabels(labels, fontsize=5.5)
            ax.tick_params(axis="x", pad=0)
        else:
            ax.set_xticklabels([])
        if is_human:
            title = cfg.human_baseline.name
        elif cfg is not None:
            title = cfg.model_name(model)
        else:
            title = _short_model_name(model)
        ax.set_title(
            textwrap.fill(title, 16, break_on_hyphens=False),
            fontsize=7 if labeled else 6.5, pad=5 if labeled else 4,
            color=color if is_human else "black",
            fontweight="bold" if labeled else None,
        )

    def _row_cy(r: int) -> float:
        return (nrows - 1 - r) * row_h + 0.45 * row_h

    for r, model in enumerate(lead):
        _draw(model, lead_w / 2, _row_cy(r), lead_d, labeled=True)

    for r, row in enumerate(rows):
        offset = (ncols - len(row)) / 2.0
        for c, model in enumerate(row):
            cx = lead_w + (offset + c + 0.5) * cell_w
            _draw(model, cx, _row_cy(r), cell_d, labeled=False)

    _save_figure(fig, results_dir / "fig5_ward_radar.pdf")
    plt.close(fig)


def fig_ward_radar_overlay(ward_df: pd.DataFrame, results_dir: Path) -> None:
    """Single radar overlaying all subject models."""
    from alienbench.analyze import _save_figure

    rates = _per_model_departure_rates(ward_df)
    order = _per_model_overall_ward(ward_df).sort_values(ascending=False).index.tolist()
    order = [m for m in order if m in rates.index]
    if not order:
        logger.warning("No subject models found; skipping fig5b_ward_radar_overlay.")
        return

    labels = [_RADAR_TICK_LABELS[d] for d in DIMENSION_IDS]
    theta = _closed_theta(len(DIMENSION_IDS))

    fig, ax = plt.subplots(figsize=(8.0, 7.2), subplot_kw={"projection": "polar"})
    cmap = plt.get_cmap("tab10")
    for i, model in enumerate(order):
        values = np.asarray([rates.loc[model, d] for d in DIMENSION_IDS], dtype=float)
        values = np.concatenate([values, values[:1]])
        color = cmap(i % cmap.N)
        ax.plot(theta, values, color=color, linewidth=1.3, label=_short_model_name(model))
        ax.fill(theta, values, color=color, alpha=0.08)

    _style_polar_axis(ax, theta, labels)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.05),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()
    _save_figure(fig, results_dir / "fig5b_ward_radar_overlay.pdf")
    plt.close(fig)


def run(config_path: str = "config.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(config_path)
    data_dir = Path(cfg.data_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ward_df = load_ward_scores(data_dir, cfg.judge_models, cfg.models, cfg.prompt_variants)
    if ward_df.empty:
        logger.error("No Ward scores found. Run 'alienbench score' first.")
        return

    from alienbench.latex_tables import load_human_scores
    fig_ward_radar_small_multiples(ward_df, results_dir, cfg, load_human_scores(cfg))
    fig_ward_radar_overlay(ward_df, results_dir)


if __name__ == "__main__":
    run()

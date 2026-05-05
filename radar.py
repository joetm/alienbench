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
    ax.set_yticklabels(["", ".5", "", "1"], fontsize=7, color="#555555")
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis="x", pad=4)
    ax.grid(True, color="#cccccc", linewidth=0.5)
    ax.spines["polar"].set_color("#888888")
    ax.spines["polar"].set_linewidth(0.5)


def fig_ward_radar_small_multiples(ward_df: pd.DataFrame, results_dir: Path) -> None:
    """Grid of per-model radar panels ordered by descending overall Ward score."""
    from alienbench.analyze import _save_figure

    rates = _per_model_departure_rates(ward_df)
    order = _per_model_overall_ward(ward_df).sort_values(ascending=False).index.tolist()
    order = [m for m in order if m in rates.index]
    if not order:
        logger.warning("No subject models found; skipping fig5_ward_radar.")
        return

    labels = [_RADAR_TICK_LABELS[d] for d in DIMENSION_IDS]
    theta = _closed_theta(len(DIMENSION_IDS))

    n = len(order)
    ncols = 5 if n >= 5 else n
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.9 * ncols, 3.2 * nrows),
        subplot_kw={"projection": "polar"},
    )
    axes = np.atleast_1d(axes).ravel()

    accent = "#d97a43"
    for ax, model in zip(axes, order):
        values = np.asarray([rates.loc[model, d] for d in DIMENSION_IDS], dtype=float)
        values = np.concatenate([values, values[:1]])
        ax.plot(theta, values, color=accent, linewidth=1.4)
        ax.fill(theta, values, color=accent, alpha=0.25)
        _style_polar_axis(ax, theta, labels)
        ax.set_title(_short_model_name(model), fontsize=9, pad=14)

    for ax in axes[len(order):]:
        ax.set_visible(False)

    fig.suptitle(
        "Ward feature departure rate by model",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
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
    ax.set_title("Ward feature departure rate by model", fontsize=11, pad=18)
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

    fig_ward_radar_small_multiples(ward_df, results_dir)
    fig_ward_radar_overlay(ward_df, results_dir)


if __name__ == "__main__":
    run()

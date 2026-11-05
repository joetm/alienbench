"""The overview figure labels models with their display names."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

# Mock krippendorff before any alienbench import; the suite shares this
# convention with test_pipeline.
_mock_krippendorff = types.ModuleType("krippendorff")
_mock_krippendorff.alpha = lambda reliability_data, level_of_measurement: 0.85
sys.modules.setdefault("krippendorff", _mock_krippendorff)

from alienbench import analyze  # noqa: E402
from alienbench.config import Config  # noqa: E402


def _row(model, gen_id, ward):
    return {"generation_id": gen_id, "subject_model": model, "judge_model": "j1",
            "prompt_variant": "baseline", "ward_score": ward}


def test_y_axis_shows_display_names_in_descending_score_order(tmp_path, monkeypatch):
    cfg = Config(
        models=[
            {"id": "org/low-model-id", "name": "Low Model"},
            {"id": "org/high-model-id", "name": "High Model"},
        ],
        judge_models=["j1"],
        prompt_variants=[{"id": "baseline", "label": "Baseline", "acronym": "Base",
                          "name": "Baseline", "text": "Imagine a creature."}],
    )
    ward_df = pd.DataFrame([
        _row("org/low-model-id", "g1", 1), _row("org/low-model-id", "g2", 2),
        _row("org/high-model-id", "g3", 8), _row("org/high-model-id", "g4", 9),
    ])

    captured = {}

    def capture(fig, path: Path) -> None:
        ax = fig.axes[0]
        captured["labels"] = {
            float(y): label.get_text()
            for y, label in zip(ax.get_yticks(), ax.get_yticklabels())
        }
        captured["path"] = path

    monkeypatch.setattr(analyze, "_save_figure", capture)
    analyze.fig_overview_scores(ward_df, tmp_path, cfg)

    assert captured["labels"] == {0.0: "High Model", 1.0: "Low Model"}
    assert captured["path"].name == "fig4_overview.pdf"


def test_plot_has_no_title(tmp_path, monkeypatch):
    cfg = Config(
        models=[{"id": "org/model-id", "name": "Model"}],
        judge_models=["j1"],
        prompt_variants=[{"id": "baseline", "label": "Baseline", "acronym": "Base",
                          "name": "Baseline", "text": "Imagine a creature."}],
    )
    ward_df = pd.DataFrame([_row("org/model-id", "g1", 1), _row("org/model-id", "g2", 2)])

    captured = {}

    def capture(fig, path: Path) -> None:
        captured["title"] = fig.axes[0].get_title()

    monkeypatch.setattr(analyze, "_save_figure", capture)
    analyze.fig_overview_scores(ward_df, tmp_path, cfg)

    assert captured["title"] == ""


def test_legend_shows_condition_names_and_keeps_the_mean_entry(tmp_path, monkeypatch):
    cfg = Config(
        models=[{"id": "org/model-id", "name": "Model"}],
        judge_models=["j1"],
        prompt_variants=[
            {"id": "baseline", "label": "Baseline", "acronym": "Base",
             "name": "Open-ended Baseline", "text": "Imagine a creature."},
            {"id": "constrained_no_light", "label": "No Light", "acronym": "NoLight",
             "name": "No-Light World", "text": "Imagine a creature in the dark."},
        ],
    )
    ward_df = pd.DataFrame([
        _row("org/model-id", "g1", 1),
        {**_row("org/model-id", "g2", 2), "prompt_variant": "constrained_no_light"},
    ])

    captured = {}

    def capture(fig, path: Path) -> None:
        captured["legend"] = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]

    monkeypatch.setattr(analyze, "_save_figure", capture)
    analyze.fig_overview_scores(ward_df, tmp_path, cfg)

    assert captured["legend"] == ["Open-ended Baseline", "No-Light World", "Mean (95% CI)"]

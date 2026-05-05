"""Tests for the LaTeX-table generator (Stage 5).

These tests build a small synthetic Ward DataFrame and assert that the three
tables consumed by the paper (tab_ward_scores, tab_ward_dimensions,
tab_reliability) compile to LaTeX strings with the structural elements the
paper relies on. They protect against silent breakage of the Stage-5 output
that the paper's ``\\input{}`` directives depend on.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest


# Mock krippendorff before any alienbench import; the suite shares this
# convention with test_pipeline.
_mock_krippendorff = types.ModuleType("krippendorff")
_mock_krippendorff.alpha = lambda reliability_data, level_of_measurement: 0.85
sys.modules.setdefault("krippendorff", _mock_krippendorff)

from alienbench.dimensions import DIMENSION_IDS  # noqa: E402
from alienbench.latex_tables import (  # noqa: E402
    _alpha_band,
    _make_reliability_table,
    _make_ward_dimensions_table,
    _make_ward_scores_table,
    run as latex_run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ward_row(*, model, gen_id, judge, variant, ward, **dims):
    base = {f"dim_{d}": 0 for d in DIMENSION_IDS}
    base.update({f"dim_{k}": v for k, v in dims.items()})
    base.update({
        "generation_id": gen_id,
        "judge_model": judge,
        "subject_model": model,
        "prompt_variant": variant,
        "ward_score": ward,
    })
    return base


@pytest.fixture()
def ward_df():
    rows = [
        _ward_row(model="anthropic/claude-3.5-sonnet", gen_id="g1", judge="j1",
                  variant="baseline", ward=4, symmetry=1, locomotion=1, body_plan=1, habitat=1),
        _ward_row(model="anthropic/claude-3.5-sonnet", gen_id="g2", judge="j2",
                  variant="baseline", ward=3, symmetry=1, locomotion=1, habitat=1),
        _ward_row(model="openai/gpt-4o", gen_id="g3", judge="j1",
                  variant="baseline", ward=2, symmetry=1, locomotion=1),
        _ward_row(model="openai/gpt-4o", gen_id="g4", judge="j2",
                  variant="baseline", ward=2, symmetry=1, locomotion=1),
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def cfg():
    """Minimal stand-in for a Config object (only attributes the tables touch)."""
    obj = types.SimpleNamespace()
    obj.models = ["anthropic/claude-3.5-sonnet", "openai/gpt-4o"]
    obj.prompt_variants = [
        types.SimpleNamespace(id="baseline", label="Baseline"),
    ]
    return obj


# ---------------------------------------------------------------------------
# _alpha_band
# ---------------------------------------------------------------------------

class TestAlphaBand:
    def test_excellent_for_high_alpha(self):
        assert _alpha_band(0.85) == "Excellent"
        assert _alpha_band(0.80) == "Excellent"

    def test_good_band(self):
        assert _alpha_band(0.79) == "Good"
        assert _alpha_band(0.60) == "Good"

    def test_moderate_band(self):
        assert _alpha_band(0.55) == "Moderate"

    def test_fair_band(self):
        assert _alpha_band(0.30) == "Fair"

    def test_poor_for_low_alpha(self):
        assert _alpha_band(0.10) == "Poor"
        assert _alpha_band(-0.05) == "Poor"

    def test_nan_returns_na(self):
        assert _alpha_band(float("nan")) == "N/A"


# ---------------------------------------------------------------------------
# _make_ward_scores_table
# ---------------------------------------------------------------------------

class TestWardScoresTable:
    def test_emits_booktabs_table_with_one_row_per_model(self, ward_df, cfg):
        tex = _make_ward_scores_table(ward_df, cfg)

        assert tex.startswith("\\begin{table}")
        assert tex.rstrip().endswith("\\end{table}")
        assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex
        assert "\\label{tab:ward_scores}" in tex
        assert "Claude 3.5 Sonnet" in tex
        assert "GPT-4o" in tex

    def test_top_model_is_bolded_per_column(self, ward_df, cfg):
        tex = _make_ward_scores_table(ward_df, cfg)
        # Claude has Ward mean 3.5, GPT-4o has 2.0; Claude should be bold in
        # the Baseline column AND Overall.
        assert "\\textbf{" in tex
        # Specifically, the bolded row should be the Claude line.
        bold_lines = [l for l in tex.splitlines() if "\\textbf{" in l]
        assert any("Claude 3.5 Sonnet" in l for l in bold_lines)


# ---------------------------------------------------------------------------
# _make_ward_dimensions_table
# ---------------------------------------------------------------------------

class TestWardDimensionsTable:
    def test_emits_one_row_per_dimension(self, ward_df, cfg):
        tex = _make_ward_dimensions_table(ward_df, cfg)
        # 10 Ward dimensions = 10 data rows
        body_rows = [
            l for l in tex.splitlines()
            if l.strip().endswith("\\\\")
            and "&" in l
            and "Dimension" not in l
        ]
        assert len(body_rows) == len(DIMENSION_IDS)

    def test_label_is_tab_ward_dimensions(self, ward_df, cfg):
        tex = _make_ward_dimensions_table(ward_df, cfg)
        assert "\\label{tab:ward_dimensions}" in tex

    def test_percentages_are_emitted(self, ward_df, cfg):
        tex = _make_ward_dimensions_table(ward_df, cfg)
        assert "\\%" in tex


# ---------------------------------------------------------------------------
# _make_reliability_table
# ---------------------------------------------------------------------------

class TestReliabilityTable:
    def test_emits_total_and_per_dimension_rows(self, ward_df):
        tex = _make_reliability_table(ward_df)
        assert "Ward Total" in tex
        # Each Ward dimension label should appear in the table; spot-check
        # two distinctive labels.
        assert "Body Symmetry" in tex
        assert "Habitat" in tex
        assert "\\label{tab:reliability_full}" in tex


# ---------------------------------------------------------------------------
# Stage-5 entry point
# ---------------------------------------------------------------------------

class TestRun:
    def _write_tmp_pipeline(self, tmp_path: Path) -> str:
        """Write the minimum on-disk state ``latex_tables.run`` consumes."""
        cfg_text = f"""
models:
  - openai/gpt-4o-mini
judge_models:
  - openai/gpt-4o-mini
samples_per_condition: 2
temperature: 1.0
max_tokens: 400
prompt_variants:
  - id: baseline
    label: Baseline
    text: "Imagine a creature that lives on an alien planet. Describe it in detail."
data_dir: {tmp_path}/data
results_dir: {tmp_path}/results
api_key_env: OPENROUTER_API_KEY
openrouter_base_url: https://openrouter.ai/api/v1
"""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(cfg_text)

        scores_dir = (
            tmp_path / "data" / "scores"
            / "openai__gpt-4o-mini" / "openai__gpt-4o-mini" / "baseline"
        )
        scores_dir.mkdir(parents=True)
        ward_jsonl = scores_dir / "ward_scores.jsonl"

        import json as _json

        per_dim = {d: 1 if d in {"symmetry", "locomotion"} else 0
                   for d in DIMENSION_IDS}
        recs = [
            {"generation_id": "g1", "judge_model": "openai/gpt-4o-mini",
             "subject_model": "openai/gpt-4o-mini", "prompt_variant": "baseline",
             "ward_score": 2, "per_dimension": per_dim},
            {"generation_id": "g2", "judge_model": "openai/gpt-4o-mini",
             "subject_model": "openai/gpt-4o-mini", "prompt_variant": "baseline",
             "ward_score": 2, "per_dimension": per_dim},
        ]
        with ward_jsonl.open("w") as f:
            for r in recs:
                f.write(_json.dumps(r) + "\n")
        return str(cfg_path)

    def test_writes_three_tex_files(self, tmp_path):
        cfg_path = self._write_tmp_pipeline(tmp_path)
        latex_run(cfg_path)
        results = tmp_path / "results"
        assert (results / "tab_ward_scores.tex").exists()
        assert (results / "tab_ward_dimensions.tex").exists()
        assert (results / "tab_reliability.tex").exists()

    def test_no_op_on_empty_scores(self, tmp_path, caplog):
        cfg_text = f"""
models:
  - openai/gpt-4o-mini
judge_models:
  - openai/gpt-4o-mini
samples_per_condition: 1
temperature: 1.0
max_tokens: 400
prompt_variants:
  - id: baseline
    label: Baseline
    text: "x"
data_dir: {tmp_path}/data
results_dir: {tmp_path}/results
api_key_env: OPENROUTER_API_KEY
openrouter_base_url: https://openrouter.ai/api/v1
"""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(cfg_text)
        latex_run(str(cfg_path))
        # No tables written when there are no scores; `run` logs an error
        # rather than raising, so the user can rerun upstream stages.
        assert not (tmp_path / "results" / "tab_ward_scores.tex").exists()

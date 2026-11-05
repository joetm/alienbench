"""Tests for the LaTeX-table generator (Stage 5).

These tests build a small synthetic Ward DataFrame and assert that the tables
consumed by the paper compile to ``tabular`` strings with the structural
elements the paper relies on. The float, caption, and label live in
``paper.tex``, so no generated table may carry them. The tests protect against
silent breakage of the Stage-5 output that the paper's ``\\input{}``
directives depend on.
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
from alienbench.config import Config  # noqa: E402
from alienbench.latex_tables import (  # noqa: E402
    _alpha_band,
    _make_boundary_table,
    _make_dimension_definitions_table,
    _make_models_table,
    _make_prompts_table,
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
    """A real Config, so the display names come from the same path as in production."""
    return Config(
        models=[
            {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet",
             "tier": "A", "resolved_version": "anthropic/claude-3.5-sonnet-20241022"},
            {"id": "openai/gpt-4o", "name": "GPT-4o", "tier": "B"},
        ],
        judge_models=["j1", "j2"],
        prompt_variants=[
            {"id": "baseline", "label": "Baseline", "acronym": "Base",
             "name": "Baseline", "text": "Imagine a creature."},
        ],
    )


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

        assert tex.startswith("\\begin{tabular}")
        assert tex.rstrip().endswith("\\end{tabular}")
        assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex
        assert "\\caption" not in tex and "\\label" not in tex
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

    def test_overall_column_precedes_prompt_columns(self, ward_df, cfg):
        tex = _make_ward_scores_table(ward_df, cfg)
        header = next(l for l in tex.splitlines() if l.strip().startswith("Model &"))
        cols = [c.strip(" \\") for c in header.split("&")]
        assert cols[1] == "Overall"


class TestWardScoresTableOpenWeightDivider:
    @staticmethod
    def _table(flags):
        models = []
        rows = []
        for i, (name, is_open) in enumerate(flags):
            entry = {"id": f"org/{name}", "name": name}
            if is_open is not None:
                entry["open_weights"] = is_open
            models.append(entry)
            rows.append(_ward_row(model=f"org/{name}", gen_id=f"g{i}", judge="j1",
                                  variant="baseline", ward=1))
        cfg = Config(
            models=models,
            judge_models=["j1"],
            prompt_variants=[{"id": "baseline", "label": "Baseline", "acronym": "Base",
                              "name": "Baseline", "text": "Imagine a creature."}],
        )
        return _make_ward_scores_table(pd.DataFrame(rows), cfg).splitlines()

    def test_divider_separates_proprietary_from_open_weight_rows(self):
        lines = self._table([("One", False), ("Two", False), ("Three", True), ("Four", True)])
        body = [l.strip() for l in lines[lines.index("  \\midrule") + 1:]]
        assert body[0].startswith("One &")
        assert body[1].startswith("Two &")
        assert body[2] == "\\midrule"
        assert body[3].startswith("Three &")
        assert body[4].startswith("Four &")

    def test_no_divider_when_all_models_share_a_group(self):
        lines = self._table([("One", False), ("Two", False)])
        assert lines.count("  \\midrule") == 1

    def test_no_divider_without_open_weight_metadata(self):
        lines = self._table([("One", None), ("Two", None)])
        assert lines.count("  \\midrule") == 1


class TestWardScoresTableLengthAdjustedColumn:
    @staticmethod
    def _tokens(**by_gen):
        models = {"g1": "anthropic/claude-3.5-sonnet", "g2": "anthropic/claude-3.5-sonnet",
                  "g3": "openai/gpt-4o", "g4": "openai/gpt-4o"}
        return pd.DataFrame([
            {"generation_id": g, "subject_model": models[g],
             "prompt_variant": "baseline", "completion_tokens": n}
            for g, n in by_gen.items()
        ])

    @staticmethod
    def _cells(tex, name):
        line = next(l for l in tex.splitlines() if l.strip().startswith(name))
        return [c.strip() for c in line.split("&")]

    def test_rate_column_follows_overall(self, ward_df, cfg):
        tokens = self._tokens(g1=200, g2=100, g3=1000, g4=500)
        tex = _make_ward_scores_table(ward_df, cfg, tokens)
        header = next(l for l in tex.splitlines() if l.strip().startswith("Model &"))
        cols = [c.strip(" \\") for c in header.split("&")]
        assert cols[:3] == ["Model", "Overall", "Ward / 100 tokens"]
        assert tex.startswith("\\begin{tabular}{lccc")

    def test_rate_is_the_mean_of_per_generation_rates(self, ward_df, cfg):
        # 4/200 and 3/100 give 2.0 and 3.0 per 100 tokens. The mean of the
        # generation rates is 2.50. The ratio of the totals would give 2.33.
        tokens = self._tokens(g1=200, g2=100, g3=1000, g4=500)
        tex = _make_ward_scores_table(ward_df, cfg, tokens)
        assert self._cells(tex, "Claude 3.5 Sonnet")[2].startswith("\\textbf{2.50\\,[")
        assert self._cells(tex, "GPT-4o")[2].startswith("0.30\\,[")

    def test_tied_rate_bolds_only_the_higher_interval(self, ward_df, cfg):
        # Both means print as 2.50. The intervals are [2.00, 3.00] and [2.50, 2.50].
        tokens = self._tokens(g1=200, g2=100, g3=80, g4=80)
        tex = _make_ward_scores_table(ward_df, cfg, tokens)
        assert "\\textbf" not in self._cells(tex, "Claude 3.5 Sonnet")[2]
        assert self._cells(tex, "GPT-4o")[2].startswith("\\textbf{2.50\\,[2.50, 2.50]")

    def test_higher_rate_outranks_a_higher_interval(self, ward_df, cfg):
        # 2.60 [2.00, 3.20] has the higher mean and the lower interval.
        tokens = self._tokens(g1=200, g2=93.75, g3=80, g4=80)
        tex = _make_ward_scores_table(ward_df, cfg, tokens)
        assert self._cells(tex, "Claude 3.5 Sonnet")[2].startswith("\\textbf{2.60\\,[")
        assert "\\textbf" not in self._cells(tex, "GPT-4o")[2]

    def test_rate_and_interval_ties_bold_every_tied_model(self, ward_df, cfg):
        # All four generations give 2.50, so the printed cells are identical.
        tokens = self._tokens(g1=160, g2=120, g3=80, g4=80)
        tex = _make_ward_scores_table(ward_df, cfg, tokens)
        assert self._cells(tex, "Claude 3.5 Sonnet")[2].startswith("\\textbf{2.50\\,[2.50, 2.50]")
        assert self._cells(tex, "GPT-4o")[2].startswith("\\textbf{2.50\\,[2.50, 2.50]")

    def test_generations_without_positive_tokens_are_excluded(self, ward_df, cfg):
        tokens = self._tokens(g1=200, g2=0, g3=1000, g4=500)
        tex = _make_ward_scores_table(ward_df, cfg, tokens)
        assert "2.00\\,[2.00, 2.00]" in self._cells(tex, "Claude 3.5 Sonnet")[2]

    def test_rate_column_is_omitted_without_token_data(self, ward_df, cfg):
        tex = _make_ward_scores_table(ward_df, cfg)
        assert "Ward / 100 tokens" not in tex


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
            and "Feature" not in l
        ]
        assert len(body_rows) == len(DIMENSION_IDS)

    def test_carries_no_float_caption_or_label(self, ward_df, cfg):
        tex = _make_ward_dimensions_table(ward_df, cfg)
        assert "\\caption" not in tex and "\\label" not in tex
        assert "\\begin{table}" not in tex

    def test_human_baseline_is_the_first_column(self, ward_df, cfg):
        human = pd.DataFrame([
            _ward_row(model="human/prolific-baseline", gen_id="h1", judge="j1",
                      variant="baseline", ward=1, symmetry=1),
        ])
        tex = _make_ward_dimensions_table(ward_df, cfg, human)
        assert "Human baseline" in tex.splitlines()[2]
        assert "100.0\\%" in tex

    def test_percentages_are_emitted(self, ward_df, cfg):
        tex = _make_ward_dimensions_table(ward_df, cfg)
        assert "\\%" in tex


# ---------------------------------------------------------------------------
# _make_reliability_table
# ---------------------------------------------------------------------------

class TestReliabilityTable:
    def test_emits_one_row_per_dimension_pair(self, ward_df):
        tex = _make_reliability_table(ward_df)
        assert "Body Symmetry" in tex
        assert "Habitat" in tex
        assert "\\label" not in tex
        # Two dimensions per row.
        rows = [l for l in tex.splitlines() if l.strip().endswith("\\\\") and "&" in l
                and "Feature" not in l]
        assert len(rows) == len(DIMENSION_IDS) // 2


# ---------------------------------------------------------------------------
# Config-driven tables
# ---------------------------------------------------------------------------

class TestConfigTables:
    def test_models_table_reads_names_and_versions_from_config(self, cfg):
        tex = _make_models_table(cfg)
        assert "Claude 3.5 Sonnet & \\texttt{anthropic/claude-3.5-sonnet}" in tex
        assert "\\texttt{anthropic/claude-3.5-sonnet-20241022}" in tex
        # No tier column, so no rules between tiers.
        assert "Tier" not in tex
        assert tex.count("\\midrule") == 1

    def test_prompts_table_uses_the_config_acronym(self, cfg):
        tex = _make_prompts_table(cfg)
        assert "Base & Baseline & Imagine a creature." in tex

    def test_definition_tables_cover_every_dimension(self):
        for tex in (_make_dimension_definitions_table(), _make_boundary_table()):
            assert tex.count("\\\\") == len(DIMENSION_IDS) + 1
            assert "\\caption" not in tex


# ---------------------------------------------------------------------------
# Stage-5 entry point
# ---------------------------------------------------------------------------

class TestRun:
    def _write_tmp_pipeline(self, tmp_path: Path) -> str:
        """Write the minimum on-disk state ``latex_tables.run`` consumes."""
        cfg_text = f"""
models:
  - id: openai/gpt-4o-mini
    name: "GPT-4o mini"
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

    def test_rate_column_uses_tokens_from_generation_records(self, tmp_path):
        import json as _json

        from alienbench.paths import generations_path

        cfg_path = self._write_tmp_pipeline(tmp_path)
        path = generations_path(tmp_path / "data", "openai/gpt-4o-mini", "baseline")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(
            _json.dumps({"id": g, "completion_tokens": n}) + "\n"
            for g, n in (("g1", 100), ("g2", 200))
        ))
        latex_run(cfg_path)
        # Both generations score 2, which gives 2.0 and 1.0 per 100 tokens.
        tex = (tmp_path / "results" / "tab_ward_scores.tex").read_text()
        assert "1.50\\,[" in tex

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


# ---------------------------------------------------------------------------
# _make_paraphrases_table
# ---------------------------------------------------------------------------

def test_paraphrases_table_lists_every_wording_verbatim():
    from alienbench.latex_tables import _make_paraphrases_table
    cfg = Config(
        models=["m/one"],
        judge_models=["j1"],
        prompt_variants=[{"id": "baseline", "label": "Baseline", "text": "Imagine a creature."}],
        prompt_paraphrases=[
            {"id": "baseline", "label": "Baseline", "text": "Imagine a creature."},
            {"id": "baseline_para_a", "label": "Paraphrase A", "text": "Picture an organism."},
        ],
    )
    tex = _make_paraphrases_table(cfg)
    assert "Baseline & Imagine a creature." in tex
    assert "Paraphrase A & Picture an organism." in tex
    assert "\\caption" not in tex and "\\label" not in tex

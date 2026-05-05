"""Tests for parse-failure tracking.

The pipeline records every extraction attempt in one of three states:
``success`` (parseable JSON), ``parse_error`` (judge replied but the reply
could not be parsed), or ``api_error`` (the judge's API call exhausted
retries without writing a record). All three are first-class results: a high
parse-failure rate on a given judge degrades inter-rater reliability and is
itself a reportable benchmark outcome.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


_mock_krippendorff = types.ModuleType("krippendorff")
_mock_krippendorff.alpha = lambda reliability_data, level_of_measurement: 0.85
sys.modules.setdefault("krippendorff", _mock_krippendorff)

from alienbench.analyze import write_parse_failure_analysis  # noqa: E402
from alienbench.paths import (  # noqa: E402
    extractions_path,
    generations_path,
    load_extraction_status,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _Variant:
    def __init__(self, id):
        self.id = id


@pytest.fixture()
def synthetic_data(tmp_path: Path):
    """Three generations under one (judge, model, prompt) cell with one of
    each status: ``success``, ``parse_error`` (judge wrote a record with
    parse_error=True), and ``api_error`` (no record at all)."""
    data_dir = tmp_path / "data"
    model = "openai/gpt-4o-mini"
    judge = "anthropic/claude-3.5-sonnet"
    variant = "baseline"

    _write_jsonl(generations_path(data_dir, model, variant), [
        {"id": "g_ok",  "model": model, "prompt_variant": variant, "response": "..."},
        {"id": "g_pe",  "model": model, "prompt_variant": variant, "response": "..."},
        {"id": "g_api", "model": model, "prompt_variant": variant, "response": "..."},
    ])

    _write_jsonl(extractions_path(data_dir, judge, model, variant), [
        {"generation_id": "g_ok", "subject_model": model,
         "prompt_variant": variant, "judge_model": judge,
         "features": {"symmetry": {"is_departure": True}},
         "parse_error": False},
        {"generation_id": "g_pe", "subject_model": model,
         "prompt_variant": variant, "judge_model": judge,
         "features": None, "parse_error": True,
         "raw_response": "garbage"},
        # No record for g_api: the judge's API call failed.
    ])
    return data_dir, [judge], [model], [_Variant(variant)]


# ---------------------------------------------------------------------------
# load_extraction_status
# ---------------------------------------------------------------------------

class TestLoadExtractionStatus:
    def test_classifies_each_generation(self, synthetic_data):
        data_dir, judges, models, variants = synthetic_data
        df = load_extraction_status(data_dir, judges, models, variants)

        statuses = dict(zip(df["generation_id"], df["status"]))
        assert statuses == {
            "g_ok":  "success",
            "g_pe":  "parse_error",
            "g_api": "api_error",
        }

    def test_empty_when_no_generations(self, tmp_path: Path):
        df = load_extraction_status(
            tmp_path / "data", ["anthropic/claude-3.5-sonnet"],
            ["openai/gpt-4o-mini"], [_Variant("baseline")],
        )
        assert df.empty

    def test_two_judges_yield_one_row_per_judge(self, tmp_path: Path):
        """A generation with two judges should produce two status rows."""
        data_dir = tmp_path / "data"
        model = "openai/gpt-4o-mini"
        variant = "baseline"

        _write_jsonl(generations_path(data_dir, model, variant), [
            {"id": "g1", "model": model, "prompt_variant": variant, "response": "..."},
        ])
        _write_jsonl(extractions_path(data_dir, "j1", model, variant), [
            {"generation_id": "g1", "subject_model": model,
             "prompt_variant": variant, "judge_model": "j1",
             "features": {"x": 1}, "parse_error": False},
        ])
        # j2 never wrote a record => api_error.
        df = load_extraction_status(
            data_dir, ["j1", "j2"], [model], [_Variant(variant)]
        )
        rows = sorted(zip(df["judge_model"], df["status"]))
        assert rows == [("j1", "success"), ("j2", "api_error")]


# ---------------------------------------------------------------------------
# write_parse_failure_analysis
# ---------------------------------------------------------------------------

class TestWriteParseFailureAnalysis:
    def test_csv_and_summary_lines(self, synthetic_data, tmp_path: Path):
        data_dir, judges, models, variants = synthetic_data
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        status_df = load_extraction_status(data_dir, judges, models, variants)
        lines = write_parse_failure_analysis(status_df, results_dir)

        csv_path = results_dir / "table_extraction_status.csv"
        assert csv_path.exists()
        cell = pd.read_csv(csv_path)
        # One cell: one judge × one model × one prompt
        assert len(cell) == 1
        row = cell.iloc[0]
        assert row["n_generations"] == 3
        assert row["success"] == 1
        assert row["parse_error"] == 1
        assert row["api_error"] == 1
        assert abs(row["success_rate"] - 1 / 3) < 1e-9

        joined = "\n".join(lines)
        assert "Extraction Reliability" in joined
        assert "anthropic/claude-3.5-sonnet" in joined
        # Both kinds of failure are surfaced explicitly.
        assert "parse_error" in joined and "api_error" in joined
        # Sub-95% cell is flagged in the summary so reviewers cannot miss it.
        assert "below 95% extraction success" in joined

    def test_handles_empty_input(self, tmp_path: Path):
        lines = write_parse_failure_analysis(pd.DataFrame(), tmp_path)
        assert any("no extraction records" in l.lower() for l in lines)

    def test_perfect_run_does_not_flag(self, tmp_path: Path):
        data_dir = tmp_path / "data"
        model = "m"
        variant = "baseline"
        _write_jsonl(generations_path(data_dir, model, variant), [
            {"id": "g1", "model": model, "prompt_variant": variant, "response": "..."},
            {"id": "g2", "model": model, "prompt_variant": variant, "response": "..."},
        ])
        _write_jsonl(extractions_path(data_dir, "j", model, variant), [
            {"generation_id": "g1", "subject_model": model,
             "prompt_variant": variant, "judge_model": "j",
             "features": {"x": 1}, "parse_error": False},
            {"generation_id": "g2", "subject_model": model,
             "prompt_variant": variant, "judge_model": "j",
             "features": {"x": 1}, "parse_error": False},
        ])
        status_df = load_extraction_status(data_dir, ["j"], [model], [_Variant(variant)])

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        lines = write_parse_failure_analysis(status_df, results_dir)
        joined = "\n".join(lines)
        assert "100.0%" in joined
        assert "below 95%" not in joined

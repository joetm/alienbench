"""Unit tests for the human validation harness."""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# Mock krippendorff before importing alienbench — matches test_pipeline.py
_mock_krippendorff = types.ModuleType("krippendorff")
_mock_krippendorff.alpha = lambda reliability_data, level_of_measurement: 0.85
sys.modules.setdefault("krippendorff", _mock_krippendorff)


CREATURE_TEXT = "A luminescent, radially symmetric drifter that metabolises geothermal flux."


@pytest.fixture()
def populated_cfg(tmp_path: Path):
    """Write a minimal config and seed two generations across two strata."""
    from alienbench.paths import generations_path

    config_text = f"""
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
  - id: departure_primed
    label: Departure-primed
    text: "Imagine a creature as different from Earth life as possible."
data_dir: {tmp_path}/data
results_dir: {tmp_path}/results
api_key_env: OPENROUTER_API_KEY
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(config_text)

    data_dir = tmp_path / "data"
    for variant_id in ("baseline", "departure_primed"):
        path = generations_path(data_dir, "openai/gpt-4o-mini", variant_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for i in range(3):
                f.write(json.dumps({
                    "id": f"{variant_id}-{i}",
                    "model": "openai/gpt-4o-mini",
                    "prompt_variant": variant_id,
                    "response": CREATURE_TEXT,
                    "completion_tokens": 120,
                }) + "\n")
    return str(cfg_path), tmp_path


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

class TestSample:
    def test_emits_ten_rows_per_generation(self, populated_cfg):
        from alienbench import human
        from alienbench.dimensions import DIMENSION_IDS

        cfg_path, tmp_path = populated_cfg
        out = human.sample(cfg_path, samples_per_stratum=2, seed=7)
        assert out.exists()

        with out.open() as fh:
            rows = list(csv.DictReader(fh))

        gens = {r["generation_id"] for r in rows}
        # 2 strata x 2 samples = 4 generations, 10 dims each = 40 rows
        assert len(gens) == 4
        assert len(rows) == 40
        for gen_id in gens:
            gen_rows = [r for r in rows if r["generation_id"] == gen_id]
            assert {r["dimension"] for r in gen_rows} == set(DIMENSION_IDS)
            assert all(r["is_departure"] == "" for r in gen_rows)

    def test_seed_is_deterministic(self, populated_cfg):
        from alienbench import human

        cfg_path, tmp_path = populated_cfg
        out_a = human.sample(cfg_path, samples_per_stratum=2, seed=123,
                             out_path=tmp_path / "a.csv")
        out_b = human.sample(cfg_path, samples_per_stratum=2, seed=123,
                             out_path=tmp_path / "b.csv")
        assert out_a.read_text() == out_b.read_text()


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def _fill_csv(src: Path, dst: Path, labels: dict[tuple[str, str], int]) -> None:
    with src.open() as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        key = (row["generation_id"], row["dimension"])
        row["is_departure"] = str(labels.get(key, 0))
        row["reasoning"] = "auto"
    with dst.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


class TestIngest:
    def test_ingest_round_trip(self, populated_cfg):
        from alienbench import human

        cfg_path, tmp_path = populated_cfg
        template = human.sample(cfg_path, samples_per_stratum=2, seed=7)
        filled = tmp_path / "filled.csv"
        # Label symmetry and metabolism as departures, everything else earth-typical.
        labels = {}
        with template.open() as fh:
            for row in csv.DictReader(fh):
                labels[(row["generation_id"], row["dimension"])] = (
                    1 if row["dimension"] in ("symmetry", "metabolism") else 0
                )
        _fill_csv(template, filled, labels)

        counts = human.ingest(cfg_path, "alice", filled)
        assert counts["generations"] == 4
        assert counts["strata"] == 2

        ward_path = (
            tmp_path / "data" / "scores" / "human__alice"
            / "openai__gpt-4o-mini" / "baseline" / "ward_scores.jsonl"
        )
        assert ward_path.exists()
        records = [json.loads(ln) for ln in ward_path.read_text().splitlines() if ln.strip()]
        assert len(records) == 2
        for rec in records:
            assert rec["judge_model"] == "human/alice"
            assert rec["ward_score"] == 2  # symmetry + metabolism
            assert rec["per_dimension"]["symmetry"] == 1
            assert rec["per_dimension"]["locomotion"] == 0

    def test_rejects_missing_dimension(self, populated_cfg):
        from alienbench import human

        cfg_path, tmp_path = populated_cfg
        template = human.sample(cfg_path, samples_per_stratum=1, seed=7)
        # Drop the 'cognition' row for one generation.
        with template.open() as fh:
            rows = [r for r in csv.DictReader(fh)]
        first_gen = rows[0]["generation_id"]
        rows = [r for r in rows
                if not (r["generation_id"] == first_gen and r["dimension"] == "cognition")]
        for r in rows:
            r["is_departure"] = "0"
        filled = tmp_path / "filled_missing.csv"
        with filled.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        with pytest.raises(ValueError, match="missing dimensions"):
            human.ingest(cfg_path, "bob", filled)

    def test_rejects_bad_is_departure(self, populated_cfg):
        from alienbench import human

        cfg_path, tmp_path = populated_cfg
        template = human.sample(cfg_path, samples_per_stratum=1, seed=7)
        with template.open() as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            r["is_departure"] = "maybe"
        filled = tmp_path / "filled_bad.csv"
        with filled.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        with pytest.raises(ValueError, match="is_departure"):
            human.ingest(cfg_path, "bob", filled)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_emits_table_and_summary_block(self, populated_cfg, monkeypatch):
        from alienbench import human
        from alienbench.paths import ward_scores_path
        import time as _time

        cfg_path, tmp_path = populated_cfg

        # Seed a fake judge ward-scores file so the analyse step has something to
        # compare humans against. The judge labels every generation as ward_score=2
        # with symmetry and metabolism flagged — matching the human labels below.
        judge_alias = "openai/gpt-4o-mini"
        data_dir = tmp_path / "data"
        per_dim_zero = {d: 0 for d in __import__("alienbench.dimensions", fromlist=["DIMENSION_IDS"]).DIMENSION_IDS}
        per_dim_dep = {**per_dim_zero, "symmetry": 1, "metabolism": 1}
        for variant_id in ("baseline", "departure_primed"):
            for i in range(3):
                wp = ward_scores_path(data_dir, judge_alias, "openai/gpt-4o-mini", variant_id)
                wp.parent.mkdir(parents=True, exist_ok=True)
                with wp.open("a") as f:
                    f.write(json.dumps({
                        "generation_id": f"{variant_id}-{i}",
                        "judge_model": judge_alias,
                        "judge_model_resolved": judge_alias,
                        "judge_call_id": "mock",
                        "subject_model": "openai/gpt-4o-mini",
                        "prompt_variant": variant_id,
                        "ward_score": 2,
                        "per_dimension": per_dim_dep,
                        "timestamp": _time.time(),
                    }) + "\n")

        # Human ingest with matching labels
        template = human.sample(cfg_path, samples_per_stratum=2, seed=7)
        filled = tmp_path / "filled.csv"
        with template.open() as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            r["is_departure"] = "1" if r["dimension"] in ("symmetry", "metabolism") else "0"
        with filled.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        human.ingest(cfg_path, "alice", filled)

        out = human.analyze(cfg_path)
        assert out is not None and out.exists()
        with out.open() as fh:
            table_rows = list(csv.DictReader(fh))
        measures = {r["Measure"] for r in table_rows}
        assert "Ward Total (0–10)" in measures
        assert any(m.startswith("Ward: symmetry") for m in measures)

        summary = (tmp_path / "results" / "summary.txt").read_text()
        assert "Human Validation" in summary
        assert "human/alice" in summary

    def test_skips_when_no_humans(self, populated_cfg, caplog):
        from alienbench import human

        cfg_path, _ = populated_cfg
        result = human.analyze(cfg_path)
        assert result is None

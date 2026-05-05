"""Tests for the run-manifest writer.

The manifest is part of the reproducibility surface: it pins the config that
produced ``results/`` and lets reviewers diff two runs to verify they used
identical settings. These tests guard the structural invariants the
downstream tooling assumes (stable hashing across runs, JSON-array history,
per-stage append).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

# Mock krippendorff before any alienbench import.
_mock_krippendorff = types.ModuleType("krippendorff")
_mock_krippendorff.alpha = lambda reliability_data, level_of_measurement: 0.85
sys.modules.setdefault("krippendorff", _mock_krippendorff)

from alienbench.config import Config  # noqa: E402
from alienbench.provenance import config_hash, write_run_manifest  # noqa: E402


def _make_config(**overrides) -> Config:
    base = dict(
        models=["openai/gpt-4o"],
        judge_models=["anthropic/claude-3.5-sonnet"],
        prompt_variants=[
            {"id": "baseline", "label": "Baseline", "text": "imagine a creature."}
        ],
        samples_per_condition=3,
        data_dir="data",
        results_dir="results",
    )
    base.update(overrides)
    return Config(**base)


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------

class TestConfigHash:
    def test_is_deterministic_within_run(self):
        cfg = _make_config()
        assert config_hash(cfg) == config_hash(cfg)

    def test_changes_when_a_field_changes(self):
        a = _make_config()
        b = _make_config(samples_per_condition=4)
        assert config_hash(a) != config_hash(b)

    def test_ignores_allowed_providers(self):
        """`allowed_providers` is excluded from the hash so a routing-only
        change does not invalidate the rest of the manifest."""
        a = _make_config()
        b = _make_config(allowed_providers=["openai"])
        assert config_hash(a) == config_hash(b)

    def test_returns_64_char_hex(self):
        h = config_hash(_make_config())
        assert len(h) == 64
        int(h, 16)  # raises if non-hex


# ---------------------------------------------------------------------------
# write_run_manifest
# ---------------------------------------------------------------------------

class TestWriteRunManifest:
    def test_creates_file_with_single_entry(self, tmp_path: Path):
        path = write_run_manifest(_make_config(), tmp_path / "results", stage="analyze")
        assert path.exists()
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert entry["stage"] == "analyze"
        assert "config_hash" in entry
        assert "config" in entry
        assert "packages" in entry
        assert "python_version" in entry
        assert entry["timestamp_utc"].endswith("Z")

    def test_appends_history_across_stages(self, tmp_path: Path):
        cfg = _make_config()
        results = tmp_path / "results"
        write_run_manifest(cfg, results, stage="analyze")
        write_run_manifest(cfg, results, stage="latex")
        data = json.loads((results / "run_manifest.json").read_text())
        assert [e["stage"] for e in data] == ["analyze", "latex"]
        # Same config means same hash on both rows.
        assert data[0]["config_hash"] == data[1]["config_hash"]

    def test_recovers_from_a_corrupted_manifest(self, tmp_path: Path):
        results = tmp_path / "results"
        results.mkdir()
        (results / "run_manifest.json").write_text("not valid json {")

        write_run_manifest(_make_config(), results, stage="analyze")
        data = json.loads((results / "run_manifest.json").read_text())
        # Corruption is silently replaced; the new entry is the only entry.
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["stage"] == "analyze"

    def test_normalises_legacy_dict_history(self, tmp_path: Path):
        results = tmp_path / "results"
        results.mkdir()
        legacy = {"stage": "analyze", "config_hash": "deadbeef"}
        (results / "run_manifest.json").write_text(json.dumps(legacy))

        write_run_manifest(_make_config(), results, stage="latex")
        data = json.loads((results / "run_manifest.json").read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["stage"] == "analyze"
        assert data[1]["stage"] == "latex"

    def test_creates_results_dir_if_absent(self, tmp_path: Path):
        target = tmp_path / "fresh" / "results"
        assert not target.exists()
        write_run_manifest(_make_config(), target, stage="analyze")
        assert (target / "run_manifest.json").exists()

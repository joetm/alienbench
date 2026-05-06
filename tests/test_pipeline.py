"""End-to-end pipeline test using mocked API responses."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the krippendorff module before any alienbench import
# ---------------------------------------------------------------------------

_mock_krippendorff = types.ModuleType("krippendorff")
_mock_krippendorff.alpha = lambda reliability_data, level_of_measurement: 0.85
sys.modules.setdefault("krippendorff", _mock_krippendorff)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_CREATURE = (
    "The Velhari is a radially symmetric organism that drifts through subsurface "
    "ammonia lakes. It has no distinct head; instead, chemoreceptive cilia distributed "
    "across its surface detect dissolved minerals. It absorbs energy directly from "
    "geothermal vents via crystalline filaments embedded in its membrane. Reproduction "
    "occurs by fragmentation — pieces detach and develop independently. The colony "
    "communicates via modulated pressure waves through the liquid."
)


def _make_extraction_response() -> str:
    """Return a valid JSON extraction response for FAKE_CREATURE."""
    features = {
        "symmetry":         {"feature_described": "radial symmetry", "is_departure": True,  "reasoning": "Radial, not bilateral."},
        "sensory_organs":   {"feature_described": "distributed chemoreceptive cilia", "is_departure": True,  "reasoning": "No head-based sensory organs."},
        "locomotion":       {"feature_described": "passive drifting", "is_departure": True,  "reasoning": "No limbs; drifts passively."},
        "body_plan":        {"feature_described": "no distinct head or torso", "is_departure": True,  "reasoning": "Lacks canonical head/torso/limb structure."},
        "skin_covering":    {"feature_described": "crystalline membrane filaments", "is_departure": True,  "reasoning": "Crystalline, not biological tissue."},
        "reproduction":     {"feature_described": "fragmentation", "is_departure": True,  "reasoning": "Asexual fragmentation, not binary-sex reproduction."},
        "metabolism":       {"feature_described": "geothermal energy absorption", "is_departure": True,  "reasoning": "Chemosynthetic, not heterotrophic."},
        "communication":    {"feature_described": "modulated pressure waves", "is_departure": False, "reasoning": "Pressure waves are a physical signal, analogous to sound."},
        "habitat":          {"feature_described": "subsurface ammonia lakes", "is_departure": True,  "reasoning": "Subsurface, not surface land or water."},
        "cognition":        {"feature_described": "not described", "is_departure": False, "reasoning": "No cognitive architecture mentioned; defaulting to Earth-typical."},
    }
    return json.dumps(features)


_FAKE_EXTRACTION_JSON = _make_extraction_response()


@pytest.fixture()
def tmp_config(tmp_path: Path):
    """Write a minimal config.yaml into a temp directory and return its path."""
    config_text = f"""
models:
  - openai/gpt-4o-mini
judge_models:
  - openai/gpt-4o-mini
samples_per_condition: 3
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
    cfg_path.write_text(config_text)
    return str(cfg_path)


# ---------------------------------------------------------------------------
# Mock factory
# ---------------------------------------------------------------------------

def make_mock_client(generation_response: str, extraction_response: str):
    """Return a mock OpenRouterClient whose .complete() returns different values per stage."""
    from alienbench.client import Response

    call_counter = {"n": 0}

    def fake_complete(model, prompt, temperature, max_tokens, system=None, seed=None):
        call_counter["n"] += 1
        if temperature == 0.0:
            return Response(extraction_response, model, 100, 200, generation_id="mock-call-id")
        return Response(generation_response, model, 50, 100, generation_id="mock-call-id")

    mock = MagicMock()
    mock.complete.side_effect = fake_complete
    return mock


def make_mock_judge(extraction_response: str):
    """Return a factory matching `make_judge(alias, cfg)` that yields a mock judge."""
    from alienbench.client import Response

    def fake_complete(prompt, temperature, max_tokens, system=None):
        return Response(extraction_response, "mock-resolved-model", 100, 200,
                        generation_id="mock-judge-call-id")

    def factory(alias, cfg):
        m = MagicMock()
        m.complete.side_effect = fake_complete
        return m

    return factory


# ---------------------------------------------------------------------------
# Module-level helpers for multiprocessing.spawn workers
# ---------------------------------------------------------------------------

class _SlowFakeClient:
    """Pickle-friendly stand-in for OpenRouterClient used by spawned workers.

    A small per-call sleep widens the window for races between the two
    workers, increasing the chance that the test catches any concurrency bug.
    """

    def __init__(self, cfg):
        self._cfg = cfg

    def complete(self, model, prompt, temperature, max_tokens, system=None, seed=None):
        import time as _time

        from alienbench.client import Response

        _time.sleep(0.01)
        return Response(FAKE_CREATURE, model, 50, 100, generation_id=f"mock-{seed}")


def _parallel_worker(config_path: str) -> None:
    import os as _os

    _os.environ["OPENROUTER_API_KEY"] = "test-key"
    from alienbench import generate

    generate.OpenRouterClient = _SlowFakeClient  # type: ignore[assignment]
    generate.run(config_path)


class _SlowFakeJudge:
    """Pickle-friendly stand-in for the extract-stage judge clients.

    A small per-call sleep widens the race window between two parallel
    extract workers, mirroring :class:`_SlowFakeClient` for generate.
    """

    def complete(self, prompt, temperature, max_tokens, system=None):
        import time as _time

        from alienbench.client import Response

        _time.sleep(0.01)
        return Response(_FAKE_EXTRACTION_JSON, "mock-resolved-model", 100, 200,
                        generation_id="mock-judge-call-id")


def _slow_judge_factory(alias, cfg):
    return _SlowFakeJudge()


def _extract_parallel_worker(config_path: str) -> None:
    import os as _os

    _os.environ["OPENROUTER_API_KEY"] = "test-key"
    from alienbench import extract

    extract.make_judge = _slow_judge_factory  # type: ignore[assignment]
    extract.run(config_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateStage:
    def test_creates_jsonl_with_correct_records(self, tmp_config, tmp_path, monkeypatch):
        from alienbench import generate

        mock_client = make_mock_client(FAKE_CREATURE, "")
        monkeypatch.setattr(generate, "OpenRouterClient", lambda cfg: mock_client)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        generate.run(tmp_config)

        out = Path(tmp_path) / "data" / "generations" / "openai__gpt-4o-mini" / "baseline" / "responses.jsonl"
        assert out.exists(), "responses.jsonl not created"
        records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert len(records) == 3
        for rec in records:
            assert rec["response"] == FAKE_CREATURE
            assert rec["model"] == "openai/gpt-4o-mini"
            assert rec["prompt_variant"] == "baseline"
            assert "id" in rec
            assert "prompt_tokens" in rec
            assert "completion_tokens" in rec
            assert "duration_seconds" in rec
            assert isinstance(rec["duration_seconds"], (int, float))
            assert rec["duration_seconds"] >= 0
            assert "sample_index" in rec
            assert isinstance(rec["sample_index"], int)
            assert 0 <= rec["sample_index"] < 3
        assert {rec["sample_index"] for rec in records} == {0, 1, 2}

    def test_checkpointing_skips_completed(self, tmp_config, tmp_path, monkeypatch):
        from alienbench import generate

        mock_client = make_mock_client(FAKE_CREATURE, "")
        monkeypatch.setattr(generate, "OpenRouterClient", lambda cfg: mock_client)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        generate.run(tmp_config)
        call_count_first = mock_client.complete.call_count

        generate.run(tmp_config)
        # No new calls on second run
        assert mock_client.complete.call_count == call_count_first

    def test_stale_reservation_reaped(self, tmp_config, tmp_path, monkeypatch):
        """A reservation file owned by a dead PID should be reaped and the index regenerated."""
        from alienbench import generate
        from alienbench.paths import reservations_dir

        mock_client = make_mock_client(FAKE_CREATURE, "")
        monkeypatch.setattr(generate, "OpenRouterClient", lambda cfg: mock_client)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        data_dir = Path(tmp_path) / "data"
        res_dir = reservations_dir(data_dir, "openai/gpt-4o-mini", "baseline")
        res_dir.mkdir(parents=True, exist_ok=True)
        # PID 999999 is essentially guaranteed not to exist on a normal Linux system.
        (res_dir / "1").write_text("999999 2000-01-01T00:00:00+00:00\n")

        generate.run(tmp_config)

        out = Path(tmp_path) / "data" / "generations" / "openai__gpt-4o-mini" / "baseline" / "responses.jsonl"
        records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert {rec["sample_index"] for rec in records} == {0, 1, 2}
        assert not any(res_dir.iterdir()), "Reservation directory should be empty after run"

    def test_parallel_workers_no_duplicates(self, tmp_path, monkeypatch):
        """Two processes running generate against the same data dir produce no duplicate (model, variant, sample_index)."""
        import multiprocessing as mp

        cfg_text = f"""
models:
  - openai/gpt-4o-mini
  - anthropic/claude-3-haiku
judge_models:
  - openai/gpt-4o-mini
samples_per_condition: 6
temperature: 1.0
max_tokens: 400
prompt_variants:
  - id: baseline
    label: Baseline
    text: "Imagine a creature."
  - id: detailed
    label: Detailed
    text: "Imagine a creature in detail."
data_dir: {tmp_path}/data
results_dir: {tmp_path}/results
api_key_env: OPENROUTER_API_KEY
openrouter_base_url: https://openrouter.ai/api/v1
"""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(cfg_text)

        ctx = mp.get_context("spawn")
        proc_a = ctx.Process(target=_parallel_worker, args=(str(cfg_path),))
        proc_b = ctx.Process(target=_parallel_worker, args=(str(cfg_path),))
        proc_a.start()
        proc_b.start()
        proc_a.join(timeout=60)
        proc_b.join(timeout=60)
        assert proc_a.exitcode == 0, f"worker A failed (exitcode={proc_a.exitcode})"
        assert proc_b.exitcode == 0, f"worker B failed (exitcode={proc_b.exitcode})"

        gen_root = Path(tmp_path) / "data" / "generations"
        all_keys = []
        for path in gen_root.rglob("responses.jsonl"):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                all_keys.append((rec["model"], rec["prompt_variant"], rec["sample_index"]))
        # 2 models * 2 variants * 6 samples = 24 records, all unique
        assert len(all_keys) == 24, f"expected 24 records, got {len(all_keys)}"
        assert len(set(all_keys)) == 24, f"duplicates detected: {len(all_keys) - len(set(all_keys))}"

        for path in gen_root.rglob("_reservations"):
            assert not any(path.iterdir()), f"leftover reservation files in {path}"


class TestExtractStage:
    def _run_generate(self, tmp_config, tmp_path, monkeypatch):
        from alienbench import generate
        mock_client = make_mock_client(FAKE_CREATURE, "")
        monkeypatch.setattr(generate, "OpenRouterClient", lambda cfg: mock_client)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        generate.run(tmp_config)

    def test_creates_features_jsonl(self, tmp_config, tmp_path, monkeypatch):
        from alienbench import extract, generate

        self._run_generate(tmp_config, tmp_path, monkeypatch)

        monkeypatch.setattr(extract, "make_judge", make_mock_judge(_make_extraction_response()))
        extract.run(tmp_config)

        out = (
            Path(tmp_path) / "data" / "extractions"
            / "openai__gpt-4o-mini" / "openai__gpt-4o-mini" / "baseline" / "features.jsonl"
        )
        assert out.exists()
        records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert len(records) == 3
        for rec in records:
            assert not rec["parse_error"]
            assert "symmetry" in rec["features"]
            assert isinstance(rec["features"]["symmetry"]["is_departure"], bool)
            assert rec["judge_model_resolved"] == "mock-resolved-model"
            assert rec["judge_call_id"] == "mock-judge-call-id"
            assert "prompt_tokens" in rec
            assert "completion_tokens" in rec
            assert "n_parse_attempts" in rec and rec["n_parse_attempts"] == 1
            assert "duration_seconds" in rec
            assert rec["duration_seconds"] >= 0

    def test_stale_extract_reservation_reaped(self, tmp_config, tmp_path, monkeypatch):
        """A reservation file owned by a dead PID should be reaped and the extraction regenerated."""
        from alienbench import extract, generate
        from alienbench.paths import extraction_reservations_dir

        self._run_generate(tmp_config, tmp_path, monkeypatch)

        # Pick a real generation_id from the responses to seed the dead reservation.
        gen_path = (
            Path(tmp_path) / "data" / "generations" / "openai__gpt-4o-mini"
            / "baseline" / "responses.jsonl"
        )
        gen_records = [
            json.loads(line) for line in gen_path.read_text().splitlines() if line.strip()
        ]
        assert len(gen_records) == 3
        target_id = gen_records[0]["id"]

        data_dir = Path(tmp_path) / "data"
        res_dir = extraction_reservations_dir(
            data_dir, "openai/gpt-4o-mini", "openai/gpt-4o-mini", "baseline"
        )
        res_dir.mkdir(parents=True, exist_ok=True)
        # PID 999999 is essentially guaranteed not to exist on a normal Linux system.
        (res_dir / target_id).write_text("999999 2000-01-01T00:00:00+00:00\n")

        monkeypatch.setattr(extract, "make_judge", make_mock_judge(_make_extraction_response()))
        extract.run(tmp_config)

        out = (
            Path(tmp_path) / "data" / "extractions"
            / "openai__gpt-4o-mini" / "openai__gpt-4o-mini" / "baseline" / "features.jsonl"
        )
        records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert {rec["generation_id"] for rec in records} == {g["id"] for g in gen_records}
        assert not any(res_dir.iterdir()), "Reservation directory should be empty after run"

    def test_parallel_extract_workers_no_duplicates(self, tmp_path, monkeypatch):
        """Two processes running extract against the same data dir produce no duplicate generation_ids."""
        import multiprocessing as mp

        cfg_text = f"""
models:
  - openai/gpt-4o-mini
  - anthropic/claude-3-haiku
judge_models:
  - openai/gpt-4o-mini
samples_per_condition: 6
temperature: 1.0
max_tokens: 400
prompt_variants:
  - id: baseline
    label: Baseline
    text: "Imagine a creature."
  - id: detailed
    label: Detailed
    text: "Imagine a creature in detail."
data_dir: {tmp_path}/data
results_dir: {tmp_path}/results
api_key_env: OPENROUTER_API_KEY
openrouter_base_url: https://openrouter.ai/api/v1
"""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(cfg_text)

        # Seed generations once via a single in-process generate run.
        from alienbench import generate
        gen_client = make_mock_client(FAKE_CREATURE, "")
        monkeypatch.setattr(generate, "OpenRouterClient", lambda cfg: gen_client)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        generate.run(str(cfg_path))

        ctx = mp.get_context("spawn")
        proc_a = ctx.Process(target=_extract_parallel_worker, args=(str(cfg_path),))
        proc_b = ctx.Process(target=_extract_parallel_worker, args=(str(cfg_path),))
        proc_a.start()
        proc_b.start()
        proc_a.join(timeout=60)
        proc_b.join(timeout=60)
        assert proc_a.exitcode == 0, f"worker A failed (exitcode={proc_a.exitcode})"
        assert proc_b.exitcode == 0, f"worker B failed (exitcode={proc_b.exitcode})"

        ext_root = Path(tmp_path) / "data" / "extractions"
        all_keys = []
        for path in ext_root.rglob("features.jsonl"):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                all_keys.append((rec["judge_model"], rec["subject_model"],
                                 rec["prompt_variant"], rec["generation_id"]))
        # 1 judge * 2 models * 2 variants * 6 samples = 24 records, all unique
        assert len(all_keys) == 24, f"expected 24 records, got {len(all_keys)}"
        assert len(set(all_keys)) == 24, f"duplicates detected: {len(all_keys) - len(set(all_keys))}"

        for path in ext_root.rglob("_reservations"):
            assert not any(path.iterdir()), f"leftover reservation files in {path}"

    def test_parse_failure_not_stored(self, tmp_config, tmp_path, monkeypatch):
        """A parse failure must not be marked complete.

        When all parse attempts fail for a generation, ``extract.run`` must
        not write a record. The next ``extract`` invocation will re-query
        the judge for the same id.
        """
        from alienbench import extract, generate

        self._run_generate(tmp_config, tmp_path, monkeypatch)

        monkeypatch.setattr(extract, "make_judge", make_mock_judge("not valid json {{{{"))
        extract.run(tmp_config)  # Should not raise

        out = (
            Path(tmp_path) / "data" / "extractions"
            / "openai__gpt-4o-mini" / "openai__gpt-4o-mini" / "baseline" / "features.jsonl"
        )
        if out.exists():
            records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
            assert records == [], f"expected no records on parse failure, got {records!r}"


class TestParseFeaturesSchema:
    """Direct unit tests for ``parse_features`` schema validation (H1/H2)."""

    def test_accepts_valid_response(self):
        from alienbench.extract import parse_features

        out = parse_features(_make_extraction_response())
        assert out is not None
        assert "symmetry" in out
        assert out["symmetry"]["is_departure"] is True

    def test_accepts_integer_zero_one(self):
        from alienbench.extract import parse_features
        from alienbench.dimensions import DIMENSION_IDS

        # Build a payload where every is_departure is the integer 0 or 1
        features = {d: {"is_departure": (1 if i % 2 == 0 else 0)}
                    for i, d in enumerate(DIMENSION_IDS)}
        out = parse_features(json.dumps(features))
        assert out is not None

    def test_rejects_string_is_departure(self):
        """A judge returning ``"is_departure": "false"`` must be a parse failure.

        Python's ``bool("false")`` is True, so accepting strings would silently
        score every "false" as a departure.
        """
        from alienbench.extract import parse_features
        from alienbench.dimensions import DIMENSION_IDS

        features = {d: {"is_departure": "false"} for d in DIMENSION_IDS}
        assert parse_features(json.dumps(features)) is None

    def test_rejects_null_is_departure(self):
        from alienbench.extract import parse_features
        from alienbench.dimensions import DIMENSION_IDS

        features = {d: {"is_departure": None} for d in DIMENSION_IDS}
        assert parse_features(json.dumps(features)) is None

    def test_rejects_missing_is_departure_field(self):
        from alienbench.extract import parse_features
        from alienbench.dimensions import DIMENSION_IDS

        features = {d: {"feature_described": "x"} for d in DIMENSION_IDS}
        assert parse_features(json.dumps(features)) is None

    def test_rejects_non_dict_dimension(self):
        from alienbench.extract import parse_features
        from alienbench.dimensions import DIMENSION_IDS

        features = {d: True for d in DIMENSION_IDS}  # primitive instead of dict
        assert parse_features(json.dumps(features)) is None

    def test_unwraps_single_element_array(self):
        """Gemini sometimes wraps the dict in a single-element JSON array.

        ``parse_features`` should unwrap ``[{...}]`` to ``{...}`` so the
        per-dimension schema check still applies. A multi-element array
        must remain a parse failure.
        """
        from alienbench.extract import parse_features

        wrapped = "[" + _make_extraction_response() + "]"
        out = parse_features(wrapped)
        assert out is not None
        assert out["symmetry"]["is_departure"] is True

        # Multi-element arrays are still rejected.
        multi = (
            "["
            + _make_extraction_response()
            + ","
            + _make_extraction_response()
            + "]"
        )
        assert parse_features(multi) is None

    def test_rejects_top_level_array(self):
        from alienbench.extract import parse_features

        assert parse_features("[1, 2, 3]") is None

    def test_rejects_missing_dimension(self):
        from alienbench.extract import parse_features
        from alienbench.dimensions import DIMENSION_IDS

        features = {d: {"is_departure": True} for d in DIMENSION_IDS[:-1]}
        assert parse_features(json.dumps(features)) is None


class TestScoreStage:
    def _run_generate_and_extract(self, tmp_config, tmp_path, monkeypatch):
        from alienbench import generate, extract
        gen_client = make_mock_client(FAKE_CREATURE, "")
        monkeypatch.setattr(generate, "OpenRouterClient", lambda cfg: gen_client)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        generate.run(tmp_config)

        monkeypatch.setattr(extract, "make_judge", make_mock_judge(_make_extraction_response()))
        extract.run(tmp_config)

    def test_ward_scores_computed_correctly(self, tmp_config, tmp_path, monkeypatch):
        from alienbench import score

        self._run_generate_and_extract(tmp_config, tmp_path, monkeypatch)

        score.run(tmp_config)

        ward_path = (
            Path(tmp_path) / "data" / "scores"
            / "openai__gpt-4o-mini" / "openai__gpt-4o-mini" / "baseline" / "ward_scores.jsonl"
        )
        assert ward_path.exists()
        records = [json.loads(line) for line in ward_path.read_text().splitlines() if line.strip()]
        assert len(records) == 3
        # 8 departures in our fake extraction (symmetry, sensory, locomotion, body_plan,
        # skin_covering, reproduction, metabolism, habitat = 8; communication=0, cognition=0)
        for rec in records:
            assert rec["ward_score"] == 8
            assert rec["per_dimension"]["symmetry"] == 1
            assert rec["per_dimension"]["communication"] == 0

    def test_score_skips_malformed_extraction(self, tmp_config, tmp_path, monkeypatch):
        """Defence-in-depth: a malformed extraction record must not crash score.run.

        Simulates the case where parse_features somehow accepted a record whose
        inner structure is broken (legacy data, manual edit, schema drift). The
        score stage should log and skip, not raise.
        """
        from alienbench import extract, generate, score
        from alienbench.paths import extractions_path

        self._run_generate_and_extract(tmp_config, tmp_path, monkeypatch)

        # Hand-craft a malformed extraction record (parse_error=False but
        # features missing is_departure on one dimension). Append it to the
        # existing features.jsonl so the score stage will encounter it.
        data_dir = Path(tmp_path) / "data"
        ext_path = extractions_path(
            data_dir, "openai/gpt-4o-mini", "openai/gpt-4o-mini", "baseline"
        )
        bad_record = {
            "generation_id": "broken-gen-id",
            "judge_model": "openai/gpt-4o-mini",
            "judge_model_resolved": "openai/gpt-4o-mini",
            "judge_call_id": "mock",
            "subject_model": "openai/gpt-4o-mini",
            "prompt_variant": "baseline",
            "timestamp": 0.0,
            "features": {"symmetry": None},  # break the inner schema
            "parse_error": False,
            "raw_response": None,
        }
        with ext_path.open("a") as f:
            f.write(json.dumps(bad_record) + "\n")

        # score.run must not raise — it should log+skip the bad record.
        score.run(tmp_config)

        ward_path = (
            data_dir / "scores" / "openai__gpt-4o-mini"
            / "openai__gpt-4o-mini" / "baseline" / "ward_scores.jsonl"
        )
        records = [
            json.loads(line) for line in ward_path.read_text().splitlines() if line.strip()
        ]
        # Three valid scores from the fixture; the malformed extraction is dropped.
        assert len(records) == 3
        assert "broken-gen-id" not in {r["generation_id"] for r in records}


class TestAnalyzeStage:
    def _run_full_pipeline(self, tmp_config, tmp_path, monkeypatch):
        from alienbench import generate, extract, score

        gen_client = make_mock_client(FAKE_CREATURE, "")
        monkeypatch.setattr(generate, "OpenRouterClient", lambda cfg: gen_client)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        generate.run(tmp_config)

        monkeypatch.setattr(extract, "make_judge", make_mock_judge(_make_extraction_response()))
        extract.run(tmp_config)

        score.run(tmp_config)

    def test_results_files_created(self, tmp_config, tmp_path, monkeypatch):
        from alienbench import analyze

        self._run_full_pipeline(tmp_config, tmp_path, monkeypatch)
        analyze.run(tmp_config)

        results = Path(tmp_path) / "results"
        assert (results / "fig1_ward_heatmap.pdf").exists()
        assert (results / "fig2_violin_scores.pdf").exists()
        assert (results / "fig5_ward_radar.pdf").exists()
        assert (results / "fig5b_ward_radar_overlay.pdf").exists()
        assert (results / "summary.txt").exists()
        # Reliability figures require >= 2 judges; test config has 1, so these are correctly absent
        assert not (results / "fig3_reliability_table.pdf").exists()
        assert not (results / "table_reliability.csv").exists()

    def test_summary_contains_model_name(self, tmp_config, tmp_path, monkeypatch):
        from alienbench import analyze

        self._run_full_pipeline(tmp_config, tmp_path, monkeypatch)
        analyze.run(tmp_config)

        summary = (Path(tmp_path) / "results" / "summary.txt").read_text()
        assert "openai/gpt-4o-mini" in summary
        assert "Ward Departure Score" in summary

    def test_length_adjusted_metric(self, tmp_config, tmp_path, monkeypatch):
        """Ward-per-100-tokens is computed, persisted, and summarised."""
        import csv

        from alienbench import analyze

        self._run_full_pipeline(tmp_config, tmp_path, monkeypatch)
        analyze.run(tmp_config)

        results = Path(tmp_path) / "results"
        la_path = results / "table_length_adjusted.csv"
        assert la_path.exists(), "table_length_adjusted.csv not created"

        with la_path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3, "expected 3 per-generation rows"
        for row in rows:
            ward = float(row["ward_score"])
            tokens = float(row["completion_tokens"])
            expected = ward / tokens * 100.0
            assert abs(float(row["ward_per_100"]) - expected) < 1e-9
            assert tokens > 0

        # Per-model table has one row carrying both metrics
        per_model_path = results / "table_length_adjusted_per_model.csv"
        assert per_model_path.exists()
        with per_model_path.open() as fh:
            per_model_rows = list(csv.DictReader(fh))
        assert len(per_model_rows) == 1
        assert per_model_rows[0]["subject_model"] == "openai/gpt-4o-mini"

        summary = (results / "summary.txt").read_text()
        assert "Length-Adjusted Ward Score" in summary


class TestRadar:
    def test_per_model_departure_rates_axes_and_values(self):
        """_per_model_departure_rates returns dimensions in DIMENSION_IDS order
        with judge-then-generation averaging."""
        import pandas as pd

        from alienbench.dimensions import DIMENSION_IDS
        from alienbench.radar import _per_model_departure_rates

        def _row(model, gen_id, judge, **dims):
            base = {f"dim_{d}": 0 for d in DIMENSION_IDS}
            base.update({f"dim_{k}": v for k, v in dims.items()})
            base.update({
                "generation_id": gen_id,
                "judge_model": judge,
                "subject_model": model,
                "prompt_variant": "baseline",
                "ward_score": sum(base[f"dim_{d}"] for d in DIMENSION_IDS),
            })
            return base

        df = pd.DataFrame([
            _row("m/a", "g1", "j1", symmetry=1, locomotion=1),
            _row("m/a", "g1", "j2", symmetry=0, locomotion=1),  # symmetry averages 0.5
            _row("m/a", "g2", "j1", symmetry=1, locomotion=0),
            _row("m/a", "g2", "j2", symmetry=1, locomotion=0),  # gen2 symmetry = 1.0
            _row("m/b", "g3", "j1", habitat=1),
            _row("m/b", "g3", "j2", habitat=1),
        ])

        rates = _per_model_departure_rates(df)

        assert list(rates.columns) == DIMENSION_IDS
        # m/a: symmetry = mean(0.5, 1.0) = 0.75; locomotion = mean(1.0, 0.0) = 0.5
        assert rates.loc["m/a", "symmetry"] == pytest.approx(0.75)
        assert rates.loc["m/a", "locomotion"] == pytest.approx(0.5)
        # m/b: only habitat departs; others 0
        assert rates.loc["m/b", "habitat"] == pytest.approx(1.0)
        assert rates.loc["m/b", "symmetry"] == pytest.approx(0.0)

    def test_radar_figure_written(self, tmp_path):
        """fig_ward_radar_small_multiples writes a PDF even with a single model."""
        import pandas as pd

        from alienbench.dimensions import DIMENSION_IDS
        from alienbench.radar import fig_ward_radar_small_multiples

        row = {f"dim_{d}": 0 for d in DIMENSION_IDS}
        row.update({
            "dim_symmetry": 1,
            "generation_id": "g1",
            "judge_model": "j1",
            "subject_model": "m/a",
            "prompt_variant": "baseline",
            "ward_score": 1,
        })
        df = pd.DataFrame([row])

        fig_ward_radar_small_multiples(df, tmp_path)

        assert (tmp_path / "fig5_ward_radar.pdf").exists()
        assert (tmp_path / "fig5_ward_radar.png").exists()


class TestMakeJudgeFactory:
    """Routing in ``make_judge`` between AI Studio and Vertex for Google."""

    def test_google_use_vertex_routes_through_vertex_client(self, monkeypatch):
        """``use_vertex=true`` must build a Vertex genai.Client and ignore api keys.

        The factory should pass ``vertexai=True``, ``project``, and
        ``location`` to ``genai.Client`` rather than ``api_key``.
        """
        from alienbench.config import Config, JudgeOverride, PromptVariant
        from alienbench.judges import make_judge

        captured: dict = {}

        class _FakeClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        # google.genai.Client is imported lazily inside _GoogleJudge.__init__.
        from google import genai
        monkeypatch.setattr(genai, "Client", _FakeClient)

        monkeypatch.setenv("GOOGLE_VERTEX_PROJECT", "my-test-project")
        monkeypatch.setenv("GOOGLE_VERTEX_LOCATION", "us-east5")
        # An ambient api key must be ignored when use_vertex is true.
        monkeypatch.setenv("GOOGLE_API_KEY", "should-not-be-used")

        cfg = Config(
            models=["x"],
            judge_models=["google/gemini-3.1-pro-preview"],
            prompt_variants=[PromptVariant(id="baseline", label="b", text="x")],
            judge_overrides={
                "google/gemini-3.1-pro-preview": JudgeOverride(
                    provider="google",
                    model_id="gemini-3.1-pro-preview",
                    use_vertex=True,
                ),
            },
        )

        make_judge("google/gemini-3.1-pro-preview", cfg)

        assert captured.get("vertexai") is True
        assert captured.get("project") == "my-test-project"
        assert captured.get("location") == "us-east5"
        assert "api_key" not in captured

    def test_google_default_routes_through_api_key_client(self, monkeypatch):
        """Without ``use_vertex``, the factory uses the AI Studio api_key path."""
        from alienbench.config import Config, JudgeOverride, PromptVariant
        from alienbench.judges import make_judge

        captured: dict = {}

        class _FakeClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        from google import genai
        monkeypatch.setattr(genai, "Client", _FakeClient)

        monkeypatch.setenv("GOOGLE_API_KEY", "studio-key")
        # Vertex env vars present but unused because use_vertex is false.
        monkeypatch.setenv("GOOGLE_VERTEX_PROJECT", "ignored")

        cfg = Config(
            models=["x"],
            judge_models=["google/gemini-3.1-pro-preview"],
            prompt_variants=[PromptVariant(id="baseline", label="b", text="x")],
            judge_overrides={
                "google/gemini-3.1-pro-preview": JudgeOverride(
                    provider="google",
                    model_id="gemini-3.1-pro-preview",
                ),
            },
        )

        make_judge("google/gemini-3.1-pro-preview", cfg)

        assert captured.get("api_key") == "studio-key"
        assert "vertexai" not in captured
        assert "project" not in captured

    def test_google_use_vertex_errors_when_project_unset(self, monkeypatch):
        """Missing project env var must surface as a clear EnvironmentError."""
        from alienbench.config import Config, JudgeOverride, PromptVariant
        from alienbench.judges import make_judge

        monkeypatch.delenv("GOOGLE_VERTEX_PROJECT", raising=False)

        cfg = Config(
            models=["x"],
            judge_models=["google/gemini-3.1-pro-preview"],
            prompt_variants=[PromptVariant(id="baseline", label="b", text="x")],
            judge_overrides={
                "google/gemini-3.1-pro-preview": JudgeOverride(
                    provider="google",
                    model_id="gemini-3.1-pro-preview",
                    use_vertex=True,
                ),
            },
        )

        with pytest.raises(EnvironmentError, match="GOOGLE_VERTEX_PROJECT"):
            make_judge("google/gemini-3.1-pro-preview", cfg)


class TestSamplesPerConditionCap:
    """``samples_per_condition_cap`` filters every stage after generate.

    Generation files on disk are not rewritten. The cap is a read-time
    filter ordered by ``sample_index``, so cap=3 means
    ``sample_index in {0, 1, 2}``.
    """

    def test_iter_generations_cap_orders_by_sample_index(self, tmp_path):
        """Cap must sort by sample_index and yield first N, regardless of write order.

        JSONL writes interleave across reservation workers in the real
        pipeline, so write order does not equal sample_index order. The
        cap path must materialise and sort before slicing.
        """
        from alienbench.paths import iter_generations

        gen_path = (
            tmp_path / "generations" / "m1" / "v1" / "responses.jsonl"
        )
        gen_path.parent.mkdir(parents=True)
        # Write records out of order to expose any bug that just takes
        # the first N JSONL lines.
        records = [
            {"id": f"id-{i}", "sample_index": i, "response": f"r{i}"}
            for i in [4, 0, 2, 1, 3]
        ]
        with open(gen_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        out = list(iter_generations(tmp_path, "m1", "v1", cap=3))
        assert [r["sample_index"] for r in out] == [0, 1, 2]
        assert [r["id"] for r in out] == ["id-0", "id-1", "id-2"]

        # cap=None yields lazily in write order (legacy behaviour).
        out_uncapped = list(iter_generations(tmp_path, "m1", "v1"))
        assert [r["sample_index"] for r in out_uncapped] == [4, 0, 2, 1, 3]

    def test_extract_respects_cap_end_to_end(self, tmp_path, monkeypatch):
        """Generate 5 samples, extract with cap=3, verify only 3 extractions.

        Disk state: generations JSONL keeps all 5 records (not rewritten),
        extractions JSONL holds 3 records corresponding to sample_index
        0, 1, 2. This is the contract the cost-cap relies on.
        """
        from alienbench import extract, generate

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            f"""
models:
  - openai/gpt-4o-mini
judge_models:
  - openai/gpt-4o-mini
samples_per_condition: 5
samples_per_condition_cap: 3
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
        )

        mock_client = make_mock_client(FAKE_CREATURE, "")
        monkeypatch.setattr(generate, "OpenRouterClient", lambda cfg: mock_client)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        generate.run(str(cfg_path))

        monkeypatch.setattr(extract, "make_judge", make_mock_judge(_FAKE_EXTRACTION_JSON))
        extract.run(str(cfg_path))

        gen_path = (
            tmp_path / "data" / "generations"
            / "openai__gpt-4o-mini" / "baseline" / "responses.jsonl"
        )
        ext_path = (
            tmp_path / "data" / "extractions"
            / "openai__gpt-4o-mini" / "openai__gpt-4o-mini" / "baseline"
            / "features.jsonl"
        )

        gen_records = [json.loads(l) for l in gen_path.read_text().splitlines() if l.strip()]
        ext_records = [json.loads(l) for l in ext_path.read_text().splitlines() if l.strip()]

        # Generations untouched: full samples_per_condition on disk.
        assert len(gen_records) == 5

        # Extractions only for sample_index < cap (cap=3).
        assert len(ext_records) == 3
        capped_gen_ids = {
            g["id"] for g in gen_records if g["sample_index"] < 3
        }
        assert {e["generation_id"] for e in ext_records} == capped_gen_ids

    def test_score_respects_cap(self, tmp_path, monkeypatch):
        """Score stage drops extractions whose generation_id is above the cap.

        Even if extractions exist on disk for sample_index >= cap (e.g.
        from a previous uncapped run), they sit dormant. The Ward
        scores file matches the cap.
        """
        from alienbench import extract, generate, score

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            f"""
models:
  - openai/gpt-4o-mini
judge_models:
  - openai/gpt-4o-mini
samples_per_condition: 5
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
        )

        mock_client = make_mock_client(FAKE_CREATURE, "")
        monkeypatch.setattr(generate, "OpenRouterClient", lambda cfg: mock_client)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        generate.run(str(cfg_path))

        # First pass: extract with no cap, producing 5 extraction records.
        monkeypatch.setattr(extract, "make_judge", make_mock_judge(_FAKE_EXTRACTION_JSON))
        extract.run(str(cfg_path))

        # Now activate the cap by rewriting the config file, and run score.
        cfg_path.write_text(cfg_path.read_text() + "\nsamples_per_condition_cap: 3\n")
        score.run(str(cfg_path))

        ward_path = (
            tmp_path / "data" / "scores"
            / "openai__gpt-4o-mini" / "openai__gpt-4o-mini" / "baseline"
            / "ward_scores.jsonl"
        )
        ward_records = [json.loads(l) for l in ward_path.read_text().splitlines() if l.strip()]

        # Despite 5 extractions on disk, score only emits 3 ward records.
        assert len(ward_records) == 3

    def test_config_rejects_cap_above_samples(self):
        """Setting cap > samples_per_condition is meaningless and must error.

        The cap can only ever filter the existing generations; allowing
        cap > samples_per_condition would silently produce a smaller
        result than the user asked for once the data is the bottleneck.
        """
        from alienbench.config import Config, PromptVariant

        with pytest.raises(ValueError, match="cannot exceed"):
            Config(
                models=["x"],
                judge_models=["y"],
                prompt_variants=[PromptVariant(id="b", label="b", text="x")],
                samples_per_condition=10,
                samples_per_condition_cap=20,
            )


class TestAnthropicBedrockFailover:
    """``_AnthropicJudge`` fails over to AWS Bedrock on a 429.

    Verifies that:
    - a successful Anthropic call returns the direct-API response
      without touching Bedrock,
    - a 429 from the direct API triggers a Bedrock call that returns
      the Bedrock response, and
    - subsequent calls within the cooldown window go straight to
      Bedrock without re-trying Anthropic.
    """

    @staticmethod
    def _stub_message(text: str, model: str):
        """Build a minimal anthropic ``Message``-shaped object."""
        block = MagicMock()
        block.type = "text"
        block.text = text
        msg = MagicMock()
        msg.content = [block]
        msg.model = model
        msg.id = f"msg_{model}"
        msg.usage = MagicMock(input_tokens=10, output_tokens=20)
        return msg

    def test_falls_over_to_bedrock_on_429(self, monkeypatch):
        from alienbench.judges import _AnthropicJudge
        import anthropic

        # Real RateLimitError so the judge's `except` clause matches.
        rate_limit = anthropic.RateLimitError(
            message="rate limit",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )

        anthropic_client = MagicMock()
        anthropic_client.messages.create.side_effect = rate_limit

        bedrock_client = MagicMock()
        bedrock_client.messages.create.return_value = self._stub_message(
            "from-bedrock", "us.anthropic.claude-opus-4-6-v1:0"
        )

        monkeypatch.setattr(anthropic, "Anthropic", lambda **_: anthropic_client)
        monkeypatch.setattr(anthropic, "AnthropicBedrock", lambda **_: bedrock_client)

        # No-sleep so retry_with_backoff does not slow the test.
        from alienbench import client as client_module
        monkeypatch.setattr(client_module.time, "sleep", lambda _: None)

        judge = _AnthropicJudge(
            "claude-opus-4-6",
            "ant-key",
            bedrock_model_id="us.anthropic.claude-opus-4-6-v1:0",
            bedrock_region="us-east-1",
            bedrock_cooldown_seconds=30.0,
        )

        resp = judge.complete(prompt="p", temperature=0.0, max_tokens=10)

        assert resp.text == "from-bedrock"
        assert resp.model == "us.anthropic.claude-opus-4-6-v1:0"
        # Anthropic was tried once (and 429'd); Bedrock served the call.
        assert anthropic_client.messages.create.call_count == 1
        assert bedrock_client.messages.create.call_count == 1

    def test_cooldown_skips_anthropic_after_429(self, monkeypatch):
        from alienbench.judges import _AnthropicJudge
        import anthropic

        rate_limit = anthropic.RateLimitError(
            message="rate limit",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )

        anthropic_client = MagicMock()
        anthropic_client.messages.create.side_effect = rate_limit

        bedrock_client = MagicMock()
        bedrock_client.messages.create.return_value = self._stub_message(
            "from-bedrock", "us.anthropic.claude-opus-4-6-v1:0"
        )

        monkeypatch.setattr(anthropic, "Anthropic", lambda **_: anthropic_client)
        monkeypatch.setattr(anthropic, "AnthropicBedrock", lambda **_: bedrock_client)
        from alienbench import client as client_module
        monkeypatch.setattr(client_module.time, "sleep", lambda _: None)

        judge = _AnthropicJudge(
            "claude-opus-4-6",
            "ant-key",
            bedrock_model_id="us.anthropic.claude-opus-4-6-v1:0",
            bedrock_region="us-east-1",
            bedrock_cooldown_seconds=30.0,
        )

        # First call: Anthropic 429 -> Bedrock failover, opens circuit.
        judge.complete(prompt="p1", temperature=0.0, max_tokens=10)
        # Second call: still within cooldown, must skip Anthropic entirely.
        judge.complete(prompt="p2", temperature=0.0, max_tokens=10)

        assert anthropic_client.messages.create.call_count == 1
        assert bedrock_client.messages.create.call_count == 2

    def test_no_failover_when_disabled(self, monkeypatch):
        """Without bedrock_model_id, a 429 propagates and is retried by retry_with_backoff."""
        from alienbench.judges import _AnthropicJudge
        import anthropic

        rate_limit = anthropic.RateLimitError(
            message="rate limit",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )

        anthropic_client = MagicMock()
        anthropic_client.messages.create.side_effect = rate_limit

        monkeypatch.setattr(anthropic, "Anthropic", lambda **_: anthropic_client)

        from alienbench import client as client_module
        monkeypatch.setattr(client_module.time, "sleep", lambda _: None)

        judge = _AnthropicJudge("claude-opus-4-6", "ant-key")

        with pytest.raises(RuntimeError, match="Failed after"):
            judge.complete(prompt="p", temperature=0.0, max_tokens=10)
        # Retried up to _MAX_RETRIES (no Bedrock to deflect to).
        assert anthropic_client.messages.create.call_count == client_module._MAX_RETRIES

    def test_config_rejects_bedrock_without_model_id(self):
        """``bedrock_fallback=True`` without a ``bedrock_model_id`` is invalid."""
        from alienbench.config import Config, JudgeOverride, PromptVariant

        with pytest.raises(ValueError, match="bedrock_model_id"):
            Config(
                models=["x"],
                judge_models=["anthropic/claude-opus-4.6"],
                prompt_variants=[PromptVariant(id="b", label="b", text="x")],
                judge_overrides={
                    "anthropic/claude-opus-4.6": JudgeOverride(
                        provider="anthropic",
                        model_id="claude-opus-4-6",
                        bedrock_fallback=True,
                    ),
                },
            )

    def test_config_rejects_bedrock_on_non_anthropic_provider(self):
        """``bedrock_fallback`` only applies to the Anthropic judge."""
        from alienbench.config import Config, JudgeOverride, PromptVariant

        with pytest.raises(ValueError, match="only defined for the Anthropic"):
            Config(
                models=["x"],
                judge_models=["openai/gpt-5"],
                prompt_variants=[PromptVariant(id="b", label="b", text="x")],
                judge_overrides={
                    "openai/gpt-5": JudgeOverride(
                        provider="openai",
                        model_id="gpt-5",
                        bedrock_fallback=True,
                        bedrock_model_id="some.bedrock.id",
                    ),
                },
            )

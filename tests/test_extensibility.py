"""Tests for the parts of the pipeline that must keep working when models are added.

Covers the bolding of table winners, the radar-figure layout, the
same-family judge analysis with several subjects per provider, and the
claim checks that flag paper sentences after the results change.
"""

from __future__ import annotations

import sys
import types

import pandas as pd

# Mock krippendorff before any alienbench import, as in the other test files.
_mock_krippendorff = types.ModuleType("krippendorff")
_mock_krippendorff.alpha = lambda reliability_data, level_of_measurement: 0.85
sys.modules.setdefault("krippendorff", _mock_krippendorff)

from alienbench.ablation_judges import (  # noqa: E402
    _panel_per_model_means,
    _same_family_map,
    _same_family_table,
)
from alienbench.claims import human_claims, model_claims, reliability_claims  # noqa: E402
from alienbench.config import Config  # noqa: E402
from alienbench.dimensions import DIMENSION_IDS  # noqa: E402
from alienbench.latex_tables import (  # noqa: E402
    _is_winner,
    _make_ward_dimensions_table,
    _make_ward_scores_table,
)
from alienbench.radar import _balanced_rows, _radar_layout  # noqa: E402


def _row(model, gen, judge, ward, variant="baseline", **dims):
    row = {f"dim_{d}": 0 for d in DIMENSION_IDS}
    row.update({f"dim_{k}": v for k, v in dims.items()})
    row.update({"generation_id": gen, "judge_model": judge, "subject_model": model,
                "prompt_variant": variant, "ward_score": ward})
    return row


def _cfg(models):
    return Config(
        models=models,
        judge_models=["j1"],
        prompt_variants=[{"id": "baseline", "label": "Baseline", "acronym": "Base",
                          "name": "Baseline", "text": "Imagine a creature."}],
    )


# ---------------------------------------------------------------------------
# Bolding of winners
# ---------------------------------------------------------------------------

class TestWinnerBolding:
    def test_ties_at_displayed_precision_are_winners(self):
        assert _is_winner(5.166, 5.174, 2)   # both print as 5.17
        assert not _is_winner(5.16, 5.17, 2)

    def test_score_table_bolds_every_model_tied_for_the_top(self):
        models = [{"id": "a/one", "name": "One"}, {"id": "b/two", "name": "Two"},
                  {"id": "c/three", "name": "Three"}]
        df = pd.DataFrame([
            _row("a/one", "g1", "j1", 5.171), _row("a/one", "g2", "j1", 5.171),
            _row("b/two", "g3", "j1", 5.169), _row("b/two", "g4", "j1", 5.169),
            _row("c/three", "g5", "j1", 2.0), _row("c/three", "g6", "j1", 2.0),
        ])
        tex = _make_ward_scores_table(df, _cfg(models))
        lines = {name: line for line in tex.splitlines()
                 for name in ("One", "Two", "Three") if line.strip().startswith(name)}
        assert "\\textbf" in lines["One"] and "\\textbf" in lines["Two"]
        assert "\\textbf" not in lines["Three"]

    def test_feature_table_bolds_ties_at_one_decimal_percent(self):
        models = [{"id": "a/one", "name": "One"}, {"id": "b/two", "name": "Two"}]
        df = pd.DataFrame([
            _row("a/one", "g1", "j1", 1, symmetry=1), _row("a/one", "g2", "j1", 1, symmetry=1),
            _row("b/two", "g3", "j1", 1, symmetry=1), _row("b/two", "g4", "j1", 1, symmetry=1),
        ])
        tex = _make_ward_dimensions_table(df, _cfg(models))
        symmetry_line = next(line for line in tex.splitlines() if "Symmetry" in line)
        assert symmetry_line.count("\\textbf") == 2


# ---------------------------------------------------------------------------
# Radar layout
# ---------------------------------------------------------------------------

class TestRadarLayout:
    def test_human_on_top_and_best_model_below_it(self):
        order = [f"m{i}" for i in range(1, 11)]   # descending overall score
        lead, rows = _radar_layout(order, "human")
        assert lead == ["human", "m1"]
        assert [m for row in rows for m in row] == order[1:]

    def test_rows_stay_balanced_and_narrow_as_models_are_added(self):
        for n in range(2, 30):
            rows = _balanced_rows([f"m{i}" for i in range(n)])
            sizes = [len(r) for r in rows]
            assert max(sizes) <= 6
            assert max(sizes) - min(sizes) <= max(sizes) - 1
            assert sum(sizes) == n

    def test_without_human_the_best_model_leads_alone(self):
        lead, _ = _radar_layout(["m1", "m2", "m3"], None)
        assert lead == ["m1"]


# ---------------------------------------------------------------------------
# Same-family pairs
# ---------------------------------------------------------------------------

class TestSameFamily:
    def test_every_subject_of_a_judge_provider_gets_a_row(self):
        judges = ["openai/judge", "google/judge"]
        subjects = ["openai/model-a", "openai/model-b", "google/model-c", "meta/model-d"]
        rows = []
        for i, s in enumerate(subjects):
            for j in judges:
                rows.append(_row(s, f"g{i}", j, ward=i + (1 if j.startswith("openai") else 0)))
        df = pd.DataFrame(rows)
        sf_map = _same_family_map(judges, subjects)
        table = _same_family_table(df, _panel_per_model_means(df), judges, sf_map)
        pairs = set(zip(table["excluded_judge"], table["sf_subject"]))
        assert pairs == {("openai/judge", "openai/model-a"),
                         ("openai/judge", "openai/model-b"),
                         ("google/judge", "google/model-c")}


# ---------------------------------------------------------------------------
# Claim checks
# ---------------------------------------------------------------------------

class TestClaims:
    def test_human_exception_claim_fails_when_there_are_two(self):
        assert human_claims({"abNHumanExceptions": "1"})[0].holds is True
        assert human_claims({"abNHumanExceptions": "2"})[0].holds is False

    def test_missing_macro_is_reported_as_unknown(self):
        assert human_claims({})[0].holds is None

    def test_reliability_claims_follow_the_band_counts(self):
        n = len(DIMENSION_IDS)
        macros = {"abNClearDims": str(n - 2), "abNExcellentDims": str(n - 3),
                  "abNGoodDims": "1", "abNModerateDims": "1"}
        below, banded = reliability_claims(macros)
        assert below.holds is False     # two features below the threshold
        assert banded.holds is False    # one feature in a band the text omits

    def test_open_weight_claims_follow_the_ranking(self):
        cfg = Config(
            models=[{"id": "o/open", "name": "Open", "open_weights": True, "tier": "A"},
                    {"id": "c/closed", "name": "Closed", "open_weights": False, "tier": "A"}],
            judge_models=["j1"],
            prompt_variants=[
                {"id": "baseline", "label": "B", "acronym": "B", "name": "B", "text": "x"},
                {"id": "detailed_description", "label": "D", "acronym": "D", "name": "D",
                 "text": "y"},
            ],
        )
        df = pd.DataFrame([
            _row("o/open", "g1", "j1", 3), _row("o/open", "g2", "j1", 3,
                                               variant="detailed_description"),
            _row("c/closed", "g3", "j1", 6), _row("c/closed", "g4", "j1", 6,
                                                 variant="detailed_description"),
        ])
        macros = {"abOpenMean": "3.00", "abClosedMean": "6.00", "abNOpenAboveClosed": "0"}
        claims = {c.text: c.holds for c in model_claims(cfg, macros, df)}
        assert claims["Open-weight models have a higher mean overall score than "
                      "proprietary models."] is False
        assert claims["Some open-weight models score higher than every proprietary "
                      "model."] is False


# ---------------------------------------------------------------------------
# Features a description does not mention
# ---------------------------------------------------------------------------

class TestUnmentioned:
    """Judge texts taken from the extraction records."""

    UNMENTIONED = [
        "not described", "Not explicitly described.", "not described in the text.",
        "no unusual cognition described", "reproduction not described",
        "cognitive architecture is not explicitly described in the text",
        "no explicit description of cognitive architecture",
        "not described; presumed centralized nervous system",
        "no communication method described", "no information given", "",
    ]
    DESCRIBED = [
        "not explicitly described beyond sensory processing",
        "mating mentioned but no mechanism described",
        "mating mentioned; mechanism unspecified",
        "asexual reproduction through budding",
        "sexual reproduction with males and females laying eggs",
        "Six thick pillar-like legs",
    ]

    def test_absent_features_are_recognized(self):
        from alienbench.omission import is_unmentioned
        assert all(is_unmentioned(t) for t in self.UNMENTIONED)

    def test_described_and_partial_features_are_not(self):
        from alienbench.omission import is_unmentioned
        assert not any(is_unmentioned(t) for t in self.DESCRIBED)

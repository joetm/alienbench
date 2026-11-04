"""The findings report: case selection, wording, and the paper comparison.

The paragraph functions are tested on synthetic facts, so the tests run on a
clean checkout without data. The last test builds the full report from the
data on disk when it exists and checks that every sentence uses only defined
macros.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

_mock_krippendorff = types.ModuleType("krippendorff")
_mock_krippendorff.alpha = lambda reliability_data, level_of_measurement: 0.85
sys.modules.setdefault("krippendorff", _mock_krippendorff)

from alienbench import findings  # noqa: E402
from alienbench.config import load_config  # noqa: E402
from alienbench.dimensions import DIMENSION_IDS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    return load_config(str(ROOT / "config.yaml"))


def make_facts(cfg, below: list | None = None, posthoc_rows: dict | None = None,
               m: dict | None = None) -> findings.Facts:
    """Facts with every model at 0.5 on every feature and humans at 0.1.

    Each (model, feature) in ``below`` gets a model rate under the human rate.
    """
    models = list(cfg.models)
    dim_rates = pd.DataFrame(0.5, index=models, columns=DIMENSION_IDS)
    human = pd.Series(0.1, index=DIMENSION_IDS)
    for x, d in below or []:
        dim_rates.loc[x, d] = 0.05
    conds = [v.id for v in cfg.prompt_variants]
    cell = pd.DataFrame(5.0, index=models, columns=conds)
    overall = pd.Series([5.0 - i * 0.1 for i in range(len(models))], index=models)
    posthoc = None
    if posthoc_rows is not None:
        posthoc = pd.DataFrame(
            [{"group_b": c, "p_corrected": p, "rank_biserial_r": r} for c, (p, r) in posthoc_rows.items()]
        ).set_index("group_b")
    cond_rates = pd.DataFrame(0.5, index=conds, columns=DIMENSION_IDS)
    return findings.Facts(cfg, dict(m or {}), models, overall, cell, posthoc, 0.0, dim_rates,
                          cond_rates, human, 2.0, Path("."))


class TestHelpers:
    def test_join(self):
        assert findings.join(["a"]) == "a"
        assert findings.join(["a", "b"], "or") == "a or b"
        assert findings.join(["a", "b", "c"]) == "a, b, and c"

    def test_define_rejects_non_letters(self):
        with pytest.raises(ValueError):
            findings.define({}, "Bad1", 1)

    def test_substitute_fails_on_an_undefined_macro(self):
        with pytest.raises(KeyError):
            findings.substitute("\\abMissing{} text", {})
        assert findings.substitute("\\abX{} and \\abX", {"abX": "1"}) == "1 and 1"

    def test_readable_fills_values_and_drops_markup(self):
        text = "(\\abA{} vs.\\ \\abB{}, \\abC{}\\% CI, \\autoref{fig:radar})"
        out = findings.readable(text, {"abA": "1", "abB": "2", "abC": "95"})
        assert out == "(1 vs. 2, 95% CI, Figure radar)"


class TestPaperComparison:
    def test_status_distinguishes_active_commented_and_new(self, tmp_path):
        tex = tmp_path / "paper.tex"
        tex.write_text("\\rev{Models score \\abMean{} overall.}\n"
                       "%Every model is higher on every feature.\n")
        m = {"abMean": "5.0"}
        texts = findings.paper_texts(tex, m)
        assert findings.status("Models score \\abMean{} overall.", texts, m) == "same"
        assert findings.status("Every model is higher on every feature.", texts, m) == "commented"
        assert findings.status("A sentence the paper lacks.", texts, m) == "new"

    def test_status_matches_on_filled_in_values(self, tmp_path):
        tex = tmp_path / "paper.tex"
        tex.write_text("Only high gravity is lower.\n")
        m = {"abNames": "high gravity"}
        texts = findings.paper_texts(tex, m)
        assert findings.status("Only \\abNames{} is lower.", texts, m) == "same"


class TestHumanExceptions:
    BASE = {"NDimLeaders": "3", "NDimLeadersWord": "three", "NDimsWord": "ten",
            "MostDim": "x", "MostDimHuman": "x", "MostDimMin": "x", "MostDimMax": "x",
            "LeastDim": "x", "LeastDimHuman": "x", "LeastDimMin": "x", "LeastDimMax": "x"}

    def macros(self):
        return {f"ab{k}": v for k, v in self.BASE.items()}

    def test_no_exception(self, cfg):
        fx = make_facts(cfg, m=self.macros())
        (p,) = findings.human_comparison(fx)
        assert p.sentences[0].startswith("Every model has a higher departure rate than the human "
                                         "participants on every feature (")
        assert fx.m["abNHumanExceptions"] == "0"

    def test_one_exception_keeps_the_paper_wording(self, cfg):
        x = cfg.models[0]
        fx = make_facts(cfg, [(x, "habitat")], m=self.macros())
        fx.m.update(abExceptionDim="habitat", abExceptionModel="M",
                    abExceptionRate="1\\%", abExceptionHumanRate="2\\%")
        (p,) = findings.human_comparison(fx)
        assert "on every feature except \\abExceptionDim{} for \\abExceptionModel{}" in p.sentences[0]

    def test_several_exceptions_are_listed(self, cfg):
        below = [(cfg.models[0], "habitat"), (cfg.models[1], "symmetry"), (cfg.models[2], "cognition")]
        fx = make_facts(cfg, below, m=self.macros())
        (p,) = findings.human_comparison(fx)
        assert "with \\abNHumanExceptionsWord{} exceptions: \\abExceptionList{}" in p.sentences[0]
        assert fx.m["abNHumanExceptionsWord"] == "three"
        assert "on habitat" in fx.m["abExceptionList"]


class TestConstrainedConditions:
    def rows(self, hg=(0.001, -0.5), am=(0.001, 0.2), nl=(0.9, 0.0)):
        return {"constrained_high_gravity": hg, "constrained_ammonia": am,
                "constrained_no_light": nl, "detailed_description": (0.9, 0.0)}

    def test_one_lower_names_it_in_the_abstract(self, cfg):
        fx = make_facts(cfg, posthoc_rows=self.rows())
        s = findings._constrained_lower_sentence(fx, "abstract")
        assert s.startswith("Only one of the \\abNConstrainedWord{} environmental constraints, "
                            "\\abConstrainedLowerNames{},")
        assert fx.m["abConstrainedLowerNames"] == "high gravity"

    def test_none_lower(self, cfg):
        fx = make_facts(cfg, posthoc_rows=self.rows(hg=(0.9, -0.1)))
        s = findings._constrained_lower_sentence(fx, "conclusion")
        assert s.startswith("None of the")

    def test_two_lower(self, cfg):
        fx = make_facts(cfg, posthoc_rows=self.rows(am=(0.001, -0.4)))
        s = findings._constrained_lower_sentence(fx, "abstract")
        assert s.startswith("\\abNConstrainedLowerCap{} of the")
        assert fx.m["abConstrainedLowerNames"] == "high gravity and ammonia oceans"

    def test_pairwise_sentence_groups_conditions(self, cfg):
        m = {"abPConstrainedNoLight": "$p = 1.0$", "abPDetailedDescription": "$p = 1.0$"}
        fx = make_facts(cfg, posthoc_rows=self.rows(), m=m)
        t = findings.condition_tests(fx)
        pairwise = t.sentences[1]
        assert "lower scores under high gravity than under the baseline" in pairwise
        assert "slightly higher scores under ammonia oceans" in pairwise
        assert ("no detectable difference for the no-light or detailed-description conditions "
                "(both \\abPConstrainedNoLight{} after correction)") in pairwise


class TestSameFamily:
    def macros(self, zero, neg, pos):
        n = zero + neg + pos
        words = findings.word
        return {"abSfNPairs": str(n), "abSfNPairsWord": words(n),
                "abSfNZero": str(zero), "abSfNZeroWord": words(zero),
                "abSfNNegative": str(neg), "abSfNNegativeWord": words(neg),
                "abSfNPositive": str(pos), "abSfNPositiveWord": words(pos),
                "abSfNPositiveCap": findings.cap(words(pos))}

    def test_all_unchanged(self, cfg):
        fx = make_facts(cfg, m=self.macros(4, 0, 0))
        assert findings._same_family(fx, "main") == [
            "Across the \\abSfNPairsWord{} same-family pairs, the rank is unchanged for all of them."]

    def test_mixed_keeps_the_paper_wording(self, cfg):
        fx = make_facts(cfg, m=self.macros(3, 1, 0))
        assert findings._same_family(fx, "main") == [
            "Across the \\abSfNPairsWord{} same-family pairs, the rank is unchanged for "
            "\\abSfNZeroWord{} and higher for \\abSfNNegativeWord{}."]

    def test_positive_change_is_reported(self, cfg):
        fx = make_facts(cfg, m=self.macros(2, 0, 1))
        last = findings._same_family(fx, "appendix")[-1]
        assert "consistent with self-preference inflation" in last


def test_full_report_uses_only_defined_macros(cfg, monkeypatch):
    """Build the report from the data on disk (skipped on a clean checkout)."""
    import re

    from alienbench.latex_tables import load_human_scores
    from alienbench.paths import load_ward_scores

    monkeypatch.chdir(ROOT)
    key_numbers = ROOT / "results" / "key_numbers.tex"
    ward = load_ward_scores(Path(cfg.data_dir), cfg.judge_models, cfg.models, cfg.prompt_variants)
    if ward.empty or not key_numbers.exists():
        pytest.skip("no scores or key_numbers.tex on disk")
    m = {}
    for line in key_numbers.read_text().splitlines():
        g = re.match(r"\\newcommand\{\\(ab[A-Za-z]+)\}\{(.*)\}$", line)
        if g:
            m[g.group(1)] = g.group(2)
    paragraphs = findings.build_findings(cfg, m, ward, load_human_scores(cfg),
                                         Path(cfg.results_dir), mentioned=False)
    assert paragraphs
    for p in paragraphs:
        assert p.title != "Section could not be written", p.case
        for s in p.sentences:
            findings.substitute(s, m)  # raises on an undefined macro

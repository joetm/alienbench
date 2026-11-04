"""The number macros of Stage 6 and the paper that consumes them.

``alienbench/paper.py`` writes ``results/key_numbers.tex``, one ``\\ab<Name>``
macro per number or name in the paper's running text. These tests pin the
formatting helpers, check that the macro file is well formed, and check that
every macro ``paper.tex`` uses is one the generator defines, so a renamed or
removed macro fails here and not in LaTeX.
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

_mock_krippendorff = types.ModuleType("krippendorff")
_mock_krippendorff.alpha = lambda reliability_data, level_of_measurement: 0.85
sys.modules.setdefault("krippendorff", _mock_krippendorff)

from alienbench import paper  # noqa: E402
from alienbench.config import Config, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAPER_TEX = ROOT.parent / "paper" / "latex" / "paper.tex"
CHECKLIST_TEX = ROOT.parent / "paper" / "latex" / "checklist.tex"
KEY_NUMBERS = ROOT / "results" / paper.OUTPUT_NAME


class TestFormatting:
    def test_word_spells_small_counts_and_numerals_beyond_twenty(self):
        assert paper.word(10) == "ten"
        assert paper.word(0) == "zero"
        assert paper.word(21) == "21"

    def test_thousands_uses_a_latex_safe_separator(self):
        assert paper.thousands(5400) == "5{,}400"
        assert paper.thousands(800) == "800"

    def test_f_drops_the_sign_of_a_negative_zero(self):
        assert paper.f(-0.001) == "0.00"
        assert paper.f(0.125, 2) == "0.12"

    def test_p_value_formats_small_and_unit_values(self):
        assert paper.p_value(1e-9) == "$p < .001$"
        assert paper.p_value(1.0) == "$p = 1.0$"
        assert paper.p_value(0.0123) == "$p = 0.0123$"

    def test_join_names(self):
        assert paper.join_names(["a"]) == "a"
        assert paper.join_names(["a", "b"]) == "a and b"
        assert paper.join_names(["a", "b", "c"]) == "a, b, and c"


class TestMacros:
    def test_names_must_be_letters_only(self):
        m = paper.Macros()
        with pytest.raises(ValueError):
            m.put("Bad2", "x")

    def test_duplicate_definitions_are_rejected(self):
        m = paper.Macros()
        m.put("Good", "x")
        with pytest.raises(ValueError):
            m.put("Good", "y")

    def test_render_emits_one_newcommand_per_macro(self):
        m = paper.Macros()
        m.put("A", "1")
        m.put("B", "two")
        text = paper.render(m)
        assert "\\newcommand{\\abA}{1}" in text
        assert "\\newcommand{\\abB}{two}" in text


class TestConfigNormalization:
    def test_mapping_entries_become_ids_plus_info(self):
        cfg = Config(
            models=[{"id": "a/b", "name": "Model B", "key": "ModelB", "tier": "A"}, "c/d"],
            judge_models=[{"id": "j/1", "name": "Judge One", "date": "2026-01-01"}],
            prompt_variants=[{"id": "baseline", "label": "Baseline", "text": "x"}],
        )
        assert cfg.models == ["a/b", "c/d"]
        assert cfg.judge_models == ["j/1"]
        assert cfg.model_name("a/b") == "Model B"
        assert cfg.model_key("a/b") == "ModelB"
        # A bare id falls back to its last path segment and a CamelCased key.
        assert cfg.model_name("c/d") == "d"
        assert cfg.model_key("c/d") == "CD"
        assert cfg.judge_name("j/1") == "Judge One"
        assert cfg.judge_info["j/1"].date == "2026-01-01"

    def test_the_shipped_config_names_every_model_and_judge(self):
        cfg = load_config(ROOT / "config.yaml")
        for model in cfg.models:
            info = cfg.model_info[model]
            assert info.name and info.key and info.tier and info.resolved_version
            assert info.key.isalpha()
            assert info.tier in cfg.tiers
            assert info.open_weights is not None
        for judge in cfg.judge_models:
            info = cfg.judge_info[judge]
            assert info.name and info.key and info.date and info.key.isalpha()
        keys = [cfg.model_key(m) for m in cfg.models]
        assert len(set(keys)) == len(keys)


@pytest.mark.skipif(not KEY_NUMBERS.exists(), reason="run `python -m alienbench paper` first")
class TestGeneratedFile:
    def test_every_macro_name_is_letters_only(self):
        names = re.findall(r"\\newcommand\{\\(\w+)\}", KEY_NUMBERS.read_text())
        assert names
        assert all(n.isalpha() for n in names)
        assert len(names) == len(set(names))

    def test_paper_uses_only_defined_macros(self):
        defined = set(re.findall(r"\\newcommand\{\\(ab\w+)\}", KEY_NUMBERS.read_text()))
        # Comments may mention macro names, so drop them before scanning.
        body = "\n".join(
            re.sub(r"(?<!\\)%.*", "", l)
            for path in (PAPER_TEX, CHECKLIST_TEX)
            for l in path.read_text().splitlines()
        )
        used = set(re.findall(r"\\(ab[A-Z][A-Za-z]*)", body))
        # `\abstract` style commands do not start with a capital after "ab".
        missing = sorted(used - defined)
        assert not missing, f"paper.tex uses undefined macros: {missing}"


class TestPaperHasNoHardcodedModelNames:
    def test_model_names_in_running_text_come_from_macros(self):
        cfg = load_config(ROOT / "config.yaml")
        body = "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in PAPER_TEX.read_text().splitlines())
        # The extraction prompt and the author block are verbatim text, not model names.
        for model in cfg.models:
            name = cfg.model_name(model)
            assert name not in body, f"{name!r} is hardcoded in paper.tex; use \\abName macros"
        for judge in cfg.judge_models:
            name = cfg.judge_name(judge)
            assert name not in body, f"{name!r} is hardcoded in paper.tex; use \\abJudgeName macros"


def test_leave_one_condition_out_keeps_the_top_model():
    """paper.tex states that the best model stays first when any condition is dropped."""
    macros = (ROOT / "results" / "key_numbers.tex").read_text()
    paper = (ROOT.parent / "paper" / "latex" / "paper.tex").read_text()
    if r"\abLocoRhoMin" in paper:
        assert r"\newcommand{\abLocoSameTop}{true}" in macros


def test_best_and_worst_models_are_open_weight_as_stated():
    """paper.tex states that the best and the worst model are both open-weight."""
    cfg = load_config(ROOT / "config.yaml")
    macros = (ROOT / "results" / "key_numbers.tex").read_text()
    paper = (ROOT.parent / "paper" / "latex" / "paper.tex").read_text()
    if "both of which are open-weight models" not in paper:
        return
    open_names = {cfg.model_name(m) for m in cfg.models if cfg.model_info[m].open_weights}
    for label in ("Best", "Worst"):
        name = re.search(rf"\\newcommand{{\\ab{label}Model}}{{(.+?)}}", macros).group(1)
        assert name in open_names

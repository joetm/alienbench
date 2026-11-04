"""The findings report: the paper's result sentences, written from the data.

``paper.py`` keeps every number in ``paper.tex`` current through macros, but a
macro cannot change the *shape* of a sentence. "Every model exceeds the human
rate on every feature, with one exception" needs other words when there is
no exception or when there are three. This module writes those sentences.

For every paragraph of the paper that reports a result, it decides which case
the current data fall in and writes the paragraph in the wording of the paper
for that case. The output is ``results/findings.md``. It is a report for the
authors to read and to copy from. Nothing in it is fed to LaTeX
automatically.

Each sentence is given twice. The LaTeX version uses the ``\\ab<Name>`` macros
of ``key_numbers.tex`` and can be pasted into ``paper.tex``. The reading
version has the values substituted. Each sentence is also marked by whether
the paper already contains it verbatim, contains it in a comment, or not at
all.

Wording follows the sentences in ``paper.tex`` (including commented-out
ones) as revised by the authors. Where the data still support a sentence it
is reproduced unchanged. A different case changes only the words that the
case contradicts.

Run through ``python -m alienbench paper``, which calls :func:`build_findings`
before it writes the macros (some sentences need macros of their own) and
:func:`write_findings` after.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

from alienbench.config import Config, _key_from
from alienbench.dimensions import DIMENSION_IDS, DIMENSION_LABELS

logger = logging.getLogger(__name__)

REPORT_NAME = "findings.md"
SIGNIFICANCE = 0.05
# Below this |r| a significant difference is called "slight", as in the paper.
SMALL_R = 0.3
# A paragraph says "nearly every feature" up to this share of exceptions.
NEARLY_ALL = 0.10

_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]

# How the paper names each condition in running text: after "on" in the
# per-condition leader sentence, after "under", and as an adjective before
# "condition". Unknown conditions fall back to their display name.
COND_TEXT = {
    "baseline": ("the open-ended baseline", "the baseline", "baseline"),
    "departure_primed": ("departure-priming", "departure priming", "departure-primed"),
    "constrained_no_light": ("the no-light constraint", "no light", "no-light"),
    "constrained_high_gravity": ("high gravity", "high gravity", "high-gravity"),
    "constrained_ammonia": ("ammonia oceans", "ammonia oceans", "ammonia-ocean"),
    "detailed_description": ("the detailed-description prompt",
                             "the detailed-description prompt", "detailed-description"),
}

# Feature names in running text.
FEATURE_TEXT = {
    "symmetry": "symmetry", "sensory_organs": "sensory organs", "locomotion": "locomotion",
    "body_plan": "body plan", "skin_covering": "skin or body covering",
    "reproduction": "reproduction", "metabolism": "metabolism",
    "communication": "communication", "habitat": "habitat", "cognition": "cognition",
}

# The feature that the Discussion names as the one each environment concerns
# ("sensory organs without light", "locomotion in ammonia").
CONCERNED_FEATURE = {
    "constrained_no_light": "sensory_organs",
    "constrained_ammonia": "locomotion",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def word(n: int) -> str:
    n = int(n)
    return _WORDS[n] if 0 <= n < len(_WORDS) else str(n)


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def join(items: list[str], conj: str = "and") -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} {conj} {items[1]}"
    return ", ".join(items[:-1]) + f", {conj} {items[-1]}"


def define(m: dict, name: str, value) -> None:
    """Add a macro the findings need. Unlike ``Macros.put``, a rerun may redefine it."""
    full = f"ab{name}"
    if not full.isalpha():
        raise ValueError(f"macro name {full!r} must contain letters only")
    m[full] = str(value)


def define_count(m: dict, name: str, n: int) -> None:
    """A count with its word and capitalized-word twins."""
    define(m, name, int(n))
    define(m, f"{name}Word", word(n))
    define(m, f"{name}Cap", cap(word(n)))


def number(m: dict, name: str) -> float:
    """A numeric macro value (percent signs and math dollars dropped)."""
    text = m[f"ab{name}"].replace("\\%", "").replace("$", "")
    return float(text)


def cond_text(cfg: Config, cond_id: str, which: int) -> str:
    if cond_id in COND_TEXT:
        return COND_TEXT[cond_id][which]
    display = next(v.display for v in cfg.prompt_variants if v.id == cond_id)
    return display.lower()


def ck(cond_id: str) -> str:
    return _key_from(cond_id)


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

@dataclass
class Paragraph:
    """One paragraph of the paper, written for the case the data fall in."""
    section: str
    title: str
    sentences: list[str] = field(default_factory=list)
    case: list[str] = field(default_factory=list)    # which case applies, and why
    check: list[str] = field(default_factory=list)   # judgments left to the authors

    def say(self, sentence: str) -> None:
        self.sentences.append(sentence)


@dataclass
class Facts:
    """Everything the paragraphs need, computed once from the data."""
    cfg: Config
    m: dict
    models: list[str]
    overall: pd.Series            # per-model overall mean, descending
    cell: pd.DataFrame            # model x condition mean
    posthoc: pd.DataFrame | None  # baseline-vs-condition rows, indexed by condition
    kruskal_p: float
    dim_rates: pd.DataFrame       # model x feature departure rate
    cond_rates: pd.DataFrame      # condition x feature departure rate
    human: pd.Series | None       # feature -> human departure rate
    human_mean: float | None
    results_dir: Path
    mentioned: pd.DataFrame | None = None  # condition x feature, mentioned-only rate

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def n_dims(self) -> int:
        return len(DIMENSION_IDS)

    def is_open(self, model: str) -> bool:
        return bool(self.cfg.model_info[model].open_weights)

    def name(self, model: str) -> str:
        return self.cfg.model_name(model)

    def sig(self, cond: str) -> bool:
        return bool(self.posthoc.loc[cond, "p_corrected"] < SIGNIFICANCE)

    def r(self, cond: str) -> float:
        return float(self.posthoc.loc[cond, "rank_biserial_r"])

    def constrained(self) -> list[str]:
        return [v.id for v in self.cfg.prompt_variants if v.id.startswith("constrained_")]

    def rate_diff(self, cond: str) -> pd.Series:
        """Per-feature departure rate under ``cond`` minus the baseline rate."""
        return self.cond_rates.loc[cond] - self.cond_rates.loc["baseline"]


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

def _mentioned_rates(cfg: Config, models: list[str]) -> pd.DataFrame:
    """Departure rate among the judge labels whose feature the description mentions."""
    from alienbench.omission import is_unmentioned
    from alienbench.paths import extractions_path, iter_generations, iter_jsonl

    data_dir = Path(cfg.data_dir)
    rows = []
    for v in cfg.prompt_variants:
        hits = dict.fromkeys(DIMENSION_IDS, 0)
        seen = dict.fromkeys(DIMENSION_IDS, 0)
        for model in models:
            ids = {g["id"] for g in iter_generations(
                data_dir, model, v.id, cap=cfg.samples_per_condition)}
            for judge in cfg.judge_models:
                path = extractions_path(data_dir, judge, model, v.id)
                if not path.exists():
                    continue
                for rec in iter_jsonl(path):
                    if rec.get("generation_id") not in ids or "features" not in rec:
                        continue
                    for d in DIMENSION_IDS:
                        feat = rec["features"].get(d) or {}
                        if is_unmentioned(feat.get("feature_described")):
                            continue
                        seen[d] += 1
                        hits[d] += bool(feat.get("is_departure"))
        rows.append({"prompt_variant": v.id,
                     **{d: hits[d] / seen[d] if seen[d] else np.nan for d in DIMENSION_IDS}})
    return pd.DataFrame(rows).set_index("prompt_variant")


def collect_facts(cfg: Config, m: dict, ward_df: pd.DataFrame, human_df: pd.DataFrame,
                  results_dir: Path, mentioned: bool = True) -> Facts:
    models = [x for x in cfg.models if x in set(ward_df["subject_model"])]
    per_gen = ward_df.groupby(["subject_model", "prompt_variant", "generation_id"])["ward_score"].mean()
    overall = (per_gen.groupby(["subject_model", "generation_id"]).mean()
               .groupby("subject_model").mean().loc[models].sort_values(ascending=False))
    cell = per_gen.groupby(["subject_model", "prompt_variant"]).mean().unstack("prompt_variant").loc[models]

    per_prompt = ward_df.groupby(["prompt_variant", "generation_id"])["ward_score"].mean()
    _, kruskal_p = sps.kruskal(*[g.values for _, g in per_prompt.groupby("prompt_variant")])

    path = results_dir / "table_posthoc.csv"
    posthoc = None
    if path.exists():
        ph = pd.read_csv(path)
        posthoc = ph[ph["group_a"] == "baseline"].set_index("group_b")

    dim_cols = [f"dim_{d}" for d in DIMENSION_IDS]
    gen_dims = ward_df.groupby(["generation_id", "subject_model", "prompt_variant"])[dim_cols].mean()
    dim_rates = gen_dims.groupby("subject_model").mean().loc[models]
    dim_rates.columns = DIMENSION_IDS
    cond_rates = gen_dims.groupby("prompt_variant").mean()
    cond_rates.columns = DIMENSION_IDS

    human = human_mean = None
    if not human_df.empty:
        from alienbench.latex_tables import _human_dimension_means
        hm = _human_dimension_means(human_df)
        human = pd.Series({d: hm[f"dim_{d}"] for d in DIMENSION_IDS})
        human_mean = human_df.groupby("generation_id")["ward_score"].mean().mean()

    facts = Facts(cfg, m, models, overall, cell, posthoc, float(kruskal_p), dim_rates,
                  cond_rates, human, human_mean, results_dir)
    if mentioned:
        facts.mentioned = _mentioned_rates(cfg, models)
    return facts


# ---------------------------------------------------------------------------
# Paragraphs. Each function mirrors one paragraph of paper.tex.
# ---------------------------------------------------------------------------

def _has(m: dict, *names: str) -> bool:
    return all(f"ab{n}" in m for n in names)


def _open_vs_closed(fx: Facts) -> int:
    """+1 when open-weight models have the higher mean, -1 when lower, 0 when equal."""
    o, c = number(fx.m, "OpenMean"), number(fx.m, "ClosedMean")
    return (o > c) - (o < c)


def _verb(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural


def _human_exceptions(fx: Facts) -> list[tuple[str, str]]:
    """(model, feature) pairs whose model rate does not exceed the human rate."""
    return [(x, d) for x in fx.models for d in DIMENSION_IDS
            if fx.dim_rates.loc[x, d] <= fx.human[d]]


def _define_exception_macros(fx: Facts, exc: list[tuple[str, str]]) -> None:
    m = fx.m
    define_count(m, "NHumanExceptions", len(exc))
    define_count(m, "NHumanCells", fx.n_models * fx.n_dims)
    define_count(m, "NHumanAboveCells", fx.n_models * fx.n_dims - len(exc))
    items = [f"{fx.name(x)} on {FEATURE_TEXT[d]} "
             f"({fx.dim_rates.loc[x, d] * 100:.1f}\\% versus {fx.human[d] * 100:.1f}\\%)"
             for x, d in exc]
    define(m, "ExceptionList", join(items))


def _human_scope(fx: Facts, exc: list) -> str:
    """'every feature', 'nearly every feature', ... for the abstract-level sentences."""
    cells = fx.n_models * fx.n_dims
    if not exc:
        return "every feature"
    if len(exc) <= max(1, NEARLY_ALL * cells):
        return "nearly every feature"
    if len(exc) < cells / 2:
        return "most features"
    return None


def abstract(fx: Facts) -> list[Paragraph]:
    m, cfg = fx.m, fx.cfg
    p = Paragraph("Abstract", "Result sentences of the abstract")
    p.say("Model scores range from \\abWorstMean{} for \\abWorstModel{} to \\abBestMean{} for \\abBestModel{}.")

    if _has(m, "OpenMean", "ClosedMean"):
        n_above = int(number(m, "NOpenAboveClosed"))
        verb = _verb(n_above, "scores", "score")
        d = _open_vs_closed(fx)
        if d > 0:
            head = ("Open-weight models have a higher mean score than proprietary models "
                    "(\\abOpenMean{} vs.\\ \\abClosedMean{})")
            p.case.append("Open-weight models have the higher mean overall score.")
        elif d < 0:
            head = ("Proprietary models have a higher mean score than open-weight models "
                    "(\\abClosedMean{} vs.\\ \\abOpenMean{})")
            p.case.append("Proprietary models have the higher mean overall score.")
        else:
            head = "Open-weight and proprietary models have the same mean score (\\abOpenMean{})"
            p.case.append("Open-weight and proprietary models have equal mean overall scores.")
        if n_above:
            link = ", and" if d > 0 else ", although" if d < 0 else ", and"
            p.say(f"{head}{link} \\abNOpenAboveClosedWord{{}} of the \\abNOpenWord{{}} open-weight "
                  f"models {verb} higher than every proprietary model.")
        else:
            p.say(f"{head}, but no open-weight model scores higher than every proprietary model.")
        p.case.append(f"{n_above} open-weight model(s) above the best proprietary model.")

    if number(m, "NCondLeaders") > 1:
        p.say("No single model scores highest under every prompt condition.")
    else:
        p.say("\\abLeaderBaseline{} scores highest under every prompt condition.")
        p.case.append("A single model leads every condition.")

    if number(m, "AlphaWard") >= number(m, "AlphaGood"):
        p.say("The LLM judges agree closely (Krippendorff's $\\alpha = \\abAlphaWard{}$).")
    else:
        p.say("The LLM judges show \\MakeLowercase{\\abAlphaWardBand{}} agreement "
              "(Krippendorff's $\\alpha = \\abAlphaWard{}$).")
        p.case.append("Judge agreement is below the good-agreement threshold.")

    if fx.human is not None:
        exc = _human_exceptions(fx)
        _define_exception_macros(fx, exc)
        scope = _human_scope(fx, exc)
        if scope:
            p.say("We find that the models depart from Earth defaults more often than a baseline "
                  f"of \\abHumanN{{}} human participants on {scope}.")
        else:
            p.say("We find that the models depart from Earth defaults more often than a baseline "
                  "of \\abHumanN{} human participants in \\abNHumanAboveCells{} of the "
                  "\\abNHumanCells{} combinations of model and feature.")
        p.case.append(f"{len(exc)} of {fx.n_models * fx.n_dims} (model, feature) pairs do not "
                      "exceed the human rate.")

    if fx.posthoc is not None:
        p.say(_constrained_lower_sentence(fx, "abstract"))
    return [p]


def _constrained_lower_sentence(fx: Facts, style: str) -> str:
    """How many constrained environments go with fewer departures than the baseline."""
    lower = [c for c in fx.constrained() if fx.sig(c) and fx.r(c) < 0]
    define_count(fx.m, "NConstrainedLower", len(lower))
    define(fx.m, "ConstrainedLowerNames", join([cond_text(fx.cfg, c, 1) for c in lower]))
    n_all = len(fx.constrained())
    if style == "abstract":
        noun, what = "environmental constraints", "fewer departures than the baseline prompt"
    else:
        noun, what = "constrained conditions", "lower scores than the baseline condition"
    if not lower:
        return f"None of the \\abNConstrainedWord{{}} {noun} is associated with {what}."
    if len(lower) == n_all:
        return f"All \\abNConstrainedWord{{}} {noun} are associated with {what}."
    if len(lower) == 1:
        if style == "abstract":
            return (f"Only one of the \\abNConstrainedWord{{}} {noun}, \\abConstrainedLowerNames{{}}, "
                    f"is associated with {what}.")
        return f"Only one of the \\abNConstrainedWord{{}} {noun} is associated with {what}."
    return (f"\\abNConstrainedLowerCap{{}} of the \\abNConstrainedWord{{}} {noun}, "
            f"\\abConstrainedLowerNames{{}}, are associated with {what}.")


def _worst_closed_sentence(fx: Facts, with_lowest: bool) -> tuple[str, str]:
    """The lowest proprietary model against the open-weight models."""
    n_below = int(number(fx.m, "NOpenBelowWorstClosed"))
    n_open = int(number(fx.m, "NOpen"))
    worst_is_open = fx.is_open(fx.overall.index[-1])
    if n_below == 0:
        return ("\\abWorstClosedModel{} (\\abWorstClosedMean{}) scores lower than every "
                "open-weight model.", "No open-weight model scores below the lowest proprietary model.")
    if n_below == n_open:
        return ("Every open-weight model scores lower than \\abWorstClosedModel{} "
                "(\\abWorstClosedMean{}).", "Every open-weight model scores below the lowest "
                "proprietary model.")
    s = ("\\abWorstClosedModel{} (\\abWorstClosedMean{}) scores lower than every open-weight "
         "model except \\abOpenBelowWorstClosedModels{}")
    if with_lowest and worst_is_open:
        s += ", which has the lowest overall score" if n_below == 1 else \
            ", and \\abWorstModel{} has the lowest overall score"
    return s + ".", f"{n_below} open-weight model(s) below the lowest proprietary model."


def _tier_c_sentence(fx: Facts, p: Paragraph) -> str | None:
    cfg, m = fx.cfg, fx.m
    if "C" not in cfg.tiers:
        return None
    tier_models = [x for x in fx.models if cfg.model_info[x].tier == "C"]
    leads = [v.id for v in cfg.prompt_variants if fx.cell[v.id].idxmax() in tier_models]
    define_count(m, "NTierCLeads", len(leads))
    p.case.append(f"Tier C leads {len(leads)} of {len(cfg.prompt_variants)} conditions "
                  f"({', '.join(leads) or 'none'}).")
    p.check.append(f"The tier-C scale in `config.yaml` is `{m.get('abTierCScale', '')}`. "
                   f"Tier C now holds {join([fx.name(x) for x in tier_models])}. "
                   "Check that the scale still describes these models.")
    if not leads:
        return ("None of the \\abTierCDesc{} (\\abTierCScale{} parameters) scores highest "
                "under any prompt condition.")
    if len(leads) == len(cfg.prompt_variants):
        head = "Under every condition"
    elif len(leads) == 1:
        head = "Under one condition"
    else:
        head = "Under some conditions"
    return f"{head}, the highest score comes from one of the \\abTierCDesc{{}} (\\abTierCScale{{}} parameters)."


def introduction(fx: Facts) -> list[Paragraph]:
    m = fx.m
    p = Paragraph("Introduction", "Results paragraph (\"We evaluate ...\")")
    p.say("We evaluate \\abNModelsWord{} LLMs (\\abNGenerations{} generations in total).")
    if number(m, "AlphaWard") >= number(m, "AlphaGood"):
        p.say("The \\abNJudgesWord{} LLM judges agree closely on the Ward Departure Score of each "
              "description (Krippendorff's $\\alpha = \\abAlphaWard{}$).")
    else:
        p.say("The \\abNJudgesWord{} LLM judges show \\MakeLowercase{\\abAlphaWardBand{}} agreement "
              "on the Ward Departure Score of each description (Krippendorff's "
              "$\\alpha = \\abAlphaWard{}$).")

    best, worst = fx.overall.index[0], fx.overall.index[-1]
    open_side = False
    if _has(m, "OpenMean", "ClosedMean"):
        d = _open_vs_closed(fx)
        if d > 0:
            p.say("On average, open-weight models have higher overall scores than proprietary "
                  "models (\\abOpenMean{} vs.\\ \\abClosedMean{}).")
        elif d < 0:
            p.say("On average, proprietary models have higher overall scores than open-weight "
                  "models (\\abClosedMean{} vs.\\ \\abOpenMean{}).")
        else:
            p.say("On average, open-weight and proprietary models have the same overall score "
                  "(\\abOpenMean{}).")
        n_above, n_open = int(number(m, "NOpenAboveClosed")), int(number(m, "NOpen"))
        if n_above == 0:
            p.say("No open-weight model has a higher overall score than all \\abNClosedWord{} "
                  "proprietary models.")
        elif n_above == n_open:
            p.say("All \\abNOpenWord{} open-weight models have higher overall scores than all "
                  "\\abNClosedWord{} proprietary models.")
        elif n_above == 1:
            p.say("One of the \\abNOpenWord{} open-weight models has a higher overall score than "
                  "all \\abNClosedWord{} proprietary models.")
        else:
            p.say("\\abNOpenAboveClosedCap{} of the \\abNOpenWord{} open-weight models have higher "
                  "overall scores than all \\abNClosedWord{} proprietary models.")

        n_led = int(number(m, "NDimOpenLeader"))
        define_count(m, "NDimClosedLeader", fx.n_dims - n_led)
        p.case.append(f"Open-weight models lead {n_led} of {fx.n_dims} features.")
        if n_led > fx.n_dims / 2:
            also = "also " if d > 0 else ""
            p.say(f"Open-weight models {also}score highest on \\abNDimOpenLeaderWord{{}} of the "
                  "\\abNDimsWord{} features.")
            open_side = True
        elif n_led * 2 == fx.n_dims:
            p.say("Open-weight and proprietary models each score highest on "
                  "\\abNDimOpenLeaderWord{} of the \\abNDimsWord{} features.")
        else:
            p.say("Proprietary models score highest on \\abNDimClosedLeaderWord{} of the "
                  "\\abNDimsWord{} features.")

        if fx.is_open(worst):
            also = "also " if open_side else ""
            p.say(f"The lowest overall score {also}comes from an open-weight model "
                  "(\\abWorstMean{} for \\abWorstModel{}).")
        else:
            p.say("The lowest overall score comes from a proprietary model "
                  "(\\abWorstMean{} for \\abWorstModel{}).")
        p.case.append(f"Lowest model: {fx.name(worst)} "
                      f"({'open-weight' if fx.is_open(worst) else 'proprietary'}).")

    n_leads = int(number(m, "NBestCondLeads"))
    n_cond = len(fx.cfg.prompt_variants)
    if n_leads == n_cond:
        p.say("\\abBestModel{} has the highest overall score among the tested models, and it "
              "scores highest under all \\abNConditionsWord{} prompt conditions.")
    elif n_leads == 0:
        p.say("\\abBestModel{} has the highest overall score among the tested models, but it "
              "does not score highest under any single prompt condition.")
    else:
        p.say("\\abBestModel{} has the highest overall score among the tested models, but it "
              "scores highest under only \\abNBestCondLeadsWord{} of the \\abNConditionsWord{} "
              "prompt conditions.")

    if _has(m, "BestClosedModel"):
        if fx.is_open(best):
            p.say("Among the proprietary models, \\abBestClosedModel{} has the highest overall "
                  "score (\\abBestClosedMean{}).")
        else:
            best_open = next(x for x in fx.overall.index if fx.is_open(x))
            define(m, "BestOpenModel", fx.name(best_open))
            define(m, "BestOpenMean", f"{fx.overall[best_open]:.2f}")
            p.say("Among the open-weight models, \\abBestOpenModel{} has the highest overall "
                  "score (\\abBestOpenMean{}).")
            p.case.append("The best model is proprietary, so the sentence names the best "
                          "open-weight model instead.")
    tier = _tier_c_sentence(fx, p)
    if tier:
        p.say(tier)
    if _has(m, "NOpenBelowWorstClosed"):
        s, case = _worst_closed_sentence(fx, with_lowest=False)
        p.say(s)
        p.case.append(case)

    if fx.human is not None:
        p.say("We also collected alien creature descriptions from \\abHumanN{} human participants "
              "under the baseline prompt.")
        exc = _human_exceptions(fx)
        _define_exception_macros(fx, exc)
        if not exc:
            p.say("Every model departs from Earth defaults more often than the human participants "
                  "on every feature.")
        elif len(exc) == 1:
            p.say("Every model departs from Earth defaults more often than the human participants "
                  "on every feature, with one exception (\\abExceptionModel{} on \\abExceptionDim{}).")
        elif _human_scope(fx, exc):
            p.say("Every model departs from Earth defaults more often than the human participants "
                  "on every feature, with \\abNHumanExceptionsWord{} exceptions (\\abExceptionList{}).")
        else:
            p.say("The models depart from Earth defaults more often than the human participants "
                  "in \\abNHumanAboveCells{} of the \\abNHumanCells{} combinations of model and "
                  "feature.")
        p.case.append(f"{len(exc)} (model, feature) pair(s) at or below the human rate.")

    c = Paragraph("Introduction", "Second contribution bullet")
    c.say("An evaluation of \\abNClosedWord{} proprietary and \\abNOpenWord{} open-weight "
          "language models.")
    if _has(m, "NOpenAboveClosed"):
        n_above = int(number(m, "NOpenAboveClosed"))
        if n_above == 0:
            c.say("Overall scores range from \\abWorstMean{} to \\abBestMean{}, and no open-weight "
                  "model has a higher overall score than all \\abNClosedWord{} proprietary models.")
        elif n_above == 1:
            c.say("Overall scores range from \\abWorstMean{} to \\abBestMean{}, and one open-weight "
                  "model has a higher overall score than the \\abNClosedWord{} proprietary models.")
        else:
            c.say("Overall scores range from \\abWorstMean{} to \\abBestMean{}, and "
                  "\\abNOpenAboveClosedWord{} open-weight models have higher overall scores than "
                  "the \\abNClosedWord{} proprietary models.")
    return [p, c]


def method(fx: Facts) -> list[Paragraph]:
    m = fx.m
    p = Paragraph("AlienBench (method)", "Prompt conditions: leave-one-condition-out sentence")
    if m.get("abLocoSameTop") == "true":
        p.say("When any one condition is removed from the mean, the Spearman correlation between "
              "the reduced and the full ranking ranges from \\abLocoRhoMin{} to \\abLocoRhoMax{}, "
              "and \\abBestModel{} remains first.")
    else:
        p.say("When any one condition is removed from the mean, the Spearman correlation between "
              "the reduced and the full ranking ranges from \\abLocoRhoMin{} to \\abLocoRhoMax{}, "
              "and \\abBestModel{} is not first in every reduced ranking.")
        p.case.append("The top model changes when some condition is removed.")

    w = Paragraph("AlienBench (method)", "Ward Departure Score: unmentioned features")
    base = {d: number(m, f"UnmentionedBaseline{_key_from(d)}") for d in DIMENSION_IDS
            if _has(m, f"UnmentionedBaseline{_key_from(d)}")}
    if len(base) >= 2:
        top = sorted(base, key=base.get, reverse=True)[:2]
        a, b = sorted(top, key=DIMENSION_IDS.index)
        w.say("Under the baseline condition, the judges' short feature descriptions mark "
              f"{FEATURE_TEXT[a]} as unmentioned in \\abUnmentionedBaseline{_key_from(a)}{{}} of labels "
              f"and {FEATURE_TEXT[b]} in \\abUnmentionedBaseline{_key_from(b)}{{}} of labels.")
        w.case.append(f"The two most often unmentioned features under the baseline: "
                      f"{FEATURE_TEXT[a]}, {FEATURE_TEXT[b]}.")
    return [p, w]


def _leaders_sentence(fx: Facts) -> str:
    groups: dict[str, list[str]] = {}
    for v in fx.cfg.prompt_variants:
        groups.setdefault(fx.cell[v.id].idxmax(), []).append(v.id)
    if len(groups) == 1:
        conds = next(iter(groups.values()))
        return ("\\abLeaderBaseline{} has the highest score under every condition ("
                + ", ".join(f"\\abLeaderMean{ck(c)}{{}}" for c in conds) + ").")
    parts = []
    for conds in groups.values():
        items = [f"{cond_text(fx.cfg, c, 0)} (\\abLeaderMean{ck(c)}{{}})" for c in conds]
        parts.append(f"\\abLeader{ck(conds[0])}{{}} on {join(items)}")
    return ("The highest score under each condition comes from \\abNCondLeadersWord{} different "
            "models: " + join(parts) + ".")


def model_comparison(fx: Facts) -> list[Paragraph]:
    m, cfg = fx.m, fx.cfg
    half = fx.n_dims / 2
    out = []

    p = Paragraph("Experiments: Model Comparison", "Opening paragraph")
    p.say("Across the \\abNModelsWord{} models, overall Ward Departure Scores range from "
          "\\abWorstMeanCI{} for \\abWorstModel{} to \\abBestMeanCI{} for \\abBestModel{} on the "
          "0--\\abWardMax{} scale (see \\autoref{tab:ward_scores}).")
    if number(m, "NCondLeaders") > 1:
        p.say("No single model scores highest in all \\abNConditionsWord{} prompt conditions "
              "(\\autoref{tab:conditions}).")
    else:
        p.say("\\abLeaderBaseline{} scores highest in all \\abNConditionsWord{} prompt conditions "
              "(\\autoref{tab:conditions}).")
    out.append(p)

    q = Paragraph("Experiments: Model Comparison", "Per-condition leaders and the overall lead")
    q.say(_leaders_sentence(fx))
    q.case.append(f"{int(number(m, 'NCondLeaders'))} different condition leaders.")
    if number(m, "CondMinDeparturePrimed") >= half:
        q.say("Models score uniformly high under the departure-primed condition "
              "(\\abCondMinDeparturePrimed{} or higher).")
    else:
        q.say("Under the departure-primed condition, scores range from \\abCondMinDeparturePrimed{} "
              "to \\abCondMaxDeparturePrimed{}.")
        q.case.append("Some model scores below half the scale under departure priming.")
    if number(m, "CondMaxConstrainedHighGravity") <= half:
        q.say("Scores are uniformly low under the high-gravity condition, where every model "
              "scores \\abCondMaxConstrainedHighGravity{} or lower.")
    else:
        q.say("Under the high-gravity condition, scores range from "
              "\\abCondMinConstrainedHighGravity{} to \\abCondMaxConstrainedHighGravity{}.")
        q.case.append("Some model scores above half the scale under high gravity.")
    n_tied = sum(f"{v:.2f}" == f"{fx.overall.iloc[1]:.2f}" for v in fx.overall.iloc[1:])
    if n_tied >= 2:
        q.say("Overall, \\abBestModel{} has the highest overall score, \\abLeadMargin{} Ward points "
              "above \\abSecondModels{} (\\abSecondTieWord{} \\abSecondMean{}).")
        q.case.append(f"{n_tied} models tie for second place.")
    else:
        q.say("Overall, \\abBestModel{} has the highest overall score, \\abLeadMargin{} Ward points "
              "above \\abSecondModels{} (\\abSecondMean{}).")
        q.case.append("No tie for second place.")
    out.append(q)

    if _has(m, "OpenMean", "ClosedMean"):
        o = Paragraph("Experiments: Model Comparison", "Open-weight and proprietary models")
        d = _open_vs_closed(fx)
        if d > 0:
            o.say("On average, open-weight models have higher overall scores than proprietary models.")
        elif d < 0:
            o.say("On average, proprietary models have higher overall scores than open-weight models.")
        else:
            o.say("On average, open-weight and proprietary models have the same overall score.")
        o.say("The mean overall score is \\abOpenMean{} for the \\abNOpenWord{} open-weight models "
              "and \\abClosedMean{} for the \\abNClosedWord{} proprietary models.")
        n_above = int(number(m, "NOpenAboveClosed"))
        if n_above == 0:
            o.say("No open-weight model scores higher than the best proprietary model, "
                  "\\abBestClosedModel{} (\\abBestClosedMean{}).")
        elif n_above == 1:
            o.say("One open-weight model (\\abOpenAboveClosedModels{}) scores higher than the best "
                  "proprietary model, \\abBestClosedModel{} (\\abBestClosedMean{}).")
        else:
            o.say("\\abNOpenAboveClosedCap{} open-weight models (\\abOpenAboveClosedModels{}) score "
                  "higher than the best proprietary model, \\abBestClosedModel{} (\\abBestClosedMean{}).")
        o.case.append(f"{n_above} open-weight model(s) above the best proprietary model.")
        s, case = _worst_closed_sentence(fx, with_lowest=True)
        o.say(s)
        o.case.append(case)
        o.say("This comparison is descriptive.")
        o.say("It covers \\abNClosedWord{} proprietary models and does not include a significance test.")
        out.append(o)

    if fx.posthoc is not None:
        out.append(condition_tests(fx))
    return out


def condition_tests(fx: Facts) -> Paragraph:
    """The Kruskal-Wallis sentence and the pairwise comparisons with the baseline."""
    cfg = fx.cfg
    t = Paragraph("Experiments: Model Comparison", "Condition effects (Kruskal-Wallis and post-hoc tests)")
    if fx.kruskal_p < SIGNIFICANCE:
        t.say("Scores differ across the \\abNConditionsWord{} prompt conditions (Kruskal-Wallis test, "
              "$H = \\abKruskalH{}$, \\abKruskalP{}).")
    else:
        t.say("Scores do not differ detectably across the \\abNConditionsWord{} prompt conditions "
              "(Kruskal-Wallis test, $H = \\abKruskalH{}$, \\abKruskalP{}).")
        t.case.append("The Kruskal-Wallis test is not significant.")

    # The paper compares the constrained and detailed-description conditions with the baseline.
    conds = [c for c in fx.constrained() + ["detailed_description"] if c in fx.posthoc.index]
    lower = [c for c in conds if fx.sig(c) and fx.r(c) < 0]
    higher = [c for c in conds if fx.sig(c) and fx.r(c) > 0]
    same = [c for c in conds if not fx.sig(c)]

    def stats(c: str) -> str:
        k = ck(c)
        return f"$r = \\abR{k}{{}}$, \\abCILevel{{}}\\% CI \\abRCI{k}{{}}, \\abP{k}{{}}"

    clauses, first = [], True
    for c in lower:
        than = " than under the baseline" if first else ""
        clauses.append(f"lower scores under {cond_text(cfg, c, 1)}{than} ({stats(c)})")
        first = False
    for c in higher:
        slight = "slightly " if abs(fx.r(c)) < SMALL_R else ""
        than = " than under the baseline" if first else ""
        clauses.append(f"{slight}higher scores under {cond_text(cfg, c, 1)}{than} ({stats(c)})")
        first = False
    if same:
        adj = join([cond_text(cfg, c, 2) for c in same], "or")
        noun = "condition" if len(same) == 1 else "conditions"
        printed = {fx.m[f"abP{ck(c)}"] for c in same}
        if len(same) == 1:
            ps = f"\\abP{ck(same[0])}{{}} after correction"
        elif len(printed) == 1:
            ps = f"{'both' if len(same) == 2 else 'all'} \\abP{ck(same[0])}{{}} after correction"
        else:
            ps = join([f"\\abP{ck(c)}{{}}" for c in same]) + ", respectively, after correction"
        clauses.append(f"no detectable difference for the {adj} {noun} ({ps})")
    t.say("Pairwise Mann-Whitney U with Bonferroni correction shows " + join(clauses) + ".")
    t.case.append("Lower: " + (", ".join(lower) or "none") + ". Higher: " + (", ".join(higher) or "none")
                  + ". Not significant: " + (", ".join(same) or "none") + ".")
    t.say("A Monte-Carlo power analysis at the Bonferroni-corrected $\\alpha' = \\abBonferroniAlpha{}$ "
          "returns a minimum detectable $|r| = \\abMinDetectableR{}$ at \\abPowerTarget{}\\% power.")
    return t


def human_comparison(fx: Facts) -> list[Paragraph]:
    m = fx.m
    p = Paragraph("Experiments: Comparison to Human Baseline", "Opening paragraph")
    if fx.human is None:
        p.case.append("No human scores were found, so this paragraph cannot be written. Restore "
                      "`data/extractions/*/human__prolific-baseline` and "
                      "`data/scores/*/human__prolific-baseline`.")
        return [p]
    exc = _human_exceptions(fx)
    _define_exception_macros(fx, exc)
    radar = "(\\autoref{fig:ward_radar})"
    if not exc:
        p.say(f"Every model has a higher departure rate than the human participants on every feature {radar}.")
    elif len(exc) == 1:
        p.say("Every model has a higher departure rate than the human participants on every feature "
              "except \\abExceptionDim{} for \\abExceptionModel{}, which scores \\abExceptionRate{} "
              f"versus \\abExceptionHumanRate{{}} for humans {radar}.")
    elif _human_scope(fx, exc):
        p.say("Every model has a higher departure rate than the human participants on every feature, "
              f"with \\abNHumanExceptionsWord{{}} exceptions: \\abExceptionList{{}} {radar}.")
    else:
        p.say("The models have a higher departure rate than the human participants in "
              "\\abNHumanAboveCells{} of the \\abNHumanCells{} combinations of model and feature "
              f"{radar}.")
    p.case.append(f"{len(exc)} (model, feature) pair(s) at or below the human rate"
                  + (": " + "; ".join(f"{fx.name(x)} on {FEATURE_TEXT[d]}" for x, d in exc) if exc else "."))
    p.say("The feature with the highest departure rate across models is \\abMostDim{} "
          "(human \\abMostDimHuman{}, model range \\abMostDimMin{}--\\abMostDimMax{}), and the "
          "feature with the lowest departure rate is \\abLeastDim{} (human \\abLeastDimHuman{}, "
          "model range \\abLeastDimMin{}--\\abLeastDimMax{}).")
    if _has(m, "GapOneRatio", "GapTwoRatio"):
        if number(m, "GapTwoRatio") > 1:
            p.say("In relative terms (mean rate across models divided by the human rate), the "
                  "model-to-human gap is largest on \\abGapOneDim{} (human \\abGapOneHuman{}, model "
                  "range \\abGapOneMin{}--\\abGapOneMax{}, ratio \\abGapOneRatio{}) and \\abGapTwoDim{} "
                  "(human \\abGapTwoHuman{}, model range \\abGapTwoMin{}--\\abGapTwoMax{}, ratio "
                  "\\abGapTwoRatio{}).")
        else:
            p.case.append("The second-largest ratio is at most 1, so 'gap' would describe models "
                          "below the human rate. Rewrite by hand.")
    n_leaders = int(number(m, "NDimLeaders"))
    if n_leaders > 1:
        p.say("Across the \\abNDimsWord{} features, the highest departure rate comes from "
              "\\abNDimLeadersWord{} different models.")
        p.say("Models with similar overall scores reach them through departures on different features.")
        p.check.append("'Models with similar overall scores reach them through departures on "
                       "different features' is read off the radar figure. Check it there.")
    else:
        define(m, "DimLeaderModel", fx.name(fx.dim_rates[DIMENSION_IDS[0]].idxmax()))
        p.say("\\abDimLeaderModel{} has the highest departure rate on all \\abNDimsWord{} features.")
    p.case.append(f"{n_leaders} different feature leaders.")
    return [p]


def reliability(fx: Facts) -> list[Paragraph]:
    m = fx.m
    p = Paragraph("Experiments: Reliability of the LLM Judge Panel", "Reliability paragraph")
    if not _has(m, "AlphaWard"):
        p.case.append("Fewer than two judges: no reliability statistics.")
        return [p]
    p.say("The Ward Departure Score reaches Krippendorff's $\\alpha = \\abAlphaWard{}$, which "
          "indicates \\MakeLowercase{\\abAlphaWardBand{}} agreement among the judges \\citep{landis1977}.")
    bands = [(b, int(number(m, f"N{b}Dims"))) for b in ("Excellent", "Good", "Moderate", "Fair", "Poor")]
    bands = [(b, n) for b, n in bands if n]
    parts = []
    for i, (b, n) in enumerate(bands):
        count = f"\\abN{b}Dims{'Cap' if i == 0 else 'Word'}{{}}"
        if i == 0:
            parts.append(f"{count} {'feature falls' if n == 1 else 'features fall'} in the {b} band")
        else:
            parts.append(f"{count} in the {b} band")
    p.say(join(parts) + ".")
    p.case.append("Bands with features: " + ", ".join(f"{b} {n}" for b, n in bands) + ".")
    n_clear = int(number(m, "NClearDims"))
    below = fx.n_dims - n_clear
    if below == 0:
        p.say("For the departure labels, all \\abNDimsWord{} features clear the "
              "$\\alpha \\geq \\abAlphaGood{}$ good-agreement threshold (lowest $\\alpha = "
              "\\abAlphaMin{}$ on \\abAlphaMinDim{}).")
    elif below == 1:
        p.say("For the departure labels, \\abNClearDimsWord{} of the \\abNDimsWord{} features clear "
              "the $\\alpha \\geq \\abAlphaGood{}$ good-agreement threshold, and only \\abBelowGoodDims{} "
              "falls below it ($\\alpha = \\abBelowGoodMin{}$).")
    elif n_clear == 0:
        p.say("For the departure labels, none of the \\abNDimsWord{} features clears the "
              "$\\alpha \\geq \\abAlphaGood{}$ good-agreement threshold.")
    else:
        p.say("For the departure labels, \\abNClearDimsWord{} of the \\abNDimsWord{} features clear "
              "the $\\alpha \\geq \\abAlphaGood{}$ good-agreement threshold, and \\abBelowGoodDims{} "
              "fall below it (lowest $\\alpha = \\abBelowGoodMin{}$).")
    p.case.append(f"{below} feature(s) below the good-agreement threshold.")
    if _has(m, "NParseErrors"):
        if int(number(m, "NParseErrors")) == 0:
            p.say("All \\abNExtractionCalls{} judge calls returned valid JSON.")
        else:
            p.say("\\abNParseErrorsCap{} of the \\abNExtractionCalls{} judge calls did not return "
                  "valid JSON.")
    return [p]


def _para_coverage(fx: Facts, p: Paragraph) -> bool:
    path = fx.results_dir / "table_ablation_prompt.csv"
    if not path.exists():
        p.case.append("`table_ablation_prompt.csv` is missing. Run `python -m alienbench ablation prompt`.")
        return False
    covered = set(pd.read_csv(path)["subject_model"])
    missing = [fx.name(x) for x in fx.models if x not in covered]
    extra = sorted(covered - set(fx.models))
    if missing or extra:
        p.check.append(
            f"The paraphrase ablation does not match the current model panel. It covers "
            f"{len(set(fx.models) & covered)} of {fx.n_models} models"
            + (f" (missing: {', '.join(missing)})" if missing else "")
            + (f" and includes models no longer in the panel ({', '.join(extra)})" if extra else "")
            + ". Rerun `python -m alienbench ablation prompt`, then `paper`, before using these "
              "sentences. Every number below comes from the old panel.")
    return True


def _para_state(fx: Facts) -> tuple[str, bool]:
    """('all' | 'most' | 'few', absolute scores sensitive?)."""
    m = fx.m
    n, total = int(number(m, "ParaNAboveNull")), int(number(m, "ParaNPairs"))
    rank = "all" if n == total else "most" if n * 2 >= total else "few"
    return rank, number(m, "ParaMadMedian") > number(m, "ParaMadBar")


def _para_conclusion(rank: str, sensitive: bool, style: str) -> str:
    stable = rank != "few"
    if style == "main":
        if stable and sensitive:
            return "Model rankings are thus stable under rewording, and absolute scores are sensitive to it."
        if stable:
            return "Model rankings and absolute scores are thus both stable under rewording."
        if sensitive:
            return "Model rankings and absolute scores thus both depend on the wording."
        return "Model rankings thus depend on the wording, and absolute scores are stable under rewording."
    if stable and sensitive:
        return "The model ranking is stable across wordings, but the score level depends on the wording."
    if stable:
        return "The model ranking and the score level are both stable across wordings."
    if sensitive:
        return "The model ranking and the score level both depend on the wording."
    return "The model ranking depends on the wording, but the score level is stable across wordings."


def _dim_loo_state(fx: Facts) -> tuple[bool, int]:
    loo = pd.read_csv(fx.results_dir / "table_ablation_dimensions_loo.csv")
    n = int(loo["rho_above_null_hi95"].sum())
    define_count(fx.m, "DimLooNAboveNull", n)
    return n == len(loo), n


def _judge_tables(fx: Facts) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    s, l = (fx.results_dir / "table_ablation_judges_single.csv",
            fx.results_dir / "table_ablation_judges_loo.csv")
    if not (s.exists() and l.exists()):
        return None
    return pd.read_csv(s), pd.read_csv(l)


def _same_family(fx: Facts, style: str) -> list[str]:
    m = fx.m
    if not _has(m, "SfNPairs"):
        return []
    n_pairs = int(number(m, "SfNPairs"))
    z, neg, pos = (int(number(m, f"SfN{k}")) for k in ("Zero", "Negative", "Positive"))
    if style == "main":
        parts = []
        for n, macro, what in ((z, "SfNZeroWord", "unchanged"), (neg, "SfNNegativeWord", "higher"),
                               (pos, "SfNPositiveWord", "lower")):
            if n == n_pairs:
                return [f"Across the \\abSfNPairsWord{{}} same-family pairs, the rank is {what} for "
                        "all of them."]
            if n:
                parts.append(f"{what} for \\ab{macro}{{}}")
        return ["Across the \\abSfNPairsWord{} same-family pairs, the rank is " + join(parts) + "."]
    out = []
    if z:
        out.append("A zero $\\Delta$, which indicates no change, occurs for \\abSfNZeroWord{} of the "
                   "\\abSfNPairsWord{} pairs (\\abSfZeroModels{}).")
    else:
        out.append("No pair has a zero $\\Delta$.")
    if neg:
        out.append("A negative $\\Delta$, the opposite direction from self-preference, occurs for "
                   "\\abSfNNegativeWord{} of the pairs (\\abSfNegativeModels{}).")
    else:
        out.append("No pair has a negative $\\Delta$, the opposite direction from self-preference.")
    if pos == 0:
        out.append("No same-family pair shows positive $\\Delta$ ($\\abSfNPositive{}$ of "
                   "$\\abSfNPairs{}$), so we find no rank-level evidence of self-preference inflation "
                   "in the panel (Panel~C of \\autoref{tab:ablation_judges}).")
    else:
        out.append("\\abSfNPositiveCap{} of the \\abSfNPairsWord{} same-family pairs "
                   f"{_verb(pos, 'shows', 'show')} a positive $\\Delta$ (\\abSfPositiveModels{{}}), "
                   "which is consistent with self-preference inflation (Panel~C of "
                   "\\autoref{tab:ablation_judges}).")
    return out


def ablations(fx: Facts) -> list[Paragraph]:
    m = fx.m
    out = []

    pp = Paragraph("Ablations and Sensitivity Analyses", "Prompt Paraphrase Sensitivity")
    if _para_coverage(fx, pp) and _has(m, "ParaNAboveNull"):
        rank, sensitive = _para_state(fx)
        pp.say("We regenerate responses under \\abNParaphrasesWord{} hand-written rewordings of the "
               "baseline prompt.")
        if rank == "all":
            pp.say("Per-model rankings are stable across rewordings: all \\abParaNPairs{} paraphrase "
                   "pairs exceed a permutation null for Spearman $\\rho$ (median $\\rho = \\abParaRhoMedian{}$).")
        elif rank == "most":
            pp.say("Per-model rankings are stable across rewordings: \\abParaNAboveNull{} of "
                   "\\abParaNPairs{} paraphrase pairs exceed a permutation null for Spearman $\\rho$ "
                   "(median $\\rho = \\abParaRhoMedian{}$).")
        else:
            pp.say("Per-model rankings are not stable across rewordings: only \\abParaNAboveNull{} of "
                   "\\abParaNPairs{} paraphrase pairs exceed a permutation null for Spearman $\\rho$ "
                   "(median $\\rho = \\abParaRhoMedian{}$).")
        pp.say("Absolute scores differ by a median of \\abParaMadMedian{} Ward points between rewordings.")
        pp.say("\\abParaOutlierModel{} scores between \\abParaOutlierMin{} and \\abParaOutlierMax{} "
               "under the rewordings and \\abParaOutlierBase{} under the baseline prompt.")
        pp.say(_para_conclusion(rank, sensitive, "main"))
        pp.case.append(f"Rank stability: {rank} pairs above the null. Median MAD "
                       f"{m['abParaMadMedian']} against the bar {m['abParaMadBar']}.")
        pp.check.append("Whether \\abParaOutlierModel{} is an outlier is a judgment. Read "
                        "`tab_ablation_prompt.tex`.")
    out.append(pp)

    fs = Paragraph("Ablations and Sensitivity Analyses", "Feature Sensitivity")
    if (fx.results_dir / "table_ablation_dimensions_loo.csv").exists() and _has(m, "DimMinKRho"):
        all_above, _ = _dim_loo_state(fx)
        fs.say("We recompute the model ranking from subsets of the \\abNDimsWord{} features.")
        if all_above:
            fs.say("Rankings without one feature correlate with the full ranking at "
                   "$\\rho \\in [\\abDimLooRhoMin{}, \\abDimLooRhoMax{}]$, and every reduced ranking "
                   "exceeds its permutation null.")
        else:
            fs.say("Rankings without one feature correlate with the full ranking at "
                   "$\\rho \\in [\\abDimLooRhoMin{}, \\abDimLooRhoMax{}]$, and \\abDimLooNAboveNullWord{} "
                   "of the \\abNDimsWord{} reduced rankings exceed their permutation null.")
        k = int(number(m, "DimMinKRho"))
        if k < 0:
            fs.say("No subset of fewer than \\abNDimsWord{} features reaches $\\rho \\geq \\abDimRhoBar{}$.")
        elif k == 1:
            fs.say("The best single feature reaches $\\rho \\geq \\abDimRhoBar{}$.")
        else:
            fs.say("The best subset of \\abDimMinKRho{} features reaches $\\rho \\geq \\abDimRhoBar{}$.")
        k_abs = m.get("abDimMadBelowOneK", "--")
        if k_abs != "--" and int(k_abs) > max(k, 0):
            fs.say("The absolute score requires more features: its mean absolute difference from the "
                   "full score falls below \\abParaMadBar{} Ward points only with \\abDimMadBelowOneK{} features.")
        elif k_abs != "--":
            fs.say("Its mean absolute difference from the full score falls below \\abParaMadBar{} Ward "
                   "points with \\abDimMadBelowOneK{} features.")
        fs.case.append(f"Smallest subset reaching the rho bar: k = {k}. Absolute score: MAD below the "
                       f"bar from k = {k_abs}.")
    out.append(fs)

    jp = Paragraph("Ablations and Sensitivity Analyses", "Judge Panel Composition")
    tables = _judge_tables(fx)
    if tables:
        single, loo = tables
        misses = single[~single["rho_heldout_above_stable_bar"]]
        jp.say("We remove judges from the panel and measure rank stability, inter-rater "
               "reliability, and same-family sensitivity.")
        jp.say("The ranking from each single judge correlates with the ranking from the remaining "
               "judges at $\\rho \\in [\\abJudgeSingleRhoMin{}, \\abJudgeSingleRhoMax{}]$.")
        if len(misses) == 0:
            jp.say("All \\abNJudgesWord{} single judges clear the preregistered bar of "
                   "$\\rho \\geq \\abJudgeRhoBar{}$.")
        elif len(misses) == 1:
            jp.say("Of the \\abNJudgesWord{} single judges, \\abJudgeSingleNClearBar{} clear the "
                   "preregistered bar of $\\rho \\geq \\abJudgeRhoBar{}$, and \\abJudgeMissName{} "
                   "reaches $\\rho = \\abJudgeMissRho{}$.")
        else:
            jp.say("Of the \\abNJudgesWord{} single judges, \\abJudgeSingleNClearBar{} clear the "
                   "preregistered bar of $\\rho \\geq \\abJudgeRhoBar{}$.")
        jp.case.append(f"{len(misses)} single judge(s) below the rank-stability bar.")
        jp.say("Panels without one judge have Krippendorff's $\\alpha$ between \\abJudgeLooAlphaMin{} "
               "and \\abJudgeLooAlphaMax{}.")
        jp.say("Self-preference would predict a lower rank for a model after removal of the judge "
               "from its own model family.")
        for s in _same_family(fx, "main"):
            jp.say(s)
    out.append(jp)
    return out


def discussion(fx: Facts) -> list[Paragraph]:
    m = fx.m
    out = []
    d = Paragraph("Discussion", "\"Models differ in structural novelty\" paragraph")
    d.say("Models differ in structural novelty.")
    if _has(m, "AlphaWard"):
        if int(number(m, "NClearDims")) == fx.n_dims:
            d.say("Krippendorff's $\\alpha = \\abAlphaWard{}$ on the Ward Departure Score places the "
                  "joint extraction in the \\abAlphaWardBand{} reliability band, and all \\abNDimsWord{} "
                  "features clear the $\\alpha \\geq \\abAlphaGood{}$ good-agreement threshold.")
        else:
            d.say("Krippendorff's $\\alpha = \\abAlphaWard{}$ on the Ward Departure Score places the "
                  "joint extraction in the \\abAlphaWardBand{} reliability band, and \\abNClearDimsWord{} "
                  "of \\abNDimsWord{} features clear the $\\alpha \\geq \\abAlphaGood{}$ good-agreement "
                  "threshold.")
    if int(number(m, "NDimLeaders")) > 1:
        d.say("No single model has the highest departure rate on every feature.")
        d.say("\\abNDimLeadersCap{} different models lead at least one feature, and the radial "
              "profiles in \\autoref{fig:ward_radar} differ in shape, not only in scale.")
        d.say("Models differ in which features they depart on, not only in how many.")
        d.check.append("'The radial profiles differ in shape, not only in scale' is read off the "
                       "figure. Check it there.")
    else:
        define(m, "DimLeaderModel", fx.name(fx.dim_rates[DIMENSION_IDS[0]].idxmax()))
        d.say("\\abDimLeaderModel{} has the highest departure rate on every feature.")
        d.case.append("A single model leads every feature.")
    out.append(d)

    if fx.human is not None:
        h = Paragraph("Discussion", "Human comparison (commented out in the paper)")
        exc = _human_exceptions(fx)
        if not exc:
            h.say("Models depart from Earth defaults more than humans on every feature.")
        else:
            h.say("Models depart from Earth defaults more than humans on every feature except "
                  "\\abExceptionDimsText{}.")
            h.check.append("\\abExceptionDimsText{} lists features on which at least one model is "
                           "at or below the human rate. 'Models depart more on every feature "
                           "except X' then holds only in the aggregate.")
        h.say("The relative gap is widest on \\abGapOneDim{} and \\abGapTwoDim{}.")
        out.append(h)

    if _has(m, "ParaRhoMedian", "DimLooRhoMin"):
        a = Paragraph("Discussion", "Stability paragraph (\"The ablations in Section ...\")")
        _para_coverage(fx, a)
        rank, _ = _para_state(fx)
        a.say("The ablations in Section~\\ref{sec:ablations} characterize the conditions under which "
              "the benchmark gives stable answers.")
        a.say({"all": "Model rankings are preserved across rewordings (median $\\rho = \\abParaRhoMedian{}$).",
               "most": "Model rankings are mostly preserved across rewordings (median $\\rho = \\abParaRhoMedian{}$).",
               "few": "Model rankings are not preserved across most rewordings (median $\\rho = \\abParaRhoMedian{}$)."}[rank])
        all_above, _ = _dim_loo_state(fx)
        if all_above:
            a.say("Rankings are robust to removing features, with leave-one-out correlations of "
                  "$\\rho \\in [\\abDimLooRhoMin{}, \\abDimLooRhoMax{}]$.")
        else:
            a.say("Rankings are robust to removing most single features, with leave-one-out "
                  "correlations of $\\rho \\in [\\abDimLooRhoMin{}, \\abDimLooRhoMax{}]$.")
        a.say("Absolute mean Ward Departure Scores do not stabilize until at least "
              "$\\abDimMadBelowOneK{}$ features are retained.")
        a.say("AlienBench is therefore intended for relative comparisons across models with a fixed "
              "set of features and defaults.")
        out.append(a)

    if fx.human is not None:
        b = Paragraph("Discussion: Is the Structured Imagination Effect present in LLMs?",
                      "Baseline condition (commented out in the paper)")
        effect = number(m, "BaselineModelMean") < fx.n_dims / 2
        weaker = number(m, "BaselineModelMean") > number(m, "HumanMean")
        if effect and weaker:
            b.say("Under the baseline condition, the models show the structured imagination effect, "
                  "but in a weaker form than observed in human participants.")
        elif effect:
            b.say("Under the baseline condition, the models show the structured imagination effect at "
                  "least as strongly as the human participants.")
        else:
            b.say("Under the baseline condition, a typical model description departs from Earth "
                  "defaults on at least half of the features.")
        b.case.append(f"Model mean under the baseline {m['abBaselineModelMean']} of {fx.n_dims}, "
                      f"humans {m['abHumanMean']}.")
        if effect:
            b.say("The mean Ward Departure Score of the models under the baseline condition is "
                  "\\abBaselineModelMean{} of \\abWardMax{}. Thus, a typical description keeps most "
                  "features Earth-typical.")
        else:
            b.say("The mean Ward Departure Score of the models under the baseline condition is "
                  "\\abBaselineModelMean{} of \\abWardMax{}.")
        b.say("Model means range from \\abCondMinBaseline{} to \\abCondMaxBaseline{}.")
        n_above = int(number(m, "NAboveHumanBaseline"))
        if n_above == fx.n_models:
            b.say("The human participants score \\abHumanMean{} under the same prompt, and all "
                  "\\abNModelsWord{} models score higher.")
        elif n_above == 0:
            b.say("The human participants score \\abHumanMean{} under the same prompt, and no model "
                  "scores higher.")
        else:
            b.say("The human participants score \\abHumanMean{} under the same prompt, and "
                  "\\abNAboveHumanBaselineWord{} of the \\abNModelsWord{} models score higher.")
        out.append(b)

    if fx.posthoc is not None:
        out.append(environments(fx))
    return out


def environments(fx: Facts) -> Paragraph:
    """Sec. 5.1: the constrained environments at the level of single features."""
    m, cfg = fx.m, fx.cfg
    e = Paragraph("Discussion: Is the Structured Imagination Effect present in LLMs?", "Constrained environments (feature level)")
    e.say("The constrained environments show the effect at the level of single features.")

    def vs(cond: str, d: str) -> str:
        return (f"\\abRate{ck(cond)}{_key_from(d)}{{}} versus \\abRateBaseline{_key_from(d)}{{}} "
                "under the baseline condition")

    pair = [c for c in ("constrained_no_light", "constrained_ammonia") if c in fx.cond_rates.index]
    tops = {c: fx.rate_diff(c).sort_values(ascending=False) for c in pair}
    concerned_top = all(CONCERNED_FEATURE[c] in list(tops[c].index[:2]) and tops[c][CONCERNED_FEATURE[c]] > 0
                        for c in pair)
    n_lower = {c: int((fx.rate_diff(c) < 0).sum()) for c in pair}
    for c in pair:
        top = tops[c]
        e.case.append(f"{cond_text(cfg, c, 1)}: largest increases "
                      + ", ".join(f"{FEATURE_TEXT[d]} {top[d] * 100:+.1f} pp" for d in top.index[:3])
                      + f". {n_lower[c]} feature(s) lower than under the baseline.")
    several = all(n >= 2 for n in n_lower.values())
    if len(pair) == 2 and concerned_top:
        tail = ", while several other features become more Earth-typical" if several else ""
        nl, am = pair
        e.say("Under no light and under ammonia oceans, the largest increases in departure occur on "
              "the features that the specific environment concerns, such as sensory organs without "
              f"light ({vs(nl, CONCERNED_FEATURE[nl])}) and locomotion in ammonia "
              f"({vs(am, CONCERNED_FEATURE[am])}){tail}.")
        e.say("The models tend to depart on the features that a constraint in the prompt concerns "
              "and retain Earth defaults elsewhere.")
    else:
        e.case.append("The feature each environment concerns (sensory organs without light, "
                      "locomotion in ammonia) is not among its two largest increases, so the "
                      "paragraph reports the largest increases instead.")
        for c in pair:
            a, b = tops[c].index[:2]
            k = ck(c)
            define_count(m, f"NLower{k}", n_lower[c])
            tail = (f", and \\abNLower{k}Word{{}} other features are lower than under the baseline "
                    "condition") if n_lower[c] else ""
            e.say(f"Under {cond_text(cfg, c, 1)}, the largest increases in departure rate occur on "
                  f"{FEATURE_TEXT[a]} ({vs(c, a)}) and {FEATURE_TEXT[b]} "
                  f"(\\abRate{k}{_key_from(b)}{{}} versus \\abRateBaseline{_key_from(b)}{{}}){tail}.")
        e.check.append("Without the concerned-feature pattern, the sentence 'The models tend to "
                       "depart where in features affected by a constraint in the prompt and "
                       "retain Earth defaults elsewhere' is not supported. Decide whether the "
                       "increases above relate to the environments.")

    hg = "constrained_high_gravity"
    if hg in fx.cond_rates.index:
        k = ck(hg)
        diff = fx.rate_diff(hg)
        lower_feats = [d for d in DIMENSION_IDS if diff[d] < 0]
        not_lower = [d for d in DIMENSION_IDS if diff[d] >= 0]
        define_count(m, "NLowerConstrainedHighGravity", len(lower_feats))
        if not not_lower:
            feats = "the departure rate is lower on every feature"
        elif len(not_lower) == 1:
            d = not_lower[0]
            feats = (f"the departure rate is lower on every feature except {FEATURE_TEXT[d]} "
                     f"({vs(hg, d)})")
        else:
            feats = ("the departure rate is lower on \\abNLowerConstrainedHighGravityWord{} of the "
                     "\\abNDimsWord{} features")
        n_below = int(number(m, f"NBelowBaseline{k}"))
        if n_below == fx.n_models:
            models = "every model scores lower"
        elif n_below == fx.n_models - 1 and _has(m, f"NotBelow{k}Model"):
            models = f"every model except \\abNotBelow{k}Model{{}} scores lower"
        else:
            models = (f"\\abNBelowBaseline{k}Word{{}} of the \\abNModelsWord{{}} models score lower")
        e.case.append(f"High gravity: {len(lower_feats)} of {fx.n_dims} features lower; "
                      f"{n_below} of {fx.n_models} models below their baseline.")
        if fx.sig(hg) and fx.r(hg) < 0:
            e.say("Under high gravity, the effect is stronger than under the baseline condition: "
                  f"{feats}, and {models} (rank-biserial $r = \\abR{k}{{}}$).")
        else:
            e.say("Under high gravity, scores do not differ detectably from the baseline condition "
                  f"(\\abP{k}{{}}): {feats}, and {models}.")

        unm = {d: number(m, f"Unmentioned{k}{_key_from(d)}") - number(m, f"UnmentionedBaseline{_key_from(d)}")
               for d in DIMENSION_IDS}
        more = sorted([d for d in DIMENSION_IDS if unm[d] > 0], key=unm.get, reverse=True)
        e.case.append("Unmentioned more often under high gravity: "
                      + (", ".join(f"{FEATURE_TEXT[d]} {unm[d]:+.1f} pp" for d in more) or "none") + ".")
        if not more:
            e.say("Under high gravity, the models do not mention any feature less often than under "
                  "the baseline condition.")
        elif len(more) <= 3:
            e.say(f"Under high gravity, the models mention {join([FEATURE_TEXT[d] for d in more])} "
                  "less often than under the baseline condition.")
        else:
            define_count(m, "NLessMentionedConstrainedHighGravity", len(more))
            e.say("Under high gravity, the models mention \\abNLessMentionedConstrainedHighGravityWord{} "
                  "of the \\abNDimsWord{} features less often than under the baseline condition, most "
                  f"of all {join([FEATURE_TEXT[d] for d in more[:3]])}.")

        if fx.mentioned is not None:
            # Pooled over features: departures among the mentioned labels.
            base_rate = float(np.nanmean(fx.mentioned.loc["baseline"]))
            hg_rate = float(np.nanmean(fx.mentioned.loc[hg]))
            define(m, "MentionedRateBaseline", f"{base_rate * 100:.1f}\\%")
            define(m, f"MentionedRate{k}", f"{hg_rate * 100:.1f}\\%")
            e.case.append(f"Departure rate among mentioned features (mean over features): baseline "
                          f"{base_rate * 100:.1f}%, high gravity {hg_rate * 100:.1f}%. New macros: "
                          f"`MentionedRateBaseline`, `MentionedRate{k}`.")
            if hg_rate < base_rate:
                e.say("Restricted to the features that the models mention, the departure rate under "
                      "high gravity remains lower than under the baseline condition.")
            else:
                e.say("Restricted to the features that the models mention, the departure rate under "
                      f"high gravity is not lower than under the baseline condition (\\abMentionedRate{k}{{}} "
                      "versus \\abMentionedRateBaseline{}).")

    retain = number(m, "CondMaxBaseline") < fx.n_dims / 2 if _has(m, "CondMaxBaseline") else False
    leaders = {fx.rate_diff(c).idxmax() for c in fx.constrained() if c in fx.cond_rates.index}
    vary = len(leaders) > 1
    if retain and vary:
        e.say("Across the baseline and the three environments, the models thus show the structured "
              "imagination effect: they retain most Earth defaults, and the features on which they "
              "depart vary with the environment.")
    elif number(m, "BaselineModelMean") < fx.n_dims / 2:
        e.say("Across the baseline and the three environments, a typical model description retains "
              "most Earth defaults.")
        e.check.append("Not every model keeps most features Earth-typical under the baseline, or "
                       "the largest increase falls on the same feature in every environment. "
                       "Decide whether 'the models thus show the structured imagination effect' "
                       "still holds.")
    e.case.append(f"Highest model mean under the baseline: {m.get('abCondMaxBaseline')}. Feature "
                  f"with the largest increase per environment: "
                  f"{', '.join(FEATURE_TEXT[d] for d in leaders)}.")
    return e


def conclusion(fx: Facts) -> list[Paragraph]:
    m, cfg = fx.m, fx.cfg
    p = Paragraph("Conclusion", "Results paragraph")
    p.say("The \\abNJudgesWord{}-judge LLM panel reaches Krippendorff's $\\alpha = \\abAlphaWard{}$ "
          "on the Ward Departure Score, and Overall Ward Departure Scores span \\abWorstMean{} to "
          "\\abBestMean{} across the \\abNModelsWord{} reference models.")
    if fx.human is not None:
        exc = _human_exceptions(fx)
        if not exc:
            p.say("Every model exceeds the human baseline on every feature, and the relative gap is "
                  "widest on \\abGapOneDim{} and \\abGapTwoDim{}.")
        else:
            p.say("Every model exceeds the human baseline on every feature except "
                  "\\abExceptionDimsText{}, and the relative gap is widest on \\abGapOneDim{} and "
                  "\\abGapTwoDim{}.")
        if number(m, "BaselineModelMean") > number(m, "HumanMean"):
            p.say("Under the baseline condition, the models show the structured imagination effect of "
                  "\\citet{ward1994} in a weaker form than the human participants (mean score "
                  "\\abBaselineModelMean{} versus \\abHumanMean{}).")
        else:
            p.say("Under the baseline condition, the models show the structured imagination effect of "
                  "\\citet{ward1994} at least as strongly as the human participants (mean score "
                  "\\abBaselineModelMean{} versus \\abHumanMean{}).")
    if fx.posthoc is not None:
        p.say(_constrained_lower_sentence(fx, "conclusion"))
        hg = "constrained_high_gravity"
        k = ck(hg)
        if int(number(m, f"NBelowBaseline{k}")) == fx.n_models:
            p.say("Under the high-gravity condition, every model scores \\abCondMaxConstrainedHighGravity{} "
                  "or lower, and all \\abNModelsWord{} models score lower than under the baseline condition.")
        else:
            p.say("Under the high-gravity condition, every model scores \\abCondMaxConstrainedHighGravity{} "
                  "or lower, and \\abNBelowBaselineConstrainedHighGravityWord{} of the \\abNModelsWord{} "
                  "models score lower than under the baseline condition.")
        clauses = []
        for c in ("constrained_no_light", "constrained_ammonia"):
            if c not in fx.posthoc.index:
                continue
            adj = cond_text(cfg, c, 2)
            if not fx.sig(c):
                clauses.append(f"scores under the {adj} condition do not differ from the baseline condition")
            elif fx.r(c) > 0:
                slight = "slightly " if abs(fx.r(c)) < SMALL_R else ""
                clauses.append(f"scores under the {adj} condition are {slight}higher")
            else:
                clauses.append(f"scores under the {adj} condition are lower")
        if clauses:
            p.say(cap(", and ".join(clauses)) + ".")
        p.say("The present results do not establish whether the effect is stronger in harsh environments.")
        if fx.sig(hg) and fx.r(hg) < 0:
            p.check.append("The Discussion says that under high gravity 'the effect is stronger than under "
                           "the baseline condition'. The last sentence here says the results do not "
                           "establish whether the effect is stronger in harsh environments. Check "
                           "that the two sentences are consistent.")

    a = Paragraph("Conclusion", "Ablation sentence (\"The feature and judge ablations show ...\")")
    tables = _judge_tables(fx)
    if tables and _has(m, "DimMinKRho", "DimMadBelowOneK"):
        _, loo = tables
        panels_ok = bool(loo["rho_heldout_above_stable_bar"].all())
        rubric_ok = 0 < int(number(m, "DimMinKRho")) < fx.n_dims
        if rubric_ok and panels_ok:
            rank = "per-model rankings are recoverable from reduced rubrics and reduced panels"
        elif rubric_ok:
            rank = "per-model rankings are recoverable from reduced rubrics but not from every reduced panel"
        elif panels_ok:
            rank = "per-model rankings are recoverable from reduced panels but not from reduced rubrics"
        else:
            rank = "per-model rankings are not recoverable from reduced rubrics or reduced panels"
        k_abs = m.get("abDimMadBelowOneK")
        if k_abs not in (None, "--") and int(k_abs) == fx.n_dims:
            abs_part = "the absolute Ward Departure Score requires all \\abNDimsWord{} features to stabilize"
        else:
            define(m, "DimMadBelowOneKWord", word(int(k_abs)))
            abs_part = ("the absolute Ward Departure Score requires at least \\abDimMadBelowOneKWord{} "
                        "of the \\abNDimsWord{} features to stabilize")
        a.say(f"The feature and judge ablations show that {rank}, while {abs_part}.")
        a.case.append(f"Rubrics: smallest subset reaching the rho bar k = {m['abDimMinKRho']}. "
                      f"Panels: all leave-one-judge-out panels clear the bar = {panels_ok}. "
                      f"Absolute score: MAD below the bar from k = {k_abs}.")
        a.check.append("The paper's version adds 'and the full three-judge panel'. No analysis in the "
                       "paper measures the absolute score under reduced panels (the judge ablation "
                       "reports rank and alpha only), so this version drops that clause.")
    return [p, a]


def appendix(fx: Facts) -> list[Paragraph]:
    m = fx.m
    out = []

    pp = Paragraph("Appendix: Ablation Details", "Prompt Paraphrase Sensitivity (results)")
    if _para_coverage(fx, pp) and _has(m, "ParaNAboveNull"):
        rank, sensitive = _para_state(fx)
        pp.say({"all": "Per-model rankings are preserved beyond chance under every rewording.",
                "most": "Per-model rankings are preserved beyond chance under most rewordings.",
                "few": "Per-model rankings are not preserved beyond chance under most rewordings."}[rank])
        pp.say("Across the \\abParaNPairs{} off-diagonal paraphrase pairs, $\\rho$ spans "
               "$[\\abParaRhoMin{}, \\abParaRhoMax{}]$ with median $\\abParaRhoMedian{}$, and "
               "\\abParaNAboveNull{} of \\abParaNPairs{} pairs exceed their null $\\abCILevel{}\\%$ upper "
               "bound (range $[\\abParaNullHiMin{}, \\abParaNullHiMax{}]$).")
        med = number(m, "ParaMadMedian")
        size = ("an absolute-score shift on the order of one Ward point" if 0.5 <= med < 1.5
                else "an absolute-score shift of about \\abParaMadMedian{} Ward points")
        pp.say("Pairwise MAD spans $[\\abParaMadMin{}, \\abParaMadMax{}]$ Ward points with median "
               f"$\\abParaMadMedian{{}}$, indicating {size} that $\\rho$ does not capture.")
        n_rho, n_mad, n_both = (int(number(m, n)) for n in ("ParaNRhoBar", "ParaNMadBar", "ParaNBothBars"))
        first = ("No pair clears" if n_rho == 0 else
                 f"\\abParaNRhoBarCap{{}} {'pair clears' if n_rho == 1 else 'pairs clear'}")
        both = (f"but only \\abParaNBothBarsWord{{}} {_verb(n_both, 'clears', 'clear')} both jointly"
                if n_both < min(n_rho, n_mad) else
                f"and \\abParaNBothBarsWord{{}} {_verb(n_both, 'clears', 'clear')} both jointly")
        pp.say(f"{first} the $\\rho \\geq \\abParaRhoBar{{}}$ bar and \\abParaNMadBarWord{{}} "
               f"{_verb(n_mad, 'clears', 'clear')} the MAD $\\leq \\abParaMadBar{{}}$ bar, {both} "
               "(\\autoref{tab:ablation_prompt}).")
        pp.say("\\abParaOutlierModel{} is a per-model outlier: its mean score is \\abParaOutlierBase{} "
               "under the baseline wording and between \\abParaOutlierMin{} and \\abParaOutlierMax{} "
               "under the \\abNParaphrasesWord{} rewordings, a larger difference than for any other model.")
        pp.say(_para_conclusion(rank, sensitive, "appendix"))
    out.append(pp)

    fl = Paragraph("Appendix: Ablation Details", "Feature Sensitivity, Leave-One-Out")
    if (fx.results_dir / "table_ablation_dimensions_loo.csv").exists() and _has(m, "DimLooRhoMin"):
        all_above, _ = _dim_loo_state(fx)
        if all_above:
            fl.say("Across the \\abNDimsWord{} features, $\\rho$ spans $[\\abDimLooRhoMin{}, "
                   "\\abDimLooRhoMax{}]$ and every feature exceeds its null $\\abCILevel{}\\%$ upper "
                   "bound of at most $\\abDimLooNullHi{}$.")
        else:
            fl.say("Across the \\abNDimsWord{} features, $\\rho$ spans $[\\abDimLooRhoMin{}, "
                   "\\abDimLooRhoMax{}]$ and \\abDimLooNAboveNullWord{} features exceed their null "
                   "$\\abCILevel{}\\%$ upper bound of at most $\\abDimLooNullHi{}$.")
        fl.say("MAD is numerically equal to each feature's departure rate and ranges from "
               "$\\abDimMadMin{}$ Ward points (\\abDimMadMinDim{}) to $\\abDimMadMax{}$ "
               "(\\abDimMadMaxDim{}), reflecting the contribution each feature makes to the absolute mean.")
        n_kept = int(number(m, "DimNRankPreserved"))
        define_count(m, "DimNRankPreserved", n_kept)
        if n_kept == 0:
            fl.say("No single feature can be removed without changing the per-model rank "
                   "(\\autoref{tab:ablation_dimensions}).")
        elif n_kept == 1:
            fl.say("\\abDimRankPreservedDims{} is the only feature whose removal preserves the "
                   "per-model rank exactly (\\autoref{tab:ablation_dimensions}).")
        else:
            fl.say("\\abDimRankPreservedDims{} are the only features whose removal preserves the "
                   "per-model rank exactly (\\autoref{tab:ablation_dimensions}).")
        fl.case.append(f"{n_kept} feature(s) whose removal preserves the rank: "
                       f"{m.get('abDimRankPreservedDims') or 'none'}.")
    out.append(fl)

    fc = Paragraph("Appendix: Ablation Details", "Feature Sensitivity, Continuous Subset Search")
    if _has(m, "DimMinKRho", "DimMadBelowOneK"):
        k = int(number(m, "DimMinKRho"))
        k_abs = m["abDimMadBelowOneK"]
        all_needed = k_abs != "--" and int(k_abs) == fx.n_dims
        need = ("still requires all \\abNDimsWord{} features" if all_needed
                else "requires at least $\\abDimMadBelowOneK{}$ features")
        if k < 0:
            fc.say("The full ranking cannot be recovered from any proper subset of the features.")
        else:
            subset = ("a single feature" if k == 1 else
                      "a subset of $\\abDimMinKRho{}$ features")
            fc.say(f"The full ranking can therefore be recovered from {subset}, but the absolute Ward Departure Score {need}: minimum MAD at "
                   "$k{=}\\abDimMinKRho{}$ is $\\abDimMinMadAtMinK{}$ Ward points and MAD falls below "
                   "$\\abParaMadBar{}$ only at $k{=}\\abDimMadBelowOneK{}$.")
        fc.case.append(f"k for the rho bar = {k}. MAD below the bar from k = {k_abs} of {fx.n_dims}.")
    out.append(fc)

    jr = Paragraph("Appendix: Ablation Details", "Judge Panel Composition, Rank Stability (results)")
    tables = _judge_tables(fx)
    if tables:
        single, loo = tables
        jr.say("Single-judge held-out $\\rho$ spans $[\\abJudgeSingleRhoMin{}, \\abJudgeSingleRhoMax{}]$ "
               "and \\abJudgeSingleNAboveNull{} of \\abNJudgesWord{} single judges exceed their null "
               "$\\abCILevel{}\\%$ upper bound of at most $\\abJudgeSingleNullHi{}$, with "
               "\\abJudgeSingleNClearBar{} of \\abNJudgesWord{} clearing this bar.")
        s_miss = list(single.loc[~single["rho_heldout_above_stable_bar"], "judge_model"])
        l_miss = list(loo.loc[~loo["rho_heldout_above_stable_bar"], "excluded_judge"])
        loo_s = ("The \\abNJudgesWord{} leave-one-judge-out sub-panels span $[\\abJudgeLooRhoMin{}, "
                 "\\abJudgeLooRhoMax{}]$ and clear the bar at \\abJudgeLooNClearBar{} of \\abNJudgesWord{}")
        if len(s_miss) == 1 and l_miss == s_miss:
            jr.say(loo_s + ".")
            jr.say("Both misses involve \\abJudgeMissName{}: the judge alone ($k{=}1$) and the panel "
                   "without it ($k{=}\\abNJudgesMinusOne{}$) each reach $\\rho{=}\\abJudgeMissRho{}$ "
                   "(Panel~A of \\autoref{tab:ablation_judges}).")
        else:
            jr.say(loo_s + " (Panel~A of \\autoref{tab:ablation_judges}).")
            if s_miss or l_miss:
                jr.check.append(f"Misses: single {s_miss}, leave-one-out {l_miss}. They do not "
                                "involve a single judge. Describe them by hand.")
        jr.case.append(f"Single-judge misses: {len(s_miss)}. Leave-one-out misses: {len(l_miss)}.")
    out.append(jr)

    ja = Paragraph("Appendix: Ablation Details", "Judge Panel Composition, Inter-Rater Reliability (results)")
    if tables:
        _, loo = tables
        ja.say("Leave-one-judge-out $\\alpha$ spans $[\\abJudgeLooAlphaMin{}, \\abJudgeLooAlphaMax{}]$, "
               "\\abJudgeLooNTentative{} of \\abNJudgesWord{} sub-panels clear the tentative-conclusions "
               "bar, and \\abJudgeLooNDropBar{} of \\abNJudgesWord{} stay within the preregistered drop bar.")
        bar = number(m, "JudgeAlphaDropBar")
        raised = loo[loo["alpha_drop_from_full"] < 0]
        lowest = loo.loc[loo["alpha_drop_from_full"].idxmax()]
        sents = []
        for _, row in raised.iterrows():
            key = fx.cfg.judge_key(row["excluded_judge"])
            slight = "slightly " if abs(row["alpha_drop_from_full"]) <= bar else ""
            sents.append(f"Without \\abJudgeName{key}{{}}, $\\alpha$ is $\\abJudgeLooAlpha{key}{{}}$, "
                         f"{slight}higher than for the full panel.")
        if lowest["alpha_drop_from_full"] > 0:
            key = fx.cfg.judge_key(lowest["excluded_judge"])
            sents.append(f"Without \\abJudgeName{key}{{}}, $\\alpha$ is $\\abJudgeLooAlpha{key}{{}}$.")
        if sents:
            sents[-1] = sents[-1][:-1] + " (Panel~B of \\autoref{tab:ablation_judges})."
            for s in sents:
                ja.say(s)
        ja.case.append("Removal raises alpha for: "
                       + (", ".join(fx.cfg.judge_name(j) for j in raised["excluded_judge"]) or "none")
                       + f". Largest drop: without {fx.cfg.judge_name(lowest['excluded_judge'])}.")
    out.append(ja)

    sf = Paragraph("Appendix: Ablation Details", "Judge Panel Composition, Same-Family Sensitivity (results)")
    for s in _same_family(fx, "appendix"):
        sf.say(s)
    if _has(m, "SfNPairs"):
        sf.case.append(f"Rank change of the same-family pairs: zero {m['abSfNZero']}, negative "
                       f"{m['abSfNNegative']}, positive {m['abSfNPositive']}.")
    out.append(sf)
    return out


SECTIONS = (abstract, introduction, method, model_comparison, human_comparison, reliability,
            ablations, discussion, conclusion, appendix)


def build_findings(cfg: Config, m: dict, ward_df: pd.DataFrame, human_df: pd.DataFrame,
                   results_dir: Path, mentioned: bool = True) -> list[Paragraph]:
    """Write every result paragraph for the current data. Adds the macros the sentences need to ``m``."""
    fx = collect_facts(cfg, m, ward_df, human_df, results_dir, mentioned=mentioned)
    paragraphs = []
    for section in SECTIONS:
        try:
            paragraphs += section(fx)
        except (KeyError, ValueError, IndexError) as exc:
            logger.warning("findings: section %s skipped (%s: %s)", section.__name__,
                           type(exc).__name__, exc)
            p = Paragraph(section.__name__, "Section could not be written")
            p.case.append(f"Missing input: `{exc}`. Rerun the stage that produces it.")
            paragraphs.append(p)
    return paragraphs


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_MACRO = re.compile(r"\\(ab[A-Za-z]+)(?:\{\})?")


def macros_in(text: str) -> list[str]:
    return _MACRO.findall(text)


def substitute(text: str, m: dict) -> str:
    """Replace every ``\\abName{}`` by its value. Raises ``KeyError`` for an undefined macro."""
    def repl(match: re.Match) -> str:
        name = match.group(1)
        if name not in m:
            raise KeyError(f"findings sentence uses undefined macro \\{name}")
        return m[name]
    return _MACRO.sub(repl, text)


def readable(text: str, m: dict) -> str:
    """The sentence as a reader sees it: values in, the commonest LaTeX markup out."""
    s = substitute(text, m)
    s = re.sub(r"\\MakeLowercase\{([^}]*)\}", lambda g: g.group(1).lower(), s)
    s = re.sub(r"\\autoref\{tab:([^}]*)\}", r"Table \1", s)
    s = re.sub(r"\\autoref\{fig:([^}]*)\}", r"Figure \1", s)
    s = re.sub(r"Section~\\ref\{([^}]*)\}", r"Section \1", s)
    s = re.sub(r"\\citet\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\citep\{([^}]*)\}", r"[\1]", s)
    s = s.replace("\\%", "%").replace("vs.\\ ", "vs. ").replace("{,}", ",").replace("~", " ")
    s = s.replace("\\pounds{}", "£")
    return s


def _norm(text: str) -> str:
    s = re.sub(r"\\(rev|ok|todo)\b", "", text)
    s = s.replace("{}", "")
    s = re.sub(r"[{}]", "", s)
    s = s.replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


def paper_texts(paper_tex: Path, m: dict) -> dict[str, tuple[str, str]]:
    """Active and commented-out text of ``paper.tex``, each as (source, values filled in)."""
    if not paper_tex.exists():
        return {"same": ("", ""), "commented": ("", "")}
    active, comments = [], []
    for line in paper_tex.read_text().splitlines():
        match = re.search(r"(?<!\\)%", line)
        if match:
            active.append(line[: match.start()])
            comments.append(line[match.end():])
        else:
            active.append(line)
    out = {}
    for label, lines in (("same", active), ("commented", comments)):
        text = " ".join(lines)
        out[label] = (_norm(text), _norm(soft_substitute(text, m)))
    return out


def soft_substitute(text: str, m: dict) -> str:
    """Like :func:`substitute`, but leaves undefined macros in place."""
    return _MACRO.sub(lambda g: m.get(g.group(1), g.group(0)), text)


def status(sentence: str, texts: dict[str, tuple[str, str]], m: dict) -> str:
    """'same', 'commented', or 'new'. A sentence matches when its LaTeX source or
    its text with values filled in occurs in the paper."""
    raw, filled = _norm(sentence), _norm(soft_substitute(sentence, m))
    for label in ("same", "commented"):
        src, values = texts[label]
        if (raw and raw in src) or (filled and filled in values):
            return label
    return "new"


_MARK = {"same": "in paper", "commented": "**commented out in paper**", "new": "**new wording**"}


def render(paragraphs: list[Paragraph], m: dict, paper_tex: Path) -> str:
    texts = paper_texts(paper_tex, m)
    counts = {"same": 0, "commented": 0, "new": 0}
    body = []
    section = None
    for p in paragraphs:
        if p.section != section:
            section = p.section
            body += [f"## {section}", ""]
        body += [f"### {p.title}", ""]
        for c in p.case:
            body.append(f"- Case: {readable(c, m)}")
        for c in p.check:
            body.append(f"- **Check by hand:** {readable(c, m)}")
        if p.case or p.check:
            body.append("")
        if not p.sentences:
            body += ["_No sentences: inputs missing (see the case notes)._", ""]
            continue
        body += ["```latex", *p.sentences, "```", ""]
        for s in p.sentences:
            st = status(s, texts, m)
            counts[st] += 1
            body.append(f"> {readable(s, m)} <sub>[{_MARK[st]}]</sub>")
            body.append(">")
        body[-1] = ""
    total = sum(counts.values())
    head = [
        "# AlienBench: findings",
        "",
        f"Generated by `python -m alienbench paper` on {_dt.date.today().isoformat()}. "
        "Do not edit: rerun the stage instead.",
        "",
        "Each paragraph below is a result paragraph of `paper.tex`, written for the case the current "
        "data fall in. The `latex` block can be pasted into the paper. Every macro it uses is defined "
        "in `key_numbers.tex`. The quoted lines show the same sentences with the values filled in.",
        "",
        "Each sentence carries a tag:",
        "",
        "- *in paper*: `paper.tex` contains the sentence verbatim in active text.",
        "- *commented out in paper*: `paper.tex` contains it only in a comment.",
        "- *new wording*: the paper does not contain it. The data no longer support the paper's "
        "wording, or the numbers moved a sentence into a different case. Compare and paste.",
        "",
        "Case notes say which case applies. *Check by hand* lists judgments the script cannot make.",
        "",
        f"Summary: {total} sentences. {counts['same']} in paper, {counts['commented']} commented out, "
        f"{counts['new']} new wording.",
        "",
    ]
    return "\n".join(head + body).rstrip() + "\n"


def write_findings(paragraphs: list[Paragraph], m: dict, results_dir: Path,
                   paper_tex: Path | None = None) -> Path:
    if paper_tex is None:
        paper_tex = Path(__file__).resolve().parent.parent / "paper" / "latex" / "paper.tex"
    out = results_dir / REPORT_NAME
    out.write_text(render(paragraphs, m, paper_tex))
    logger.info("Wrote the findings report to %s", out)
    return out

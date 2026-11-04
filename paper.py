"""Stage 6: the paper's number macros.

Every number, model name, and judge name that ``paper/latex/paper.tex`` prints
in running text is a macro defined in ``results/key_numbers.tex``. This module
writes that file. Nothing in the paper restates a value that the data or
``config.yaml`` fixes, so a rerun of the pipeline changes the paper's
sentences and tables together.

Sources, all under ``alienbench/``:

    config.yaml                         names, tiers, judge dates, design constants
    data/scores/                        per-judge Ward scores (the main analysis)
    results/table_posthoc.csv           analyze.py, pairwise rank-biserial effects
    results/table_extraction_status.csv analyze.py, per-cell parse and API errors
    results/table_ablation_*.csv        the three ablation stages

A section whose source is absent is skipped, and its macros are then missing
from the output. ``\\abMissing`` is not defined by this file, so a paper that
cites a skipped macro fails to compile and names it, instead of printing a
wrong number.

A macro cannot change the wording of a sentence when the shape of the
results changes (no exception instead of one, a tie that dissolves). For
those sentences, ``findings.py`` writes ``results/findings.md``: every result
paragraph of the paper, worded for the current data, for the authors to read
and paste. It also adds the few macros these sentences need.

Macro names carry the ``ab`` prefix and letters only, as LaTeX requires. A
subject model or judge enters a name through the ``key`` field of its
``config.yaml`` entry, and a prompt condition through its CamelCased id.

Usage:
    python -m alienbench paper [--config alienbench/config.yaml]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

from alienbench.ablation_dimensions import (
    LOO_NULL_N,
    RHO_THRESHOLD as DIM_RHO_THRESHOLD,
    SUBSET_NULL_N,
    TAU_THRESHOLD as DIM_TAU_THRESHOLD,
)
from alienbench.ablation_judges import (
    ALPHA_DROP_THRESHOLD,
    ALPHA_TENTATIVE,
    NULL_N as JUDGE_NULL_N,
    RHO_STABLE_THRESHOLD as JUDGE_RHO_THRESHOLD,
)
from alienbench.ablation_prompt import (
    MAD_STABLE_THRESHOLD,
    N_PERMUTATIONS as PARA_NULL_N,
    RHO_STABLE_THRESHOLD as PARA_RHO_THRESHOLD,
)
from alienbench.config import Config, _key_from, load_config
from alienbench.dimensions import DIMENSION_IDS, DIMENSION_LABELS, PAPER_TEXT
from alienbench.extract import JUDGE_TEMPERATURE
from alienbench.latex_tables import (
    _ALPHA_BANDS,
    _alpha_band,
    _human_dimension_means,
    dimension_alphas,
    load_human_scores,
)
from alienbench.paths import iter_jsonl, load_ward_scores
from alienbench.analyze import TOKENS_PER_RATE_UNIT
from alienbench.extract import _USER_TEMPLATE
from alienbench.stats import BOOTSTRAP_N, CI_LEVEL, bootstrap_ci, krippendorff_alpha

logger = logging.getLogger(__name__)

OUTPUT_NAME = "key_numbers.tex"

# Krippendorff's alpha at which the paper calls inter-judge agreement good
# (the lower edge of the Good band of Landis and Koch, 1977).
ALPHA_GOOD = 0.60
# Retrospective power analysis of ``analyze.write_summary``.
POWER_TARGET = 0.80

_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def word(n: int) -> str:
    """A count as an English word up to twenty, as a numeral beyond."""
    n = int(n)
    return _WORDS[n] if 0 <= n < len(_WORDS) else str(n)


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def f(x: float, d: int = 2) -> str:
    """A number at fixed decimals, with the sign artifact of rounding removed."""
    return f"{round(float(x), d) + 0.0:.{d}f}"


def thousands(n: float) -> str:
    """An integer with a LaTeX-safe thousands separator, ``5{,}400``."""
    return f"{int(round(n)):,}".replace(",", "{,}")


def pct(x: float) -> str:
    """A proportion as a one-decimal percentage."""
    return f"{float(x) * 100:.1f}\\%"


def interval(lo: float, hi: float, d: int = 2, math_mode: bool = False) -> str:
    text = f"[{f(lo, d)}, {f(hi, d)}]"
    return f"${text}$" if math_mode else text


def p_value(p: float) -> str:
    """A p-value as the paper prints it: ``p < .001`` or ``p = 0.05``."""
    if p < 0.001:
        return "$p < .001$"
    return f"$p = {p:.3g}$" if p < 1 else "$p = 1.0$"


def join_names(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def latex_text(text: str) -> str:
    """Escape the characters of a config string that LaTeX treats as markup."""
    return text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


class Macros(dict):
    """Ordered ``name -> value`` map. Names are checked to be letters only."""

    def put(self, name: str, value) -> None:
        full = f"ab{name}"
        if not full.isalpha():
            raise ValueError(f"macro name {full!r} must contain letters only")
        if full in self:
            raise ValueError(f"macro {full!r} defined twice")
        self[full] = str(value)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _condition_key(variant) -> str:
    return _key_from(variant.id)


def design_macros(m: Macros, cfg: Config, ward_df: pd.DataFrame) -> None:
    """Counts and settings fixed by the config and the data volume."""
    n_models, n_judges = len(cfg.models), len(cfg.judge_models)
    n_conditions, n_dims = len(cfg.prompt_variants), len(DIMENSION_IDS)
    n_generations = ward_df["generation_id"].nunique()
    for name, n in (("Models", n_models), ("Judges", n_judges),
                    ("Conditions", n_conditions), ("Dims", n_dims)):
        m.put(f"N{name}", n)
        m.put(f"N{name}Word", word(n))
    m.put("NDimsMinusOneWord", word(n_dims - 1))
    m.put("NJudgesMinusOne", n_judges - 1)
    m.put("NJudgesMinusOneWord", word(n_judges - 1))
    m.put("NParaphrases", len(cfg.prompt_paraphrases) - 1)
    m.put("NParaphrasesWord", word(len(cfg.prompt_paraphrases) - 1))
    m.put("WardMax", n_dims)
    constrained = [v for v in cfg.prompt_variants if v.id.startswith("constrained_")]
    m.put("NConstrainedWord", word(len(constrained)))
    m.put("NTiersWord", word(len(cfg.tiers)))
    m.put("NPerCell", cfg.samples_per_condition)
    m.put("NPerModel", cfg.samples_per_condition * n_conditions)
    m.put("NGenerations", thousands(n_generations))
    m.put("NExtractions", thousands(len(ward_df)))
    m.put("Temperature", f(cfg.temperature, 1))
    # Output-token limits: the start limit and the ceiling it doubles toward
    # on each cut-off, with a macro per model or judge that overrides either.
    m.put("MaxTokens", thousands(cfg.max_tokens))
    m.put("MaxTokensCeiling", thousands(cfg.max_tokens_ceiling))
    for model in cfg.models:
        key = cfg.model_key(model)
        start, ceiling = cfg.max_tokens_for(model), cfg.max_tokens_ceiling_for(model)
        if start != cfg.max_tokens:
            m.put(f"MaxTokens{key}", thousands(start))
        if ceiling != cfg.max_tokens_ceiling:
            m.put(f"MaxTokensCeiling{key}", thousands(ceiling))
        if start != cfg.max_tokens or ceiling != cfg.max_tokens_ceiling:
            m.put(f"MaxTokensModel{key}", cfg.model_name(model))
    m.put("JudgeTemperature", f(JUDGE_TEMPERATURE, 0))
    m.put("JudgeMaxTokens", thousands(cfg.judge_max_tokens))
    m.put("JudgeMaxTokensCeiling", thousands(cfg.judge_max_tokens_ceiling))
    for judge in cfg.judge_models:
        key = cfg.judge_key(judge)
        start, ceiling = cfg.judge_max_tokens_for(judge), cfg.judge_max_tokens_ceiling_for(judge)
        if start != cfg.judge_max_tokens:
            m.put(f"JudgeMaxTokens{key}", thousands(start))
        if ceiling != cfg.judge_max_tokens_ceiling:
            m.put(f"JudgeMaxTokensCeiling{key}", thousands(ceiling))
    # Records produced at a raised limit, over all subject models and judges.
    from alienbench.claims import limit_rows
    model_limits, judge_limits = limit_rows(cfg)
    m.put("NRaisedGenerations", thousands(sum(r["raised"] for r in model_limits)))
    m.put("NRaisedExtractions", thousands(sum(r["raised"] for r in judge_limits)))
    m.put("BootstrapB", thousands(BOOTSTRAP_N))
    m.put("CILevel", f(CI_LEVEL * 100, 0))
    # A two-sided interval at this level has its upper bound at this percentile.
    m.put("UpperPercentile", f(100 - (100 - CI_LEVEL * 100) / 2, 1))
    m.put("TokensPerRateUnit", TOKENS_PER_RATE_UNIT)
    # The extraction prompt's per-feature word limit ("brief, <=15 words").
    limit = re.search(r"(\d+) words", _USER_TEMPLATE)
    if limit:
        m.put("FeatureMaxWords", limit.group(1))
    m.put("AlphaGood", f(ALPHA_GOOD))
    m.put("ScoreEarth", 0)
    m.put("ScoreDeparture", 1)
    m.put("RateAxisMax", f(1, 0))
    m.put("MinRaters", 2)
    legs = re.search(r"(\w+) or \w+ legs", PAPER_TEXT["locomotion"]["boundary_rule"])
    if legs:
        m.put("ExampleLegsWord", legs.group(1).lower())
    for key, value in cfg.literature.items():
        m.put(f"Lit{_key_from(key)}", value)
    m.put("PowerTarget", f(POWER_TARGET * 100, 0))

    panel = [f"{cfg.judge_name(j)} ({cfg.judge_info[j].date})" for j in cfg.judge_models]
    m.put("JudgePanel", join_names(panel))
    for j in cfg.judge_models:
        m.put(f"JudgeName{cfg.judge_key(j)}", cfg.judge_name(j))
        m.put(f"JudgeDate{cfg.judge_key(j)}", cfg.judge_info[j].date)

    for tier, info in cfg.tiers.items():
        m.put(f"Tier{tier}Desc", info.description)
        if info.scale:
            m.put(f"Tier{tier}Scale", info.scale.replace("~", "$\\sim$"))
    for model in cfg.models:
        m.put(f"Name{cfg.model_key(model)}", cfg.model_name(model))
    for v in cfg.prompt_variants:
        m.put(f"CondName{_condition_key(v)}", v.display)
    # Table-header key: "Base = baseline, ..., NoLight = constrained (no light), ...".
    entries = []
    for v in cfg.prompt_variants:
        name = re.sub(r"^(\w+): (.*)$", r"\1 (\2)", v.display[:1].lower() + v.display[1:])
        entries.append(f"{v.short} = {name}")
    m.put("ConditionKey", join_names(entries))
    # The gravity multiplier is part of the constrained prompt's wording.
    for v in cfg.prompt_variants:
        found = re.search(r"(\d+)x Earth's gravity", v.text)
        if found:
            m.put("GravityMultiplier", found.group(1))

    # Krippendorff's alpha bands (Landis and Koch, 1977), lower edges.
    for edge, band in _ALPHA_BANDS[:-1]:
        m.put(f"Band{band}", f(edge))


def compute_macros(m: Macros, cfg: Config) -> None:
    """Token volume of every record in the released JSONL artifacts.

    Sums the per-call ``prompt_tokens`` and ``completion_tokens`` of every
    generation and extraction record under ``data/``, including the human
    baseline and earlier runs kept on disk. Records of the paraphrase
    ablation are totalled separately, by their prompt-variant directory.
    """
    data_dir = Path(cfg.data_dir)
    para_ids = {p.id for p in cfg.prompt_paraphrases} - {p.id for p in cfg.prompt_variants}
    totals = {(para, kind): 0 for para in (False, True) for kind in ("in", "out")}
    files = [(f, f.parent.name) for f in (data_dir / "generations").glob("*/*/responses.jsonl")]
    files += [(f, f.parent.name) for f in (data_dir / "extractions").glob("*/*/*/features.jsonl")]
    if not files:
        logger.warning("no generation or extraction records; skipping the token macros")
        return
    for path, variant in files:
        para = variant in para_ids
        for rec in iter_jsonl(path):
            totals[(para, "in")] += rec.get("prompt_tokens") or 0
            totals[(para, "out")] += rec.get("completion_tokens") or 0
    main_in, main_out = totals[(False, "in")], totals[(False, "out")]
    m.put("TokensInputM", f(main_in / 1e6))
    m.put("TokensOutputM", f(main_out / 1e6))
    m.put("TokensTotalM", f((main_in + main_out) / 1e6))
    m.put("TokensParaphraseM", f((totals[(True, "in")] + totals[(True, "out")]) / 1e6))


def human_macros(m: Macros, cfg: Config) -> None:
    """The human baseline's collection protocol, from ``config.yaml``."""
    h = cfg.human_baseline
    m.put("HumanN", h.n_participants)
    m.put("HumanPlatform", h.platform)
    m.put("HumanApproval", h.min_approval_rate)
    m.put("HumanPriorTasks", h.min_prior_tasks)
    m.put("HumanPay", h.hourly_pay.replace("£", "\\pounds{}") if h.hourly_pay else "")
    m.put("HumanMinutes", h.median_minutes)
    m.put("HumanSeconds", h.median_seconds)
    m.put("HumanExcludedWord", word(h.n_excluded))


def _per_generation(ward_df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return ward_df.groupby(["generation_id", *by])["ward_score"].mean().reset_index()


def model_macros(m: Macros, cfg: Config, ward_df: pd.DataFrame) -> dict[str, float]:
    """Per-model overall means with bootstrap CIs, and per-condition cell means."""
    models = [x for x in cfg.models if x in set(ward_df["subject_model"])]
    per_model = _per_generation(ward_df, ["subject_model"])
    per_cell = _per_generation(ward_df, ["subject_model", "prompt_variant"])

    overall: dict[str, float] = {}
    for model in models:
        vals = per_model.loc[per_model["subject_model"] == model, "ward_score"].values
        mean, lo, hi = bootstrap_ci(vals, n_boot=BOOTSTRAP_N)
        key = cfg.model_key(model)
        overall[model] = mean
        m.put(f"Mean{key}", f(mean))
        m.put(f"Lo{key}", f(lo))
        m.put(f"Hi{key}", f(hi))
        m.put(f"CI{key}", interval(lo, hi))
        m.put(f"MeanCI{key}", f"{f(mean)} {interval(lo, hi)}")
        for v in cfg.prompt_variants:
            cell = per_cell[(per_cell["subject_model"] == model)
                            & (per_cell["prompt_variant"] == v.id)]["ward_score"]
            m.put(f"Cell{key}{_condition_key(v)}", f(cell.mean()))

    ranked = sorted(models, key=lambda x: overall[x], reverse=True)
    best, worst = ranked[0], ranked[-1]
    for label, model in (("Best", best), ("Worst", worst)):
        key = cfg.model_key(model)
        m.put(f"{label}Model", cfg.model_name(model))
        m.put(f"{label}Mean", f(overall[model]))
        m.put(f"{label}MeanCI", m[f"abMeanCI{key}"])
    # The runners-up that tie at the displayed precision.
    second_value = f(overall[ranked[1]])
    tied = [cfg.model_name(x) for x in ranked[1:] if f(overall[x]) == second_value]
    m.put("SecondModels", join_names(tied))
    m.put("SecondMean", second_value)
    m.put("SecondTieWord", "both" if len(tied) == 2 else "all")
    m.put("LeadMargin", f(overall[best] - overall[ranked[1]], 1))

    # Open-weight against proprietary models, compared at the displayed precision.
    shown = {x: float(f(overall[x])) for x in models}
    open_models = [x for x in ranked if cfg.model_info[x].open_weights]
    closed_models = [x for x in ranked if cfg.model_info[x].open_weights is False]
    if open_models and closed_models:
        best_closed, worst_closed = closed_models[0], closed_models[-1]
        for label, group in (("Open", open_models), ("Closed", closed_models)):
            m.put(f"N{label}", len(group))
            m.put(f"N{label}Word", word(len(group)))
            # Unweighted mean of the per-model overall scores in the group.
            m.put(f"{label}Mean", f(np.mean([overall[x] for x in group])))
        for label, model in (("BestClosed", best_closed), ("WorstClosed", worst_closed)):
            m.put(f"{label}Model", cfg.model_name(model))
            m.put(f"{label}Mean", f(overall[model]))
        above = [x for x in open_models if shown[x] > shown[best_closed]]
        m.put("NOpenAboveClosed", len(above))
        m.put("NOpenAboveClosedWord", word(len(above)))
        m.put("OpenAboveClosedModels", join_names([cfg.model_name(x) for x in above]))
        below = [x for x in open_models if shown[x] < shown[worst_closed]]
        m.put("NOpenBelowWorstClosed", len(below))
        m.put("NOpenBelowWorstClosedWord", word(len(below)))
        m.put("OpenBelowWorstClosedModels", join_names([cfg.model_name(x) for x in below]))

    # Per-condition leaders and extremes.
    cell_means = per_cell.groupby(["subject_model", "prompt_variant"])["ward_score"].mean()
    leaders = set()
    n_best_leads = 0
    for v in cfg.prompt_variants:
        ck = _condition_key(v)
        col = {x: cell_means.loc[(x, v.id)] for x in models}
        top = max(col, key=col.get)
        leaders.add(top)
        n_best_leads += top == best
        m.put(f"Leader{ck}", cfg.model_name(top))
        m.put(f"LeaderMean{ck}", f(col[top]))
        m.put(f"CondMin{ck}", f(min(col.values())))
        m.put(f"CondMax{ck}", f(max(col.values())))
    m.put("NCondLeaders", len(leaders))
    m.put("NCondLeadersWord", word(len(leaders)))
    m.put("NBestCondLeads", n_best_leads)
    m.put("NBestCondLeadsWord", word(n_best_leads))

    # Sensitivity of the overall ranking to the equal weighting of conditions:
    # drop one condition at a time and correlate with the full ranking.
    cell_table = cell_means.unstack("prompt_variant").loc[models]
    full_mean = cell_table.mean(axis=1)
    loco_rho, loco_top = [], set()
    for v in cfg.prompt_variants:
        reduced = cell_table.drop(columns=v.id).mean(axis=1)
        loco_rho.append(full_mean.corr(reduced, method="spearman"))
        loco_top.add(reduced.idxmax())
    m.put("LocoRhoMin", f(min(loco_rho)))
    m.put("LocoRhoMax", f(max(loco_rho)))
    m.put("LocoSameTop", "true" if loco_top == {full_mean.idxmax()} else "false")

    # Models scoring below their own baseline under each constrained condition.
    base = {x: cell_means.loc[(x, "baseline")] for x in models}
    for v in cfg.prompt_variants:
        if v.id.startswith("constrained_"):
            ck = _condition_key(v)
            below = [x for x in models if cell_means.loc[(x, v.id)] < base[x]]
            above = [x for x in models if x not in below]
            m.put(f"NBelowBaseline{ck}", len(below))
            m.put(f"NBelowBaseline{ck}Word", word(len(below)))
            if len(above) == 1:
                x = above[0]
                m.put(f"NotBelow{ck}Model", cfg.model_name(x))
                m.put(f"NotBelow{ck}Mean", f(cell_means.loc[(x, v.id)]))
                m.put(f"NotBelow{ck}Base", f(base[x]))
    return overall


def condition_effect_macros(m: Macros, cfg: Config, ward_df: pd.DataFrame, results_dir: Path) -> None:
    """Kruskal-Wallis, pairwise effects, and the retrospective power analysis."""
    path = results_dir / "table_posthoc.csv"
    if not path.exists():
        logger.warning("%s missing; skipping the condition-effect macros", path)
        return
    posthoc = pd.read_csv(path)
    per_prompt = _per_generation(ward_df, ["prompt_variant"])
    groups = [g["ward_score"].values for _, g in per_prompt.groupby("prompt_variant")]
    h, p = sps.kruskal(*groups)
    m.put("KruskalH", f(h))
    m.put("KruskalP", p_value(p))

    for _, row in posthoc.iterrows():
        a, b = row["group_a"], row["group_b"]
        if a != "baseline":
            continue
        ck = _key_from(b)
        m.put(f"R{ck}", f(row["rank_biserial_r"]))
        m.put(f"RCI{ck}", interval(row["r_ci_lo"], row["r_ci_hi"], math_mode=True))
        m.put(f"P{ck}", p_value(row["p_corrected"]))
    n_comparisons = len(posthoc)
    m.put("NComparisons", n_comparisons)
    m.put("BonferroniAlpha", f(0.05 / n_comparisons, 3))

    from alienbench.stats import retrospective_power
    power = retrospective_power(
        posthoc["rank_biserial_r"].abs().tolist(),
        n_per_group=int(min(posthoc["n_a"].min(), posthoc["n_b"].min())),
        n_comparisons=n_comparisons,
        target_power=POWER_TARGET,
    )
    m.put("MinDetectableR", f(power["min_detectable_r"]))


def dimension_macros(m: Macros, cfg: Config, ward_df: pd.DataFrame,
                     human_df: pd.DataFrame) -> None:
    """Per-dimension departure rates of models against the human baseline."""
    models = [x for x in cfg.models if x in set(ward_df["subject_model"])]
    dim_cols = [f"dim_{d}" for d in DIMENSION_IDS]
    per_gen = ward_df.groupby(["generation_id", "subject_model"])[dim_cols].mean().reset_index()
    rates = per_gen.groupby("subject_model")[dim_cols].mean().loc[models]

    leaders = set()
    for d, col in zip(DIMENSION_IDS, dim_cols):
        leaders.add(rates[col].idxmax())
        m.put(f"DimMin{_key_from(d)}", pct(rates[col].min()))
        m.put(f"DimMax{_key_from(d)}", pct(rates[col].max()))
    m.put("NDimLeaders", len(leaders))
    m.put("NDimLeadersWord", word(len(leaders)))
    # Dimensions on which the top model has open weights.
    n_open_led = sum(bool(cfg.model_info[rates[col].idxmax()].open_weights) for col in dim_cols)
    m.put("NDimOpenLeader", n_open_led)
    m.put("NDimOpenLeaderWord", word(n_open_led))

    if human_df.empty:
        logger.warning("no human scores; skipping the human-baseline macros")
        return
    human = _human_dimension_means(human_df)
    m.put("HumanNGenerations", human_df["generation_id"].nunique())
    for d, col in zip(DIMENSION_IDS, dim_cols):
        m.put(f"Human{_key_from(d)}", pct(human[col]))

    # Ward Departure Score under the baseline prompt, the condition the human
    # participants completed: judges averaged per generation, then generations.
    human_mean = human_df.groupby("generation_id")["ward_score"].mean().mean()
    base = ward_df[ward_df["prompt_variant"] == "baseline"]
    base_means = (base.groupby(["subject_model", "generation_id"])["ward_score"].mean()
                  .groupby("subject_model").mean().reindex(models))
    n_above = int((base_means > human_mean).sum())
    m.put("HumanMean", f(human_mean))
    m.put("BaselineModelMean", f(base_means.mean()))
    m.put("NAboveHumanBaseline", n_above)
    m.put("NAboveHumanBaselineWord", word(n_above))

    # Cells where a model does not exceed the human rate.
    exceptions = [(x, d) for x in models for d, col in zip(DIMENSION_IDS, dim_cols)
                  if rates.loc[x, col] <= human[col]]
    m.put("NHumanExceptions", len(exceptions))
    if exceptions:
        x, d = exceptions[0]
        m.put("ExceptionModel", cfg.model_name(x))
        m.put("ExceptionDim", DIMENSION_LABELS[d].lower())
        m.put("ExceptionRate", pct(rates.loc[x, f"dim_{d}"]))
        m.put("ExceptionHumanRate", pct(human[f"dim_{d}"]))
    exception_dims = sorted({d for _, d in exceptions})
    m.put("ExceptionDimsText", join_names([PAPER_TEXT[d]["short"].lower() for d in exception_dims]))

    # Most and least departed dimension across the panel.
    panel_mean = rates.mean()
    most = panel_mean.idxmax().removeprefix("dim_")
    least = panel_mean.idxmin().removeprefix("dim_")
    for label, d in (("Most", most), ("Least", least)):
        col = f"dim_{d}"
        m.put(f"{label}Dim", PAPER_TEXT[d]["short"].lower())
        m.put(f"{label}DimHuman", pct(human[col]))
        m.put(f"{label}DimMin", pct(rates[col].min()))
        m.put(f"{label}DimMax", pct(rates[col].max()))

    # The two dimensions with the largest relative gap: the panel's mean rate
    # divided by the human rate. (An absolute gap ranks different dimensions.)
    ratio = {c: rates[c].mean() / human[c] for c in dim_cols if human[c] > 0}
    for rank, col in enumerate(sorted(ratio, key=ratio.get, reverse=True)[:2], start=1):
        d = col.removeprefix("dim_")
        tag = f"Gap{word(rank).capitalize()}"
        m.put(f"{tag}Dim", DIMENSION_LABELS[d].lower())
        m.put(f"{tag}Human", pct(human[col]))
        m.put(f"{tag}Min", pct(rates[col].min()))
        m.put(f"{tag}Max", pct(rates[col].max()))
        m.put(f"{tag}Ratio", f(ratio[col], 1))


def condition_feature_macros(m: Macros, cfg: Config, ward_df: pd.DataFrame) -> None:
    """Departure rate and unmentioned rate of each feature under each condition.

    Rates pool the subject models: judges are averaged per generation, then
    generations per condition. ``Rate<Condition><Feature>`` is the departure
    rate, and ``Unmentioned<Condition><Feature>`` the share of judge labels
    whose feature the description does not mention (labeled Earth-typical by
    the boundary rules).
    """
    from alienbench.omission import unmentioned_rates

    dim_cols = [f"dim_{d}" for d in DIMENSION_IDS]
    per_gen = ward_df.groupby(["prompt_variant", "generation_id"])[dim_cols].mean()
    rates = per_gen.groupby("prompt_variant").mean()
    models = [x for x in cfg.models if x in set(ward_df["subject_model"])]
    unmentioned = unmentioned_rates(cfg, models)
    for v in cfg.prompt_variants:
        ck = _condition_key(v)
        for d in DIMENSION_IDS:
            dk = _key_from(d)
            m.put(f"Rate{ck}{dk}", pct(rates.loc[v.id, f"dim_{d}"]))
            m.put(f"Unmentioned{ck}{dk}", pct(unmentioned.loc[v.id, d]))


def reliability_macros(m: Macros, cfg: Config, ward_df: pd.DataFrame,
                       results_dir: Path) -> None:
    if ward_df["judge_model"].nunique() < 2:
        return
    total = krippendorff_alpha(ward_df, "judge_model", "generation_id", "ward_score",
                               level="interval")
    m.put("AlphaWard", f(total, 3))
    m.put("AlphaWardBand", _alpha_band(total))
    alphas = dimension_alphas(ward_df)
    lo_d = min(alphas, key=alphas.get)
    hi_d = max(alphas, key=alphas.get)
    m.put("AlphaMin", f(alphas[lo_d], 3))
    m.put("AlphaMinDim", PAPER_TEXT[lo_d]["short"].lower())
    m.put("AlphaMax", f(alphas[hi_d], 3))
    m.put("AlphaMaxDim", PAPER_TEXT[hi_d]["short"].lower())
    bands = [_alpha_band(a) for a in alphas.values()]
    for band in ("Excellent", "Good", "Moderate", "Fair", "Poor"):
        m.put(f"N{band}Dims", bands.count(band))
        m.put(f"N{band}DimsWord", word(bands.count(band)))
    n_clear = sum(a >= ALPHA_GOOD for a in alphas.values())
    m.put("NClearDims", n_clear)
    m.put("NClearDimsWord", word(n_clear))
    below = [PAPER_TEXT[d]["short"].lower() for d, a in alphas.items() if a < ALPHA_GOOD]
    m.put("BelowGoodDims", join_names(below))
    m.put("BelowGoodMin", f(min(alphas.values()), 3))

    status = results_dir / "table_extraction_status.csv"
    if status.exists():
        s = pd.read_csv(status)
        m.put("NExtractionCalls", thousands(s["n_generations"].sum()))
        m.put("NParseErrors", int(s["parse_error"].sum()))
        m.put("NApiErrors", int(s["api_error"].sum()))
        m.put("NParseErrorsWord", word(int(s["parse_error"].sum())))


def prompt_ablation_macros(m: Macros, cfg: Config, results_dir: Path) -> None:
    null_path = results_dir / "table_ablation_prompt_null.csv"
    means_path = results_dir / "table_ablation_prompt.csv"
    if not null_path.exists() or not means_path.exists():
        logger.warning("prompt-ablation CSVs missing; skipping their macros")
        return
    null = pd.read_csv(null_path)
    off = null[null["pair_a"] != null["pair_b"]]
    n_pairs = len(off)
    m.put("ParaN", cfg.prompt_ablation_samples_per_condition)
    m.put("ParaNPairs", n_pairs)
    m.put("ParaNPermutations", thousands(PARA_NULL_N))
    m.put("ParaRhoBar", f(PARA_RHO_THRESHOLD, 1))
    m.put("ParaMadBar", f(MAD_STABLE_THRESHOLD, 1))
    m.put("ParaMadBarPct", f(MAD_STABLE_THRESHOLD / len(DIMENSION_IDS) * 100, 0))
    m.put("ParaRhoMin", f(off["rho_obs"].min()))
    m.put("ParaRhoMax", f(off["rho_obs"].max()))
    m.put("ParaRhoMedian", f(off["rho_obs"].median()))
    above = int((off["rho_obs"] > off["null_hi95"]).sum())
    m.put("ParaNAboveNull", above)
    m.put("ParaNullHiMin", f(off["null_hi95"].min()))
    m.put("ParaNullHiMax", f(off["null_hi95"].max()))
    m.put("ParaMadMin", f(off["mad"].min()))
    m.put("ParaMadMax", f(off["mad"].max()))
    m.put("ParaMadMedian", f(off["mad"].median()))
    m.put("ParaNRhoBar", int(off["rho_above_stable_bar"].sum()))
    m.put("ParaNRhoBarWord", word(int(off["rho_above_stable_bar"].sum())))
    m.put("ParaNMadBar", int(off["mad_below_stable_bar"].sum()))
    m.put("ParaNMadBarWord", word(int(off["mad_below_stable_bar"].sum())))
    m.put("ParaNBothBars", int(off["both_bars_pass"].sum()))
    m.put("ParaNBothBarsWord", word(int(off["both_bars_pass"].sum())))
    # The ordered-pair counts above double-count each unordered pair. The
    # caption reports the same off-diagonal convention as the text.

    means = pd.read_csv(means_path, index_col=0)
    shift = (means.drop(columns="baseline").sub(means["baseline"], axis=0)).abs().mean(axis=1)
    outlier = shift.idxmax()
    para = means.loc[outlier].drop("baseline")
    m.put("ParaOutlierModel", cfg.model_name(outlier))
    m.put("ParaOutlierBase", f(means.loc[outlier, "baseline"]))
    m.put("ParaOutlierMin", f(para.min()))
    m.put("ParaOutlierMax", f(para.max()))


def dimension_ablation_macros(m: Macros, cfg: Config, results_dir: Path) -> None:
    loo_path = results_dir / "table_ablation_dimensions_loo.csv"
    sub_path = results_dir / "table_ablation_dimensions_minsubset.csv"
    null_path = results_dir / "table_ablation_dimensions_null_min_k.csv"
    if not (loo_path.exists() and sub_path.exists() and null_path.exists()):
        logger.warning("dimension-ablation CSVs missing; skipping their macros")
        return
    loo = pd.read_csv(loo_path)
    sub = pd.read_csv(sub_path)
    null = pd.read_csv(null_path)
    m.put("DimNPermutations", thousands(LOO_NULL_N))
    m.put("DimNSubsetPermutations", thousands(SUBSET_NULL_N))
    m.put("DimRhoBar", f(DIM_RHO_THRESHOLD, 1))
    m.put("DimTauBar", f(DIM_TAU_THRESHOLD, 1))
    m.put("DimLooRhoMin", f(loo["spearman_rho"].min()))
    m.put("DimLooRhoMax", f(loo["spearman_rho"].max()))
    m.put("DimLooNAboveNull", int(loo["rho_above_null_hi95"].sum()))
    m.put("DimLooAllAboveNull", "every" if loo["rho_above_null_hi95"].all() else "not every")
    m.put("DimLooNullHi", f(loo["null_rho_hi95"].max()))
    lo_row = loo.loc[loo["mad_vs_full"].idxmin()]
    hi_row = loo.loc[loo["mad_vs_full"].idxmax()]
    m.put("DimMadMin", f(lo_row["mad_vs_full"]))
    m.put("DimMadMinDim", DIMENSION_LABELS[lo_row["dropped_dimension"]].lower())
    m.put("DimMadMax", f(hi_row["mad_vs_full"]))
    m.put("DimMadMaxDim", DIMENSION_LABELS[hi_row["dropped_dimension"]].lower())
    kept = loo[loo["rank_preserved"]]["dropped_dimension"].tolist()
    m.put("DimNRankPreserved", len(kept))
    m.put("DimRankPreservedDims", join_names([DIMENSION_LABELS[d] for d in kept]))

    def min_k(col: str, bar: float) -> int:
        hit = sub[sub[col] >= bar]
        return int(hit["k"].min()) if len(hit) else -1

    k_rho, k_tau = min_k("best_rho", DIM_RHO_THRESHOLD), min_k("best_tau", DIM_TAU_THRESHOLD)
    for name, k, col in (("Rho", k_rho, "min_k_rho_null"), ("Tau", k_tau, "min_k_tau_null")):
        m.put(f"DimMinK{name}", k)
        m.put(f"DimMinK{name}NullP", f(np.percentile(null[col], 5), 0))
        m.put(f"DimMinK{name}PValue", f(float(np.mean(null[col] <= k)), 3))
    at2 = sub[sub["k"] == k_rho].iloc[0]
    m.put("DimMinMadAtMinK", f(at2["min_mad"]))
    below_one = sub[sub["min_mad"] < 1.0]
    m.put("DimMadBelowOneK", int(below_one["k"].min()) if len(below_one) else "--")
    m.put("DimNSubsetDims", len(DIMENSION_IDS))


def judge_ablation_macros(m: Macros, cfg: Config, results_dir: Path) -> None:
    single_path = results_dir / "table_ablation_judges_single.csv"
    loo_path = results_dir / "table_ablation_judges_loo.csv"
    if not (single_path.exists() and loo_path.exists()):
        logger.warning("judge-ablation CSVs missing; skipping their macros")
        return
    single = pd.read_csv(single_path)
    loo = pd.read_csv(loo_path)
    n = len(cfg.judge_models)
    m.put("JudgeNPermutations", thousands(JUDGE_NULL_N))
    m.put("JudgeRhoBar", f(JUDGE_RHO_THRESHOLD, 1))
    m.put("JudgeAlphaTentative", f(ALPHA_TENTATIVE, 3))
    m.put("JudgeAlphaDropBar", f(ALPHA_DROP_THRESHOLD, 1))
    m.put("JudgeKMax", word(n - 1))

    for label, df in (("Single", single), ("Loo", loo)):
        m.put(f"Judge{label}RhoMin", f(df["rho_vs_heldout"].min()))
        m.put(f"Judge{label}RhoMax", f(df["rho_vs_heldout"].max()))
        m.put(f"Judge{label}NAboveNull", word(int(df["rho_above_null_hi95"].sum())))
        m.put(f"Judge{label}NClearBar", word(int(df["rho_heldout_above_stable_bar"].sum())))
        m.put(f"Judge{label}NullHi", f(df["null_rho_hi95"].max()))
        m.put(f"Judge{label}NMissBar", int((~df["rho_heldout_above_stable_bar"]).sum()))
    missed = single[~single["rho_heldout_above_stable_bar"]]
    if len(missed):
        row = missed.iloc[0]
        m.put("JudgeMissName", cfg.judge_name(row["judge_model"]))
        m.put("JudgeMissRho", f(row["rho_vs_heldout"]))

    m.put("JudgeLooAlphaMin", f(loo["alpha_cross_model"].min()))
    m.put("JudgeLooAlphaMax", f(loo["alpha_cross_model"].max()))
    m.put("JudgeLooNTentative", word(int((loo["alpha_cross_model"] >= ALPHA_TENTATIVE).sum())))
    m.put("JudgeLooNDropBar", word(int(loo["alpha_drop_within_bar"].sum())))
    for _, row in loo.iterrows():
        key = cfg.judge_key(row["excluded_judge"])
        m.put(f"JudgeLooAlpha{key}", f(row["alpha_cross_model"]))
        m.put(f"JudgeLooAlphaDrop{key}", f(row["alpha_drop_from_full"]))

    # Same-family pairs: one row per (judge, subject) that share a provider.
    # A provider may have several subject models, so the macros summarize the
    # pairs by the sign of the rank change instead of naming one per judge.
    sf_path = results_dir / "table_ablation_judges_same_family.csv"
    if not sf_path.exists():
        logger.warning("%s missing; skipping the same-family macros", sf_path)
        return
    sf = pd.read_csv(sf_path)
    m.put("SfNPairs", len(sf))
    m.put("SfNPairsWord", word(len(sf)))
    for label, mask in (
        ("Positive", sf["sf_rank_delta"] > 0),
        ("Zero", sf["sf_rank_delta"] == 0),
        ("Negative", sf["sf_rank_delta"] < 0),
    ):
        n = int(mask.sum())
        m.put(f"SfN{label}", n)
        m.put(f"SfN{label}Word", word(n))
        m.put(f"SfN{label}Cap", cap(word(n)))
        m.put(f"Sf{label}Models", join_names([cfg.model_name(x) for x in sf[mask]["sf_subject"]]))
    for _, row in sf.iterrows():
        key = cfg.model_key(row["sf_subject"])
        m.put(f"SfFull{key}", int(row["sf_rank_in_full"]))
        m.put(f"SfLoo{key}", int(row["sf_rank_in_loo"]))
        m.put(f"SfDelta{key}", f(row["sf_rank_delta"], 0))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_macros(cfg: Config) -> Macros:
    data_dir, results_dir = Path(cfg.data_dir), Path(cfg.results_dir)
    ward_df = load_ward_scores(data_dir, cfg.judge_models, cfg.models, cfg.prompt_variants)
    if ward_df.empty:
        raise RuntimeError("No Ward scores found. Run 'alienbench score' first.")
    human_df = load_human_scores(cfg)

    m = Macros()
    design_macros(m, cfg, ward_df)
    human_macros(m, cfg)
    compute_macros(m, cfg)
    model_macros(m, cfg, ward_df)
    condition_effect_macros(m, cfg, ward_df, results_dir)
    dimension_macros(m, cfg, ward_df, human_df)
    condition_feature_macros(m, cfg, ward_df)
    reliability_macros(m, cfg, ward_df, results_dir)
    prompt_ablation_macros(m, cfg, results_dir)
    dimension_ablation_macros(m, cfg, results_dir)
    judge_ablation_macros(m, cfg, results_dir)
    # A capitalized twin of every count word, for sentence-initial use.
    for name in [n for n in m if n.endswith("Word")]:
        twin = name[: -len("Word")] + "Cap"
        if twin not in m:
            m[twin] = cap(m[name])
    return m


def render(macros: Macros) -> str:
    lines = [
        "% Generated by alienbench/paper.py. Do not edit by hand.",
        "% One macro per number or name that paper.tex prints in running text.",
    ]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    return "\n".join(lines) + "\n"


def run(config_path: str = "config.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(config_path)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    macros = build_macros(cfg)
    ward_df = load_ward_scores(Path(cfg.data_dir), cfg.judge_models, cfg.models, cfg.prompt_variants)
    human_df = load_human_scores(cfg)

    # The result paragraphs, written for the case the data fall in. Some of
    # their sentences need macros of their own, so this runs before the
    # macro file is written.
    from alienbench.findings import build_findings, write_findings
    paragraphs = build_findings(cfg, macros, ward_df, human_df, results_dir)

    out = results_dir / OUTPUT_NAME
    out.write_text(render(macros))
    logger.info("Wrote %d macros to %s", len(macros), out)
    write_findings(paragraphs, macros, results_dir)

    # Flag sentences whose wording no longer matches the results.
    from alienbench.claims import write_report
    write_report(cfg, macros, ward_df)

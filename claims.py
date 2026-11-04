"""Checks that the paper's qualitative claims still hold for the current results.

The macros in ``key_numbers.tex`` keep every number in the paper current, but
some sentences also make a claim about the *shape* of the results: that the
top model is open-weight, that exactly one model-feature cell falls below the
human baseline, that two models tie for second place, and so on. A macro
cannot rewrite such a sentence when the results change. This module tests each
of these claims and writes ``results/claims_check.md`` so that, after the
model line-up changes, the sentences that need new wording are listed with
their location in the paper.

The report also lists, per subject model, missing generations, missing judge
scores, and responses cut off at the token limit, and it flags result files
that are older than the newest scores (a stage that still needs a rerun).

Run through ``python -m alienbench paper``, which calls :func:`write_report`
after it writes the macros.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alienbench.config import Config
from alienbench.dimensions import DIMENSION_IDS
from alienbench.paths import (
    ENDINGS,
    cut_offs_path,
    ending,
    extractions_path,
    generations_path,
    model_dir_name,
    ward_scores_path,
)

logger = logging.getLogger(__name__)

REPORT_NAME = "claims_check.md"
SIGNIFICANCE = 0.05


@dataclass
class Claim:
    """One qualitative statement in the paper and whether the data support it."""
    where: str      # location in the paper
    text: str       # the claim, paraphrased
    holds: bool | None  # None when the inputs are missing
    detail: str = ""


class MissingMacro(KeyError):
    """A claim needs a macro that ``key_numbers.tex`` does not define."""


def _num(macros: dict, name: str) -> float:
    """A macro as a number. Raises :class:`MissingMacro` when it is absent.

    Call it inside a claim's condition, so that :func:`_claim` reports the
    claim as unknown instead of as failing.
    """
    try:
        return float(macros[f"ab{name}"])
    except (KeyError, ValueError):
        raise MissingMacro(name) from None


def _pct(macros: dict, name: str) -> float:
    """A percentage macro (``78.4\\%``) as a number. Raises :class:`MissingMacro`."""
    try:
        return float(macros[f"ab{name}"].replace("\\%", ""))
    except (KeyError, ValueError):
        raise MissingMacro(name) from None


def _claim(where: str, text: str, condition, detail: str = "") -> Claim:
    """Build a claim. ``condition`` is a zero-argument callable or a bool."""
    try:
        holds = condition() if callable(condition) else condition
    except (KeyError, TypeError, ValueError, IndexError):
        holds = None
    return Claim(where, text, None if holds is None else bool(holds), detail)


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

def model_claims(cfg: Config, macros: dict, ward_df: pd.DataFrame) -> list[Claim]:
    """Claims about the overall ranking and the open-weight comparison."""
    per_gen = ward_df.groupby(["subject_model", "generation_id"])["ward_score"].mean()
    overall = per_gen.groupby("subject_model").mean().reindex(cfg.models).dropna()
    ranked = overall.sort_values(ascending=False)
    best, worst = ranked.index[0], ranked.index[-1]
    is_open = {x: bool(cfg.model_info[x].open_weights) for x in ranked.index}

    second = f"{ranked.iloc[1]:.2f}"
    n_tied_second = sum(f"{v:.2f}" == second for v in ranked.iloc[1:])

    per_cell = (
        ward_df.groupby(["subject_model", "prompt_variant", "generation_id"])["ward_score"]
        .mean().groupby(["subject_model", "prompt_variant"]).mean()
    )
    leaders = {
        v.id: per_cell.xs(v.id, level="prompt_variant").reindex(ranked.index).idxmax()
        for v in cfg.prompt_variants
    }
    tier_c_leads = [v for v, x in leaders.items() if cfg.model_info[x].tier == "C"]

    below_names = macros.get("abOpenBelowWorstClosedModels", "")
    n_dims = len(DIMENSION_IDS)

    return [
        _claim("Abstract; Introduction; Sec. 4.1 (open-weight paragraph)",
               "Open-weight models have a higher mean overall score than proprietary models.",
               lambda: _num(macros, "OpenMean") > _num(macros, "ClosedMean"),
               f"open {macros.get('abOpenMean')} vs. proprietary {macros.get('abClosedMean')}"),
        _claim("Abstract; Introduction; contributions; Sec. 4.1",
               "Some open-weight models score higher than every proprietary model.",
               lambda: _num(macros, "NOpenAboveClosed") >= 1,
               f"{macros.get('abNOpenAboveClosed')} open-weight models above the best "
               "proprietary model"),
        _claim("Introduction",
               "Open-weight models score highest on most of the features.",
               lambda: _num(macros, "NDimOpenLeader") > n_dims / 2,
               f"{macros.get('abNDimOpenLeader')} of {n_dims} features"),
        _claim("Introduction",
               "The lowest overall score comes from an open-weight model.",
               is_open[worst], f"lowest: {cfg.model_name(worst)}"),
        _claim("Introduction; Sec. 4.1 (open-weight paragraph)",
               "The lowest proprietary model scores below every open-weight model except "
               "named ones (the 'except' list is not empty).",
               lambda: _num(macros, "NOpenBelowWorstClosed") >= 1,
               f"except: {below_names or '(none)'}"),
        _claim("Sec. 4.1 (open-weight paragraph)",
               "The open-weight model(s) below the lowest proprietary model have the lowest "
               "overall score ('which has the lowest overall score', singular).",
               lambda: _num(macros, "NOpenBelowWorstClosed") == 1
               and below_names == cfg.model_name(worst),
               f"except: {below_names}; lowest: {cfg.model_name(worst)}"),
        _claim("Introduction (best model)",
               "The best model is open-weight (stated as 'the highest overall score comes "
               "from an open-weight model' or implied by the open-weight framing).",
               is_open[best], f"best: {cfg.model_name(best)}"),
        _claim("Abstract; Introduction",
               "No single model scores highest under every prompt condition.",
               lambda: _num(macros, "NCondLeaders") > 1,
               f"{macros.get('abNCondLeaders')} different condition leaders"),
        _claim("Introduction",
               "Under some conditions, the highest score comes from a tier-C (small) model.",
               bool(tier_c_leads), f"conditions led by tier C: {tier_c_leads or '(none)'}"),
        _claim("Sec. 3 (prompt conditions)",
               "The best model remains first when any one condition is removed.",
               macros.get("abLocoSameTop") == "true"),
        _claim("Sec. 4.1 (per-condition leaders)",
               "The baseline leader also leads the detailed-description condition "
               "(the sentence names them together).",
               leaders["baseline"] == leaders.get("detailed_description"),
               f"baseline: {cfg.model_name(leaders['baseline'])}; detailed: "
               f"{cfg.model_name(leaders.get('detailed_description', ''))}"),
        _claim("Sec. 4.1 (overall lead)",
               "Two or more models tie for second place ('the second-place tie of ...').",
               n_tied_second >= 2, f"{n_tied_second} model(s) at {second}"),
    ]


def condition_claims(cfg: Config, macros: dict, results_dir: Path) -> list[Claim]:
    """Claims about the prompt conditions and the constrained environments."""
    claims = []
    half = len(DIMENSION_IDS) / 2
    claims.append(_claim(
        "Sec. 4.1", "Scores are uniformly high under the departure-primed condition.",
        lambda: _num(macros, "CondMinDeparturePrimed") >= half,
        f"lowest model: {macros.get('abCondMinDeparturePrimed')}"))
    claims.append(_claim(
        "Sec. 4.1; Conclusion", "Scores are uniformly low under the high-gravity condition.",
        lambda: _num(macros, "CondMaxConstrainedHighGravity") <= half,
        f"highest model: {macros.get('abCondMaxConstrainedHighGravity')}"))

    n_models = len(cfg.models)
    claims.append(_claim(
        "Sec. 5.1 (structured imagination)",
        "Every model except one scores lower under high gravity than under the baseline.",
        lambda: _num(macros, "NBelowBaselineConstrainedHighGravity") == n_models - 1,
        f"{macros.get('abNBelowBaselineConstrainedHighGravity')} of {n_models} below baseline"))

    path = results_dir / "table_posthoc.csv"
    if not path.exists():
        claims.append(Claim("Sec. 4.1; Sec. 5.1; Conclusion",
                            "Condition effects (post-hoc tests)", None, f"{path.name} missing"))
        return claims
    posthoc = pd.read_csv(path)
    vs_base = posthoc[posthoc["group_a"] == "baseline"].set_index("group_b")

    def sig(cond: str) -> bool:
        return vs_base.loc[cond, "p_corrected"] < SIGNIFICANCE

    def r(cond: str) -> float:
        return vs_base.loc[cond, "rank_biserial_r"]

    claims += [
        _claim("Sec. 4.1", "High gravity: large negative effect relative to the baseline "
               "(r < 0, |r| >= 0.5, significant).",
               lambda: sig("constrained_high_gravity") and r("constrained_high_gravity") <= -0.5,
               f"r = {r('constrained_high_gravity'):.2f}"),
        _claim("Sec. 4.1; Sec. 5.1; Conclusion", "Ammonia oceans: small positive effect "
               "(0 < r < 0.3, significant).",
               lambda: sig("constrained_ammonia") and 0 < r("constrained_ammonia") < 0.3,
               f"r = {r('constrained_ammonia'):.2f}"),
        _claim("Sec. 4.1; Sec. 5.1; Conclusion", "No light: no detectable difference from "
               "the baseline (not significant).",
               lambda: not sig("constrained_no_light"),
               f"p = {vs_base.loc['constrained_no_light', 'p_corrected']:.3g}"),
        _claim("Sec. 4.1", "Detailed description: no detectable difference from the "
               "baseline, with the same printed p-value as no light ('both ...').",
               lambda: (not sig("detailed_description"))
               and macros.get("abPDetailedDescription") == macros.get("abPConstrainedNoLight"),
               f"p = {vs_base.loc['detailed_description', 'p_corrected']:.3g}"),
    ]
    constrained = [v.id for v in cfg.prompt_variants if v.id.startswith("constrained_")]
    n_lower = sum(sig(c) and r(c) < 0 for c in constrained)
    claims.append(_claim(
        "Abstract; Conclusion",
        "Only one constrained condition is associated with lower scores than the baseline.",
        n_lower == 1, f"{n_lower} of {len(constrained)}"))
    return claims


FEATURE_KEYS = {
    "symmetry": "Symmetry", "sensory_organs": "SensoryOrgans", "locomotion": "Locomotion",
    "body_plan": "BodyPlan", "skin_covering": "SkinCovering", "reproduction": "Reproduction",
    "metabolism": "Metabolism", "communication": "Communication", "habitat": "Habitat",
    "cognition": "Cognition",
}


def environment_feature_claims(macros: dict) -> list[Claim]:
    """Claims of Sec. 5.1 about single features under the constrained environments."""
    where = "Sec. 5.1 (feature rates)"

    def diff(cond: str, feature: str) -> float:
        key = FEATURE_KEYS[feature]
        return _pct(macros, f"Rate{cond}{key}") - _pct(macros, f"RateBaseline{key}")

    def largest(cond: str, k: int) -> list[str]:
        return sorted(FEATURE_KEYS, key=lambda d: diff(cond, d), reverse=True)[:k]

    def n_lower(cond: str) -> int:
        return sum(diff(cond, d) < 0 for d in FEATURE_KEYS)

    def unmentioned(cond: str, feature: str) -> float:
        return _pct(macros, f"Unmentioned{cond}{FEATURE_KEYS[feature]}")

    nl, am, hg = "ConstrainedNoLight", "ConstrainedAmmonia", "ConstrainedHighGravity"
    return [
        _claim(where, "No light: the two largest increases are sensory organs and communication.",
               lambda: set(largest(nl, 2)) == {"sensory_organs", "communication"},
               lambda_detail(lambda: f"largest: {largest(nl, 2)}")),
        _claim(where, "Ammonia: locomotion and metabolism are higher and among the three "
               "largest increases.",
               lambda: {"locomotion", "metabolism"} <= set(largest(am, 3)),
               lambda_detail(lambda: f"largest: {largest(am, 3)}")),
        _claim(where, "No light and ammonia: several other features are lower than under the "
               "baseline ('several other features become more Earth-typical').",
               lambda: n_lower(nl) >= 2 and n_lower(am) >= 2,
               lambda_detail(lambda: f"lower: no light {n_lower(nl)}, ammonia {n_lower(am)}")),
        _claim(where, "High gravity: the departure rate is lower on every feature.",
               lambda: n_lower(hg) == len(FEATURE_KEYS),
               lambda_detail(lambda: f"{n_lower(hg)} of {len(FEATURE_KEYS)} lower")),
        _claim(where, "High gravity: reproduction, communication, and cognition are unmentioned "
               "more often than under the baseline.",
               lambda: all(unmentioned(hg, d) > unmentioned("Baseline", d)
                           for d in ("reproduction", "communication", "cognition"))),
    ]


def lambda_detail(fn) -> str:
    """Evaluate a detail string, or return an empty one when its inputs are missing."""
    try:
        return fn()
    except (KeyError, TypeError, ValueError):
        return ""


def human_claims(macros: dict) -> list[Claim]:
    return [_claim(
        "Abstract ('nearly every feature'); Introduction ('with one exception'); "
        "Sec. 4.2; Sec. 5; Conclusion",
        "Exactly one (model, feature) pair does not exceed the human baseline.",
        lambda: _num(macros, "NHumanExceptions") == 1,
        f"{macros.get('abNHumanExceptions')} pair(s)"),
        _claim("Sec. 5.1 (structured imagination); Conclusion",
               "Under the baseline condition, the models show the effect in a weaker form "
               "than the humans (higher mean score).",
               lambda: _num(macros, "BaselineModelMean") > _num(macros, "HumanMean"),
               f"models {macros.get('abBaselineModelMean')} vs. humans "
               f"{macros.get('abHumanMean')}"),
        _claim("Sec. 5.1 (structured imagination)",
               "Under the baseline condition, a typical model description keeps most "
               "features Earth-typical (mean below half the scale).",
               lambda: _num(macros, "BaselineModelMean") < len(DIMENSION_IDS) / 2,
               f"mean {macros.get('abBaselineModelMean')}"),
    ]


def reliability_claims(macros: dict) -> list[Claim]:
    n_dims = len(DIMENSION_IDS)
    def n_banded() -> float:
        return sum(float(macros.get(f"abN{b}Dims", 0)) for b in ("Excellent", "Good", "Moderate"))

    return [
        _claim("Sec. 4.3", "Exactly one feature falls below the good-agreement threshold "
               "('... and X falls below it').",
               lambda: n_dims - _num(macros, "NClearDims") == 1,
               f"{macros.get('abNClearDims')} of {n_dims} clear it"),
        _claim("Sec. 4.3", "Every feature is in the Excellent, Good, or Moderate band "
               "(the sentence lists only these three).",
               lambda: n_banded() == n_dims, f"{n_banded():.0f} of {n_dims} in these bands"),
    ]


def ablation_claims(cfg: Config, macros: dict, results_dir: Path) -> list[Claim]:
    claims = [
        _claim("Sec. 4.4 (feature sensitivity)",
               "Exactly one feature preserves the ranking when removed ('the only feature').",
               lambda: _num(macros, "DimNRankPreserved") == 1,
               f"{macros.get('abDimNRankPreserved')} feature(s)"),
        _claim("Sec. 4.4 (judge panel)",
               "Exactly one single judge misses the rank-stability bar ('The bar misses "
               "concern X').",
               lambda: _num(macros, "JudgeSingleNMissBar") == 1,
               f"{macros.get('abJudgeSingleNMissBar')} judge(s)"),
        _claim("Sec. 4.4 (judge panel)",
               "Removing the Claude Opus judge raises alpha, and removing the GPT judge "
               "lowers it.",
               lambda: _num(macros, "JudgeLooAlphaDropClaudeOpus") < 0
               < _num(macros, "JudgeLooAlphaDropGPT")),
        _claim("Sec. 4.4 (same family)",
               "No same-family pair shows a positive rank change.",
               lambda: _num(macros, "SfNPositive") == 0,
               f"{macros.get('abSfNPositive')} positive"),
        _claim("Sec. 4.4 (same family)",
               "At least one same-family pair has a zero and one a negative rank change "
               "(both sentences name models).",
               lambda: _num(macros, "SfNZero") >= 1 and _num(macros, "SfNNegative") >= 1,
               f"zero: {macros.get('abSfNZero')}, negative: {macros.get('abSfNNegative')}"),
    ]
    path = results_dir / "table_ablation_prompt.csv"
    if path.exists():
        covered = set(pd.read_csv(path)["subject_model"])
        missing = [cfg.model_name(x) for x in cfg.models if x not in covered]
        claims.append(_claim(
            "Sec. 4.4 (prompt paraphrase)",
            "The paraphrase ablation covers every subject model.",
            not missing, f"missing: {', '.join(missing)}" if missing else ""))
    claims.append(Claim(
        "Sec. 4.4 (prompt paraphrase); Sec. 5",
        "The paraphrase outlier is still far outside the other models' shifts, and "
        "'rankings are robust to prompt paraphrase' still fits the numbers.",
        None, "judgment call: read tab_ablation_prompt.tex"))
    return claims


# ---------------------------------------------------------------------------
# Data completeness and staleness
# ---------------------------------------------------------------------------

def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def completeness_rows(cfg: Config) -> list[dict]:
    """Per subject model: generations and judge scores."""
    data_dir = Path(cfg.data_dir)
    n = cfg.samples_per_condition
    rows = []
    for model in cfg.models:
        n_gen = 0
        n_scored = {j: 0 for j in cfg.judge_models}
        for v in cfg.prompt_variants:
            gens = [g for g in _jsonl(generations_path(data_dir, model, v.id))
                    if g.get("sample_index", n) < n]
            n_gen += len(gens)
            ids = {g["id"] for g in gens}
            for j in cfg.judge_models:
                scored = {s.get("generation_id") for s in
                          _jsonl(ward_scores_path(data_dir, j, model, v.id))}
                n_scored[j] += len(ids & scored)
        rows.append({
            "model": model,
            "expected": n * len(cfg.prompt_variants),
            "generations": n_gen,
            "scored": n_scored,
        })
    return rows


def truncation_rows(cfg: Config) -> list[dict]:
    """Per subject model: how each generated response ended.

    Covers the main conditions and the paraphrase ablation. Each response
    falls in one category of :data:`alienbench.paths.ENDINGS`.
    """
    data_dir = Path(cfg.data_dir)
    prompt_ids = list(dict.fromkeys(
        [v.id for v in cfg.prompt_variants] + [p.id for p in cfg.prompt_paraphrases]))
    rows = []
    for model in cfg.models:
        endings = [ending(g) for pid in prompt_ids
                   for g in _jsonl(generations_path(data_dir, model, pid))]
        row = {"model": model, "responses": len(endings)}
        row.update({e: endings.count(e) for e in ENDINGS})
        rows.append(row)
    return rows


def _limit_row(name: str, start: int, ceiling: int, records: list[dict],
               cut_offs: list[dict], key: str) -> dict:
    """Summary of one model's or judge's output-token limits.

    ``key`` names the field that links a cut-off entry to its record
    (``sample_index`` for generations, ``generation_id`` for extractions).
    """
    raised = [r for r in records if r.get("n_cut_offs")]
    recorded = {(r.get("prompt_variant"), r.get(key)) for r in records}
    unresolved = {(c.get("prompt_variant"), c.get(key)) for c in cut_offs} - recorded
    return {
        "name": name,
        "start": start,
        "ceiling": ceiling,
        "records": len(records),
        "raised": len(raised),
        "highest": max((r.get("max_tokens") or 0 for r in records), default=0),
        "discarded": len(cut_offs),
        "unresolved": len(unresolved),
    }


def limit_rows(cfg: Config) -> tuple[list[dict], list[dict]]:
    """Output-token limits per subject model and per judge.

    Subject models cover the main conditions and the paraphrase ablation.
    Judges cover their extractions of the subject models' responses.
    """
    data_dir = Path(cfg.data_dir)
    prompt_ids = list(dict.fromkeys(
        [v.id for v in cfg.prompt_variants] + [p.id for p in cfg.prompt_paraphrases]))
    models = []
    for model in cfg.models:
        paths = [generations_path(data_dir, model, pid) for pid in prompt_ids]
        models.append(_limit_row(
            cfg.model_name(model), cfg.max_tokens_for(model),
            cfg.max_tokens_ceiling_for(model),
            [r for p in paths for r in _jsonl(p)],
            [c for p in paths for c in _jsonl(cut_offs_path(p))], "sample_index"))
    judges = []
    for judge in cfg.judge_models:
        paths = [extractions_path(data_dir, judge, model, pid)
                 for model in cfg.models for pid in prompt_ids]
        judges.append(_limit_row(
            cfg.judge_name(judge), cfg.judge_max_tokens_for(judge),
            cfg.judge_max_tokens_ceiling_for(judge),
            [r for p in paths for r in _jsonl(p)],
            [c for p in paths for c in _jsonl(cut_offs_path(p))], "generation_id"))
    return models, judges


def limit_section(cfg: Config) -> tuple[list[str], list[str]]:
    """Report lines on output-token limits, and one sentence per problem."""
    models, judges = limit_rows(cfg)
    lines = ["## Output-token limits", "",
             "A call starts at the start limit. A response cut off at the limit "
             "is discarded, and the call is repeated with the limit doubled, up "
             "to the ceiling. Raised: records produced at a raised limit. "
             "Highest: the largest limit any record was produced under. "
             "Discarded: cut-off calls, logged in `cut_offs.jsonl`. Unresolved: "
             "samples still cut off at the ceiling, with no record.", ""]
    problems = []
    for kind, rows in (("Subject model", models), ("Judge", judges)):
        lines += [f"| {kind} | Start | Ceiling | Records | Raised | Highest | Discarded | Unresolved |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            unresolved = f"**{r['unresolved']}**" if r["unresolved"] else "0"
            lines.append(
                f"| {r['name']} | {r['start']} | {r['ceiling']} | {r['records']} | "
                f"{r['raised']} | {r['highest']} | {r['discarded']} | {unresolved} |")
            if r["unresolved"]:
                problems.append(
                    f"{r['unresolved']} {'sample' if r['unresolved'] == 1 else 'samples'}"
                    f" of {r['name']} cut off at the ceiling of {r['ceiling']} tokens")
        lines.append("")
    return lines, problems


# Result files and the stage that writes them. A file older than the newest
# score record is stale and its stage needs a rerun.
STAGE_OUTPUTS = {
    "analyze": ["table_posthoc.csv", "table_reliability.csv", "fig5_ward_radar.pdf"],
    "latex": ["tab_ward_scores.tex", "tab_ward_dimensions.tex", "tab_models.tex"],
    "ablation prompt": ["table_ablation_prompt.csv", "tab_ablation_prompt.tex"],
    "ablation dimensions": ["table_ablation_dimensions_loo.csv", "tab_ablation_dimensions.tex"],
    "ablation judges": ["table_ablation_judges_loo.csv",
                        "table_ablation_judges_same_family.csv", "tab_ablation_judges.tex"],
}


def stale_outputs(cfg: Config) -> list[tuple[str, str]]:
    # Only subject-model scores feed these outputs. Human-baseline scores
    # are excluded so that rescoring the humans does not flag every stage.
    data_dir = Path(cfg.data_dir)
    paths = [ward_scores_path(data_dir, j, x, v.id)
             for j in cfg.judge_models for x in cfg.models for v in cfg.prompt_variants]
    newest = max((p.stat().st_mtime for p in paths if p.exists()), default=0)
    results_dir = Path(cfg.results_dir)
    stale = []
    for stage, names in STAGE_OUTPUTS.items():
        for name in names:
            path = results_dir / name
            if not path.exists() or path.stat().st_mtime < newest:
                stale.append((stage, name))
    return stale


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# Problem endings: (key, column header, singular sentence, plural sentence).
_PROBLEMS = [
    ("truncated", "Truncated", "response ended at `max_tokens`",
     "responses ended at `max_tokens`"),
    ("no_finish_reason", "No finish reason", "response has no finish reason",
     "responses have no finish reason"),
    ("refusal", "Refusal", "response was stopped by a provider refusal",
     "responses were stopped by a provider refusal"),
    ("other", "Other", "response ended for another reason",
     "responses ended for another reason"),
]


def truncation_section(cfg: Config) -> tuple[list[str], list[str]]:
    """Report lines on how responses ended, and one sentence per problem."""
    # Every model must end its description on its own stop token. Every
    # other ending is reported.
    rows = truncation_rows(cfg)
    lines = ["## Responses that did not end on the model's stop token", "",
             "Main conditions and paraphrase ablation.", ""]
    problems = []
    for key, _, one, many in _PROBLEMS:
        n = sum(r[key] for r in rows)
        if n:
            problems.append(f"{n} {one if n == 1 else many}")
    if problems:
        lines += [f"**{'. '.join(problems)}.** Regenerate these responses "
                  "before reporting results.", ""]
    else:
        lines += ["Every response ended on the model's own stop token.", ""]
    lines += ["| Model | Responses | Stop | " + " | ".join(h for _, h, _, _ in _PROBLEMS) + " |",
              "|---|---|---|" + "---|" * len(_PROBLEMS)]
    for row in rows:
        cells = [str(row[key]) if not row[key] else f"**{row[key]}**"
                 for key, _, _, _ in _PROBLEMS]
        lines.append(f"| {cfg.model_name(row['model'])} | {row['responses']} | "
                     f"{row['stop']} | " + " | ".join(cells) + " |")
    lines += ["",
              "Truncated: cut off at `max_tokens`. Generation discards these "
              "responses and retries at a higher limit, so a record shows this "
              "only if it predates that rule. "
              "No finish reason: the provider's stream ended before its final "
              "chunk, so the text stops early. Refusal: the provider's safety "
              "system stopped the text (`native_finish_reason: refusal`), which "
              "OpenRouter reports as `stop`. Other: any other finish reason, such "
              "as a content filter.", ""]
    return lines, problems


def _mark(holds: bool | None) -> str:
    return {True: "OK", False: "**FAILS**", None: "check by hand"}[holds]


def write_report(cfg: Config, macros: dict, ward_df: pd.DataFrame) -> list[Claim]:
    results_dir = Path(cfg.results_dir)
    claims = (
        model_claims(cfg, macros, ward_df)
        + condition_claims(cfg, macros, results_dir)
        + environment_feature_claims(macros)
        + human_claims(macros)
        + reliability_claims(macros)
        + ablation_claims(cfg, macros, results_dir)
    )

    lines = ["# AlienBench: checks before updating the paper", ""]
    lines += ["## Data completeness per subject model", "",
              "| Model | Generations | Scored per judge |",
              "|---|---|---|"]
    for row in completeness_rows(cfg):
        scored = ", ".join(f"{model_dir_name(j).split('__')[-1]} {n}"
                           for j, n in row["scored"].items())
        flag = "" if row["generations"] == row["expected"] else " **incomplete**"
        lines.append(
            f"| {cfg.model_name(row['model'])} | {row['generations']}/{row['expected']}{flag}"
            f" | {scored} |")
    lines.append("")

    section, problems = truncation_section(cfg)
    lines += section
    section, limit_problems = limit_section(cfg)
    lines += section

    stale = stale_outputs(cfg)
    lines += ["## Result files older than the newest scores", ""]
    if stale:
        lines += [f"- `{name}`: rerun `python -m alienbench {stage}`" for stage, name in stale]
    else:
        lines.append("None.")
    lines.append("")

    failed = [c for c in claims if c.holds is False]
    lines += ["## Claims in the paper", "",
              f"{len(failed)} of {len(claims)} claims fail. Each failing claim marks a "
              "sentence whose wording, not only its numbers, must change.", "",
              "| Status | Where | Claim | Detail |", "|---|---|---|---|"]
    for c in sorted(claims, key=lambda c: {False: 0, None: 1, True: 2}[c.holds]):
        lines.append(f"| {_mark(c.holds)} | {c.where} | {c.text} | {c.detail} |")

    out = results_dir / REPORT_NAME
    out.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", out)
    for problem in problems:
        logger.warning("Truncation check: %s; see %s", problem, out)
    for problem in limit_problems:
        logger.warning("Output-token limit: %s; see %s", problem, out)
    for c in failed:
        logger.warning("Claim fails (%s): %s [%s]", c.where, c.text, c.detail)
    if stale:
        logger.warning("%d result file(s) are older than the newest scores; see %s",
                       len(stale), out)
    return claims

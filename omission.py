"""Features that a creature description does not mention.

The boundary rules label a feature that a description does not mention as
Earth-typical (0). A low departure rate can therefore come from Earth-typical
features or from features left out. This module tells the two apart. Each
judge writes a short ``feature_described`` text per feature, and for a feature
the description does not cover, this text says so ("not described", "no
communication described", "not explicitly described in the text").
:func:`is_unmentioned` recognizes these texts, and :func:`unmentioned_rates`
reports their share per prompt condition and feature.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from alienbench.config import Config
from alienbench.dimensions import DIMENSION_IDS
from alienbench.paths import extractions_path, iter_generations, iter_jsonl

# "not described", "not explicitly described in the text", "cognition is not
# described", "no communication described", "no explicit cognitive
# architecture described", "unspecified". A text that goes on to describe
# something ("not explicitly described beyond social behavior", "mating
# mentioned but no mechanism described") reports a partial description and
# does not count.
_NEGATED = re.compile(
    r"^(?:[a-z/ -]+? )?(?:is |are )?"
    r"(?:not|no|none)\b[a-z ,-]*?\b"
    r"(?:described|mentioned|specified|addressed|stated|discussed|given|provided)\b",
)
_UNSPECIFIED = re.compile(r"^(?:[a-z/ -]+ )?(?:unspecified|unknown|n/?a)\b")
# "no explicit description of cognitive architecture", "no information given".
_NO_DESCRIPTION = re.compile(
    r"^(?:[a-z/ -]+? )?no (?:explicit |specific |clear )?(?:description|mention|information|details?)\b"
)
_PARTIAL = re.compile(r"\b(?:beyond|but|except|although|though)\b")


def is_unmentioned(feature_described: str | None) -> bool:
    """True when a judge's ``feature_described`` text says the feature is absent."""
    text = re.sub(r"\s+", " ", (feature_described or "").strip().lower())
    if not text:
        return True
    if _PARTIAL.search(text):
        return False
    return bool(_NEGATED.match(text) or _UNSPECIFIED.match(text) or _NO_DESCRIPTION.match(text))


def unmentioned_rates(cfg: Config, models: list[str] | None = None) -> pd.DataFrame:
    """Share of judge labels whose feature the description does not mention.

    Rows are prompt conditions and columns are features. Every (judge,
    generation) label counts once, pooled over the subject models, and only
    the first ``samples_per_condition`` generations per cell are used, as in
    the rest of the pipeline.
    """
    data_dir = Path(cfg.data_dir)
    models = list(cfg.models) if models is None else models
    rows = []
    for v in cfg.prompt_variants:
        counts = dict.fromkeys(DIMENSION_IDS, 0)
        n = 0
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
                    n += 1
                    for d in DIMENSION_IDS:
                        feat = rec["features"].get(d) or {}
                        counts[d] += is_unmentioned(feat.get("feature_described"))
        rows.append({"prompt_variant": v.id,
                     **{d: counts[d] / n if n else float("nan") for d in DIMENSION_IDS}})
    return pd.DataFrame(rows).set_index("prompt_variant")

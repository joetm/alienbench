"""Human validation harness for the Ward rubric.

The paper (sec:analysis) commits to a human validation study in which a
stratified sample of subject generations is coded by human annotators under
the same Ward rubric used by the LLM judges. This module provides the three
pieces needed to run that study:

1. :func:`sample` draws a stratified sample across every
   (subject_model, prompt_variant) cell and writes a CSV template the
   annotators fill in.
2. :func:`ingest` validates a filled CSV and stores it in the same on-disk
   format used by the LLM-judge pipeline, under a ``human/<annotator_id>``
   alias (sanitized to ``human__<annotator_id>`` on disk). This means the
   existing DataFrame loaders and analysis utilities treat humans as just
   another rater.
3. :func:`analyze` computes Krippendorff's alpha between humans and judges
   at the Ward total and per-dimension levels, writes
   ``table_human_validation.csv``, and appends a summary block to
   ``summary.txt``.

CSV template columns (one row per generation x dimension):

    generation_id, subject_model, prompt_variant, dimension, earth_default,
    departure_examples, boundary_note, description, is_departure, reasoning

``is_departure`` is blank in the template; the annotator fills 0 or 1.
``reasoning`` is optional.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from alienbench.config import load_config
from alienbench.dimensions import DIMENSION_IDS, WARD_DIMENSIONS, compute_ward_score
from alienbench.paths import (
    extractions_path,
    iter_jsonl,
    load_ward_scores,
    ward_scores_path,
)
from alienbench.stats import krippendorff_alpha

logger = logging.getLogger(__name__)


HUMAN_PREFIX = "human/"
CSV_COLUMNS = [
    "generation_id",
    "subject_model",
    "prompt_variant",
    "dimension",
    "earth_default",
    "departure_examples",
    "boundary_note",
    "description",
    "is_departure",
    "reasoning",
]
_TRUE_TOKENS = {"1", "true", "t", "yes", "y"}
_FALSE_TOKENS = {"0", "false", "f", "no", "n"}


def _human_alias(annotator_id: str) -> str:
    """Return the internal judge-alias for a human annotator.

    Sanitization: strip whitespace, lowercase, replace spaces with ``_``.
    Rejects empty strings and strings containing ``/`` (which would collide
    with the ``provider/model`` convention used for LLM judges).
    """
    aid = annotator_id.strip().lower().replace(" ", "_")
    if not aid:
        raise ValueError("annotator_id must be non-empty")
    if "/" in aid:
        raise ValueError("annotator_id must not contain '/'")
    return HUMAN_PREFIX + aid


def _parse_is_departure(raw: str) -> int:
    s = str(raw).strip().lower()
    if s in _TRUE_TOKENS:
        return 1
    if s in _FALSE_TOKENS:
        return 0
    raise ValueError(f"is_departure must be 0/1 (or true/false), got {raw!r}")


# ---------------------------------------------------------------------------
# 1. Stratified sampling
# ---------------------------------------------------------------------------

def sample(
    config_path: str = "config.yaml",
    samples_per_stratum: int = 5,
    seed: int = 42,
    out_path: str | Path | None = None,
) -> Path:
    """Draw a stratified sample of generations and write a CSV template.

    One row per (generation, dimension) pair, so an annotator reads each
    dimension's rubric on the same row as the feature it applies to.
    """
    cfg = load_config(config_path)
    data_dir = Path(cfg.data_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = results_dir / "human_sample.csv"
    out_path = Path(out_path)

    rng = np.random.default_rng(seed)
    dim_by_id = {d["id"]: d for d in WARD_DIMENSIONS}

    picked_rows: list[dict] = []
    total_picked = 0
    for model in cfg.models:
        for variant in cfg.prompt_variants:
            gen_path = (
                data_dir / "generations"
                / model.replace("/", "__") / variant.id / "responses.jsonl"
            )
            if not gen_path.exists():
                logger.warning(
                    "No generations for %s / %s; skipping stratum", model, variant.id
                )
                continue
            records = [r for r in iter_jsonl(gen_path) if r.get("id") and r.get("response")]
            if not records:
                continue
            k = min(samples_per_stratum, len(records))
            idx = rng.choice(len(records), size=k, replace=False)
            total_picked += k
            for i in sorted(int(j) for j in idx):
                rec = records[i]
                for dim_id in DIMENSION_IDS:
                    d = dim_by_id[dim_id]
                    picked_rows.append({
                        "generation_id": rec["id"],
                        "subject_model": model,
                        "prompt_variant": variant.id,
                        "dimension": dim_id,
                        "earth_default": d["earth_default"],
                        "departure_examples": d["departure_examples"],
                        "boundary_note": d["boundary_note"],
                        "description": rec["response"],
                        "is_departure": "",
                        "reasoning": "",
                    })

    if not picked_rows:
        raise RuntimeError(
            "No generations available to sample. Run the generate stage first."
        )

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in picked_rows:
            writer.writerow(row)

    logger.info(
        "Wrote sample of %d generations (%d rows) to %s",
        total_picked, len(picked_rows), out_path,
    )
    return out_path


# ---------------------------------------------------------------------------
# 2. Ingest filled CSV → features.jsonl + ward_scores.jsonl
# ---------------------------------------------------------------------------

def ingest(
    config_path: str,
    annotator_id: str,
    csv_path: str | Path,
) -> dict[str, int]:
    """Convert a filled annotation CSV into the on-disk judge format.

    Writes two artifacts per (subject_model, prompt_variant) stratum touched:
    a ``features.jsonl`` under ``data/extractions/human__<annotator>/...``
    (for provenance and round-trip compatibility) and a ``ward_scores.jsonl``
    under ``data/scores/human__<annotator>/...`` (read directly by
    :func:`analyze`). The existing :func:`compute_ward_score` is reused so
    the scoring rule is identical to the LLM-judge pipeline.

    Returns a dict with counts: ``generations``, ``strata``, ``rows``.
    """
    cfg = load_config(config_path)
    data_dir = Path(cfg.data_dir)
    alias = _human_alias(annotator_id)

    csv_path = Path(csv_path)
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        missing = set(CSV_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV is missing columns: {sorted(missing)}. "
                f"Expected {CSV_COLUMNS}."
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")

    known_dims = set(DIMENSION_IDS)
    by_gen: dict[tuple, dict] = {}
    for i, row in enumerate(rows, start=2):  # row 1 is header
        dim = row["dimension"].strip()
        if dim not in known_dims:
            raise ValueError(f"row {i}: unknown dimension {dim!r}")
        try:
            dep = _parse_is_departure(row["is_departure"])
        except ValueError as e:
            raise ValueError(f"row {i}: {e}") from e
        key = (row["generation_id"], row["subject_model"], row["prompt_variant"])
        bucket = by_gen.setdefault(key, {"features": {}, "description": row["description"]})
        if dim in bucket["features"]:
            raise ValueError(f"row {i}: duplicate dimension {dim!r} for generation {key[0]}")
        bucket["features"][dim] = {
            "feature_described": "",
            "is_departure": bool(dep),
            "reasoning": row.get("reasoning", ""),
        }

    for key, entry in by_gen.items():
        missing_dims = known_dims - set(entry["features"])
        if missing_dims:
            raise ValueError(
                f"generation {key[0]!r}: missing dimensions "
                f"{sorted(missing_dims)}"
            )

    strata: dict[tuple, list[tuple[str, dict]]] = {}
    for (gen_id, model, variant), entry in by_gen.items():
        strata.setdefault((model, variant), []).append((gen_id, entry))

    ts = time.time()
    for (model, variant), entries in strata.items():
        extr_p = extractions_path(data_dir, alias, model, variant)
        extr_p.parent.mkdir(parents=True, exist_ok=True)
        ward_p = ward_scores_path(data_dir, alias, model, variant)
        ward_p.parent.mkdir(parents=True, exist_ok=True)

        with extr_p.open("w") as f_extr, ward_p.open("w") as f_ward:
            for gen_id, entry in entries:
                extraction = {
                    "generation_id": gen_id,
                    "judge_model": alias,
                    "judge_model_resolved": alias,
                    "judge_call_id": None,
                    "subject_model": model,
                    "prompt_variant": variant,
                    "timestamp": ts,
                    "features": entry["features"],
                    "parse_error": False,
                    "raw_response": None,
                }
                f_extr.write(json.dumps(extraction) + "\n")

                ward = compute_ward_score(entry["features"])
                ward_record = {
                    "generation_id": gen_id,
                    "judge_model": alias,
                    "judge_model_resolved": alias,
                    "judge_call_id": None,
                    "subject_model": model,
                    "prompt_variant": variant,
                    "ward_score": ward["total"],
                    "per_dimension": ward["per_dimension"],
                    "timestamp": ts,
                }
                f_ward.write(json.dumps(ward_record) + "\n")

    counts = {
        "generations": len(by_gen),
        "strata": len(strata),
        "rows": len(rows),
    }
    logger.info(
        "Ingested %(generations)d generations across %(strata)d strata for annotator %(alias)s",
        {**counts, "alias": alias},
    )
    return counts


# ---------------------------------------------------------------------------
# 3. Human–judge Krippendorff alpha
# ---------------------------------------------------------------------------

def _discover_human_aliases(data_dir: Path) -> list[str]:
    """Return sorted human judge-aliases present under ``data/scores``."""
    scores_dir = data_dir / "scores"
    if not scores_dir.exists():
        return []
    aliases = []
    prefix = HUMAN_PREFIX.replace("/", "__")
    for sub in sorted(scores_dir.iterdir()):
        if sub.is_dir() and sub.name.startswith(prefix):
            aliases.append(sub.name.replace("__", "/", 1))
    return aliases


def _load_human_ward_df(
    data_dir: Path,
    human_aliases: list[str],
    models: list[str],
    prompt_variants,
) -> pd.DataFrame:
    """Load Ward score records written by :func:`ingest`.

    Mirrors :func:`alienbench.paths.load_ward_scores` but walks the
    ``human__<annotator>`` prefix and exposes the same columns so it can be
    concatenated with the judge DataFrame.
    """
    rows: list[dict] = []
    for alias in human_aliases:
        for model in models:
            for variant in prompt_variants:
                path = ward_scores_path(data_dir, alias, model, variant.id)
                if not path.exists():
                    continue
                for rec in iter_jsonl(path):
                    row = {
                        "generation_id": rec["generation_id"],
                        "judge_model": rec["judge_model"],
                        "subject_model": rec["subject_model"],
                        "prompt_variant": rec["prompt_variant"],
                        "ward_score": rec["ward_score"],
                    }
                    row.update({
                        f"dim_{d}": rec["per_dimension"].get(d, 0)
                        for d in DIMENSION_IDS
                    })
                    rows.append(row)
    return pd.DataFrame(rows)


def analyze(
    config_path: str = "config.yaml",
    human_aliases: Iterable[str] | None = None,
) -> Path | None:
    """Compute human–judge Krippendorff's alpha at total and per-dimension levels.

    When ``human_aliases`` is ``None`` the set is discovered from
    ``data/scores/human__*``. Writes ``table_human_validation.csv`` and
    appends a ``## Human Validation`` block to ``summary.txt`` so it is
    present in the artifact that the paper prose cites.
    """
    cfg = load_config(config_path)
    data_dir = Path(cfg.data_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if human_aliases is None:
        human_aliases = _discover_human_aliases(data_dir)
    else:
        human_aliases = list(human_aliases)

    if not human_aliases:
        logger.warning(
            "No human annotators found under %s/scores/human__*; skipping.",
            data_dir,
        )
        return None

    judge_df = load_ward_scores(data_dir, cfg.judge_models, cfg.models, cfg.prompt_variants)
    human_df = _load_human_ward_df(data_dir, human_aliases, cfg.models, cfg.prompt_variants)

    if human_df.empty:
        logger.warning("No human ward-score records loaded; skipping human analysis.")
        return None

    judge_df = judge_df.assign(rater_type="judge")
    human_df = human_df.assign(rater_type="human")
    combined = pd.concat([judge_df, human_df], ignore_index=True)

    measures = [("Ward Total (0–10)", "ward_score", "interval")]
    measures += [(f"Ward: {d}", f"dim_{d}", "nominal") for d in DIMENSION_IDS]

    rows = []
    for label, col, level in measures:
        if col not in combined.columns:
            continue
        alpha_hj = krippendorff_alpha(
            combined, "judge_model", "generation_id", col, level=level
        )
        alpha_j = (
            krippendorff_alpha(judge_df, "judge_model", "generation_id", col, level=level)
            if not judge_df.empty else float("nan")
        )
        alpha_h = (
            krippendorff_alpha(human_df, "judge_model", "generation_id", col, level=level)
            if human_df["judge_model"].nunique() >= 2 else float("nan")
        )
        rows.append({
            "Measure": label,
            "α (human+judge)": alpha_hj,
            "α (judges only)": alpha_j,
            "α (humans only)": alpha_h,
        })

    table_df = pd.DataFrame(rows)
    out_path = results_dir / "table_human_validation.csv"
    table_df.to_csv(out_path, index=False)
    logger.info("Saved %s", out_path)

    lines = ["\n## Human Validation (Human–Judge Krippendorff α)\n"]
    lines.append(f"  Human annotators: {', '.join(human_aliases)}")
    lines.append(f"  Judges: {', '.join(cfg.judge_models)}")
    lines.append(f"  Human-annotated generations: {human_df['generation_id'].nunique()}")
    for _, r in table_df.iterrows():
        def _fmt(x):
            try:
                v = float(x)
            except (TypeError, ValueError):
                return "n/a"
            return f"{v:.3f}" if not np.isnan(v) else "n/a"
        lines.append(
            f"  {r['Measure']}: "
            f"α(h+j)={_fmt(r['α (human+judge)'])}, "
            f"α(j)={_fmt(r['α (judges only)'])}, "
            f"α(h)={_fmt(r['α (humans only)'])}"
        )

    summary_path = results_dir / "summary.txt"
    if summary_path.exists():
        with summary_path.open("a") as fh:
            fh.write("\n".join(lines) + "\n")
    else:
        summary_path.write_text("\n".join(lines) + "\n")
    logger.info("Appended human-validation block to %s", summary_path)

    return out_path

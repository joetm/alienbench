"""Shared file-path helpers and JSONL loaders.

Used by the main pipeline (generate/extract/score/analyze) and by the
ablation modules. Centralises the on-disk layout under ``data_dir`` and the
conventions for reading each stage's JSONL outputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model / path naming
# ---------------------------------------------------------------------------

def model_dir_name(model_id: str) -> str:
    """Convert 'openai/gpt-4o' -> 'openai__gpt-4o' for use as a directory name."""
    return model_id.replace("/", "__")


def generations_path(data_dir: Path, model_id: str, prompt_id: str) -> Path:
    return data_dir / "generations" / model_dir_name(model_id) / prompt_id / "responses.jsonl"


def extractions_path(data_dir: Path, judge_model: str, model_id: str, prompt_id: str) -> Path:
    return (
        data_dir
        / "extractions"
        / model_dir_name(judge_model)
        / model_dir_name(model_id)
        / prompt_id
        / "features.jsonl"
    )


def ward_scores_path(data_dir: Path, judge_model: str, model_id: str, prompt_id: str) -> Path:
    return (
        data_dir / "scores" / model_dir_name(judge_model)
        / model_dir_name(model_id) / prompt_id / "ward_scores.jsonl"
    )


# ---------------------------------------------------------------------------
# JSONL iteration + checkpointing
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield records from a JSONL file, logging a warning for malformed lines."""
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Malformed JSON in %s (line %d): %s", path, lineno, e)


def count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def load_extracted_ids(path: Path) -> set[str]:
    """Return set of generation_ids already extracted in this file."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    for rec in iter_jsonl(path):
        try:
            ids.add(rec["generation_id"])
        except KeyError:
            pass
    return ids


def load_scored_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for rec in iter_jsonl(path):
        try:
            ids.add(rec["generation_id"])
        except KeyError:
            pass
    return ids


def iter_generations(data_dir: Path, model_id: str, prompt_id: str) -> Iterator[dict]:
    path = generations_path(data_dir, model_id, prompt_id)
    if not path.exists():
        return
    yield from iter_jsonl(path)


def iter_extractions(data_dir: Path, judge: str, model_id: str, prompt_id: str) -> Iterator[dict]:
    """Yield extraction records, skipping those flagged ``parse_error``.

    Use :func:`iter_all_extractions` when you need to count failed
    extractions (e.g. for the parse-failure analysis in
    :func:`alienbench.analyze.write_parse_failure_analysis`).
    """
    path = extractions_path(data_dir, judge, model_id, prompt_id)
    if not path.exists():
        return
    for rec in iter_jsonl(path):
        if not rec.get("parse_error"):
            yield rec


def iter_all_extractions(data_dir: Path, judge: str, model_id: str, prompt_id: str) -> Iterator[dict]:
    """Yield every extraction record, including those flagged ``parse_error``."""
    path = extractions_path(data_dir, judge, model_id, prompt_id)
    if not path.exists():
        return
    yield from iter_jsonl(path)


# ---------------------------------------------------------------------------
# DataFrame loaders
# ---------------------------------------------------------------------------

def load_generation_tokens(data_dir: Path, models: list[str], prompt_variants) -> pd.DataFrame:
    """Load completion_tokens from generation records.

    Returns a DataFrame with columns: subject_model, prompt_variant,
    generation_id, completion_tokens.
    """
    rows = []
    for model in models:
        for variant in prompt_variants:
            path = generations_path(data_dir, model, variant.id)
            if not path.exists():
                continue
            for rec in iter_jsonl(path):
                gen_id = rec.get("id")
                ct = rec.get("completion_tokens")
                if gen_id and ct is not None:
                    rows.append({
                        "generation_id": gen_id,
                        "subject_model": model,
                        "prompt_variant": variant.id,
                        "completion_tokens": ct,
                    })
    return pd.DataFrame(rows)


def load_extraction_tokens(
    data_dir: Path,
    judge_models: list[str],
    models: list[str],
    prompt_variants,
) -> pd.DataFrame:
    """Load token counts and timing from extraction records.

    Returns a DataFrame with columns: judge_model, subject_model,
    prompt_variant, generation_id, prompt_tokens, completion_tokens,
    duration_seconds, n_parse_attempts. Records missing any of these
    fields (e.g. produced before the compute-tracking change) are
    dropped silently.
    """
    rows = []
    for judge in judge_models:
        for model in models:
            for variant in prompt_variants:
                for rec in iter_all_extractions(data_dir, judge, model, variant.id):
                    pt = rec.get("prompt_tokens")
                    ct = rec.get("completion_tokens")
                    ds = rec.get("duration_seconds")
                    if pt is None or ct is None or ds is None:
                        continue
                    rows.append({
                        "judge_model": judge,
                        "subject_model": model,
                        "prompt_variant": variant.id,
                        "generation_id": rec.get("generation_id"),
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "duration_seconds": ds,
                        "n_parse_attempts": rec.get("n_parse_attempts", 1),
                    })
    return pd.DataFrame(rows)


def load_extraction_status(
    data_dir: Path,
    judge_models: list[str],
    models: list[str],
    prompt_variants,
) -> pd.DataFrame:
    """Return one row per (judge, generation) attempt with success/failure flags.

    A generation can be in three states with respect to a given judge:

    * ``success``: an extraction record exists with ``parse_error=False``.
    * ``parse_error``: an extraction record exists with ``parse_error=True``
      (the judge returned text that could not be parsed as the expected JSON
      structure).
    * ``api_error``: the generation has no extraction record for this judge
      (the judge's API call failed and was not retryable, so no record was
      written; see :func:`alienbench.extract.run`).

    The resulting DataFrame has columns: ``judge_model``, ``subject_model``,
    ``prompt_variant``, ``generation_id``, ``status``.
    """
    rows = []
    for model in models:
        for variant in prompt_variants:
            gen_path = generations_path(data_dir, model, variant.id)
            if not gen_path.exists():
                continue
            gen_ids = [rec["id"] for rec in iter_jsonl(gen_path) if "id" in rec]
            for judge in judge_models:
                ext_records = {
                    rec["generation_id"]: rec
                    for rec in iter_all_extractions(data_dir, judge, model, variant.id)
                    if "generation_id" in rec
                }
                for gid in gen_ids:
                    rec = ext_records.get(gid)
                    if rec is None:
                        status = "api_error"
                    elif rec.get("parse_error"):
                        status = "parse_error"
                    else:
                        status = "success"
                    rows.append({
                        "judge_model": judge,
                        "subject_model": model,
                        "prompt_variant": variant.id,
                        "generation_id": gid,
                        "status": status,
                    })
    return pd.DataFrame(rows)


def load_ward_scores(data_dir: Path, judge_models: list[str], models: list[str], prompt_variants) -> pd.DataFrame:
    """Load Ward score records for every (judge, model, prompt) triple.

    Each row carries the total Ward score and a ``dim_<id>`` column per
    dimension in :data:`alienbench.dimensions.DIMENSION_IDS`.
    """
    from alienbench.dimensions import DIMENSION_IDS

    rows = []
    for judge in judge_models:
        for model in models:
            for variant in prompt_variants:
                path = ward_scores_path(data_dir, judge, model, variant.id)
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
                    row.update({f"dim_{d}": rec["per_dimension"].get(d, 0) for d in DIMENSION_IDS})
                    rows.append(row)
    return pd.DataFrame(rows)

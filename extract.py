"""Stage 2: Extract Ward feature dimensions from creature descriptions.

Multiple ``python -m alienbench extract`` processes may run concurrently
against the same ``data/`` directory. Duplication prevention is delegated
to :mod:`alienbench.reservation`: each ``generation_id`` within a
(judge, subject_model, prompt_variant) cell is claimed via an atomic
reservation file, JSONL appends are guarded by ``fcntl.flock``, and stale
reservations from crashed workers are reaped on each pass.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path

from alienbench.config import Config, load_config
from alienbench.judges import make_judge
from alienbench.dimensions import DIMENSION_IDS, WARD_DIMENSIONS
from alienbench.paths import (
    extraction_reservations_dir,
    extractions_path,
    iter_generations,
    load_extracted_ids,
    model_dir_name,
)
from alienbench.reservation import (
    append_jsonl_locked,
    release_reservation,
    try_reserve,
)

logger = logging.getLogger(__name__)

MAX_PARSE_RETRIES = 3

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a precise scientific analyst evaluating descriptions of alien creatures. \
Your task is to extract structured information about 10 biological feature dimensions. \
Always output valid JSON and nothing else.\
"""

_DIMENSION_BLOCK = "\n".join(
    f'{i+1}. "{d["id"]}": {d["label"]}\n'
    f'   Earth default: {d["earth_default"]}\n'
    f'   Departure examples: {d["departure_examples"]}\n'
    f'   Boundary rule: {d["boundary_note"]}'
    for i, d in enumerate(WARD_DIMENSIONS)
)

_USER_TEMPLATE = """\
Analyse this alien creature description and extract the 10 Ward biological feature dimensions.

CREATURE DESCRIPTION:
{description}

DIMENSIONS TO EXTRACT:
{dimensions}

For each dimension output:
- "feature_described": what the creature actually has (brief, ≤15 words)
- "is_departure": true if it departs from the Earth default, false otherwise
- "reasoning": one sentence justifying your decision

Respond with ONLY this JSON structure (no markdown, no explanation):
{{
  "symmetry":         {{"feature_described": "...", "is_departure": true/false, "reasoning": "..."}},
  "sensory_organs":   {{"feature_described": "...", "is_departure": true/false, "reasoning": "..."}},
  "locomotion":       {{"feature_described": "...", "is_departure": true/false, "reasoning": "..."}},
  "body_plan":        {{"feature_described": "...", "is_departure": true/false, "reasoning": "..."}},
  "skin_covering":    {{"feature_described": "...", "is_departure": true/false, "reasoning": "..."}},
  "reproduction":     {{"feature_described": "...", "is_departure": true/false, "reasoning": "..."}},
  "metabolism":       {{"feature_described": "...", "is_departure": true/false, "reasoning": "..."}},
  "communication":    {{"feature_described": "...", "is_departure": true/false, "reasoning": "..."}},
  "habitat":          {{"feature_described": "...", "is_departure": true/false, "reasoning": "..."}},
  "cognition":        {{"feature_described": "...", "is_departure": true/false, "reasoning": "..."}}
}}
"""


def build_extraction_prompt(description: str) -> str:
    return _USER_TEMPLATE.format(description=description, dimensions=_DIMENSION_BLOCK)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_features(raw: str) -> dict | None:
    """Parse JSON from the judge response. Returns None on failure.

    A response is rejected when the inner structure does not match the schema
    expected by :func:`alienbench.dimensions.compute_ward_score`: the top
    level must be a dict with every Ward dimension as a key, each dimension
    must itself be a dict, and each must contain an ``is_departure`` field
    that is a bool or the integer 0/1. Strict validation here ensures that
    a malformed-but-syntactically-valid response cannot crash the score
    stage downstream — it triggers the parse-retry loop instead.
    """
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    # Some judges (notably Gemini) occasionally wrap the dict in a
    # single-element array. Unwrap so the per-dimension schema check
    # below still applies; the inner dict must still match the schema.
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]

    if not isinstance(data, dict):
        return None

    for dim_id in DIMENSION_IDS:
        if dim_id not in data:
            return None
        entry = data[dim_id]
        if not isinstance(entry, dict):
            return None
        if "is_departure" not in entry:
            return None
        val = entry["is_departure"]
        # Accept Python bool and the integers 0/1; reject everything else
        # (strings like "false", None, lists, etc.). Note: isinstance(True, int)
        # is True in Python, so the bool check must come first.
        if not isinstance(val, bool) and val not in (0, 1):
            return None
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _all_done(cells: list[tuple]) -> bool:
    for _judge, _model, _prompt_id, out_path, pending_by_id in cells:
        remaining = set(pending_by_id) - load_extracted_ids(out_path)
        if remaining:
            return False
    return True


def run(
    config_path: str = "config.yaml",
    human_models: list[str] | None = None,
) -> None:
    """Run feature extraction for all subject models and any pre-collected human models.

    ``human_models`` lists model keys whose generation records are provided
    externally (e.g. ``["human/prolific-baseline"]``). They are appended to
    ``cfg.models`` for extraction but are not generated by Stage 1. Pass
    ``None`` (default) to process only the models in ``config.yaml``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(config_path)
    data_dir = Path(cfg.data_dir)
    judge_clients = {alias: make_judge(alias, cfg) for alias in cfg.judge_models}
    rng = random.Random()
    # human_models are pre-collected external responses; they extend cfg.models
    # without being generated by Stage 1.
    models_to_process = list(cfg.models) + list(human_models or [])

    # Build per-cell work descriptors. ``pending_by_id`` snapshots the full
    # generation records keyed by id so the inner loop can reach the
    # description after reserving without re-iterating the JSONL.
    cells: list[tuple] = []
    for judge in cfg.judge_models:
        for model in models_to_process:
            for variant in cfg.prompt_variants:
                out_path = extractions_path(data_dir, judge, model, variant.id)
                # Clip to the canonical per-cell N (``samples_per_condition``,
                # see config.py). Generations with ``sample_index >=
                # samples_per_condition`` (e.g. from an earlier run at a
                # higher N) remain on disk and are silently skipped here.
                pending_by_id = {
                    gen["id"]: gen
                    for gen in iter_generations(
                        data_dir,
                        model,
                        variant.id,
                        cap=cfg.samples_per_condition,
                    )
                }
                if not pending_by_id:
                    continue
                cells.append((judge, model, variant.id, out_path, pending_by_id))

    if not cells:
        logger.info("No generations available for extraction.")
        return

    # Count remaining as |pending - done| per cell. A naive
    # n_existing - n_total can go negative when the on-disk JSONL
    # contains records for ids outside the current N (e.g.
    # ``samples_per_condition`` was lowered after extractions were
    # written), which inflates n_existing without contributing to
    # in-window progress.
    n_total = sum(len(p) for _, _, _, _, p in cells)
    n_remaining = sum(
        len(set(p) - load_extracted_ids(out)) for _, _, _, out, p in cells
    )
    n_done = n_total - n_remaining
    logger.info(
        "Extraction: %d/%d already complete; %d remaining across %d cells",
        n_done, n_total, n_remaining, len(cells),
    )

    n_success = 0
    n_parse_error = 0
    n_api_error = 0

    # Ids that failed in this run, keyed by (judge, model, prompt_id). They
    # are excluded from re-reservation within the current run so a worker
    # does not loop on the same id; the next ``extract`` invocation will
    # pick them up again because no record was written for them.
    failed_this_run: dict[tuple, set] = {}

    while True:
        progress = False
        rng.shuffle(cells)
        for judge, model, prompt_id, out_path, pending_by_id in cells:
            res_dir = extraction_reservations_dir(data_dir, judge, model, prompt_id)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cell_key = (judge, model, prompt_id)

            key = try_reserve(
                res_dir,
                candidates=list(pending_by_id),
                completed_fn=lambda p=out_path, k=cell_key: (
                    load_extracted_ids(p) | failed_this_run.get(k, set())
                ),
                rng=rng,
            )
            if key is None:
                continue
            progress = True
            gen = pending_by_id[key]

            try:
                prompt = build_extraction_prompt(gen["response"])
                total_prompt_tokens = 0
                total_completion_tokens = 0
                call_start = time.monotonic()
                response = None
                features = None
                attempt = 0
                for attempt in range(MAX_PARSE_RETRIES):
                    response = judge_clients[judge].complete(
                        prompt=prompt,
                        temperature=0.0,
                        # 8000 budget: thinking-mode judges (e.g. gemini-3.1-
                        # pro-preview) count internal reasoning tokens against
                        # max_output_tokens, so the previous 2500 ceiling
                        # could leave the JSON body truncated mid-string.
                        max_tokens=8000,
                        system=_SYSTEM,
                    )
                    total_prompt_tokens += response.prompt_tokens
                    total_completion_tokens += response.completion_tokens
                    features = parse_features(response.text)
                    if features is not None:
                        break
                    logger.warning(
                        "Parse attempt %d/%d failed for generation %s"
                        " with judge %s — raw: %.500s",
                        attempt + 1, MAX_PARSE_RETRIES,
                        gen["id"], judge, response.text,
                    )
                duration_seconds = time.monotonic() - call_start

                if features is None:
                    # All parse attempts failed. Do not write a record so
                    # the next ``extract`` run retries this generation;
                    # within this run, mark it failed so we do not loop.
                    logger.error(
                        "All %d parse attempts failed for generation %s"
                        " with judge %s. No record written; will retry on"
                        " next run. Last raw: %.500s",
                        MAX_PARSE_RETRIES, gen["id"], judge, response.text,
                    )
                    failed_this_run.setdefault(cell_key, set()).add(gen["id"])
                    n_parse_error += 1
                else:
                    record = {
                        "generation_id": gen["id"],
                        "judge_model": judge,
                        "judge_model_resolved": response.model,
                        "judge_call_id": response.generation_id,
                        "subject_model": gen["model"],
                        "prompt_variant": gen["prompt_variant"],
                        "timestamp": time.time(),
                        "features": features,
                        "parse_error": False,
                        "raw_response": None,
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "n_parse_attempts": attempt + 1,
                        "duration_seconds": duration_seconds,
                    }
                    append_jsonl_locked(out_path, record)
                    n_success += 1
                    logger.info(
                        "Wrote %s ← %s/%s gen_id=%s (%d this run)",
                        model_dir_name(judge), model_dir_name(model), prompt_id,
                        gen["id"], n_success + n_parse_error,
                    )
            except Exception as e:
                # API call exhausted retries (or another error). No record
                # is written; defer to the next run.
                logger.error(
                    "Error for generation %s / judge %s: %s. No record"
                    " written; will retry on next run.",
                    gen["id"], judge, e,
                )
                failed_this_run.setdefault(cell_key, set()).add(gen["id"])
                n_api_error += 1
            finally:
                release_reservation(res_dir, key)

        if not progress:
            break

    total_attempts = n_success + n_parse_error + n_api_error
    if total_attempts == 0:
        logger.info("All extractions already complete.")
    elif n_parse_error + n_api_error == 0:
        logger.info(
            "Extraction complete: %d/%d succeeded.", n_success, total_attempts,
        )
    else:
        failure_pct = 100 * (n_parse_error + n_api_error) / max(total_attempts, 1)
        log = logger.error if failure_pct >= 10 else logger.warning
        log(
            "Extraction complete: %d succeeded, %d parse errors, %d API errors"
            " (%.1f%% failure rate). Parse failures are included in the analysis summary.",
            n_success, n_parse_error, n_api_error, failure_pct,
        )

    if not _all_done(cells):
        logger.warning(
            "Worker exiting with cells still incomplete (%d new extractions written, "
            "%d parse errors, %d API errors). Re-run to retry.",
            n_success, n_parse_error, n_api_error,
        )

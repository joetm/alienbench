"""Stage 2: Extract Ward feature dimensions from creature descriptions."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from tqdm import tqdm

from alienbench.config import Config, load_config
from alienbench.judges import make_judge
from alienbench.dimensions import DIMENSION_IDS, WARD_DIMENSIONS
from alienbench.paths import (
    extractions_path,
    iter_generations,
    load_extracted_ids,
    model_dir_name,
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
    # human_models are pre-collected external responses; they extend cfg.models
    # without being generated by Stage 1.
    models_to_process = list(cfg.models) + list(human_models or [])

    # Count total work
    tasks = []
    for judge in cfg.judge_models:
        for model in models_to_process:
            for variant in cfg.prompt_variants:
                out_path = extractions_path(data_dir, judge, model, variant.id)
                done_ids = load_extracted_ids(out_path)
                pending = [
                    gen for gen in iter_generations(data_dir, model, variant.id)
                    if gen["id"] not in done_ids
                ]
                if pending:
                    tasks.append((judge, model, variant.id, out_path, pending))
                else:
                    logger.info("Skipping %s / %s / %s — already complete", judge, model, variant.id)

    if not tasks:
        logger.info("All extractions already complete.")
        return

    total = sum(len(p) for _, _, _, _, p in tasks)
    logger.info("Extracting features for %d generation/judge pairs", total)

    n_success = 0
    n_parse_error = 0
    n_api_error = 0

    with tqdm(total=total, unit="extraction") as bar:
        for judge, model, prompt_id, out_path, pending in tasks:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            bar.set_description(f"{model_dir_name(judge)} ← {model_dir_name(model)}/{prompt_id}")

            with open(out_path, "a") as f:
                for gen in pending:
                    try:
                        prompt = build_extraction_prompt(gen["response"])
                        total_prompt_tokens = 0
                        total_completion_tokens = 0
                        call_start = time.monotonic()
                        for attempt in range(MAX_PARSE_RETRIES):
                            response = judge_clients[judge].complete(
                                prompt=prompt,
                                temperature=0.0,
                                max_tokens=2500,
                                system=_SYSTEM,
                            )
                            total_prompt_tokens += response.prompt_tokens
                            total_completion_tokens += response.completion_tokens
                            features = parse_features(response.text)
                            if features is not None:
                                break
                            logger.warning(
                                "Parse attempt %d/%d failed for generation %s"
                                " with judge %s — raw: %.100s",
                                attempt + 1, MAX_PARSE_RETRIES,
                                gen["id"], judge, response.text,
                            )
                        else:
                            logger.error(
                                "All %d parse attempts failed for generation %s"
                                " with judge %s",
                                MAX_PARSE_RETRIES, gen["id"], judge,
                            )
                        duration_seconds = time.monotonic() - call_start

                        record = {
                            "generation_id": gen["id"],
                            "judge_model": judge,
                            "judge_model_resolved": response.model,
                            "judge_call_id": response.generation_id,
                            "subject_model": gen["model"],
                            "prompt_variant": gen["prompt_variant"],
                            "timestamp": time.time(),
                            "features": features,
                            "parse_error": features is None,
                            "raw_response": response.text if features is None else None,
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                            "n_parse_attempts": attempt + 1,
                            "duration_seconds": duration_seconds,
                        }
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                        if features is None:
                            n_parse_error += 1
                        else:
                            n_success += 1
                    except Exception as e:
                        logger.error("Error for generation %s / judge %s: %s", gen["id"], judge, e)
                        n_api_error += 1

                    bar.update(1)

    total_attempts = n_success + n_parse_error + n_api_error
    if n_parse_error + n_api_error == 0:
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

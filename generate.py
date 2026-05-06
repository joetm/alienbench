"""Stage 1: Generate creature descriptions from each model.

Multiple ``python -m alienbench generate`` processes may run concurrently
against the same ``data/`` directory. Duplication prevention is delegated
to :mod:`alienbench.reservation`: each ``sample_index`` is claimed via an
atomic reservation file, JSONL appends are guarded by ``fcntl.flock``,
and stale reservations from crashed workers are reaped on each pass.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
import uuid
from pathlib import Path

from alienbench.client import OpenRouterClient
from alienbench.config import Config, load_config
from alienbench.paths import (
    generations_path,
    load_completed_sample_indices,
    model_dir_name,
    reservations_dir,
)
from alienbench.reservation import (
    append_jsonl_locked,
    release_reservation,
    try_reserve,
)

logger = logging.getLogger(__name__)

# Range used for the deterministic per-sample seed. OpenAI's chat.completions
# API documents ``seed`` as a 64-bit integer; we restrict to 32 bits to stay
# within the range every upstream provider accepts.
_SEED_MOD = 2 ** 31


def _sample_seed(model: str, variant_id: str, sample_index: int) -> int:
    """Deterministic seed for the ``sample_index``-th call on (model, variant).

    Different samples receive different seeds so the temperature-1.0 sampling
    yields a distribution rather than a single repeated response, while a
    fixed (model, variant_id, sample_index) tuple always yields the same seed
    so a re-run reproduces the same generation order. The seed is derived
    from a stable cryptographic hash to avoid Python's per-process hash
    randomisation. Upstream providers honour the seed on a best-effort basis;
    see Limitations in the paper.
    """
    payload = f"{model}|{variant_id}|{sample_index}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") % _SEED_MOD


def _all_done(cfg: Config, data_dir: Path) -> bool:
    for model in cfg.models:
        for variant in cfg.prompt_variants:
            path = generations_path(data_dir, model, variant.id)
            if len(load_completed_sample_indices(path)) < cfg.samples_per_condition:
                return False
    return True


def run(config_path: str = "config.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(config_path)
    client = OpenRouterClient(cfg)
    data_dir = Path(cfg.data_dir)
    rng = random.Random()

    cells = [(model, variant) for model in cfg.models for variant in cfg.prompt_variants]

    # Count remaining as |target_indices - completed_indices| per cell so
    # that on-disk records with sample_index >= samples_per_condition
    # (left over from a prior, larger run) cannot push the displayed
    # remaining count negative.
    target = set(range(cfg.samples_per_condition))
    n_total = cfg.samples_per_condition * len(cells)
    n_remaining = sum(
        len(target - load_completed_sample_indices(generations_path(data_dir, m, v.id)))
        for m, v in cells
    )
    n_done = n_total - n_remaining
    logger.info(
        "Generation: %d/%d samples already complete; %d remaining across %d cells",
        n_done, n_total, n_remaining, len(cells),
    )

    n_done_this_run = 0
    n_failed_this_run = 0
    # A sample_index that errors this run is excluded from re-reservation
    # so a persistent API error does not loop forever. Nothing is written
    # to the JSONL on failure, so the next ``generate`` run retries it.
    # Mirrors the same mechanism in ``extract.run``.
    failed_this_run: dict[tuple[str, str], set[str]] = {}

    while True:
        progress = False
        rng.shuffle(cells)
        for model, variant in cells:
            res_dir = reservations_dir(data_dir, model, variant.id)
            responses_path = generations_path(data_dir, model, variant.id)
            responses_path.parent.mkdir(parents=True, exist_ok=True)
            cell_key = (model, variant.id)

            candidates = [str(i) for i in range(cfg.samples_per_condition)]
            key = try_reserve(
                res_dir,
                candidates=candidates,
                completed_fn=lambda p=responses_path, k=cell_key: (
                    {str(i) for i in load_completed_sample_indices(p)}
                    | failed_this_run.get(k, set())
                ),
                rng=rng,
            )
            if key is None:
                continue
            i = int(key)
            progress = True

            seed = _sample_seed(model, variant.id, i)
            try:
                call_start = time.monotonic()
                response = client.complete(
                    model=model,
                    prompt=variant.text,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    seed=seed,
                )
                duration_seconds = time.monotonic() - call_start
                record = {
                    "id": str(uuid.uuid4()),
                    "model": model,
                    "model_resolved": response.model,
                    "call_id": response.generation_id,
                    "prompt_variant": variant.id,
                    "prompt_text": variant.text,
                    "response": response.text,
                    "temperature": cfg.temperature,
                    "seed": seed,
                    "sample_index": i,
                    "timestamp": time.time(),
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "duration_seconds": duration_seconds,
                }
                append_jsonl_locked(responses_path, record)
                n_done_this_run += 1
                logger.info(
                    "Wrote %s / %s sample_index=%d (%d this run)",
                    model_dir_name(model), variant.id, i, n_done_this_run,
                )
            except Exception as e:
                n_failed_this_run += 1
                failed_this_run.setdefault(cell_key, set()).add(key)
                logger.error(
                    "Error for %s / %s sample_index=%d: %s. No record"
                    " written; will retry on next run.",
                    model, variant.id, i, e,
                )
            finally:
                release_reservation(res_dir, key)

        if not progress:
            break

    if _all_done(cfg, data_dir):
        logger.info(
            "Generation complete: %d new samples written this run (%d API failures).",
            n_done_this_run, n_failed_this_run,
        )
    else:
        logger.warning(
            "Worker exiting with cells still incomplete (%d new samples written, "
            "%d API failures). Re-run to retry.",
            n_done_this_run, n_failed_this_run,
        )

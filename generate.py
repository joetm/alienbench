"""Stage 1: Generate creature descriptions from each model."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path

from tqdm import tqdm

from alienbench.client import OpenRouterClient
from alienbench.config import Config, load_config
from alienbench.paths import count_existing, generations_path, model_dir_name

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


def run(config_path: str = "config.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(config_path)
    client = OpenRouterClient(cfg)
    data_dir = Path(cfg.data_dir)

    # Build work list, skipping already-completed runs
    tasks = []
    for model in cfg.models:
        for variant in cfg.prompt_variants:
            path = generations_path(data_dir, model, variant.id)
            existing = count_existing(path)
            remaining = cfg.samples_per_condition - existing
            if remaining > 0:
                tasks.append((model, variant, path, existing, remaining))
            else:
                logger.info("Skipping %s / %s — already complete (%d samples)", model, variant.id, existing)

    if not tasks:
        logger.info("All generations already complete.")
        return

    total = sum(r for _, _, _, _, r in tasks)
    logger.info("Generating %d samples across %d model/prompt combinations", total, len(tasks))

    with tqdm(total=total, unit="sample") as bar:
        for model, variant, path, existing, remaining in tasks:
            path.parent.mkdir(parents=True, exist_ok=True)
            bar.set_description(f"{model_dir_name(model)} / {variant.id}")

            with open(path, "a") as f:
                for i in range(remaining):
                    sample_index = existing + i
                    seed = _sample_seed(model, variant.id, sample_index)
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
                            "timestamp": time.time(),
                            "prompt_tokens": response.prompt_tokens,
                            "completion_tokens": response.completion_tokens,
                            "duration_seconds": duration_seconds,
                        }
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                    except Exception as e:
                        logger.error(
                            "Error for %s / %s sample %d: %s",
                            model, variant.id, existing + i + 1, e,
                        )

                    bar.update(1)

    n_expected = 0
    n_shortfall = 0
    for model, variant, path, existing, remaining in tasks:
        expected = existing + remaining
        actual = count_existing(path)
        shortfall = expected - actual
        n_expected += expected
        if shortfall > 0:
            n_shortfall += shortfall
            logger.error(
                "Generation shortfall for %s / %s: expected %d, got %d (%d failed).",
                model, variant.id, expected, actual, shortfall,
            )
    if n_shortfall > 0:
        logger.error(
            "Generation complete with %d/%d failures. Re-run to retry failed samples.",
            n_shortfall, n_expected,
        )
    else:
        logger.info("Generation complete: %d/%d samples succeeded.", n_expected, n_expected)

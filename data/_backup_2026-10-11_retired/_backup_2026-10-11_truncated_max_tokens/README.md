# Generations cut off at max_tokens (moved 2026-10-11)

Every subject-model generation whose finish_reason was `length` (cut off at
`max_tokens`, 800 at the time, 4000 for Gemini main conditions) was removed
from `data/generations`, together with its extraction and Ward-score records
from all judges. The next `python -m alienbench run` regenerates these
samples (same model, prompt, and sample_index, hence the same seed) at the
new cap.

The finish reason came from the generation record (`finish_reason`) where it
was recorded, and otherwise from OpenRouter's generation-metadata endpoint
(`GET /api/v1/generation?id=<call_id>`). Kept records without a recorded
finish reason were backfilled from the same lookup
(`finish_reason_source: openrouter_generation_lookup`).

- `manifest.csv`: one row per removed generation.
- `removed/`: the removed records, in the original directory layout.
- `original_files/`: full copies of every file before modification. To
  restore, copy `original_files/*` back over `data/`.

# Generations that ended without a finish reason (moved 2026-10-11)

Four subject-model generations ended without a finish reason. OpenRouter's
generation metadata (`GET /api/v1/generation?id=<call_id>`) shows HTTP 200,
not cancelled, a streamed upstream response, and finish_reason and
native_finish_reason both null. The upstream stream ended before its final
chunk, so each text stops mid-sentence. They were removed from
`data/generations`, together with their extraction and Ward-score records
from all judges. The next `python -m alienbench run` regenerates them.

- `manifest.csv`: one row per removed generation.
- `removed/`: the removed records, in the original directory layout.
- `original_files/`: full copies of every file before modification. To
  restore, copy `original_files/*` back over `data/`.

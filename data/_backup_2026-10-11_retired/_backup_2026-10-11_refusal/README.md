# Generations interrupted by a provider refusal (moved 2026-10-11)

Every subject-model generation whose `native_finish_reason` was `refusal`
was removed from `data/generations`, together with its extraction and
Ward-score records from all judges. OpenRouter reports these calls as
finish_reason `stop`, but the provider's safety system stopped the text
mid-description. The next `python -m alienbench run` regenerates them.

- `manifest.csv`: one row per removed generation.
- `removed/`: the removed records, in the original directory layout.
- `original_files/`: full copies of every file before modification. To
  restore, copy `original_files/*` back over `data/`.

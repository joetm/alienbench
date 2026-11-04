# All collected data retired (moved 2026-10-13)

The study restarts from scratch with `max_tokens: 16000`. Regenerating only
the samples cut off at the old cap (800 tokens, 4000 for Gemini main
conditions) would have kept samples that ended under the cap by chance next
to unconditional new draws. That mix favors short responses. All model data
collected so far was therefore retired.

Contents, in the original `data/` layout:

- `generations/`: every subject-model generation.
- `extractions/`, `scores/`: every judge extraction and Ward score, including
  the judge results on the human Prolific baseline.
- `_backup_2026-10-11_truncated_max_tokens/`, `_backup_2026-10-11_refusal/`,
  `_backup_2026-10-11_no_finish_reason/`: the earlier cleanups, each with its
  own README and manifest.
- `_quarantine/`: the earlier quarantined Gemini run.

Not retired: `data/generations/human__prolific-baseline/` (100 human
responses) stays in place. The judges score it again in the fresh run.

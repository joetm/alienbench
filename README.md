# AlienBench

A creativity benchmark for LLMs based on Ward's (1994) Structured Imagination task. Models are prompted to describe creatures on an alien planet; their outputs are scored on one dimension:

- **Ward Departure Score (0–10)**: how many of 10 Earth-typical biological features the creature departs from (symmetry, locomotion, metabolism, etc.)

Multiple judge models extract features independently, enabling inter-rater reliability analysis.

---

## About this folder

**This `alienbench/` folder is the self-contained benchmark.** Everything required to replicate the benchmark lives inside it: source code, configuration (`config.yaml`, `test_config.yaml`), Python dependencies (`requirements.txt`), tests (`tests/`), the API-key template (`.env.sample`), and the runtime output directories (`data/`, `results/`). The folder is what gets released as the anonymized code artifact for camera-ready submission, and it is the unit a reviewer downloads to reproduce the benchmark from a clean checkout.

Nothing the benchmark needs lives outside this folder. The canonical invocation runs from the repository root that contains `alienbench/`:

```bash
python3 -m alienbench run
```

`config.yaml` is found inside the package by default, and `data_dir` / `results_dir` resolve relative to the config file's directory, so outputs always land next to the config that produced them — regardless of the working directory.

---

## Setup

**Requirements:** Python 3.10+

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r alienbench/requirements.txt
```

### API keys

Subject models are called through [OpenRouter](https://openrouter.ai). Judge models are called directly through their native provider SDKs with pinned, dated model IDs, so the judge panel is reproducible at submission time (see `judge_overrides` in `config.yaml`).

You therefore need up to four API keys:

| Variable | Used for |
|---|---|
| `OPENROUTER_API_KEY` | Subject models (Llama, Qwen, Mistral, etc.) and any judge without a native override |
| `ANTHROPIC_API_KEY` | Native judge calls to Anthropic models |
| `OPENAI_API_KEY` | Native judge calls to OpenAI models |
| `GOOGLE_API_KEY` | Native judge calls to Google models |

```bash
cp alienbench/.env.sample alienbench/.env
```

Open `alienbench/.env` and paste your keys. The file is gitignored and never committed.

---

## Configuration

Everything is controlled by `config.yaml`. The key sections:

### Subject models

```yaml
models:
  # Tier A: Frontier flagships (full models; no mini / lite tiers)
  - anthropic/claude-opus-4.7
  - openai/gpt-5.4
  - google/gemini-3.1-pro-preview
  - deepseek/deepseek-v3
  # Tier B: Other frontier open-weight
  - meta-llama/llama-4-maverick
  - qwen/qwen-3-72b-instruct
  - mistralai/mistral-large-2411
  # Tier C: Smaller open-weight (~8B)
  - meta-llama/llama-3.1-8b-instruct
  - qwen/qwen-3-8b-instruct
  - mistralai/ministral-8b-instruct
```

Use any [OpenRouter model ID](https://openrouter.ai/models). These are the subject models whose structural novelty is being measured. The canonical baseline is tiered by frontier capability rather than by openness: Tier A collects the four frontier flagships from each top-tier provider (Anthropic, OpenAI, Google, DeepSeek), Tier B adds the remaining major frontier open-weight providers (Meta, Alibaba, Mistral), and Tier C gives a ~8B small-model baseline from the three Tier-B providers so the benchmark exposes a within-provider scaling axis.

### Judge panel

```yaml
judge_models:
  - anthropic/claude-opus-4.6
  - google/gemini-2.5-pro
  - openai/gpt-5.4-2026-03-05

judge_overrides:
  anthropic/claude-opus-4.6:
    provider: anthropic
    model_id: claude-opus-4-6
  google/gemini-2.5-pro:
    provider: google
    model_id: gemini-2.5-pro-preview-03-25
  openai/gpt-5.4-2026-03-05:
    provider: openai
    model_id: gpt-5.4-2026-03-05
```

These models evaluate the generated descriptions. All judges score every subject model, including models from the same provider family as one judge. Self-preference bias on same-family pairs is diluted across the two cross-family judges via the panel-of-judges design (Verga et al., 2024); residual bias is documented in the paper's Limitations section. Using 2–3 judges enables inter-rater reliability analysis (Krippendorff's α).

The `judge_models` aliases serve as panel keys (used in path names and dataframes). The `judge_overrides` map routes each alias to a native provider SDK with a pinned dated `model_id`, so the upstream model is fixed at submission time rather than floating with OpenRouter's alias routing. Aliases without an override fall back to OpenRouter using the alias as the request slug. The resolved upstream model is recorded per-call as `judge_model_resolved` on every extraction row, alongside the per-call `judge_call_id`, providing a per-row provenance audit trail.

### Sample count

```yaml
samples_per_condition: 50
```

Number of creature descriptions generated per model per prompt variant. 50 is recommended for distributional analysis; use 3–5 for a quick smoke test.

### Prompt variants

```yaml
prompt_variants:
  - id: baseline
    label: "Baseline"
    text: "Imagine a creature that lives on an alien planet. Describe it in detail."

  - id: departure_primed
    label: "Departure-primed"
    text: "Imagine a creature on an alien planet that is as different from any Earth creature as possible. Describe it in detail."
  # ... (see config.yaml for all 6 variants)
```

Each variant gets its own condition. Add, remove, or modify variants freely — each needs a unique `id` (used as directory name), a `label` (for figures), and a `text` (the actual prompt).

The six default variants are:

| ID | Description |
|---|---|
| `baseline` | Open-ended, minimal instruction |
| `departure_primed` | Explicitly instructed to differ from Earth |
| `constrained_no_light` | Planet has no light |
| `constrained_high_gravity` | Planet has 10× Earth gravity |
| `constrained_ammonia` | Planet has ammonia oceans |
| `detailed_description` | Longer, more complete description without naming biological aspects |

### Prompt paraphrases

```yaml
prompt_paraphrases:
  - id: baseline
    label: "Baseline"
    text: "Imagine a creature that lives on an alien planet. Describe it in detail."
  - id: baseline_para_a
    label: "Paraphrase A"
    text: "Picture an organism native to a planet other than Earth. Write a detailed description of it."
  - id: baseline_para_b
    label: "Paraphrase B"
    text: "Describe, in detail, a lifeform that inhabits a world beyond our solar system."
```

The `prompt_paraphrases:` block is read only by the prompt paraphrase sensitivity ablation (see below). Each entry must share intent with the baseline: imagine an alien creature and describe it in detail, without priming departure or imposing constraints. At least two entries are required. Paraphrase `id`s must be disjoint from `prompt_variants` ids, except that `baseline` is allowed in both pools and shares its on-disk generations, extractions, and scores so the ablation's correlation matrix anchors against the wording used in the primary analysis.

### Provider routing

Subject models are addressed by their OpenRouter slug (`meta-llama/llama-3.1-70b-instruct`, etc.) and OpenRouter selects an upstream provider. The canonical config does not restrict routing: the model IDs are pinned, and the per-call `model_resolved` / `call_id` fields on each generation row record which upstream actually served the request, giving a per-row audit trail. If a deployment needs hard-locked routing, add an `allowed_providers:` list (and optionally `allow_provider_fallbacks: false`) to the config.

### Generation parameters

```yaml
temperature: 1.0   # Controls randomness; 1.0 recommended for creativity tasks
max_tokens: 800    # Max length of each generated description
```

---

## Running the benchmark

The full pipeline is a single command:

```bash
python3 -m alienbench run
```

This chains the five stages (generate → extract → score → analyze → latex). All stages are **checkpointed**: re-running picks up where an interrupted run left off, so repeating `run` after a network failure is safe.

To start from or stop at a specific stage:

```bash
python3 -m alienbench run --from extract            # skip generation
python3 -m alienbench run --to score                # stop before analysis
```

Stages can also be run individually:

```bash
python3 -m alienbench generate
python3 -m alienbench extract
python3 -m alienbench score
python3 -m alienbench analyze
python3 -m alienbench latex
```

To use a non-default config file, pass `--config` either before or after the subcommand:

```bash
python3 -m alienbench run --config my_config.yaml
python3 -m alienbench generate --config my_config.yaml
```

After `analyze` or `latex` runs, a `run_manifest.json` is written into `results_dir`. It records the resolved config, a SHA-256 config hash, package versions, and timestamps for each stage run, so any figure or table can be traced back to the configuration that produced it.

### Quick test (no API key cost)

```bash
pip install -r alienbench/requirements-dev.txt
python3 -m pytest alienbench/tests
```

Runs the full pipeline end-to-end with mocked API responses, plus unit tests
for the LaTeX-table generator, the provenance manifest writer, and the
parse-failure analysis. No API key needed.

### Smoke test with real API (minimal cost)

```bash
python3 -m alienbench run --config test_config.yaml
```

`test_config.yaml` uses one cheap model as both subject and judge to minimise cost, with 1 prompt and 3 samples — enough to verify the full pipeline against the real API at near-zero cost.

---

## Ablation studies

Three ablations accompany the primary analysis and are documented in §5 of the paper. They are supplementary to the main pipeline and run through a dedicated `ablation` subcommand.

```bash
python3 -m alienbench ablation prompt
python3 -m alienbench ablation dimensions
python3 -m alienbench ablation judges
```

The `prompt` ablation reruns `generate → extract → score` on a paraphrase set before aggregating. The `dimensions` and `judges` ablations are pure re-aggregations of the Ward score records already written by the main pipeline, so they do not call any API.

| Ablation | Prerequisite | Reads |
|---|---|---|
| `prompt` | OpenRouter + judge API keys available | `prompt_paraphrases:` in the config |
| `dimensions` | `score` has completed | `data/scores/**/ward_scores.jsonl` |
| `judges` | `score` has completed; ≥2 judges in the scored data | `data/scores/**/ward_scores.jsonl` |

`--config` behaves the same as for other subcommands.

### Prompt paraphrase sensitivity

Reports pairwise Spearman rank correlations of per-model mean Ward scores across surface paraphrases of the baseline prompt with identical intent. Paraphrases come from the `prompt_paraphrases:` config block (see above). The ablation writes a shadow config whose `prompt_variants` equals the paraphrase list, then invokes the main pipeline stages. New artefacts land under `data/generations/<model>/<paraphrase_id>/`, `data/extractions/.../<paraphrase_id>/`, and `data/scores/.../<paraphrase_id>/`, which do not collide with the main pipeline because paths are keyed by prompt id.

Outputs in `results/`:

| File | Contents |
|---|---|
| `table_ablation_prompt.csv` | Per-model × paraphrase mean Ward scores (wide) |
| `table_ablation_prompt_corr.csv` | Pairwise Spearman ρ across paraphrases |
| `table_ablation_prompt_corr_n.csv` | Pair counts for each ρ cell |
| `tab_ablation_prompt.tex` | LaTeX table for `\input{}` into the paper |
| `summary_ablation_prompt.txt` | Text summary: ρ and MAD ranges, permutation null, and preregistered stability-bar pass counts ($\rho \geq 0.9$, MAD $\leq 1.0$) |

### Dimension sensitivity

Reports leave-one-out Spearman ρ between each 9-of-10 reduced per-model mean and the full 10-dimension mean, and the minimum dimension subset that preserves the full-score model ranking under an exhaustive search over all $2^{10}-1$ subsets. The ablation only reads existing Ward score records and never touches an API.

Outputs in `results/`:

| File | Contents |
|---|---|
| `table_ablation_dimensions_loo.csv` | One row per dropped dimension (departure rate, ρ vs full, rank changes) |
| `table_ablation_dimensions_minsubset.csv` | One row per subset size `k` with counts and best ρ |
| `table_ablation_dimensions_rank_preserving.csv` | All rank-preserving subsets at the smallest `k` (written when such a `k` exists) |
| `tab_ablation_dimensions.tex` | LaTeX table for `\input{}` into the paper |
| `summary_ablation_dimensions.txt` | Text summary |

### Judge panel composition

Compares single-judge, leave-one-judge-out, and size-`k` subset panels against the full-panel aggregate. Reports Spearman ρ of per-model means and stratified cross-model Krippendorff α on each reduced panel. Requires the scored data to contain at least two judges.

Outputs in `results/`:

| File | Contents |
|---|---|
| `table_ablation_judges_single.csv` | One row per single judge (ρ vs panel, rank changes) |
| `table_ablation_judges_loo.csv` | One row per excluded judge (ρ vs full, α on reduced panel) |
| `table_ablation_judges_sizesweep.csv` | One row per panel size `k` with aggregate ρ and α |
| `tab_ablation_judges.tex` | LaTeX table for `\input{}` into the paper |
| `summary_ablation_judges.txt` | Text summary: held-out ρ ranges, α ranges, and preregistered-threshold pass counts ($\rho_{\text{heldout}} \geq 0.9$, $\alpha_{\text{full}} - \alpha_{\text{reduced}} \leq 0.1$, $\alpha \geq 0.667$) |

---

## Human validation

The paper anchors the LLM-judge measurements with a human validation study
(§3.5). The harness is exposed via the `human` subcommand:

```bash
python3 -m alienbench human sample [--samples-per-stratum N] [--seed S] [--out PATH]
python3 -m alienbench human ingest --annotator <id> path/to/filled.csv
python3 -m alienbench human analyze [--annotator <id> ...]
```

The workflow is:

1. **Sample.** Draws a stratified sample across every (subject_model,
   prompt_variant) cell from the existing `data/generations/` and writes a
   CSV template to `results/human_sample.csv` (or `--out PATH`). One row per
   (generation, dimension) pair, so each row carries the rubric for the
   dimension it applies to.

   ```bash
   python3 -m alienbench human sample --samples-per-stratum 5 --seed 42
   ```

   `--seed` is reproducible: identical configs and seeds draw the same
   sample. Default seed is 42; use distinct seeds to draw disjoint samples.

2. **Annotate.** Each annotator fills the CSV in place. The schema is:

   | Column | Filled by | Notes |
   |---|---|---|
   | `generation_id`, `subject_model`, `prompt_variant`, `dimension`, `earth_default`, `departure_examples`, `boundary_note`, `description` | sampler | rubric context |
   | `is_departure` | annotator | `0`/`1` (or `true`/`false`); required |
   | `reasoning` | annotator | optional one-line justification |

3. **Ingest.** Validates the filled CSV and writes it to disk in the same
   format the LLM-judge pipeline uses, under a `human/<annotator_id>` alias
   so the analysis utilities treat humans as just another rater. Annotator
   ids are sanitized (lowercased, spaces become `_`, `/` is rejected).

   ```bash
   python3 -m alienbench human ingest --annotator alice filled_alice.csv
   python3 -m alienbench human ingest --annotator bob   filled_bob.csv
   ```

4. **Analyze.** Computes Krippendorff's α between every annotator and the
   LLM-judge panel at the Ward total (interval level) and per-dimension
   (nominal level), writes `results/table_human_validation.csv`, and appends
   a `## Human Validation` block to `results/summary.txt`. By default
   discovers every `human/*` annotator on disk; restrict the comparison with
   `--annotator <id>` (repeatable).

   ```bash
   python3 -m alienbench human analyze
   ```

The `analyze` step requires that the LLM-judge pipeline has already run
through Stage 3, since it compares humans against the existing
`data/scores/<judge>/` records.

---

## Human baseline

The `--human` flag processes pre-collected human responses through Stages 2–3
(feature extraction and Ward scoring). Human generation records are provided
externally — the pipeline does not call any API to produce them. The generate
stage is unaffected by `--human`.

### Place the generation records on disk

Write one JSONL record per participant to:

```
alienbench/data/generations/human__prolific-baseline/baseline/responses.jsonl
```

Each line is one JSON object. The schema must match the format written by
`generate.py`:

```json
{
  "id":                "<uuid>",
  "model":             "human/prolific-baseline",
  "model_resolved":    "human/prolific-baseline",
  "call_id":           "<uuid>",
  "prompt_variant":    "baseline",
  "prompt_text":       "Imagine a creature that lives on an alien planet. Describe it in detail.",
  "response":          "<participant text>",
  "temperature":       0.0,
  "seed":              0,
  "timestamp":         1234567890.0,
  "prompt_tokens":     0,
  "completion_tokens": 150,
  "duration_seconds":  0.0
}
```

Set `completion_tokens` to the word count of the participant's response. The
pipeline uses this field to compute the length-adjusted Ward metric
(`ward_per_100`), so 0 would produce an undefined rate. A word-count proxy is
the appropriate estimate for human text.

### Run feature extraction and scoring

```bash
python3 -m alienbench run --human --from extract --to score
```

Or stages individually:

```bash
python3 -m alienbench extract --human
python3 -m alienbench score   --human
```

The `--human` flag appends `human/prolific-baseline` to the list of models
processed by the extraction and scoring stages. It does not affect the generate
stage, since human generations are provided externally.

### Outputs

```
alienbench/data/extractions/<judge_model>/human__prolific-baseline/baseline/features.jsonl
alienbench/data/scores/<judge_model>/human__prolific-baseline/baseline/ward_scores.jsonl
```

The `human__prolific-baseline` directory name distinguishes human outputs from
LLM outputs at a glance. The records carry the same schema as LLM extraction
and scoring records, so the existing analysis utilities (`analyze`, `latex`)
can consume them if `human/prolific-baseline` is added to `models` in
`config.yaml` for a combined analysis run.

---

## Output structure

All outputs live inside `alienbench/` so the folder remains self-contained.

```
alienbench/data/
  generations/
    {model}/
      {prompt_variant}/
        responses.jsonl          # Raw generated descriptions
                                 # incl. model_resolved + call_id (provenance)

  extractions/
    {judge_model}/
      {model}/
        {prompt_variant}/
          features.jsonl         # Ward features extracted per description
                                 # incl. judge_model_resolved + judge_call_id

  scores/
    {judge_model}/
      {model}/
        {prompt_variant}/
          ward_scores.jsonl      # Ward departure scores (0–10) + per-dimension breakdown
                                 # judge_model_resolved propagated from extraction

alienbench/results/
  fig1_ward_heatmap.pdf                    # Departure rates per model × dimension
  fig2_violin_scores.pdf                   # Ward score distributions by model × prompt variant
  fig3_reliability_table.pdf               # Inter-rater reliability (Krippendorff's α)
  fig4_reliability_table.pdf               # Per-dimension reliability breakdown
  table_reliability.csv                    # Reliability table as CSV
  table_departure_freq.csv                 # Departure frequencies per dimension
  table_posthoc.csv                        # Post-hoc pairwise comparisons
  table_token_covariate.csv                # Token-length covariate analysis
  table_token_covariate_per_condition.csv  # Token covariate broken out by condition
  table_extraction_status.csv              # Per (judge × model × prompt) extraction success/parse-error/api-error counts
  table_human_validation.csv               # Per-measure α between humans and the LLM-judge panel (after `human analyze`)
  summary.txt                              # Means, 95% CIs, Kruskal-Wallis p-values, extraction reliability
  findings.md                              # Key findings in human-readable form
  tab_reliability.tex                      # LaTeX table: inter-rater reliability
  tab_ward_dimensions.tex                  # LaTeX table: Ward feature departure rates
  tab_ward_scores.tex                      # LaTeX table: full scores
  run_manifest.json                        # Provenance: config, config hash, package versions, timestamps

  # Ablations (written only when the corresponding ablation has been run)
  table_ablation_prompt.csv                # Prompt ablation: per-model × paraphrase means
  table_ablation_prompt_corr.csv           # Prompt ablation: pairwise Spearman ρ
  table_ablation_prompt_corr_n.csv         # Prompt ablation: pair counts
  tab_ablation_prompt.tex                  # Prompt ablation: LaTeX table
  summary_ablation_prompt.txt              # Prompt ablation: text summary
  table_ablation_dimensions_loo.csv        # Dimension ablation: leave-one-out
  table_ablation_dimensions_minsubset.csv  # Dimension ablation: size-k search summary
  table_ablation_dimensions_rank_preserving.csv  # Dimension ablation: rank-preserving subsets at min k
  tab_ablation_dimensions.tex              # Dimension ablation: LaTeX table
  summary_ablation_dimensions.txt          # Dimension ablation: text summary
  table_ablation_judges_single.csv         # Judge ablation: per-judge vs panel
  table_ablation_judges_loo.csv            # Judge ablation: leave-one-judge-out
  table_ablation_judges_sizesweep.csv      # Judge ablation: size-k sweep
  tab_ablation_judges.tex                  # Judge ablation: LaTeX table
  summary_ablation_judges.txt              # Judge ablation: text summary
```

Model IDs with `/` are stored as `__` in directory names (e.g., `openai/gpt-4o` → `openai__gpt-4o`).

---

## Ward feature dimensions

The 10 dimensions scored for Earth-typical defaults:

| # | Dimension | Earth default |
|---|---|---|
| 1 | Body symmetry | Bilateral |
| 2 | Sensory organs | Eyes/ears/nose on a head |
| 3 | Locomotion | Legs |
| 4 | Body plan | Head + torso + limbs |
| 5 | Skin/covering | Skin, fur, scales, or feathers |
| 6 | Reproduction | Sexual, binary sexes |
| 7 | Metabolism | Eats organic matter, breathes O₂ |
| 8 | Communication | Sound or visual signals |
| 9 | Habitat | Land surface or water |
| 10 | Cognition | Centralised brain/nervous system |

A dimension scores 1 (departure) only for a fundamentally different mechanism, not a variation on the Earth default (e.g., 6 legs = still uses legs = score 0).

---

## Scoring notes

- **Ward scoring** is programmatic: Stage 2 extracts each dimension into JSON; Stage 3 reads the `is_departure` field and sums.
- Scores are checkpointed and computed per judge model — disagreements between judges are quantified in the reliability analysis.
- Records where the judge returns unparseable output are flagged with `parse_error: true` and stored with the raw response for inspection.

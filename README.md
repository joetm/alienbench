# AlienBench

A creativity benchmark for LLMs based on Ward's (1994) Structured Imagination task. Models are prompted to describe creatures on an alien planet; their outputs are scored on one dimension:

- **Ward Departure Score (0–10)**: how many of 10 Earth-typical biological features the creature departs from (symmetry, locomotion, metabolism, etc.)

Three judge models extract the features.

---

## About this folder

**This `alienbench/` folder is the self-contained benchmark.** Everything required to replicate the benchmark lives inside it: source code, configuration (`config.yaml`, `test_config.yaml`), Python dependencies (`requirements.txt`), tests (`tests/`), the API-key template (`.env.sample`), and the runtime output directories (`data/`, `results/`). The folder is what gets released as the anonymized code artifact for camera-ready submission, and it is the unit a reviewer downloads to reproduce the benchmark from a clean checkout.

Nothing the benchmark needs lives outside this folder. The canonical invocation runs from the repository root that contains `alienbench/`:

```bash
python3 -m alienbench run
```

The script `alienbench/run.py` is an equivalent entry point for callers who prefer a script to a module invocation, and it runs from any working directory (see [Running the benchmark](#running-the-benchmark)).

`config.yaml` is found inside the package by default, and `data_dir` / `results_dir` resolve relative to the config file's directory, so outputs always land next to the config that produced them, regardless of the working directory.

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
  - id: anthropic/claude-opus-4.7         # OpenRouter id: the key for paths and dataframes
    name: "Claude Opus 4.7"               # the name tables, figures, and the paper print
    key: ClaudeOpus                       # letters only: stem of the paper's \ab... macros
    resolved_version: anthropic/claude-4.7-opus-20260416   # dated snapshot, for the record
  # ... one entry per subject model (Tier A frontier flagships, Tier B other
  # frontier open-weight, Tier C smaller open-weight)
```

A bare id (`- openai/gpt-4o`) is still accepted. The display name then falls back to the id's last path segment. `config.yaml` is the single source of model information: the paper's model names, the models appendix table, and the per-model macros are all generated from these fields.

Use any [OpenRouter model ID](https://openrouter.ai/models). These are the subject models whose structural novelty is being measured. The canonical baseline is tiered by frontier capability rather than by openness: Tier A collects the four frontier flagships from each top-tier provider (Anthropic, OpenAI, Google, DeepSeek), Tier B adds the remaining major frontier open-weight providers (Meta, Alibaba, Mistral), and Tier C gives a ~8B small-model baseline from the three Tier-B providers so the benchmark exposes a within-provider scaling axis.

### Judge panel

```yaml
judge_models:
  - id: anthropic/claude-opus-4.6
    name: "Claude Opus 4.6"
    key: ClaudeOpus
    date: "2026-02-04"                    # release date of the pinned snapshot, as the paper prints it
  - id: google/gemini-3.1-pro-preview
    name: "Gemini 3.1 Pro"
    key: Gemini
    date: "2025-12-17"
  - id: openai/gpt-5.4
    name: "GPT-5.4"
    key: GPT
    date: "2026-03-05"

judge_overrides:
  anthropic/claude-opus-4.6:
    provider: anthropic
    model_id: claude-opus-4-6
  google/gemini-3.1-pro-preview:
    provider: google
    model_id: gemini-3.1-pro-preview
  openai/gpt-5.4:
    provider: openai
    model_id: gpt-5.4
```

These models evaluate the generated descriptions. All judges score every subject model, including models from the same provider family as one judge. Self-preference bias on same-family pairs is diluted across the two cross-family judges via the panel-of-judges design (Verga et al., 2024); residual bias is documented in the paper's Limitations section. Using 2–3 judges enables inter-rater reliability analysis (Krippendorff's α).

The `judge_models` aliases serve as panel keys (used in path names and dataframes). The `judge_overrides` map routes each alias to a native provider SDK with a pinned dated `model_id`, so the upstream model is fixed at submission time rather than floating with OpenRouter's alias routing. Aliases without an override fall back to OpenRouter using the alias as the request slug. The resolved upstream model is recorded per-call as `judge_model_resolved` on every extraction row, alongside the per-call `judge_call_id`, providing a per-row provenance audit trail.

### Sample count

```yaml
samples_per_condition: 30
```

Canonical number of creature descriptions per (subject model, prompt variant) cell. Every pipeline stage (generate, extract, score, analyze, human) clips to the first `samples_per_condition` records, ordered ascending by `sample_index`. Use 3–5 for a quick smoke test.

If a previous run wrote records at a higher N (e.g. 50), those records remain on disk but are silently skipped by downstream stages once `samples_per_condition` is lowered. The parse-failure denominator in the analyze stage matches the current N, so failure rates are not misreported. Canonical implementation reference: `Config.samples_per_condition` in `alienbench/config.py`.

The Prompt Paraphrase Sensitivity ablation uses a smaller per-cell N controlled by `prompt_ablation_samples_per_condition` (default 10), because each paraphrase requires a full generate+extract+score pass over every (subject model, judge) cell.

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

Subject models are addressed by their OpenRouter slug (`meta-llama/llama-3.1-70b-instruct`, etc.) and OpenRouter selects an upstream provider. The canonical config does not restrict routing: the model IDs are pinned, and the per-call `model_resolved` / `call_id` fields on each generation row record which upstream actually served the request, giving a per-row audit trail. If a deployment needs hard-locked routing, add an `allowed_providers:` list (and optionally `allow_provider_fallbacks: false`) to the config. A model entry's `provider` pins that model to one upstream with fallbacks off. Set `allow_fallbacks: true` on the entry to let OpenRouter route elsewhere when the pinned upstream is unavailable. Each generation row stores the upstream that served it in `served_provider`. Rows written before this field existed do not have it.

### Generation parameters

```yaml
temperature: 1.0                # Controls randomness; 1.0 recommended for creativity tasks
max_tokens: 16000               # Output-token limit of the first generation call
max_tokens_ceiling: 64000       # Largest limit a generation call may escalate to
judge_max_tokens: 8000          # Output-token limit of the first extraction call
judge_max_tokens_ceiling: 32000 # Largest limit an extraction call may escalate to
```

Every response must end on its own stop token. A subject-model or judge response cut off at the limit is discarded and logged in `cut_offs.jsonl` next to `responses.jsonl` or `features.jsonl`. The call is repeated with the limit doubled, up to the ceiling. A response still cut off at the ceiling is not recorded, so the next run retries it. Each generation and extraction record stores the limit it was produced under (`max_tokens`) and the number of discarded calls (`n_cut_offs`). `model_overrides` sets either limit per subject model, and a `judge_models` entry sets either limit per judge. Lower a ceiling to a provider's output limit where needed.

---

## Running the benchmark

The full pipeline is a single command:

```bash
python3 -m alienbench run
```

This chains the six stages (generate → extract → score → analyze → latex → paper). All stages are **checkpointed**: re-running picks up where an interrupted run left off, so repeating `run` after a network failure is safe.

### The `run.py` script

`run.py` is an executable wrapper around the same CLI, provided for callers who prefer a script to a module invocation. It puts the directory containing the package on `sys.path` before delegating, so it works from any working directory and does not require `alienbench` to be on `PYTHONPATH` or installed:

```bash
python3 alienbench/run.py            # full pipeline, from the repository root
python3 run.py                       # full pipeline, from inside alienbench/
./run.py --config test_config.yaml   # smoke test
```

Called with no arguments, `run.py` executes the full pipeline, which is the `run` subcommand. All arguments are forwarded verbatim, so every subcommand and flag documented below is available through the script as well:

```bash
python3 alienbench/run.py run --from extract --to score
python3 alienbench/run.py extract --human
python3 alienbench/run.py ablation judges
python3 alienbench/run.py human sample --seed 42
```

A leading option with no subcommand is treated as an option to `run`. `python3 alienbench/run.py --config test_config.yaml` is therefore equivalent to `python3 -m alienbench run --config test_config.yaml`.

Interrupting the script with Ctrl-C exits with status 130. Because every stage is checkpointed, reissuing the same command resumes the run.

Everything in the remainder of this section applies to both entry points. The examples use the `python3 -m alienbench` form.

To start from or stop at a specific stage:

```bash
python3 -m alienbench run --from extract            # skip generation
python3 -m alienbench run --to score                # stop before analysis
```

To run the benchmark for selected subject models, pass `--model` with a model's id, `key`, or `name` from `config.yaml`. The flag may be repeated:

```bash
python3 -m alienbench run --model GPTSixAstra                    # one model
python3 -m alienbench run --model openai/gpt-6-astra --model x-ai/grok-4.7
python3 -m alienbench ablation prompt --model GPTSixAstra        # its paraphrase responses
```

`--model` restricts generation, extraction, and scoring to the named models. Every judge in the panel scores their responses. The analysis stages (`analyze`, `latex`, `paper`) and the ablation tables always cover every model in the config, because every result compares models. `--model` also works on the single stages `generate`, `extract`, and `score`. The `dimensions` and `judges` ablations make no API calls and do not accept it.

Stages can also be run individually:

```bash
python3 -m alienbench generate
python3 -m alienbench extract
python3 -m alienbench score
python3 -m alienbench analyze
python3 -m alienbench latex
python3 -m alienbench paper
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

Three ablations accompany the primary analysis and are documented in §4 of the paper. They are supplementary to the main pipeline and run through a dedicated `ablation` subcommand.

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
| `tab_ablation_judges_rank.tex` | LaTeX table of rank stability for `\input{}` into the paper |
| `tab_ablation_judges_reliability.tex` | LaTeX table of inter-rater reliability for `\input{}` into the paper |
| `tab_ablation_judges_same_family.tex` | LaTeX table of same-family rank changes for `\input{}` into the paper |
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

### Per-dimension departure rates (Table 4 column)

The main `latex` stage only generates Table 4 for LLM subject models. To compute
the equivalent per-dimension departure rates for the human baseline, run the
standalone script after Stages 2–3 have completed:

```bash
python3 alienbench/human_dimension_rates.py
# or, from inside alienbench/:
python3 human_dimension_rates.py --config config.yaml
```

The script applies the same averaging logic as `_make_ward_dimensions_table`:
judges are averaged first within each generation, then averaged across
generations. Sample output:

```
Human baseline: human/prolific-baseline
Generations: 200  |  Judges: 3 of 3

Dimension                       Rate
----------------------------------------
Body Symmetry                    7.2%
Sensory Organs                  22.4%
Locomotion                      18.8%
Body Plan                       11.3%
Skin / Body Covering             9.6%
Reproduction                     4.1%
Metabolism / Energy Source      13.7%
Communication                    6.9%
Habitat                          5.4%
Cognitive Architecture           8.1%
----------------------------------------
Mean Ward score (0–10)           1.08
```

These values are the human-baseline column for Table 4 in the paper.

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
                                 # and request: the API call exactly as sent
        cut_offs.jsonl           # Discarded calls cut off at max_tokens, each with its request

  extractions/
    {judge_model}/
      {model}/
        {prompt_variant}/
          features.jsonl         # Ward features extracted per description
                                 # incl. judge_model_resolved + judge_call_id,
                                 # request (the API call exactly as sent), and
                                 # raw_response (the judge's text)
          cut_offs.jsonl         # Discarded judge calls cut off at max_tokens, each with its request

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
  tab_models.tex, tab_prompts.tex          # LaTeX tables generated from config.yaml (models, prompt conditions)
  tab_ward_definitions.tex, tab_boundary.tex  # LaTeX tables: Ward dimension definitions and boundary criteria
  key_numbers.tex                          # \ab<Name> macros: every number and name in the paper's running text
  findings.md                              # Result paragraphs of the paper, worded for the current data (read and paste)
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
  tab_ablation_judges_rank.tex             # Judge ablation: rank stability table
  tab_ablation_judges_reliability.tex      # Judge ablation: reliability table
  tab_ablation_judges_same_family.tex      # Judge ablation: same-family table
  summary_ablation_judges.txt              # Judge ablation: text summary
```

### Paper artifacts

The paper contains no hand-typed number, model name, or table body. Three kinds of generated file feed `paper/latex/paper.tex`:

- **Tables.** Every `tab_*.tex` file holds a `tabular` alone. The float, `\resizebox`, caption, and label are authored in `paper.tex`, so regenerating a table cannot overwrite the wording of a caption.
- **Number macros.** `python3 -m alienbench paper` writes `results/key_numbers.tex`, one `\ab<Name>` macro per number or name that the running text and the captions print (means with confidence intervals, test statistics, reliability coefficients, ablation summaries, model and judge names, design constants such as sample counts and token limits). The statistics are recomputed from `data/scores/` and the result CSVs, and the names come from `config.yaml`. A macro whose source is absent is left undefined, so a paper that cites it fails to compile instead of printing a stale number.
- **Names.** Model, judge, tier, and prompt-condition names live only in `config.yaml` (see Configuration).

After a change to the data or the config, regenerate in this order: `latex`, `analyze` (for `table_posthoc.csv`), the three ablations, then `paper`. `tests/test_paper.py` checks that every macro `paper.tex` uses is defined and that no model name is hardcoded in the paper.

- **Claim checks.** `python3 -m alienbench paper` also writes `results/claims_check.md`. Macros keep the numbers current, but some sentences state the shape of a result (the top model is open-weight, exactly one model-feature pair falls below the human baseline, two models tie for second place). The report tests each such claim and names the sentence's location when it fails. It also lists missing generations and judge scores per model, every response that did not end on the model's own stop token (cut off at `max_tokens`, no finish reason, or stopped by a provider refusal), and result files older than the newest scores.
- **Findings report.** `python3 -m alienbench paper` also writes `results/findings.md`. Where `claims_check.md` says that a sentence fails, the findings report gives the replacement. For every result paragraph of the paper (abstract, introduction, experiments, ablations, discussion, conclusion, appendix), it decides which case the data fall in (no exception, one exception, several, a tie or none, and so on) and writes the paragraph in the paper's wording for that case. Each sentence appears as LaTeX built from `\ab<Name>` macros, ready to paste, and with its values filled in. A tag says whether `paper.tex` already contains the sentence, contains it only in a comment, or not at all. Nothing in the report is fed into LaTeX: the authors read it and copy what they need. The few macros that only these sentences use (e.g. `\abExceptionList`, `\abConstrainedLowerNames`) are added to `key_numbers.tex`.

### Extending the model line-up

Every stage caches its work, so adding models only processes the new cells. Generation skips `(model, condition, sample_index)` records already on disk, and extraction and scoring skip `(judge, generation_id)` pairs already written. Existing results are not recomputed or billed again.

1. **Add the models to `config.yaml`.** Give each entry an `id`, `name`, a letters-only `key` that no other model uses, a `tier`, the `resolved_version`, and `open_weights`. A model whose provider prefix matches a judge (e.g. `openai/...` with the GPT judge) becomes a same-family pair automatically. Several subject models from one provider are supported.
2. **Check how responses ended.** Every model must end its description on its own stop token. Generation logs a warning for each response that does not, and a summary when it finishes. The section "Responses that did not end on the model's stop token" in `results/claims_check.md` gives the counts per model: no finish reason (the provider's stream ended early) and stopped by a provider refusal. Regenerate these before reporting results. Cut-off responses are never recorded. The section "Output-token limits" lists, per subject model and judge, the start limit, the ceiling, the records produced at a raised limit, and the samples still cut off at the ceiling. Raise the ceiling for any model or judge with unresolved samples and rerun.
3. **Run the main pipeline:** `python3 alienbench/run.py`. It runs `generate`, `extract`, `score`, `analyze`, `latex`, and `paper`. The human baseline is already scored and needs no rerun, because the judge panel does not change. Its extractions and scores live under `data/extractions/<judge>/human__prolific-baseline/` and `data/scores/<judge>/human__prolific-baseline/`. Keep them in place when archiving model data: the responses cannot be collected again, and without these files every human-baseline macro is undefined.
4. **Rerun the ablations.** `ablation prompt` generates paraphrase responses for the new models (API cost). `ablation dimensions` and `ablation judges` re-aggregate the scores. Then run `paper` again.
5. **Read `results/claims_check.md` and `results/findings.md`.** Every row marked **FAILS** in the claims check names a sentence whose wording, not only its numbers, must change. The findings report gives the paragraph in its new wording: paste the sentences tagged *new wording*. Items marked "check by hand" need a judgment call. The "Result files older than the newest scores" section must be empty.
6. **Recompile the paper.** Tables, the radar figure, and all macros update on their own. The radar figure keeps the human baseline top left and the best model below it, and the other models fill rows of at most six in score order. Table highlighting is computed from the data, and every model tied for the top at the printed precision is bolded.

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

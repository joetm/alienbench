# AlienBench: checks before updating the paper

## Data completeness per subject model

| Model | Generations | Scored per judge |
|---|---|---|
| Claude Opus 5.5 | 180/180 | claude-opus-4.6 180, gemini-3.1-pro-preview 180, gpt-5.4 180 |
| GPT-6 Astra | 180/180 | claude-opus-4.6 180, gemini-3.1-pro-preview 180, gpt-5.4 180 |
| GPT-6 Luna | 180/180 | claude-opus-4.6 180, gemini-3.1-pro-preview 180, gpt-5.4 180 |
| Gemini 3.8 Flash | 180/180 | claude-opus-4.6 180, gemini-3.1-pro-preview 180, gpt-5.4 180 |
| Grok 4.7 | 180/180 | claude-opus-4.6 180, gemini-3.1-pro-preview 180, gpt-5.4 180 |
| DeepSeek V4.1 Flash 1017 | 180/180 | claude-opus-4.6 180, gemini-3.1-pro-preview 180, gpt-5.4 180 |
| Llama 4 Maverick | 180/180 | claude-opus-4.6 180, gemini-3.1-pro-preview 180, gpt-5.4 180 |
| Mistral Large 4 | 180/180 | claude-opus-4.6 180, gemini-3.1-pro-preview 180, gpt-5.4 180 |
| Qwen3.8 27B fp8 | 180/180 | claude-opus-4.6 180, gemini-3.1-pro-preview 180, gpt-5.4 180 |
| Ministral 3 14B 2512 | 180/180 | claude-opus-4.6 180, gemini-3.1-pro-preview 180, gpt-5.4 180 |

## Responses that did not end on the model's stop token

Main conditions and paraphrase ablation.

**1 response ended for another reason.** Regenerate these responses before reporting results.

| Model | Responses | Stop | Truncated | No finish reason | Refusal | Other |
|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 230 | 230 | 0 | 0 | 0 | 0 |
| GPT-6 Astra | 230 | 230 | 0 | 0 | 0 | 0 |
| GPT-6 Luna | 230 | 230 | 0 | 0 | 0 | 0 |
| Gemini 3.8 Flash | 230 | 230 | 0 | 0 | 0 | 0 |
| Grok 4.7 | 230 | 230 | 0 | 0 | 0 | 0 |
| DeepSeek V4.1 Flash 1017 | 230 | 230 | 0 | 0 | 0 | 0 |
| Llama 4 Maverick | 230 | 230 | 0 | 0 | 0 | 0 |
| Mistral Large 4 | 230 | 230 | 0 | 0 | 0 | 0 |
| Qwen3.8 27B fp8 | 230 | 229 | 0 | 0 | 0 | **1** |
| Ministral 3 14B 2512 | 230 | 230 | 0 | 0 | 0 | 0 |

Truncated: cut off at `max_tokens`. Generation discards these responses and retries at a higher limit, so a record shows this only if it predates that rule. No finish reason: the provider's stream ended before its final chunk, so the text stops early. Refusal: the provider's safety system stopped the text (`native_finish_reason: refusal`), which OpenRouter reports as `stop`. Other: any other finish reason, such as a content filter.

## Output-token limits

A call starts at the start limit. A response cut off at the limit is discarded, and the call is repeated with the limit doubled, up to the ceiling. Raised: records produced at a raised limit. Highest: the largest limit any record was produced under. Discarded: cut-off calls, logged in `cut_offs.jsonl`. Unresolved: samples still cut off at the ceiling, with no record.

| Subject model | Start | Ceiling | Records | Raised | Highest | Discarded | Unresolved |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 64000 | 64000 | 230 | 0 | 64000 | 0 | 0 |
| GPT-6 Astra | 64000 | 64000 | 230 | 0 | 64000 | 0 | 0 |
| GPT-6 Luna | 64000 | 64000 | 230 | 0 | 64000 | 0 | 0 |
| Gemini 3.8 Flash | 64000 | 64000 | 230 | 0 | 64000 | 0 | 0 |
| Grok 4.7 | 64000 | 64000 | 230 | 0 | 64000 | 0 | 0 |
| DeepSeek V4.1 Flash 1017 | 64000 | 64000 | 230 | 0 | 64000 | 0 | 0 |
| Llama 4 Maverick | 16384 | 16384 | 230 | 0 | 16384 | 0 | 0 |
| Mistral Large 4 | 64000 | 64000 | 230 | 0 | 64000 | 0 | 0 |
| Qwen3.8 27B fp8 | 64000 | 64000 | 230 | 0 | 64000 | 0 | 0 |
| Ministral 3 14B 2512 | 64000 | 64000 | 230 | 0 | 64000 | 0 | 0 |

| Judge | Start | Ceiling | Records | Raised | Highest | Discarded | Unresolved |
|---|---|---|---|---|---|---|---|
| Claude Opus 4.6 | 8000 | 32000 | 2300 | 0 | 8000 | 0 | 0 |
| Gemini 3.1 Pro | 8000 | 32000 | 2300 | 0 | 8000 | 0 | 0 |
| GPT-5.4 | 8000 | 32000 | 2300 | 0 | 8000 | 0 | 0 |

## Result files older than the newest scores

- `table_ablation_dimensions_loo.csv`: rerun `python -m alienbench ablation dimensions`
- `tab_ablation_dimensions.tex`: rerun `python -m alienbench ablation dimensions`
- `table_ablation_judges_loo.csv`: rerun `python -m alienbench ablation judges`
- `table_ablation_judges_same_family.csv`: rerun `python -m alienbench ablation judges`
- `tab_ablation_judges.tex`: rerun `python -m alienbench ablation judges`

## Claims in the paper

13 of 37 claims fail. Each failing claim marks a sentence whose wording, not only its numbers, must change.

| Status | Where | Claim | Detail |
|---|---|---|---|
| **FAILS** | Introduction | The lowest overall score comes from an open-weight model. | lowest: GPT-6 Luna |
| **FAILS** | Introduction; Sec. 4.1 (open-weight paragraph) | The lowest proprietary model scores below every open-weight model except named ones (the 'except' list is not empty). | except: (none) |
| **FAILS** | Sec. 4.1 (open-weight paragraph) | The open-weight model(s) below the lowest proprietary model have the lowest overall score ('which has the lowest overall score', singular). | except: ; lowest: GPT-6 Luna |
| **FAILS** | Sec. 4.1 (overall lead) | Two or more models tie for second place ('the second-place tie of ...'). | 1 model(s) at 5.42 |
| **FAILS** | Sec. 4.1; Sec. 5.1; Conclusion | Ammonia oceans: small positive effect (0 < r < 0.3, significant). | r = 0.37 |
| **FAILS** | Sec. 4.1 | Detailed description: no detectable difference from the baseline, with the same printed p-value as no light ('both ...'). | p = 2.39e-05 |
| **FAILS** | Sec. 5.1 (feature rates) | No light: the two largest increases are sensory organs and communication. | largest: ['communication', 'locomotion'] |
| **FAILS** | Sec. 5.1 (feature rates) | High gravity: the departure rate is lower on every feature. | 9 of 10 lower |
| **FAILS** | Sec. 5.1 (feature rates) | High gravity: reproduction, communication, and cognition are unmentioned more often than under the baseline. |  |
| **FAILS** | Abstract ('nearly every feature'); Introduction ('with one exception'); Sec. 4.2; Sec. 5; Conclusion | Exactly one (model, feature) pair does not exceed the human baseline. | 0 pair(s) |
| **FAILS** | Sec. 4.4 (feature sensitivity) | Exactly one feature preserves the ranking when removed ('the only feature'). | 2 feature(s) |
| **FAILS** | Sec. 4.4 (judge panel) | Exactly one single judge misses the rank-stability bar ('The bar misses concern X'). | 0 judge(s) |
| **FAILS** | Sec. 4.4 (same family) | At least one same-family pair has a zero and one a negative rank change (both sentences name models). | zero: 4, negative: 0 |
| check by hand | Sec. 4.4 (prompt paraphrase); Sec. 5 | The paraphrase outlier is still far outside the other models' shifts, and 'rankings are robust to prompt paraphrase' still fits the numbers. | judgment call: read tab_ablation_prompt.tex |
| OK | Abstract; Introduction; Sec. 4.1 (open-weight paragraph) | Open-weight models have a higher mean overall score than proprietary models. | open 5.40 vs. proprietary 4.58 |
| OK | Abstract; Introduction; contributions; Sec. 4.1 | Some open-weight models score higher than every proprietary model. | 1 open-weight models above the best proprietary model |
| OK | Introduction | Open-weight models score highest on most of the features. | 6 of 10 features |
| OK | Introduction (best model) | The best model is open-weight (stated as 'the highest overall score comes from an open-weight model' or implied by the open-weight framing). | best: Ministral 3 14B 2512 |
| OK | Abstract; Introduction | No single model scores highest under every prompt condition. | 4 different condition leaders |
| OK | Introduction | Under some conditions, the highest score comes from a tier-C (small) model. | conditions led by tier C: ['baseline', 'constrained_no_light', 'detailed_description'] |
| OK | Sec. 3 (prompt conditions) | The best model remains first when any one condition is removed. |  |
| OK | Sec. 4.1 (per-condition leaders) | The baseline leader also leads the detailed-description condition (the sentence names them together). | baseline: Ministral 3 14B 2512; detailed: Ministral 3 14B 2512 |
| OK | Sec. 4.1 | Scores are uniformly high under the departure-primed condition. | lowest model: 8.68 |
| OK | Sec. 4.1; Conclusion | Scores are uniformly low under the high-gravity condition. | highest model: 4.18 |
| OK | Sec. 5.1 (structured imagination) | Every model except one scores lower under high gravity than under the baseline. | 9 of 10 below baseline |
| OK | Sec. 4.1 | High gravity: large negative effect relative to the baseline (r < 0, |r| >= 0.5, significant). | r = -0.53 |
| OK | Sec. 4.1; Sec. 5.1; Conclusion | No light: no detectable difference from the baseline (not significant). | p = 1 |
| OK | Abstract; Conclusion | Only one constrained condition is associated with lower scores than the baseline. | 1 of 3 |
| OK | Sec. 5.1 (feature rates) | Ammonia: locomotion and metabolism are higher and among the three largest increases. | largest: ['locomotion', 'metabolism', 'body_plan'] |
| OK | Sec. 5.1 (feature rates) | No light and ammonia: several other features are lower than under the baseline ('several other features become more Earth-typical'). | lower: no light 5, ammonia 3 |
| OK | Sec. 5.1 (structured imagination); Conclusion | Under the baseline condition, the models show the effect in a weaker form than the humans (higher mean score). | models 4.10 vs. humans 1.93 |
| OK | Sec. 5.1 (structured imagination) | Under the baseline condition, a typical model description keeps most features Earth-typical (mean below half the scale). | mean 4.10 |
| OK | Sec. 4.3 | Exactly one feature falls below the good-agreement threshold ('... and X falls below it'). | 9 of 10 clear it |
| OK | Sec. 4.3 | Every feature is in the Excellent, Good, or Moderate band (the sentence lists only these three). | 10 of 10 in these bands |
| OK | Sec. 4.4 (judge panel) | Removing the Claude Opus judge raises alpha, and removing the GPT judge lowers it. |  |
| OK | Sec. 4.4 (same family) | No same-family pair shows a positive rank change. | 0 positive |
| OK | Sec. 4.4 (prompt paraphrase) | The paraphrase ablation covers every subject model. |  |

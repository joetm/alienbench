# AlienBench: Test Run Findings

**Date:** 2026-04-07  
**Run type:** Feasibility / pipeline test (`samples_per_condition=3`)

---

## 1. Study Overview

| Parameter | Value |
|---|---|
| Subject models | `openai/gpt-4o`, `google/gemini-2.0-flash-001` |
| Judge models | `openai/gpt-4o`, `google/gemini-2.0-flash-001` |
| Prompt conditions | 6 (see below) |
| Samples per condition | 3 |
| Total generations | 72 (2 models × 6 conditions × 3 samples × 2 judges) |
| Total scored records | 144 Ward + 144 Creativity |
| Parse errors | 0 |

**Prompt conditions:**

| ID | Label |
|---|---|
| `baseline` | Open-ended: "Imagine a creature that lives on an alien planet. Describe it in detail." |
| `departure_primed` | "…as different from any Earth creature as possible…" |
| `constrained_no_light` | Planet with no light at all |
| `constrained_high_gravity` | Planet with 10× Earth gravity |
| `constrained_ammonia` | Oceans of liquid ammonia |
| `elaboration_prompted` | Body / movement / feeding / reproduction / communication |

---

## 2. Ward Departure Scores by Model × Condition

Scores are on a 0–10 integer scale (one point per Ward dimension departed from Earth default), averaged across both judges and all 3 samples.

| Condition | GPT-4o | Gemini 2.0 Flash | Average |
|---|---|---|---|
| Baseline | 3.33 | 5.67 | 4.50 |
| Departure-primed | **8.83** | **9.33** | **9.08** |
| Constrained: No Light | 4.00 | 3.50 | 3.75 |
| Constrained: High Gravity | 0.83 | 2.50 | 1.67 |
| Constrained: Ammonia Oceans | 3.00 | 5.50 | 4.25 |
| Elaboration-prompted | 6.00 | 6.17 | 6.08 |
| **Overall** | **4.33** | **5.44** | **4.89** |

Overall 95% CIs (averaged across judges, all conditions):

- GPT-4o: 4.33 [2.96, 5.71] (n=18)
- Gemini 2.0 Flash: 5.44 [4.06, 6.83] (n=18)

---

## 3. Creativity Scores by Model × Condition

Scores are on a 5–25 integer scale (five 1–5 rubric dimensions summed), averaged across both judges.

| Condition | GPT-4o | Gemini 2.0 Flash | Average |
|---|---|---|---|
| Baseline | 23.67 | 23.67 | 23.67 |
| Departure-primed | 24.83 | **25.00** | **24.92** |
| Constrained: No Light | 22.17 | 22.67 | 22.42 |
| Constrained: High Gravity | 21.17 | 23.00 | 22.08 |
| Constrained: Ammonia Oceans | 22.33 | 23.50 | 22.92 |
| Elaboration-prompted | 23.67 | 23.67 | 23.67 |
| **Overall** | **22.97** | **23.58** | **23.27** |

Overall 95% CIs:

- GPT-4o: 22.97 [22.24, 23.71] (n=18)
- Gemini 2.0 Flash: 23.58 [23.16, 24.00] (n=18)

---

## 4. Statistical Findings

**Prompt effect on Ward scores (Kruskal-Wallis):**  
H = 22.30, p = 0.0005 — highly significant. Prompt manipulation is the dominant driver of departure scores; which model is used is secondary.

**Ward–Creativity correlation (Spearman):**  
ρ = 0.804, p < 0.0001 (n = 36) — strong positive correlation. Creatures that depart more structurally from Earth anatomy are also rated more creative by LLM judges.

---

## 5. Inter-Rater Reliability (Krippendorff's α)

Two LLM judges (`gpt-4o` and `gemini-2.0-flash-001`) rated every generation independently.

| Measure | α | Band |
|---|---|---|
| **Ward Total (0–10)** | **0.802** | Good |
| Body Symmetry | 0.924 | Excellent |
| Sensory Organs | 0.606 | Good |
| Locomotion | 0.560 | Moderate |
| Body Plan | 0.507 | Moderate |
| Skin / Covering | 0.601 | Good |
| Reproduction | 0.889 | Excellent |
| Metabolism | 0.945 | Excellent |
| Communication | 0.613 | Good |
| Habitat | 0.357 | Fair |
| Cognition | 0.683 | Good |
| **Creativity Total (5–25)** | **0.552** | Moderate |
| Novelty | 0.577 | Moderate |
| Elaboration | −0.044 | Poor ⚠ |
| Internal Consistency | −0.014 | Poor ⚠ |
| Imaginative Reach | 0.566 | Moderate |
| Functional Integration | 1.000 | Excellent† |

†Perfect agreement on this small n; treat with caution until verified at scale.  
⚠ Near-zero (worse than chance). These dimensions require rubric revision before the full run.

Bands follow Landis & Koch (1977): Poor (<0.20), Fair (0.20–0.40), Moderate (0.40–0.60), Good (0.60–0.80), Excellent (≥0.80).

---

## 6. Key Observations and Implications for Full Run

**1. Departure-primed is the strongest manipulation and should serve as a benchmark anchor.**  
Both models nearly max out the Ward scale (8.83 / 9.33) when explicitly asked to differ from Earth. This validates the benchmark's sensitivity to prompting style and is theoretically consistent with Ward's path-of-least-resistance hypothesis.

**2. The constraint paradox: physical constraints suppress rather than expand departures.**  
High-gravity is the worst condition for both models (0.83 and 2.50). No-light and ammonia also underperform baseline for Gemini. This is a substantively interesting finding: models appear to anchor heavily on the stated constraint and then conserve all other features, reducing overall departure. It directly illustrates Ward's (1994) structured imagination effect. This pattern merits dedicated analysis in the full run.

**3. Creativity scores show ceiling effects and low variance across conditions.**  
All cells fall in the 21–25 range on a 5–25 scale. The judges are over-generous or under-calibrated. Options for the full run: (a) revise anchor descriptions to pull scores lower, (b) use pairwise comparative ranking instead of absolute ratings, or (c) report only the Ward score as the primary metric and treat creativity as secondary.

**4. Habitat and Body Plan have the weakest Ward reliability (α = 0.357 and 0.507).**  
These dimensions have the most ambiguous boundary criteria. The extraction prompt for both should be tightened before the full run. Candidate revision: add a concrete negative example (what does NOT count as a departure).

**5. Elaboration and Internal Consistency have near-zero creativity reliability.**  
α ≈ 0 means judges are essentially random on these two dimensions. These should either be dropped from the creativity total or have their rubric anchors substantially revised. The current anchors may be too abstract for LLM judges to apply consistently.

**6. Sample size is too small for model comparison.**  
CIs for GPT-4o and Gemini overlap substantially on both Ward and Creativity. The overall Ward gap (4.33 vs. 5.44) is directionally consistent but requires n=50 for adequate power. The full run is needed before any model conclusions can be drawn.

**7. Pipeline is clean.**  
Zero parse errors across 144 scored records. Checkpointing works. The full run can proceed.

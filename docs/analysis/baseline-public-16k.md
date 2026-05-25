# Baseline model analysis — pub-002 (16k tokens, public split)

Evaluated on `data/public.jsonl` (1,126 rows) using saved outputs in `data/full_public_16k.jsonl` and `data/full_public_16k.responses.jsonl`. Topic breakdowns from `data/full_public_16k_topics.json`. Companion to [`baseline-public-8k.md`](baseline-public-8k.md).

---

## Setup

| Setting | Value |
|--------|--------|
| Model | `Qwen/Qwen3-4B-Thinking-2507` (bfloat16, vLLM, A100) |
| Max generation tokens | **16,384** (doubled from pub-001) |
| Sampling | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Prompting | Adaptive per question: MCQ and single-blank free-form use baseline prompts; free-form with 2+ `[ANS]` placeholders uses §1.3 multi-blank format (`\boxed{a}, \boxed{b}, …`) |
| Analysis notebook | `notebooks/baseline_analysis_16k.ipynb` |

---

## Headline metrics

| Split | N | Correct | Accuracy | Δ vs 8k (pub-001) |
|-------|--:|--------:|---------:|------------------:|
| **Overall** | 1,126 | 697 | **61.90%** | **+9.24 pp** |
| MCQ | 375 | 270 | **72.00%** | **+21.60 pp** |
| Free-form | 751 | 427 | **56.86%** | **+3.07 pp** |

Doubling the token budget from 8k to 16k yielded the largest single inference gain in this project.

**Truncation status:**

| Group | N | Accuracy |
|-------|--:|--------:|
| Think finished (`</think>` present) | 1,048 | **66.3%** |
| Truncated mid-think (no `</think>`) | 78 | **2.6%** |

Truncation rate fell from 25.6% (288/1126) at 8k to **6.9%** (78/1126) at 16k. The accuracy gap (63.7 pp) confirms that nearly all truncated responses produce no usable answer.

---

## Strengths

1. **MCQ dramatically improved.** 50.40% → 72.00% (+21.6 pp) by giving the model room to finish its chain-of-thought before committing to a letter.

2. **Truncation mostly solved.** 78 truncated responses remain vs 288 at 8k. The remaining 6.9% are concentrated in MCQ (11.7% MCQ truncated vs 4.5% FF).

3. **Every topic improved.** The weakest gains (+4.1 pp polynomials/algebra) are still positive; several hard topics gained 20+ pp.

4. **Integration and derivatives stand out.** Integration at 87.3% (55 items), derivatives at 83.3% (12 items) — both are clean reasoning chains with little multi-blank complexity.

5. **When finished, model is accurate.** 66.3% on the finished cohort (1,048 responses). Given that harder problems now finish and dilute this pool (see §6), the raw reasoning quality is strong.

---

## Weaknesses

### 1. Reasoning errors are now the dominant MCQ failure mode

At 8k, truncation caused **84.4%** of wrong MCQ responses. At 16k, the failure-mode distribution has inverted:

| Failure mode | 8k n | 8k % | 16k n | 16k % |
|---|---:|---:|---:|---:|
| Truncated mid-think (no `</think>`) | 157 | 84.4% | 43 | 41.0% |
| Think done, no `\boxed{Letter}` | 5 | 2.7% | 8 | 7.6% |
| Think done, wrong `\boxed{Letter}` | 24 | 12.9% | 54 | 51.4% |
| **Total wrong MCQ** | 186 | 100% | 105 | 100% |

The absolute wrong-MCQ count fell from 186 to 105, a meaningful improvement. But **51.4% of remaining wrong MCQ** are now pure reasoning failures: the model finishes, formats correctly, and still picks the wrong answer. Truncation interventions cannot address these.

**Implication:** SFT on hard MCQ problems (geometry, sequences/recurrences, "other") is the next highest-leverage intervention for MCQ.

### 2. MCQ reasoning errors concentrate in geometry and sequences

Slicing the 54 "think-finished, wrong-boxed" cases by topic:

| Topic | Wrong | Total finished-boxed | Error rate |
|---|---:|---:|---:|
| Other (catch-all) | 22 | 88 | 25% |
| Sequences / recurrences | 12 | 50 | 24% |
| Geometry | 7 | 32 | 22% |
| Integration | 3 | 51 | **6%** |
| Polynomials / algebra | 3 | 28 | 11% |

Geometry (50.4% overall) remains the weakest topic and is now bottlenecked by reasoning, not tokens. Distribution by question-length quartile is roughly uniform (14 / 18 / 17 / 5 cases), so harder problems across all lengths fail — it is not just the longest questions.

### 3. Multi-blank free-form accuracy collapses with blank count

The §1.3 multi-blank prompt (`\boxed{a}, \boxed{b}, …`) is already active for all 2+ `[ANS]` questions. These numbers are post-prompt.

| Blank count | N | Accuracy |
|---|--:|---:|
| 1 | 337 | 60.2% |
| 2 | 171 | 58.5% |
| 3 | 90 | 62.2% |
| 4 | 59 | 55.9% |
| 5 | 31 | 41.9% |
| 6–10 | 53 | **34.0%** |
| 11+ | 10 | 40.0% |

Items with ≥6 blanks score 34%, nearly half the single-blank rate. The remaining gap is a reasoning problem — the model must track and output many independent answers — not a format problem that prompting can solve.

### 4. Long questions still fail at high rates

| Question length quartile | N | Accuracy |
|---|--:|---:|
| Q1 (short, ≤150 chars) | 283 | 78.8% |
| Q2 | 281 | 65.5% |
| Q3 | 281 | 59.4% |
| Q4 (long, ≥435 chars) | 281 | **43.8%** |

Q4 accuracy is 35 pp below Q1. Mean question length: 297 chars (correct) vs 498 chars (wrong). Long problems are inherently harder — SFT on long-context math problems would directly address this gap.

### 5. Think-finished accuracy diluted (pool composition shift)

Think-finished accuracy dropped from 69.5% (8k) to 66.3% (16k). This is a composition effect: at 8k, the 25.6% of responses that were truncated disproportionately represented hard problems (mean truncated question length: longer than finished). At 16k, most of those problems now finish, lowering the pool average. It does not indicate degraded reasoning — the overall accuracy still rose +9.24 pp.

### 6. Response length still separates correct from wrong

| Split | Correct mean | Wrong mean |
|---|---:|---:|
| MCQ | 16,981 chars | 31,219 chars |
| Free-form | 7,308 chars | 18,151 chars |

Wrong MCQ responses are nearly 2× as long as correct ones. Long chains that never commit to a confident answer, or that reach the token cap mid-reasoning, remain a failure mode. Mean truncated response: 39,087 chars (~16k chars/token ratio of 2.39).

---

## Per-topic accuracy: 8k → 16k delta

**Caveat:** "Other" is 581 / 1126 rows (**51.6% of the dataset**) — a heterogeneous catch-all, not a coherent topic. Several named topics have n ≤ 25 with 95% CI bands of ±20–26 pp; their accuracy numbers and 8k→16k deltas are noise and should not drive SFT targeting decisions.

| Topic | n | 8k % | 16k % | Δ pp | 95% CI ± (16k) |
|---|--:|---:|---:|---:|---:|
| other | 581 | 56.1% | 62.1% | +6.0 | ±3.9 |
| polynomials / algebra | 146 | 59.6% | 63.7% | +4.1 | ±7.8 |
| **geometry** | 115 | 35.6% | **50.4%** | +14.8 | ±9.1 |
| probability / stats | 82 | 50.0% | 58.5% | +8.5 | ±10.7 |
| sequences / recurrences | 75 | 34.7% | 58.7% | +24.0 | ±11.1 |
| integration | 55 | 74.5% | 87.3% | +12.8 | ±8.8 |
| linear algebra | 23 | 52.2% | 60.9% | +8.7 | ±19.9 |
| number theory | 23 | 30.4% | 56.5% | +26.1 | ±20.3 |
| limits | 14 | 35.7% | 57.1% | +21.4 | ±25.9 |
| derivatives | 12 | 58.3% | 83.3% | +25.0 | ±21.1 |

Only **geometry** has a CI tight enough to call a clean weakness vs the 61.9% overall. The headline "number theory +26 pp" and "derivatives +25 pp" gains are inside their own ±20 pp confidence bands — those deltas could easily be sampling noise. The §2 "wrong-finished-boxed" error rates per topic (geometry 22%, sequences 24%, "other" 25%) suffer the same small-n problem on top of being conditioned on a 54-row slice.

**Implication for targeting:** SFT mix should oversample long-context (§4) and high-blank-count (§3) problems where N is large and the gap is real, plus weighted geometry where the topic signal survives CI scrutiny. Other named topics provide no statistically reliable targeting signal.

---

## MCQ letter-choice bias

No systematic position bias detected. Gold letters A–J are roughly uniformly distributed (23–44 per letter). Wrong-case predictions spread across all letters with no single over-selected choice. Per-letter accuracy ranges from 73.9% (H) to 92.6% (I), with no structural pattern — errors appear to be reasoning failures, not bias toward early/late options.

---

## Priority order for next experiments

1. **SFT — hard MCQ / geometry / sequences.** 51.4% of wrong MCQ are now reasoning failures with the format prompt already applied. QLoRA fine-tuning on Numina hard problems addresses this directly.

2. **Long-question SFT.** Q4 (longest 25%) scores 43.8%. SFT corpus should oversample long-context problems.

3. **High-blank-count free-form.** 6–10 blank items score 34%. The multi-blank format prompt is already active; the remaining gap is reasoning. SFT examples with many blanks would help.

4. **32k token budget: rejected.** dev-009 showed no lift vs 16k (−0.89 pp overall). 16k is the effective sweet spot.

5. **Concise-think prompting: rejected.** dev-006 showed −0.2 pp; truncation is structural, not prompt-addressable without fine-tuning.

---

## Experiment registry

- Run ID: [`pub-002`](../log/experiments.md#pub-002) in `log/experiments.md`

## Files referenced

- Results: `data/full_public_16k.jsonl` — fields `id`, `is_mcq`, `correct`, `gold`
- Responses: `data/full_public_16k.responses.jsonl` — fields `id`, `response`
- Topic breakdown: `data/full_public_16k_topics.json`
- Data: `data/public.jsonl`
- Free-form scoring: `judger.py` (`Judger.auto_judge`)
- Analysis notebook: `notebooks/baseline_analysis_16k.ipynb`

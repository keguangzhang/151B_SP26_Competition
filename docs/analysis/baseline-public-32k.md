# Baseline model analysis — pub-003 (32k tokens, public split)

Evaluated on `data/public.jsonl` (1,126 rows) using saved outputs in `data/full_public_32k.jsonl` and `data/full_public_32k.responses.jsonl`. Topic breakdowns from `data/full_public_32k_topics.json` (classifier `weighted_v1` in [`scripts/topic_classify.py`](../../scripts/topic_classify.py)). Companion to [`baseline-public-16k.md`](baseline-public-16k.md).

---

## Setup

| Setting | Value |
|--------|--------|
| Model | `Qwen/Qwen3-4B-Thinking-2507` (bfloat16, vLLM, A100) |
| Max generation tokens | **32,768** (doubled from pub-002) |
| Sampling | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Prompting | Adaptive per question: MCQ and single-blank free-form use baseline prompts; free-form with 2+ `[ANS]` placeholders uses §1.3 multi-blank format (`\boxed{a}, \boxed{b}, …`) |
| Analysis notebook | `notebooks/baseline_analysis_32k.ipynb` |
| Topic classifier | `scripts/topic_classify.py` (`weighted_v1` — weighted regex scoring; 14 topics + `other`) |

---

## Headline metrics

| Split | N | Correct | Accuracy | Δ vs 16k (pub-002) |
|-------|--:|--------:|---------:|-------------------:|
| **Overall** | 1,126 | 741 | **65.81%** | **+3.91 pp** |
| MCQ | 375 | 304 | **81.07%** | **+9.07 pp** |
| Free-form | 751 | 437 | **58.19%** | **+1.33 pp** |

Doubling from 16k to 32k yielded a smaller but meaningful gain, driven overwhelmingly by MCQ (+9 pp). Free-form improved only marginally (+1.3 pp).

**Truncation status:**

| Group | N | Accuracy |
|-------|--:|--------:|
| Think finished (`</think>` present) | 1,115 | **66.3%** |
| Truncated mid-think (no `</think>`) | 11 | **18.2%** |

Truncation rate fell from 6.9% (78/1126) at 16k to **1.0%** (11/1126) at 32k — nearly eliminated. The accuracy gap (48.1 pp) confirms truncated responses are near-useless, but the small n (11) means this is no longer a meaningful failure mode.

---

## Strengths

1. **MCQ dramatically improved.** 72.00% → 81.07% (+9.07 pp). The additional budget lets hard MCQ problems finish chain-of-thought reliably.

2. **Truncation essentially eliminated.** Only 11 truncated responses remain (1.0%). At 32k, truncation is no longer a meaningful driver of failure.

3. **Number theory and sequences made large jumps.** Number theory +15.8 pp (57.9% → 73.7%), sequences/recurrences +10.7 pp (62.5% → 73.2%). Both are MCQ-heavy and benefit from finished reasoning.

4. **MCQ format compliance high.** 97.9% of MCQ responses contain a `\boxed{Letter}` — format is not an issue.

5. **Reasoning errors now dominate MCQ failures.** 91.5% of wrong MCQ responses finish their chain-of-thought and produce a correctly-formatted but wrong letter. This signals the model has genuine reasoning capability; improvements require better training signal, not inference tricks.

---

## Weaknesses

### 1. Reasoning errors are the dominant MCQ failure mode

At 32k, the failure-mode distribution is almost entirely reasoning-driven:

| Failure mode | 16k n | 16k % | 32k n | 32k % |
|---|---:|---:|---:|---:|
| Truncated mid-think (no `</think>`) | 43 | 41.0% | 4 | 5.6% |
| Think done, no `\boxed{Letter}` | 8 | 7.6% | 2 | 2.8% |
| Think done, wrong `\boxed{Letter}` | 54 | 51.4% | 65 | 91.5% |
| **Total wrong MCQ** | 105 | 100% | 71 | 100% |

Wrong MCQ count dropped from 105 to 71. But **91.5% of remaining failures are pure reasoning errors** — the model finishes, formats correctly, and picks the wrong letter. Token-budget increases cannot address these. SFT on hard MCQ problems is the primary remaining lever.

### 2. Free-form gains are modest; wrong-value errors dominate

Wrong free-form breakdown (n=314):

| Failure mode | n | % |
|---|---:|---:|
| Truncated mid-think | 5 | 1.6% |
| Misformat (count ≠ blanks) | 28 | 8.9% |
| Count OK — wrong values | 285 | 90.8% |
| Count OK — wrong order (n≥2) | 1 | 0.3% |

Among the 285 wrong-value cases: 54.4% wrong on all blanks, 40.7% partial (some blanks correct). Primary error type: numeric close but outside tolerance (44.2%) and conceptual/math errors (48.4%).

### 3. Multi-blank accuracy collapses with blank count

| Blanks | N | Accuracy |
|---|--:|---:|
| 1 | 337 | 61.7% |
| 2 | 171 | 57.9% |
| 3 | 90 | 65.6% |
| 4 | 59 | 59.3% |
| 5 | 31 | 45.2% |
| 6–10 | 53 | **32.1%** |
| 11+ | 10 | 50.0% |

Items with 6–10 blanks score 32.1%, roughly half the single-blank rate. This gap is a reasoning problem — the multi-blank prompt is already active; what's needed is SFT examples with many blanks.

**Overthinking regression in multi-blank:** Among R→W (16k correct → 32k wrong) cases, several multi-blank regressions involve the model inserting an extra `\boxed{}` value (e.g., id 396, 980, 883) — the longer trace re-introduces answer material before the final answer group, producing a "too many boxes" misformat or an off-by-one extraction error.

### 4. Long questions still fail at high rates

| Question length quartile | N | Accuracy |
|---|--:|---:|
| Q1 (short, ≤150 chars) | 283 | 79.9% |
| Q2 | 281 | 71.5% |
| Q3 | 281 | 63.3% |
| Q4 (long, ≥435 chars) | 281 | **48.4%** |

Q4 accuracy is 31.5 pp below Q1. Mean question length: 303 chars (correct) vs 511 chars (wrong). This gap is unchanged from 16k and reflects reasoning difficulty, not token limits.

### 5. Response length still separates correct from wrong

| Split | Correct mean | Wrong mean | Ratio |
|---|---:|---:|---:|
| MCQ | 20,485 chars | 33,853 chars | 1.65× |
| Free-form | 7,081 chars | 18,780 chars | 2.65× |

Wrong responses are substantially longer. The 16k→32k length delta is small (median +52 chars, mean +673 chars), confirming most traces fit within 16k. The long tail (p95: +13,856 chars) consists mainly of hard problems that were already wrong at 16k.

**Budget utilization:** Only 6/1126 responses (0.5%) reach the 32k cap (≥79k chars). The extra budget is not being used by most responses — truncation was already rare at 16k.

### 6. Paired 16k→32k flips: gains concentrated in MCQ

| Slice | W→R | R→W | Net |
|---|---:|---:|---:|
| Overall | 60 | 16 | +44 |
| MCQ | 41 | 7 | +34 |
| Free-form | 19 | 9 | +10 |

MCQ benefited most (+34 net). Free-form net gain is only +10, with 9 regressions (R→W) — many from overthinking-style failures in multi-blank items. **Diminishing returns are visible:** 8k→16k yielded +9.24 pp overall vs 16k→32k +3.91 pp.

---

## Per-topic accuracy: 16k → 32k delta

| Topic | n | 16k % | 32k % | Δ pp |
|---|--:|---:|---:|---:|
| number theory | 57 | 57.9% | **73.7%** | +15.8 |
| complex analysis | 13 | 69.2% | **84.6%** | +15.4 |
| sequences/recurrences | 56 | 62.5% | **73.2%** | +10.7 |
| limits | 13 | 61.5% | 69.2% | +7.7 |
| geometry | 108 | 52.8% | 57.4% | +4.6 |
| linear algebra | 22 | 72.7% | 77.3% | +4.5 |
| polynomials/algebra | 163 | 68.1% | 71.8% | +3.7 |
| arithmetic/word problems | 146 | 60.3% | 63.7% | +3.4 |
| other | 167 | 73.0% | 76.0% | +3.0 |
| probability/stats | 205 | 49.8% | 51.7% | +2.0 |
| integration | 59 | 84.8% | 84.8% | +0.0 |
| logs/exponents | 31 | 51.6% | 51.6% | +0.0 |
| derivatives | 21 | 71.4% | 71.4% | +0.0 |
| trigonometry | 65 | 53.9% | 53.9% | +0.0 |

**Reliable weakness signals (n large, below 65.8% overall):**

| Topic | n | 32k acc | Notes |
|---|--:|---:|---|
| **probability/stats** | 205 | **51.7%** | Largest named bucket; MCQ subset is strong (86.7%) but free-form dominates |
| **geometry** | 108 | **57.4%** | Still 8+ pp below mean despite two token doublings |
| trigonometry | 65 | **53.9%** | Flat vs 16k — fully reasoning-limited |
| logs/exponents | 31 | **51.6%** | Small n; flat vs 16k |

Topics with zero delta (integration, logs/exponents, derivatives, trigonometry) are fully reasoning-limited. Extra budget provides no benefit here.

---

## MCQ letter-choice bias

No systematic position bias detected. Gold letters A–J are roughly uniformly distributed. Per-letter accuracy ranges widely but shows no structural pattern — errors are reasoning failures, not preference for early/late options.

---

## Priority order for next experiments

1. **SFT — probability/stats, geometry, hard MCQ.** 91.5% of wrong MCQ are pure reasoning errors. Topic buckets: probability/stats (51.7%, n=205), geometry (57.4%, n=108), trigonometry (53.9%, n=65). SFT is the primary lever.

2. **Long-question SFT.** Q4 (longest 25%) scores 48.4%. Oversample long-context problems in SFT corpus.

3. **High-blank-count free-form SFT.** 6–10 blank items score 32.1%. Multi-blank format prompt is already active; reasoning examples with many blanks needed. Address the extra-box regression (avoid mid-trace answer leakage).

4. **32k token budget: marginal returns.** 8k→16k: +9.24 pp. 16k→32k: +3.91 pp. Only 11 truncated responses remain at 32k. Further token increases will not meaningfully improve results — focus effort on SFT.

5. **Overthinking / extra-box regression.** 16 R→W regressions at 32k; several are multi-blank items where the longer trace introduces extra `\boxed{}` values. SFT should include examples that keep answer groups clean and final.

---

## Experiment registry

- Run ID: [`pub-003`](../log/experiments.md) in `log/experiments.md`

## Files referenced

- Results: `data/full_public_32k.jsonl` — fields `id`, `is_mcq`, `correct`, `gold`
- Responses: `data/full_public_32k.responses.jsonl` — fields `id`, `response`
- Topic breakdown: `data/full_public_32k_topics.json` (see `scripts/topic_classify.py`)
- Data: `data/public.jsonl`
- Free-form scoring: `judger.py` (`Judger.auto_judge`)
- Analysis notebook: `notebooks/baseline_analysis_32k.ipynb`

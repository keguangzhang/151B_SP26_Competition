# Baseline model analysis — pub-002 (16k tokens, public split)

Evaluated on `data/public.jsonl` (1,126 rows) using saved outputs in `data/full_public_16k.jsonl` and `data/full_public_16k.responses.jsonl`. Topic breakdowns from `data/full_public_16k_topics_weighted_v1.json` (classifier `weighted_v1` in [`scripts/topic_classify.py`](../../scripts/topic_classify.py)). Companion to [`baseline-public-8k.md`](baseline-public-8k.md).

---

## Setup

| Setting | Value |
|--------|--------|
| Model | `Qwen/Qwen3-4B-Thinking-2507` (bfloat16, vLLM, A100) |
| Max generation tokens | **16,384** (doubled from pub-001) |
| Sampling | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Prompting | Adaptive per question: MCQ and single-blank free-form use baseline prompts; free-form with 2+ `[ANS]` placeholders uses §1.3 multi-blank format (`\boxed{a}, \boxed{b}, …`) |
| Analysis notebook | `notebooks/baseline_analysis_16k.ipynb` |
| Topic classifier | `scripts/topic_classify.py` (`weighted_v1` — weighted regex scoring; 14 topics + `other`) |

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

3. **Most topics improved vs 8k.** Under the same `weighted_v1` classifier, every topic with n ≥ 20 gained except arithmetic/word problems (+0.7 pp). Largest lifts: sequences/recurrences (+26.8 pp), number theory (+26.3 pp), geometry (+16.7 pp).

4. **Integration remains a strength.** Integration at **84.8%** (59 items); derivatives at **71.4%** (21 items). Both are mostly MCQ-heavy and benefit from finished chain-of-thought.

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

**Implication:** SFT on hard MCQ problems (geometry, sequences/recurrences, residual `other`) is the next highest-leverage intervention for MCQ.

### 2. MCQ reasoning errors concentrate in sequences, geometry, and residual `other`

Slicing the 54 "think-finished, wrong-boxed" cases by topic (`weighted_v1`):

| Topic | Wrong | Total finished-boxed | Error rate |
|---|---:|---:|---:|
| Other (residual) | 16 | 43 | 37% |
| Sequences / recurrences | 7 | 36 | 19% |
| Geometry | 5 | 33 | 15% |
| Trigonometry | 3 | 15 | 20% |
| Derivatives | 3 | 15 | 20% |
| Integration | 5 | 55 | 9% |

At 8k there were only **24** wrong-boxed MCQ cases (truncation dominated); at 16k the pool is **54**, so topic-level error rates are still small-n but directionally useful.

**Overall topic weaknesses (not just MCQ):** probability/stats is the weakest large bucket at **49.8%** (n=205), then geometry **52.8%** (n=108), trigonometry **53.9%** (n=65). Geometry is still below the 61.9% overall mean and is now reasoning-limited rather than token-limited. Question-length quartile for wrong-boxed cases remains roughly uniform across Q1–Q4.

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

**Classifier revision (2026-05-28):** Topics now use `weighted_v1` ([`scripts/topic_classify.py`](../../scripts/topic_classify.py)) — weighted regex scoring across 14 named topics. Residual **`other` is 167 / 1126 (14.8%)**, down from 581 (51.6%) under the old first-match keyword list. Both runs below use the same classifier so 8k→16k deltas are comparable.

Rebuild: `python3 scripts/topic_classify.py aggregate --results data/full_public_{8k,16k}.jsonl --force`

| Topic | n | 8k % | 16k % | Δ pp | 95% CI ± (16k) |
|---|--:|---:|---:|---:|---:|
| probability / stats | 205 | 44.4% | **49.8%** | +5.4 | ±6.8 |
| other (residual) | 167 | 64.1% | 73.0% | +9.0 | ±6.7 |
| polynomials / algebra | 163 | 65.6% | 68.1% | +2.5 | ±7.2 |
| arithmetic / word problems | 146 | 59.6% | 60.3% | +0.7 | ±7.9 |
| **geometry** | 108 | 36.1% | **52.8%** | +16.7 | ±9.4 |
| trigonometry | 65 | 43.1% | 53.9% | +10.8 | ±12.1 |
| integration | 59 | 71.2% | **84.8%** | +13.6 | ±9.2 |
| number theory | 57 | 31.6% | 57.9% | +26.3 | ±12.8 |
| sequences / recurrences | 56 | 35.7% | 62.5% | +26.8 | ±12.7 |
| logs / exponents | 31 | 48.4% | 51.6% | +3.2 | ±17.6 |
| linear algebra | 22 | 59.1% | 72.7% | +13.6 | ±18.6 |
| derivatives | 21 | 52.4% | 71.4% | +19.1 | ±19.3 |
| complex analysis | 13 | 61.5% | 69.2% | +7.7 | ±25.1 |
| limits | 13 | 53.9% | 61.5% | +7.7 | ±26.4 |

**Reliable weakness signals (n large, below 61.9% overall):**

| Topic | n | 16k acc | Notes |
|---|--:|---:|---|
| **probability / stats** | 205 | **49.8%** | Largest named bucket; MCQ subset is strong (86.7%) but free-form dominates and drags overall |
| **geometry** | 108 | **52.8%** | Still well below mean despite +16.7 pp vs 8k |
| trigonometry | 65 | 53.9% | Moderate n; wide CI |
| logs / exponents | 31 | 51.6% | Small n |

**Implication for targeting:** Prioritize **probability/stats** and **geometry** in SFT mix (both have n ≥ 100). Sequences/recurrences rose to 62.5% at 16k — still watch MCQ wrong-boxed rate (19%) but less urgent than at 8k (30.4% MCQ acc). Continue oversampling **long-context** (§4) and **high-blank-count** (§3) slices. Ignore residual `other` except for MCQ error mining (37% wrong-boxed error rate on finished responses).

---

## MCQ letter-choice bias

No systematic position bias detected. Gold letters A–J are roughly uniformly distributed (23–44 per letter). Wrong-case predictions spread across all letters with no single over-selected choice. Per-letter accuracy ranges from 73.9% (H) to 92.6% (I), with no structural pattern — errors appear to be reasoning failures, not bias toward early/late options.

---

## Priority order for next experiments

1. **SFT — probability/stats, geometry, hard MCQ.** 51.4% of wrong MCQ are reasoning failures. Topic buckets: probability/stats (49.8%, n=205), geometry (52.8%, n=108), plus MCQ wrong-boxed concentration in sequences and residual `other`.

2. **Long-question SFT.** Q4 (longest 25%) scores 43.8%. SFT corpus should oversample long-context problems.

3. **High-blank-count free-form.** 6–10 blank items score 34%. The multi-blank format prompt is already active; the remaining gap is reasoning. SFT examples with many blanks would help.

4. **32k token budget: rejected.** dev-009 showed no lift vs 16k (−0.89 pp overall). 16k is the effective sweet spot.

5. **Concise-think prompting: rejected.** dev-006 showed −0.2 pp; truncation is structural, not prompt-addressable without fine-tuning.

---

## Revision notes

| Date | Change |
|------|--------|
| 2026-05-28 | Recomputed all topic tables with `weighted_v1` classifier (`scripts/topic_classify.py`). Residual `other` 14.8% (was 51.6%). Added probability/stats, trigonometry, arithmetic/word problems, logs/exponents, complex analysis buckets. |

---

## Experiment registry

- Run ID: [`pub-002`](../log/experiments.md#pub-002) in `log/experiments.md`

## Files referenced

- Results: `data/full_public_16k.jsonl` — fields `id`, `is_mcq`, `correct`, `gold`
- Responses: `data/full_public_16k.responses.jsonl` — fields `id`, `response`
- Topic breakdown: `data/full_public_16k_topics_weighted_v1.json` (see `scripts/topic_classify.py`)
- Data: `data/public.jsonl`
- Free-form scoring: `judger.py` (`Judger.auto_judge`)
- Analysis notebook: `notebooks/baseline_analysis_16k.ipynb`

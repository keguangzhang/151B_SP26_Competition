# Baseline model analysis (public split)

This document summarizes strengths and weaknesses of the **8k-token run** evaluated on `data/public.jsonl`, using saved outputs in `data/full_public_8k.jsonl` and `data/full_public_8k.responses.jsonl` (1,126 examples). Topic breakdowns are from `data/full_public_8k_topics.json`.

## Baseline setup (what was measured)

| Setting | Value |
|--------|--------|
| Model | `Qwen/Qwen3-4B-Thinking-2507` |
| Inference | vLLM with BitsAndBytes **INT8** weights |
| Max generation tokens | **8,192** per question |
| Sampling | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Prompting | Separate system prompts for MCQ vs free-form; answers requested in `\boxed{}` |

**Scoring (as implemented in the notebook):**

- **MCQ:** `extract_letter()` first looks for `\boxed{X}` with a single letter. If that is missing, it falls back to the **last** standalone capital letter matched by `\b([A-Z])\b` in the full response (including chain-of-thought).
- **Free-form:** `Judger.auto_judge()` from `judger.py` — symbolic/numeric equivalence and structured checks, not raw string equality.

The numbers below are **exactly** what the starter pipeline computes.

---

## Headline metrics

| Split | Count | Accuracy |
|-------|------:|---------:|
| **Overall** | 1,126 | **52.66%** (593 / 1,126) |
| MCQ (`options` present) | 375 | **50.40%** (189 / 375) |
| Free-form | 751 | **53.79%** (404 / 751) |

Compared to the 4k-token run (40.76% overall, 26.13% MCQ, 48.07% free-form), doubling the token budget yielded a **+11.9 pp overall gain**, with **+24.3 pp on MCQ** — the dominant lever.

---

## Strengths

1. **MCQ accuracy nearly doubled vs 4k run.** Going from 4,096 → 8,192 max tokens drove MCQ from 26% → 50%. The token budget was the primary bottleneck for MCQ, not reasoning quality.

2. **Format compliance is the remaining MCQ lever.** Of 375 MCQ items, 205 (54.7%) contain a `\boxed{Letter}` pattern; of those, **88.3%** are scored correct. The 170 items without a boxed letter score only **4.7%** (almost all are fallback-letter guesses). Format compliance is now the bottleneck, not truncation.

3. **Free-form remains solid and improved.** Single-blank items: **63.0%** (206 / 327). Multi-blank items: **46.7%** (198 / 424). Both up from 4k run (~59% / ~41%).

4. **Shorter problems still easier.** Mean question character length: **287** for correct vs **470** for incorrect. The model handles compact prompts better than long multi-part stems.

5. **Integration is a standout topic.** 74.5% accuracy (55 / 55 tagged items), far above the global average. MCQ accuracy in this bucket also 74.5%.

---

## Weaknesses

### 1. MCQ format compliance is the main bottleneck

Token capping is **no longer the dominant MCQ failure mode** at 8k tokens. Only **6 / 186** wrong MCQ responses (~3.2%) are near the 8,192-token cap. The new primary failure is **missing `\boxed{Letter}`**:

| Bucket | Count | % of wrong MCQ |
|--------|------:|---------------:|
| No `\boxed{Letter}` (fallback extraction) | **162** | **87.1%** |
| Has `\boxed{Letter}` (wrong letter) | **24** | **12.9%** |
| Near token cap (≥28k chars) | **6** | **3.2%** |

**The model reasons but doesn't commit.** 87% of wrong MCQ responses trail off or use bold/text emphasis instead of a `\boxed{X}` final answer. Improving format compliance (prompt engineering, constrained decoding, post-process) is the highest-value MCQ intervention remaining.

The fallback letter heuristic buys a small gain: strict `\boxed{Letter}` accuracy is **48.3%** (181/375) vs reported **50.4%** (189/375) — only ~2 pp delta, much less than in the 4k run.

### 2. MCQ 10-option items still hard

10-option MCQs (336/375): **50.6%** correct. Other MCQs (39/375): **48.7%** correct. No longer a strong gap — the difficulty is now format-driven across both types.

### 3. Multi-answer free-form remains harder

Items with multiple blanks (gold `answer` list length > 1): **424 items**, **46.7%** accuracy vs **63.0%** for single-blank. Ordering, per-subpart errors, and `\boxed{}` formatting of multiple values all hurt.

### 4. Long free-form generations correlate with failure

Incorrect free-form responses: mean **14,094** characters. Correct free-form responses: mean **6,204** characters. Long traces indicate the model is stuck or reconsidering rather than resolving cleanly.

### 5. Topic-level gaps

| Topic | Count | Accuracy | MCQ accuracy |
|-------|------:|---------:|-------------:|
| Number theory | 23 | **30.4%** | 28.6% |
| Sequences / recurrences | 75 | **34.7%** | 34.4% |
| Geometry | 115 | **35.6%** | 28.2% |
| Limits | 14 | 35.7% | 71.4% |
| Probability / stats | 82 | 50.0% | 69.0% |
| Linear algebra | 23 | 52.2% | 63.2% |
| Other (catch-all) | 581 | 56.1% | 42.2% |
| Derivatives | 12 | 58.3% | 58.3% |
| Polynomials / algebra | 146 | 59.6% | 76.7% |
| Integration | 55 | **74.5%** | 74.5% |

Number theory, sequences/recurrences, and geometry are the weakest topics (all ≤36%). Limits MCQ accuracy (71.4%) is deceptively high due to small n=7.

### 6. Model capacity ceiling

INT8 + 4B parameters prioritizes feasibility on one GPU over peak reasoning score. Both quantization and scale limit accuracy on harder items.

---

## Interpretation for improving beyond 8k run

- **MCQ format:** The primary gain is now **getting `\boxed{X}` into every MCQ response** — prompt engineering ("end your response with `\boxed{X}` where X is the letter"), constrained decoding, or a two-pass answer extraction. Token budget is no longer the lever.
- **Free-form multi-blank:** Explicit "Answer 1: `\boxed{...}`, Answer 2: `\boxed{...}`" instructions may improve the 46.7% multi-blank rate.
- **Hard topics:** Sequences/recurrences and geometry (both ~35%) need specialized prompting or verification. Symbolic recurrence checkers and diagram-aware prompting may help.
- **Model scale:** Upgrading to a larger (7B+) or BF16 checkpoint addresses the capacity ceiling for the remaining hard items.

---

## Experiment registry

- Run ID: [pub-001](../log/runs/pub-001-full-public-8k.md) in [`log/experiments.md`](../log/experiments.md)

## Files referenced

- Results: `data/full_public_8k.jsonl` — fields `id`, `is_mcq`, `correct`
- Responses: `data/full_public_8k.responses.jsonl` — fields `id`, `response`
- Topic breakdown: `data/full_public_8k_topics.json`
- Data: `data/public.jsonl`
- Free-form scoring: `judger.py` (`Judger.auto_judge`)

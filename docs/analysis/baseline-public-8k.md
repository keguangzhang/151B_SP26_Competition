# Baseline model analysis (public split)

This document summarizes strengths and weaknesses of the **8k-token run** evaluated on `data/public.jsonl`, using saved outputs in `data/full_public_8k.jsonl` and `data/full_public_8k.responses.jsonl` (1,126 examples). Topic breakdowns are from `data/full_public_8k_topics_weighted_v1.json` (classifier `weighted_v1` in [`scripts/topic_classify.py`](../../scripts/topic_classify.py)).

> **2026-05-23 revision:** truncation detection corrected from char-length heuristic to `</think>` tag presence. Several conclusions in §Weaknesses reversed — see §1 below.

> **2026-05-28 revision:** topic tables recomputed with `weighted_v1` classifier (same as 16k analysis). Residual `other` is 14.8% of rows, not 51.6%.

## Baseline setup (what was measured)

| Setting | Value |
|--------|--------|
| Model | `Qwen/Qwen3-4B-Thinking-2507` |
| Inference | vLLM with BitsAndBytes **INT8** weights |
| Max generation tokens | **8,192** per question |
| Sampling | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Prompting | Separate system prompts for MCQ vs free-form; answers requested in `\boxed{}` |
| Topic classifier | `scripts/topic_classify.py` (`weighted_v1`) |

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

**Accuracy splits by truncation status** (see §1 for methodology):

| Group | N | Accuracy |
|-------|--:|--------:|
| Think finished (`</think>` present) | 838 | **69.5%** |
| Truncated mid-think (no `</think>`) | 288 | **3.8%** |

The 65+ pp gap confirms that truncation — not reasoning quality — is the primary failure driver.

---

## Strengths

1. **MCQ accuracy nearly doubled vs 4k run.** Going from 4,096 → 8,192 max tokens drove MCQ from 26% → 50%. The token budget was the primary bottleneck for MCQ, not reasoning quality.

2. **When the model finishes thinking, it is highly accurate.** 838 responses that reached `</think>` scored **69.5%** overall. The model's reasoning quality is strong when given enough tokens.

3. **Free-form remains solid.** Single-blank items: **63%** (corrected figure). Multi-blank items: **47.8%**. Both up substantially from the 4k run.

4. **Shorter problems still easier.** Mean question character length: **287** for correct vs **470** for incorrect.

5. **Integration is a standout topic.** 71.2% accuracy (59 items under `weighted_v1`).

---

## Weaknesses

### 1. Token truncation is the dominant failure mode — not format compliance (REVISED)

The original analysis used a **28k-character threshold** to detect responses near the 8,192-token cap (assuming ~3.5 chars/token). This was **badly wrong for math content**.

Qwen3-4B-Thinking generates dense LaTeX and Unicode math. The actual chars/token ratio for truncated responses is approximately **2.55 chars/token** — meaning a 14k-character response can represent ~5,500 tokens, and a 21k-character response exceeds the 8k cap.

**Correct truncation detection: check for `</think>` tag.**

The model wraps its chain-of-thought in `<think>…</think>`. If `</think>` is absent, the model was cut off mid-reasoning and produced no final answer.

| Bucket | Count | % of wrong MCQ |
|--------|------:|---------------:|
| **Truncated mid-think (no `</think>`)** | **157** | **84.4%** |
| Think finished, no `\boxed{Letter}` | 5 | 2.7% |
| Think finished, wrong `\boxed{Letter}` | 24 | 12.9% |

The original report attributed 87% of wrong MCQ to "format non-compliance" (no `\boxed{X}`). In reality, **84.4% of wrong MCQ responses were truncated before any answer was written** — there was nothing to format. Only **5 responses** (2.7%) finished thinking but failed to produce a boxed letter.

**Overall truncation:**

| Split | Truncated (no `</think>`) |
|-------|------------------------:|
| Overall | 288 / 1,126 (25.6%) |
| MCQ | 165 / 375 (44.0%) |
| Free-form | 123 / 751 (16.4%) |

MCQ is far more truncation-prone because these problems tend to be harder and require longer chains of thought before the model commits to a letter answer.

The old char-heuristic detected only **6 truncated MCQ** (3.2%). The real count is **157** (84.4%) — the heuristic undercounted by 26×.

### 2. Increasing max_tokens is the highest-value intervention

Since 84% of wrong MCQ and 44% of all MCQ responses are truncated, more token budget directly unlocks correct answers the model has already reasoned toward but couldn't write. The 69.5% accuracy of finished responses vs 3.8% for truncated responses quantifies the ceiling.

Estimated gains from increasing `max_tokens`:
- If MCQ truncation rate drops from 44% to 0% and accuracy among finished responses holds at ~88.3% (the boxed-response rate): MCQ accuracy would approach **~78%**, overall accuracy **~65%**.
- Even halving truncation (44% → 22%) would yield several pp overall.

### 3. Multi-answer free-form remains harder

Items with multiple blanks (gold `answer` list length > 1): **414 items**, **47.8%** accuracy vs **63%** for single-blank. Ordering, per-subpart errors, and `\boxed{}` formatting of multiple values all hurt.

### 4. Long free-form generations correlate with failure

Incorrect free-form responses: mean **14,094** characters. Correct free-form responses: mean **6,204** characters. Long traces indicate the model is stuck or truncated rather than resolving cleanly.

### 5. Topic-level gaps

Classifier: `weighted_v1` — see [`baseline-public-16k.md`](baseline-public-16k.md) for 8k→16k deltas on the same taxonomy.

| Topic | n | Accuracy | MCQ accuracy |
|-------|--:|---------:|-------------:|
| Number theory | 57 | **31.6%** | 28.6% |
| Geometry | 108 | **36.1%** | 30.0% |
| Sequences / recurrences | 56 | **35.7%** | 30.4% |
| Probability / stats | 205 | **44.4%** | 66.7% |
| Trigonometry | 65 | 43.1% | 68.8% |
| Logs / exponents | 31 | 48.4% | 50.0% |
| Derivatives | 21 | 52.4% | 56.2% |
| Limits | 13 | 53.9% | 66.7% |
| Arithmetic / word problems | 146 | 59.6% | 66.7% |
| Linear algebra | 22 | 59.1% | 61.9% |
| Polynomials / algebra | 163 | 65.6% | 66.7% |
| Other (residual) | 167 | 64.1% | 38.3% |
| Complex analysis | 13 | 61.5% | 66.7% |
| Integration | 59 | **71.2%** | 71.2% |

Weakest overall buckets: **number theory**, **geometry**, **sequences/recurrences**, and **probability/stats** (largest weak bucket at n=205). Many of these also have high MCQ truncation rates. Residual `other` is only 14.8% of rows but has a low MCQ accuracy (38.3%) — useful for error mining, not a majority catch-all.

### 6. Model capacity ceiling

INT8 + 4B parameters prioritizes feasibility on one GPU over peak reasoning score. Both quantization and scale limit accuracy on harder items.

---

## Revised interpretation for improving beyond 8k run

Priority order has changed from the original analysis:

1. **More tokens (highest leverage).** 44% of MCQ responses are truncated before writing any answer. Moving to `max_tokens=16384` or higher directly fixes the dominant failure mode. The model reasons correctly when it finishes — 69.5% accuracy vs 3.8% for truncated responses.

2. **Thinking token efficiency.** If 16k tokens is not feasible, prompting the model to reason more concisely (e.g., `/no_think` mode, or a shorter system prompt) may let more responses finish within 8k tokens.

3. **Free-form multi-blank.** Explicit "Answer 1: `\boxed{...}`, Answer 2: `\boxed{...}`" instructions may improve the 47.8% multi-blank rate.

4. **Hard topics.** Geometry, sequences/recurrences, number theory (~32–36%), and probability/stats (44.4%, n=205) need targeted data — much of MCQ weakness on these is truncation-driven at 8k.

5. **MCQ format (now low priority).** Only 5 finished-think responses failed to produce a boxed letter. Format engineering addresses 2.7% of wrong MCQ, not 87% as originally believed.

6. **Model scale.** Upgrading to a larger (7B+) or BF16 checkpoint addresses the capacity ceiling for harder items.

---

## Revision notes

| Date | Change |
|------|--------|
| 2026-05-23 | Corrected truncation detection from char-length heuristic to `</think>` presence. Reversed §1 conclusion: token truncation (84% of wrong MCQ) is the dominant failure, not format compliance (2.7%). Updated priority order accordingly. |
| 2026-05-28 | Topic table recomputed with `weighted_v1` classifier (`scripts/topic_classify.py`); aligned with 16k analysis taxonomy. |

---

## Experiment registry

- Run ID: [pub-001](../log/runs/pub-001-full-public-8k.md) in [`log/experiments.md`](../log/experiments.md)

## Files referenced

- Results: `data/full_public_8k.jsonl` — fields `id`, `is_mcq`, `correct`
- Responses: `data/full_public_8k.responses.jsonl` — fields `id`, `response`
- Topic breakdown: `data/full_public_8k_topics_weighted_v1.json` (see `scripts/topic_classify.py`)
- Data: `data/public.jsonl`
- Free-form scoring: `judger.py` (`Judger.auto_judge`)
- Analysis notebook: `notebooks/baseline_analysis.ipynb`

# Baseline model analysis (public split)

This document summarizes strengths and weaknesses of the **starter baseline** defined in `starter_code_cse151b_comp.ipynb`, evaluated on `data/public.jsonl` using saved outputs in `results/starter_results.jsonl` (1,126 examples; one JSON object per line).

## Baseline setup (what was measured)

| Setting | Value |
|--------|--------|
| Model | `Qwen/Qwen3-4B-Thinking-2507` |
| Inference | vLLM with BitsAndBytes **INT8** weights |
| Max generation tokens | 4,096 per question |
| Sampling | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Prompting | Separate system prompts for MCQ vs free-form; answers requested in `\boxed{}` |

**Scoring (as implemented in the notebook):**

- **MCQ:** `extract_letter()` first looks for `\boxed{X}` with a single letter. If that is missing, it falls back to the **last** standalone capital letter matched by `\b([A-Z])\b` in the full response (including chain-of-thought). See `extract_letter` in the scoring cell of the notebook.
- **Free-form:** `Judger.auto_judge()` from `judger.py` — symbolic/numeric equivalence and structured checks, not raw string equality.

The numbers below are **exactly** what the starter pipeline computes; they are not an independent reimplementation of the judger.

---

## Headline metrics

| Split | Count | Accuracy |
|-------|------:|---------:|
| **Overall** | 1,126 | **40.76%** (459 / 1,126) |
| MCQ (`options` present) | 375 | **26.13%** (98 / 375) |
| Free-form | 751 | **48.07%** (361 / 751) |

Rough composition of `public.jsonl`: about **33%** MCQ and **67%** free-form. The baseline is **much stronger on free-form than on MCQ** under this evaluation protocol.

---

## Strengths

1. **Free-form dominates overall accuracy.** Nearly half of non-MCQ items are graded correct. The symbolic judger accepts algebraically equivalent answers, which rewards coherent final expressions even when formatting differs from the reference.

2. **Single-blank free-form is relatively solid.** For items whose gold `answer` is a **one-element list**, accuracy is about **59%** (327 items). Many of these are short numerical or symbolic answers where the model can converge after reasoning.

3. **When MCQ answers include a proper `\boxed{A}`-style letter, accuracy is very high.** Among responses that contain `\boxed{` followed by a single letter (same pattern the notebook prioritizes), roughly **91%** of those items are scored correct (92 such responses in the file). So the model often **knows** the letter when it commits to a boxed choice.

4. **Shorter problems are easier.** Mean character length of the **question text** is lower for correct items than incorrect ones (about **266 vs 448** characters). The baseline copes better with compact prompts than with long, multi-part stems.

5. **Reasonable performance on generic “other” bucket.** A coarse keyword tagger assigns many problems to a catch-all **other** category (~905 row-tags); accuracy there is about **45%**, above the global average once duplicate tagging is considered — indicating breadth across standard calculus/algebra tasks without extreme specialization.

---

## Weaknesses

### 1. MCQ is the main bottleneck

Reported MCQ accuracy is **~26%**, far below free-form. Contributing factors:

- **Thinking-style outputs are long** (often thousands of tokens). The model frequently debates options without emitting `\boxed{X}`.
- **Fallback letter extraction is brittle.** Taking the **last** isolated `A`–`Z` in the entire trace can align with the gold letter by accident or drift from the model’s true conclusion — but it cannot fix the majority of failures. Only **~22.4%** of MCQ items would be correct if credit required a **strict** `\boxed{Letter}` match (84 / 375), versus **26.1%** reported — so the heuristic buys a few points but does not fix format compliance.
- **Most items are 10-option MCQs** (336 / 375); accuracy in that bucket stays near **25%**, consistent with difficulty and weak letter extraction combined.

### 2. Multi-answer free-form is harder

Items with **multiple blanks** (gold `answer` is a list with length **> 1**) number **414**; accuracy is about **41%**, versus **~59%** for single-blank list answers. Ordering, comma-separated `\boxed{}` usage, and per-subpart errors all hurt.

### 3. Topic-level gaps (heuristic keywords on question text)

These tags are **overlapping and approximate**; they still highlight systematic pain points:

| Rough topic signal | Approx. count (tagged rows) | Accuracy |
|--------------------|------------------------------:|---------:|
| Sequences / recurrences (`a_n`, “sequence”, …) | 57 | **~7%** |
| Geometry (“triangle”, circumcircle, …) | 21 | **~5%** |
| Integration (`\int`, “integral”, …) | 58 | **~31%** |
| Stats / probability (“probability”, Normal, …) | 50 | **~38%** |
| Derivatives | 15 | **~27%** |
| Limits | 15 | **~20%** |
| Linear algebra (“matrix”, “determinant”) | 20 | **~25%** |

Sequences and Euclidean geometry stand out as **very weak** relative to the global rate.

### 4. Failure correlates with very long generations (free-form)

For free-form items, **incorrect** runs have **much longer** outputs on average (mean length ≈ **9.3k** characters vs ≈ **4.4k** for correct). Long traces often indicate the model is stuck, reconsidering, or never stabilizing on a judger-friendly `\boxed{}` final answer — consistent with a thinking model hitting confusion rather than a short resolution path.

### 5. Quantization and model size

INT8 reduces memory and speeds iteration, but it is a **capacity/compression** tradeoff versus BF16 or larger models. Together with **4B** parameters, the baseline prioritizes **feasibility on one GPU** over peak reasoning score.

---

## Interpretation for improving the baseline

- **MCQ:** Constrain decoding or post-process so a **single final `\boxed{A}`** (or constrained JSON) appears reliably — or fine-tune / prompt specifically to **stop after the boxed letter**. Reducing temperature for MCQ-only runs may help. The current metric mixes **reasoning quality** with **extractability**; improving format alone could lift MCQ substantially without changing “math ability.”
- **Free-form:** Target **multi-blank** prompts with explicit “Answer 1, Answer 2” structure; consider shorter `max_tokens` with stronger “final answer only” instructions to reduce runaway chains where the judger never sees a clean box.
- **Hard topics:** Extra data or specialized verification (e.g., symbolic check for recurrences, diagram reasoning) may be needed for sequences and geometry.

---

## Files referenced

- Results: `results/starter_results.jsonl` — fields `id`, `is_mcq`, `gold`, `response`, `correct`
- Data: `data/public.jsonl`
- Pipeline and prompts: `starter_code_cse151b_comp.ipynb`
- Free-form scoring: `judger.py` (`Judger.auto_judge`)

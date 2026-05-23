# dev-006 — §1.2 concise-prompt experiment

**Date:** 2026-05-23  
**Eval set:** `data/dev.jsonl` (20% stratified, 225 rows: 75 MCQ / 150 free-form)  
**Notebook:** `notebooks/dev.ipynb`, `PROMPT_VARIANT = "concise"`  
**Artifacts:** `results/dev_results_concise.jsonl`, `results/dev_results_concise.responses.jsonl`

## Setup

Added two "concise" system prompt variants (vs baseline):

**MCQ:**
> Think step-by-step. Keep reasoning focused and non-repetitive: do not re-derive steps already completed, avoid going in circles. Once you identify the best answer, commit immediately and output ONLY its letter inside `\boxed{}`.

**Free-form:**
> Think step-by-step. Keep reasoning focused and non-repetitive. Same `\boxed{}` format.

All other settings identical to pub-001: `max_tokens=8192`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, guided decoding enabled for MCQ.

## Results

| Group | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| MCQ | 36 | 75 | 48.00% |
| Free-form | 82 | 150 | 54.67% |
| Overall | 118 | 225 | 52.44% |

## Comparison

| Metric | pub-001 (baseline) | dev-005 (guided, 20%) | dev-006 (concise) |
|--------|-------------------:|----------------------:|------------------:|
| MCQ | 50.40% | 53.33% | **48.00%** |
| Free-form | 53.79% | 52.00% | **54.67%** |
| Overall | 52.66% | 52.44% | **52.44%** |

## Takeaway

No improvement. MCQ accuracy dropped ~5 pp vs dev-005 (within n=75 noise: 1 answer = 1.33 pp, so ~4 answers). Free-form is flat. Overall unchanged.

The concise instruction cannot fix structural truncation: at 8k tokens, 44% of MCQ responses hit the limit mid-think regardless of how efficiently they reason. The prompt may even be counterproductive for hard items requiring long chains.

**Decision:** §1.2 rejected. §1.1 (16k tokens, pub-002) remains the highest-priority next step.

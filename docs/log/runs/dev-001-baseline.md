# dev-001 — Dev baseline (starter-style)

**Registry:** [dev-001](../experiments.md#dev-001) · **Status:** baseline

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/dev.jsonl` (112 rows: 37 MCQ, 75 free-form) |
| Notebook | `notebooks/dev.ipynb` |
| Prompt | Starter MCQ / free-form system prompts |
| Decoding | `temperature=0.6`, `top_p=0.95` (all rows) |
| `max_tokens` | 4096 path (starter default before 8k ablation) |

## Results

| Split | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| MCQ | 11 | 37 | **29.73%** |
| Free-form | 39 | 75 | **52.00%** |
| **Overall** | **50** | **112** | **44.64%** |

## Artifacts

- `results/dev_results.responses.jsonl` (if present from early dev run)

## Takeaway

Reference point for dev ablations. MCQ is severely truncated-limited at 4k; see [dev-003](dev-003-max-tokens-8k.md).

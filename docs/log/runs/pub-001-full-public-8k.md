# pub-001 — Full public, 8k baseline

**Registry:** [pub-001](../experiments.md#pub-001) · **Decision:** [D001](../decisions.md#d001) · **Status:** shipped

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/public.jsonl` — **1,126** rows |
| Model | `Qwen/Qwen3-4B-Thinking-2507`, INT8, vLLM |
| `max_tokens` | **8192** |
| Sampling | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Prompting | Separate MCQ / free-form system prompts; `\boxed{}` requested |

## Results

| Split | Count | Accuracy |
|-------|------:|---------:|
| **Overall** | 1,126 | **52.66%** |
| MCQ | 375 | **50.40%** |
| Free-form | 751 | **53.79%** |

vs 4k full-public run: **+11.9 pp overall**, **+24.3 pp MCQ**.

## Artifacts

- `data/full_public_8k.jsonl` — `id`, `is_mcq`, `correct`
- `data/full_public_8k.responses.jsonl` — `id`, `response`
- `data/full_public_8k_topics_weighted_v1.json` — topic breakdown (`scripts/topic_classify.py`)

## Deep analysis

[`analysis/baseline-public-8k.md`](../../analysis/baseline-public-8k.md) — format compliance, multi-blank gap, topic table.

## Takeaway

**Current inference baseline.** MCQ bottleneck is missing `\boxed{Letter}` (87% of wrong MCQ), not truncation. When boxed letter is present, ~**88.3%** MCQ accuracy.

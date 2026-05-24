# dev-007 — `max_tokens` 16384 (16k)

**Date:** 2026-05-23  
**Registry:** [dev-007](../experiments.md#dev-007) · **Roadmap:** [§1.1](../../roadmap.md#11-lift-max_tokens-to-16384--next--highest-priority)  
**Status:** done (dev validated; full public → pub-002 pending)

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/dev.jsonl` — 20% stratified, seed 42 (**225 rows**: 75 MCQ, 150 free-form) |
| Change | **`max_tokens` 8192 → 16384** only |
| Prompt / decoding | Baseline (pub-001 prompts); `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Model | `Qwen/Qwen3-4B-Thinking-2507`, bf16, `max_model_len=32768` — see [vLLM A100 profile](../../infra/vllm-inference-config.md#notebooksdevipynb--a100-optimized-load-7) |
| Notebook | `notebooks/dev.ipynb` — `MAX_TOKENS=16384`, `PROMPT_VARIANT="baseline"` |

## Results

| Split | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| MCQ | 53 | 75 | **70.67%** |
| Free-form | 82 | 150 | **54.67%** |
| **Overall** | 135 | 225 | **60.00%** |

## Comparison (same 20% dev slice unless noted)

| Metric | pub-001 (public 8k) | dev-005 (guided 8k) | dev-006 (concise 8k) | **dev-007 (16k)** |
|--------|--------------------:|--------------------:|---------------------:|------------------:|
| MCQ | 50.40% | 53.33% | 48.00% | **70.67%** |
| Free-form | 53.79% | 52.00% | 54.67% | **54.67%** |
| Overall | 52.66% | 52.44% | 52.44% | **60.00%** |

| Δ vs | Overall | MCQ |
|------|--------:|----:|
| dev-006 (8k concise, same slice) | **+7.56 pp** | **+22.67 pp** |
| dev-005 (8k guided, same slice) | **+7.56 pp** | **+17.34 pp** |
| pub-001 (8k public) | **+7.34 pp** | **+20.27 pp** |

Free-form is flat vs dev-006 (54.67%); nearly all lift is MCQ, consistent with truncation being the dominant 8k failure mode ([analysis doc](../../analysis/baseline-public-8k.md)).

## Artifacts

- `results/dev_results_baseline_16k.jsonl` (new naming in notebook §2)
- Colab run also saved as `results/dev_results_baseline.jsonl` on Drive (pre-rename)

## Follow-up

Run full public at 16k → **pub-002** before shipping as inference baseline.

## Takeaway

16k tokens on dev strongly validates roadmap §1.1: MCQ +20 pp vs pub-001, overall +7.3 pp. Matches hypothesis that truncation (44% of MCQ at 8k) was the bottleneck, not format or prompting.

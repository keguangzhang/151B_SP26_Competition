# dev-013-verify — Verification-forcing prompt (holdout_20p)

**Date:** 2026-05-28  
**Registry:** [dev-013-verify](../experiments.md#dev-013-verify)  
**Status:** rejected — **0.00 pp** overall vs [sft-eval-001](sft-eval-001-baseline-holdout-20p.md) on same 225-row slice

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/eval/holdout_20p.jsonl` — **20%** stratified, seed 42 (**225 rows**: 75 MCQ, 150 free-form) |
| Change | **`PROMPT_VARIANT="verify_prompt"`** — multi-blank `\boxed{}` format + verify-before-box nudge (substitute back, units/range/signs, independent check) |
| Decoding | `max_tokens=16384`, `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Model | `Qwen/Qwen3-4B-Thinking-2507`, bf16 — A100 |
| Notebook | `notebooks/dev.ipynb` — `SELF_CONSISTENCY=False`, `PHP_ENABLED=False`, `BUDGET_FORCING=False`, `SMOKE_TEST=False` |

### Prompt delta vs `multi_blank`

Math system prompt adds:

> Verify your answer before boxing it: substitute back, check units/range/signs, or confirm with an independent method when possible.

MCQ system prompt adds:

> Verify your choice before boxing it: substitute back, eliminate inconsistent options, or check constraints when possible.

Multi-blank user hint and `\boxed{}` layout unchanged from §1.3.

## Results

| Split | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| MCQ | 59 | 75 | **78.67%** |
| Free-form | 86 | 150 | **57.33%** |
| Multi-blank (2+ `[ANS]`) | 42 | 82 | **51.22%** |
| Single-blank | 44 | 68 | **64.71%** |
| **Overall** | 145 | 225 | **64.44%** |

## Comparison

| Anchor | Eval | N | Overall | MCQ | FF | Multi-blank | Single-blank |
|--------|------|--:|--------:|----:|---:|------------:|-------------:|
| [sft-eval-001](sft-eval-001-baseline-holdout-20p.md) | holdout_20p | 225 | **64.44%** | 77.33% | 58.00% | 52.44% | 64.71% |
| **dev-013-verify** (this run) | holdout_20p | 225 | **64.44%** | **78.67%** | **57.33%** | **51.22%** | **64.71%** |

| Δ vs sft-eval-001 (`multi_blank`, same 225 ids) | Overall | MCQ | FF | Multi-blank | Single-blank |
|--------------------------------------------------|--------:|----:|---:|------------:|-------------:|
| | **0.00 pp** | **+1.34 pp** | **−0.67 pp** | **−1.22 pp** | **0.00 pp** |

**Item-level:** 145/225 correct on both runs — identical headline accuracy. Verification nudge trades one MCQ gain for one free-form loss (net 0).

| Δ vs [dev-008](dev-008-multi-blank-16k.md) (10% slice, `multi_blank`) | Overall | Notes |
|------------------------------------------------------------------------|--------:|-------|
| **−0.74 pp** | 64.44% vs 65.18% | Different eval size (225 vs 112); not directly comparable |

## Artifacts

| Path | Role |
|------|------|
| `results/dev_results_verify_prompt_16k.jsonl` | Judged results (Colab Drive: `MyDrive/CSE151B/results/`) |
| `results/dev_results_verify_prompt_16k.responses.jsonl` | Generation checkpoint |

## Takeaway

Verification forcing on top of multi-blank format is **flat** on holdout_20p: same **64.44%** as the `multi_blank` anchor ([sft-eval-001](sft-eval-001-baseline-holdout-20p.md)). Sub-slice movement (+1 MCQ, −1 FF, −1 multi-blank) is within noise at this N. Do not ship `verify_prompt` for public/private; keep **`multi_blank`** as the inference prompt. See [D011](../decisions.md#d011--verify_prompt-rejected-on-holdout_20p).

## Follow-up

- No full-public run warranted unless a stronger verify variant is designed (e.g. MCQ-only verify clause).
- Reasoning-quality gains remain better targeted by SFT ([roadmap](../../roadmap.md) §2) than prompt-only verify nudges on a thinking model.

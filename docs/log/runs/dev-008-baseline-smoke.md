# dev-008-baseline-smoke — §1.3 multi-blank smoke (baseline arm)

**Date:** 2026-05-24  
**Registry:** [dev-008-baseline-smoke](../experiments.md#dev-008-baseline-smoke) · **Roadmap:** [§1.3](../../roadmap.md#13-multi-blank-free-form-structure--high-value-independent)  
**Status:** done (smoke reference for A/B vs `dev-008-smoke`)

## Setup

| Field | Value |
|-------|--------|
| Eval | **20** multi-blank free-form rows from `data/dev.jsonl` (10% stratified, seed 42) |
| Smoke ids | `21, 128, 139, 152, 158, 203, 217, 256, 293, 312, 391, 410, 422, 429, 475, 482, 489, 536, 545, 547` |
| Change | **Baseline** pub-001 prompts only (`PROMPT_VARIANT="baseline"`) |
| Decoding | `max_tokens=16384`, `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Model | `Qwen/Qwen3-4B-Thinking-2507`, bf16 — [A100 profile](../../infra/vllm-inference-config.md) |
| Notebook | `notebooks/dev.ipynb` — `SMOKE_TEST=True`, `SMOKE_N=20` |

## Results

| Split | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| Multi-blank (smoke slice) | 6 | 20 | **30.00%** |
| Free-form (same 20) | 6 | 20 | **30.00%** |
| **Overall** | 6 | 20 | **30.00%** |

## Context

- Control arm for roadmap §1.3: compare against `dev-008-smoke` (`PROMPT_VARIANT="multi_blank"`) on **identical ids**.
- pub-001 multi-blank on full public: **47.8%** (n=414) — not directly comparable at n=20; treat ±~10 pp as noise at this slice size.
- Prior broken multi_blank smoke (labeled `Answer N:` format) scored **0/20** due to judger extract mismatch, not model quality — see analysis of `data/dev_results_multi_blank_16k_smoke.jsonl`.

## Artifacts

- `results/dev_results_baseline_16k_smoke.jsonl` (Colab Drive: `MyDrive/CSE151B/results/`)

## Follow-up

Compared to [dev-008-smoke](dev-008-smoke.md): **30% → 40%** (+2 items, +10 pp on same ids). Full dev run pending.

## Takeaway

Baseline on the §1.3 smoke slice: **6/20 (30%)**. A/B control for the multi_blank prompt variant.

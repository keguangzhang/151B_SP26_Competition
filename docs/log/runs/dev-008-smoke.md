# dev-008-smoke — §1.3 multi-blank smoke (fixed prompt)

**Date:** 2026-05-24  
**Registry:** [dev-008-smoke](../experiments.md#dev-008-smoke) · **Roadmap:** [§1.3](../../roadmap.md#13-multi-blank-free-form-structure--high-value-independent)  
**Status:** smoke done — full dev → [dev-008](dev-008-multi-blank-16k.md)

## Setup

| Field | Value |
|-------|--------|
| Eval | **20** multi-blank free-form rows from `data/dev.jsonl` (same ids as [dev-008-baseline-smoke](dev-008-baseline-smoke.md)) |
| Smoke ids | `21, 128, 139, 152, 158, 203, 217, 256, 293, 312, 391, 410, 422, 429, 475, 482, 489, 536, 545, 547` |
| Change | **`PROMPT_VARIANT="multi_blank"`** — separate `\boxed{a}, \boxed{b}, ...` comma-separated (no `Answer N:` labels; judger-compatible) |
| Decoding | `max_tokens=16384`, `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Model | `Qwen/Qwen3-4B-Thinking-2507`, bf16 — [A100 profile](../../infra/vllm-inference-config.md) |
| Notebook | `notebooks/dev.ipynb` — `SMOKE_TEST=True`, `SMOKE_N=20` |

## Results

| Split | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| Multi-blank (smoke slice) | 8 | 20 | **40.00%** |
| Free-form (same 20) | 8 | 20 | **40.00%** |
| **Overall** | 8 | 20 | **40.00%** |

## Comparison (same 20 ids)

| Run | Multi-blank | Δ |
|-----|------------:|--:|
| [dev-008-baseline-smoke](dev-008-baseline-smoke.md) | 30.00% (6/20) | — |
| **dev-008-smoke** (this run) | **40.00% (8/20)** | **+10.0 pp** |
| dev-008-smoke v0 (broken `Answer N:` prompt) | 0.00% (0/20) | judger extract fail |

At n=20 one answer ≈ 5 pp; +10 pp = +2 items — promising but not conclusive. Roadmap expected +3–6 pp on full free-form.

## Artifacts

- `results/dev_results_multi_blank_16k_smoke.jsonl` (Colab Drive: `MyDrive/CSE151B/results/`)

## Follow-up

Full 10% dev recorded as [dev-008](dev-008-multi-blank-16k.md): **50% multi-blank (19/38)**, **65.18% overall**.

## Takeaway

Fixed multi-blank prompt beats baseline **30% → 40%** on smoke (+2 correct). Full dev confirms **50% multi-blank**, +4.5 pp overall vs 16k baseline on same slice.

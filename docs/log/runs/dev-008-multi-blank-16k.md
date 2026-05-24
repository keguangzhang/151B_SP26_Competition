# dev-008 — §1.3 multi_blank prompt (full 10% dev)

**Date:** 2026-05-24  
**Registry:** [dev-008](../experiments.md#dev-008) · **Roadmap:** [§1.3](../../roadmap.md#13-multi-blank-free-form-structure--high-value-independent)  
**Status:** done (10% dev validated; public eval pending)

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/dev.jsonl` — **10%** stratified, seed 42 (**112 rows**: 37 MCQ, 75 free-form) |
| Change | **`PROMPT_VARIANT="multi_blank"`** — separate `\boxed{a}, \boxed{b}, ...` comma-separated (judger-compatible; no `Answer N:` labels) |
| Decoding | `max_tokens=16384`, `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Model | `Qwen/Qwen3-4B-Thinking-2507`, bf16 — [A100 profile](../../infra/vllm-inference-config.md) |
| Notebook | `notebooks/dev.ipynb` — `SMOKE_TEST=False`, `DEV_FRACTION=0.10` |

## Results

| Split | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| MCQ | 29 | 37 | **78.38%** |
| Free-form | 44 | 75 | **58.67%** |
| Multi-blank | 19 | 38 | **50.00%** |
| Single-blank | 25 | 37 | **67.57%** |
| **Overall** | 73 | 112 | **65.18%** |

## Comparison

| Metric | pub-001 (public 8k) | dev-007 (20% dev, baseline 16k) | 10% dev baseline 16k† | **dev-008 (10% dev, multi_blank 16k)** |
|--------|--------------------:|--------------------------------:|----------------------:|---------------------------------------:|
| MCQ | 50.40% | 70.67% | ~75.68% | **78.38%** |
| Free-form | 53.79% | 54.67% | ~53.33% | **58.67%** |
| Multi-blank | 47.8% (n=414) | — | — | **50.00% (n=38)** |
| Overall | 52.66% | 60.00% | ~60.71% | **65.18%** |

†Unregistered Colab run: `PROMPT_VARIANT="baseline"`, `MAX_TOKENS=16384`, same 112-row dev slice (notebook output in `dev.ipynb`).

| Δ vs | Overall | Free-form | Multi-blank |
|------|--------:|----------:|------------:|
| 10% baseline 16k† | **+4.47 pp** | **+5.34 pp** | — |
| [dev-008-smoke](dev-008-smoke.md) (same prompt, n=20) | — | +18.67 pp | **+10.0 pp** |
| pub-001 multi-blank (public) | — | — | **+2.2 pp** |

Smoke → full dev: multi-blank **40% → 50%** (+10 pp on broader slice). Free-form lift (+5.3 pp vs 10% baseline 16k) exceeds roadmap §1.3 estimate (+3–6 pp).

## Artifacts

- `results/dev_results_multi_blank_16k.jsonl` (Colab Drive: `MyDrive/CSE151B/results/`)

Prior smoke runs: [dev-008-baseline-smoke](dev-008-baseline-smoke.md), [dev-008-smoke](dev-008-smoke.md).

## Follow-up

- **32k ablation:** [dev-009](dev-009-max-tokens-32k.md) — no lift vs this run; stay at 16k ([D008](../decisions.md#d008--32k-max_tokens-rejected-stay-at-16k)).
- Optional: register 10% baseline 16k as separate run for clean A/B on same slice.
- Run full public with multi_blank + 16k before shipping §1.3.
- Consider combining with §1.1 shipped baseline (16k + multi_blank) for submission path.

## Takeaway

§1.3 multi_blank prompt validated on full 10% dev: **+4.5 pp overall** vs 16k baseline on same slice, **50% multi-blank** (vs 47.8% pub-001). MCQ unchanged by design; lift is free-form. Prompt fix (judger-compatible `\boxed{}` list) was required — labeled `Answer N:` format scored 0/20 in smoke.

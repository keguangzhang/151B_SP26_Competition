# dev-014-precision — Exact-form / no-round prompt (holdout_20p)

**Date:** 2026-05-31  
**Registry:** [dev-014-precision](../experiments.md#dev-014-precision)  
**Notebook `RUN_ID`:** `dev-precision-prompt`  
**Status:** done — **+1.78 pp** overall vs [sft-eval-001](sft-eval-001-baseline-holdout-20p.md) (`multi_blank` anchor, same 225 ids)

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/eval/holdout_20p.jsonl` — **20%** stratified, seed 42 (**225 rows**: 75 MCQ, 150 free-form) |
| Change | **`PROMPT_VARIANT="precision"`** — exact-form / no-round clause + grader-format hygiene + §1.3 multi-blank `\boxed{}` layout |
| Decoding | `max_tokens=16384`, `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Model | `Qwen/Qwen3-4B-Thinking-2507`, bf16 — A100 |
| Notebook | `notebooks/dev.ipynb` — `SELF_CONSISTENCY=False`, `PHP_ENABLED=False`, `BUDGET_FORCING=False`, `SMOKE_TEST=False` |

### Motivation

[`scripts/measure_precision_strict.py`](../../../scripts/measure_precision_strict.py) on public FF: **67/751** wrong items fail **only** because the model rounds (e.g. `62.78` vs gold `62.777778`); the grader accepts exact forms via its symbolic path at **1e-8** relative tolerance. The precision variant instructs fractions/radicals when possible, ≥8 sig figs if a decimal is unavoidable, and never to round — plus box-hygiene rules (no units in box, bracketed tuples, probability as decimal not `%`).

### Prompt delta vs `multi_blank`

Math system prompt replaces the generic multi-blank instruction with:

- **Exact-form clause:** report `\boxed{565/9}` or `\boxed{3\sqrt{10}/10}` instead of rounded decimals; decimals only when no exact form exists (≥8 sig figs, no rounding).
- **Grader-format clause:** value-only boxes (no units), probability as decimal (`0.5` not `50%`), tuples/intervals in one box with brackets, Yes/No without extra words.
- **Multi-blank layout:** unchanged from §1.3 (separate `\boxed{}` per `[ANS]`, comma-separated, no labels).

MCQ system prompt: **unchanged** from baseline (`_MCQ_PRECISION = _MCQ_BASELINE`).

## Results

| Split | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| MCQ | 57 | 75 | **76.00%** |
| Free-form | 92 | 150 | **61.33%** |
| Multi-blank (2+ `[ANS]`) | 45 | 82 | **54.88%** |
| Single-blank | 47 | 68 | **69.12%** |
| **Overall** | 149 | 225 | **66.22%** |

## Comparison

| Anchor | Eval | N | Overall | MCQ | FF | Multi-blank | Single-blank |
|--------|------|--:|--------:|----:|---:|------------:|-------------:|
| [sft-eval-001](sft-eval-001-baseline-holdout-20p.md) | holdout_20p | 225 | **64.44%** | 77.33% | 58.00% | 52.44% | 64.71% |
| [dev-013-verify](dev-013-verify-holdout-20p.md) | holdout_20p | 225 | **64.44%** | 78.67% | 57.33% | 51.22% | 64.71% |
| **dev-014-precision** (this run) | holdout_20p | 225 | **66.22%** | **76.00%** | **61.33%** | **54.88%** | **69.12%** |

| Δ vs sft-eval-001 (`multi_blank`, same 225 ids) | Overall | MCQ | FF | Multi-blank | Single-blank |
|--------------------------------------------------|--------:|----:|---:|------------:|-------------:|
| | **+1.78 pp** | **−1.33 pp** | **+3.33 pp** | **+2.44 pp** | **+4.41 pp** |

| Δ vs dev-013-verify (same slice) | Overall | MCQ | FF | Multi-blank | Single-blank |
|----------------------------------|--------:|----:|---:|------------:|-------------:|
| | **+1.78 pp** | **−2.67 pp** | **+4.00 pp** | **+3.66 pp** | **+4.41 pp** |

**Item-level vs sft-eval-001:** **149/225** correct vs **145/225** — **+4 net** (MCQ −1, free-form +5: multi-blank +2, single-blank +3).

| Δ vs [dev-012-sc5](dev-012-sc5.md) (10% slice, SC K=5) | Overall | Notes |
|--------------------------------------------------------|--------:|-------|
| **−0.74 pp** | 66.22% vs 66.96% | Different eval size (225 vs 112) and method (prompt-only vs SC); not directly comparable |

## Artifacts

| Path | Role |
|------|------|
| `results/dev_results_precision_16k.jsonl` | Judged results (Colab Drive: `MyDrive/CSE151B/results/`) |
| `results/dev_results_precision_16k.responses.jsonl` | Generation checkpoint |

Local repo copy not synced at log time.

## Takeaway

Exact-form / no-round prompting on top of the shipped multi-blank layout is the **first prompt-only holdout_20p lift** since token scaling: **66.22% overall (+1.78 pp, +4 items)** vs the `multi_blank` anchor. Gains concentrate in **free-form** (+3.33 pp, especially single-blank +4.41 pp) as hypothesized from precision-strict error analysis; **MCQ regresses −1.33 pp** (likely noise at n=75). Do **not** ship on holdout alone — run **full public** (`pub-004` or similar) to confirm before replacing `multi_blank` in the submission path. If public confirms, this isolates the “precision” component conflated in the pub-001→pub-002 delta ([roadmap §1.4](../../roadmap.md)).

## Follow-up

- Full-public eval with `PROMPT_VARIANT="precision"` at 16k (compare vs pub-002 / pub-003 baselines).
- ~~Item-level diff vs sft-eval-001 jsonl when Drive artifacts are synced locally.~~ → done against `full_public_16k` baseline (same 225 ids): **[`docs/analysis/dev-014-error-analysis.md`](../../analysis/dev-014-error-analysis.md)** — 76 wrong items bucketed, motivates `precision_v2` variant (8 clauses, +8 to +15 items estimated).
- Run `PROMPT_VARIANT="precision_v2"` on holdout_20p; A/B vs this run before any public eval.
- Optional: MCQ-only ablation (precision math + baseline MCQ already in code) if MCQ regression persists on public.

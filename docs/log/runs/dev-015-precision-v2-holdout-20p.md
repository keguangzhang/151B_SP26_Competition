# dev-015-precision-v2 — Composite 6-fix prompt on holdout_20p

**Date:** 2026-05-31
**Registry:** [dev-015-precision-v2](../experiments.md#dev-015-precision-v2) · **Decision:** [D013](../decisions.md#d013--precision_v2--precision_v3-rejected-precision-v1-is-the-prompt-only-ceiling) · **Status:** **rejected** — −2 items vs `precision` (v1); inside sampling noise

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/eval/holdout_20p.jsonl` — 225 rows (75 MCQ, 150 free-form), seed 42 |
| Change | **`PROMPT_VARIANT="precision_v2"`** — adds 7 clauses to `precision` derived from [dev-014 error analysis](../../analysis/dev-014-error-analysis.md): final-block-only boxes, plain `, ` separator, multi-select `\boxed{AB}`, integer no-pad, domain-conditional decimal, slot-order + letter-only, budget hint at 12k chars |
| Decoding | `max_tokens=16384`, `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Model | `Qwen/Qwen3-4B-Thinking-2507`, bf16 — A100 |
| Notebook | `notebooks/dev.ipynb` — `SELF_CONSISTENCY=False`, `PHP_ENABLED=False`, `BUDGET_FORCING=False` |
| MCQ prompt | **unchanged** (`_MCQ_PRECISION_V2 = _MCQ_BASELINE`) — same as v1 |

System-prompt length: **2529 chars** (v1 was ~1300).

## Results

| Split | Correct | N | Accuracy | Δ vs [dev-014-precision](dev-014-precision-holdout-20p.md) |
|-------|--------:|--:|--------:|-----------------------------------------------------------:|
| MCQ | 55 | 75 | 73.33% | **−2.67 pp** |
| Free-form | 92 | 150 | 61.33% | **0.00 pp** |
| Multi-blank | 44 | 82 | 53.66% | **−1.22 pp** |
| Single-blank | 48 | 68 | 70.59% | **+1.47 pp** |
| **Overall** | **147** | 225 | **65.33%** | **−0.89 pp** |

## Item-level diff vs `precision` (v1)

10 fixed, 12 broken, **22 items flipped, −2 net**.

| Bucket | Fixed | Broken |
|--------|------:|------:|
| MCQ (identical prompt — pure noise) | 4 | 6 |
| Multi-blank FF | 2 | 3 |
| Single-blank FF | 4 | 3 |

**MCQ flips are 100% sampling noise** — both v1 and v2 use `_MCQ_BASELINE` verbatim. 10/75 MCQ items flipped between runs (~13% churn) consistent with `temp=0.6` variance at n=75.

## Clause-level audit

Walked every item targeted by a new clause:

| Targeted | ids | v2 outcome |
|----------|-----|------------|
| Final-block / no boxed bullets | 44, 250 | Both ✗→✗ (still 3/4 boxes vs ans_count=2/4) — clause ignored |
| `\quad` separator | 547 | ✗→✗ — newly truncated this run (0 boxes) |
| Multi-select `AB` | 545 | ✗→✗ — newly truncated this run (0 boxes) |
| Integer no-pad | 20 | ✗→✗ — still emitted 15 `.000`-padded boxes |
| Domain decimal | 44, 482, 495, 509, 754, 806, 895 | **id=495 ✓ FIXED**; others ✗→✗ |
| Slot order / letter-only | 80, 250, 358, 391 | All ✗→✗ |
| Budget hint at 12k | 148, 762, 100 | All ✗→✗ (148 = 41k chars, 762 = 45k) |

**Only 1 confirmed clause-attributable fix** (id=495, domain-decimal). Remaining 9 fixes and all 12 breaks indistinguishable from sampling churn.

## Artifacts

| Path | Role |
|------|------|
| `data/dev_results_precision_v2_16k.jsonl` | Judged results (local copy) |

## Takeaway

Composite 7-clause additive prompt **did not move the headline** (−2 items, well inside the ±2 noise floor at n=225, temp=0.6). Per-clause audit shows the model ignored most format/order/budget instructions when buried in a 2500-char system prompt — only the domain-decimal clause demonstrably fired (1 item). The "MCQ regression" is artifactual: the MCQ system prompt is byte-identical to v1's, and the 10-item MCQ flip count matches expected sampling churn. Rejected; led to slimmer [dev-016 (`precision_v3`)](dev-016-precision-v3-holdout-20p.md) which dropped 4 ignored clauses but also landed at 147/225.

## Follow-up

- See [dev-016-precision-v3](dev-016-precision-v3-holdout-20p.md) for the slimmed retest.
- Joint write-up in [`docs/analysis/dev-014-error-analysis.md`](../../analysis/dev-014-error-analysis.md) (retrospective section).
- Decision recorded as [D013](../decisions.md#d013--precision_v2--precision_v3-rejected-precision-v1-is-the-prompt-only-ceiling).

# dev-005 — Guided decoding, 20% dev slice

**Registry:** [dev-005](../experiments.md#dev-005) · **Decision:** [D003](../decisions.md#d003) · **Status:** done

## Setup

| Field | Value |
|-------|--------|
| Eval | 20% stratified dev (`DEV_FRACTION=0.20`, seed 42) — **225 rows** (75 MCQ, 150 free-form) |
| Config | Same as [dev-004](dev-004-guided-decoding-10pct.md) |
| Notebook | `notebooks/dev.ipynb` |

## Results

| Split | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| MCQ | 40 | 75 | **53.33%** |
| Free-form | 78 | 150 | **52.00%** |
| **Overall** | **118** | **225** | **52.44%** |

## Comparison

- pub-001 MCQ: **50.40%** on 375 items — guided decoding ~flat at larger dev n.
- Expected +10–20 pp from roadmap §1.1 **not observed** on dev.

## Takeaway

Larger dev slice still shows no meaningful lift. Prioritize **full `public.jsonl`** eval or alternate §1.1 implementations (two-pass tail, `guided_choice`) before more dev tuning.

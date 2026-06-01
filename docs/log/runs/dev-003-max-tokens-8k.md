# dev-003 — `max_tokens` 8192

**Registry:** [dev-003](../experiments.md#dev-003) · **Decision:** [D001](../decisions.md#d001) · **Status:** shipped

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/dev.jsonl` (112 rows) |
| Change | **`max_tokens` 4096 → 8192** only |
| Prompt / decoding | Same as dev-001 (`0.6` / `0.95`) |
| Notebook | `notebooks/dev.ipynb` |

## Results

| Split | Accuracy | Δ vs dev-001 |
|-------|----------|--------------|
| MCQ | **54.05%** | **+24.32 pp** |
| Free-form | **54.67%** | +2.67 pp |
| **Overall** | **54.46%** | **+9.82 pp** |

## Follow-up

Confirmed on full public → [pub-001](pub-001-full-public-8k.md). Shipped as inference baseline.

## Takeaway

Large MCQ gain on dev; primary lever before format-compliance work. Truncation largely addressed at 8k on public (see analysis doc).

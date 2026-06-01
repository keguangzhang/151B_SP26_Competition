# dev-002 — MCQ prompt + temperature (§1.3)

**Registry:** [dev-002](../experiments.md#dev-002) · **Decision:** [D002](../decisions.md#d002) · **Status:** rejected

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/dev.jsonl` (112 rows) |
| Change | Stronger `\boxed{}` final-line clause; ~1500-token reasoning hint on MCQ |
| Decoding | MCQ-only: `temperature=0.2`, `top_p=0.9`; free-form unchanged (`0.6` / `0.95`) |
| Roadmap | [roadmap.md](../../roadmap.md) §1.3 |

## Results

| Split | Accuracy | Δ vs dev-001 |
|-------|----------|--------------|
| MCQ | **27.03%** | −2.70 pp |
| Free-form | **54.67%** | +2.67 pp |
| **Overall** | **45.54%** | +0.90 pp |

## Takeaway

Essentially flat overall at n=112. Tested on 4k-era baseline where truncation dominated; **re-scope on 8k** before final rejection — see roadmap §1.3.

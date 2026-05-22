# dev-004 — Guided decoding, 10% dev slice

**Registry:** [dev-004](../experiments.md#dev-004) · **Decision:** [D003](../decisions.md#d003) · **Status:** done

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/dev.jsonl` (112 rows) |
| `max_tokens` | 8192 |
| MCQ only | `StructuredOutputsParams(regex=r"(?s:.)*\\boxed{[A-J]}")` |
| Free-form | Unchanged vs 8k dev baseline |

## Results

| Split | Accuracy | Δ vs dev-003 (8k dev) |
|-------|----------|------------------------|
| MCQ | **51.35%** | −2.70 pp |
| Free-form | **54.67%** | 0.00 pp |
| **Overall** | **53.57%** | −0.89 pp |

## Takeaway

n=37 MCQ is too small for ±3 pp conclusions. No free-form regression. **Run full public** (375 MCQ) before adopting or rejecting §1.1.

## Notes

- Regex may constrain too late in the thinking trace.
- Compare emission rate of `\boxed{Letter}` vs pub-001 when public run exists.

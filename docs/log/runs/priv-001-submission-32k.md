# priv-001 — First private submission (32k baseline)

**Date:** 2026-05-30  
**Registry:** [priv-001](../experiments.md)  
**Status:** done (leaderboard interim score recorded)

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/private.jsonl` — **943** rows (300 MCQ, 643 free-form) |
| Model | `Qwen/Qwen3-4B-Thinking-2507` |
| Change | Same inference path as **pub-003**: `max_tokens=32768`, adaptive multi-blank prompts |
| Artifact | `results/submission_32k.csv` |
| Analysis | `notebooks/submission_analysis.ipynb`; [`analysis/private-submission-32k-priv-001.md`](../../analysis/private-submission-32k-priv-001.md) |

## Results

| Split | N | Accuracy | Notes |
|-------|--:|---------:|-------|
| **Overall (leaderboard)** | ~283† | **48.00%** | Interim ~30% of private until finals |
| MCQ | — | — | No local labels |
| Free-form | — | — | No local labels |

† Course leaderboard evaluates a subset; full private n=943.

## Comparison vs pub-003 (public 32k)

| Metric | pub-003 | priv-001 LB | Δ |
|--------|--------:|------------:|--:|
| Overall | 65.81% | 48.00% | **−17.8 pp** |
| Bucket-model expectation‡ | 63.24% | 48.00% | **−15.2 pp** |

‡ Private bucket mix × public per-bucket accuracy ([analysis doc](../../analysis/private-submission-32k-priv-001.md#gap-decomposition-format-risk-buckets)).

## Format QA (full private, pre-grade)

| Metric | Value |
|--------|------:|
| Integrity | PASS (943/943) |
| Truncated | 29 (3.1%) |
| FF misformat | 45 |
| MCQ boxed | 97.7% |
| FF count_ok | 93.0% |
| Flagged ids | 75 → `results/submission_flags.jsonl` |

## Takeaway

First private score is **~18 pp below pub-003** on public. Format/truncation explains **~3 pp** of that; the rest is reasoning/generalization on clean traces. Next lever: **SFT weak topics**, not format fixes or token budget alone.

## Follow-up

- Update `LEADERBOARD_SCORE` in `submission_analysis.ipynb` §1 after each upload.
- Track finals score on full private when released.

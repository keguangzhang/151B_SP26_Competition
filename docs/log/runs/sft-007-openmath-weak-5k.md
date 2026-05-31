# sft-007 — OpenMath weak-topic + anchor ~5k (gentle LoRA)

**Date:** 2026-05-30  
**Registry:** [sft-007](../experiments.md#sft-007)  
**Design:** [2026-05-30-sft-007-openmath-weak-topic-anchor-design.md](../../superpowers/specs/2026-05-30-sft-007-openmath-weak-topic-anchor-design.md)  
**Status:** **planned** — corpus build + train/eval on Colab A100 pending

## Goal

~5k all-OpenMath SFT (geometry / probability-statistics / trigonometry + general anchor) with **gentle LoRA** (lr 5e-6, α=32, dropout 0.1, `MAX_SEQ_LENGTH=16384`). Ship private only if holdout gates pass and a weak-topic dev slice improves vs pub-002.

## Setup

| Field | Value |
|-------|--------|
| **Corpus build** | `scripts/build_sft_corpus_sft007.py` or `notebooks/sft_data_prep.ipynb` §10.5 (`RUN_SFT007_BUILD=True`) |
| **Corpus** | `data/sft_corpus_sft007_openmath_5k.jsonl` + manifest |
| **Train** | `notebooks/sft_train.ipynb` — `RUN_NAME=openmath_sft007_5k` |
| **Inline eval** | Same notebook §8 — `RUN_EVAL_AFTER_TRAIN=True` |
| **Fallback submit** | pub-002 base (`results/submission.csv`) |

### Target slice quotas (manifest records actuals)

| Slice | Target |
|-------|--------|
| probability/stats | ~1300 |
| geometry | ~1200 |
| trigonometry | ~1000 (+ backfill rule if short) |
| general anchor | ~1500 |

### Training (sft-007 vs sft-006)

| Param | sft-006 | sft-007 |
|-------|---------|---------|
| `LEARNING_RATE` | 1e-5 | **5e-6** |
| `LORA_ALPHA` | 64 | **32** |
| `LORA_DROPOUT` | 0.05 | **0.1** |
| `MAX_SEQ_LENGTH` | 8192 | **16384** |

## Eval protocol (§8 in `sft_train.ipynb`)

| Slice | JSONL | Role |
|-------|-------|------|
| holdout | `data/eval/holdout_20p.jsonl` (225) | **Gate:** overall ≥ 64.44%, MCQ ≥ 77% |
| geometry | `data/eval/geometry_dev.jsonl` (133) | Upside vs pub-002 ~53.38% |
| prob_stats | `data/eval/prob_stats_dev.jsonl` (205) | Upside vs pub-002 ~49.8% (full-public topic) |

Watch sets (holdout only, descriptive): Q4 long, multi-blank ≥3.

**Artifacts:** `results/sft_eval/openmath_sft007_5k/eval_{slice}_0.*` on Drive.

## Results

*(Fill after Colab run.)*

| Slice | Overall | MCQ | Notes |
|-------|---------|-----|-------|
| holdout | — | — | gate |
| geometry_dev | — | — | vs 53.38% |
| prob_stats_dev | — | — | vs 49.8% |

## Decision

*(Pending §8 gate + dev-slice upside.)*

- **Ship LoRA** if gates pass and geometry or prob_stats improves.
- **Keep pub-002** if gates fail or flat with no topical gain.

## Local prep (done in repo)

- `scripts/build_eval_prob_stats_set.py` → `data/eval/prob_stats_dev.jsonl`
- `scripts/build_sft_corpus_sft007.py`, `scripts/openmath_qualify.py`
- `notebooks/sft_train.ipynb` §8 inline eval (no `sft_eval.ipynb` changes)

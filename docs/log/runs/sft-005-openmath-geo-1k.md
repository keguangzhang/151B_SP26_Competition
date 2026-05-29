# sft-005 — OpenMathReasoning geometry-only 1k (narrow-domain SFT probe)

**Date:** 2026-05-28  
**Registry:** [sft-005](../experiments.md#sft-005)  
**Status:** done — geometry_dev + holdout_20p eval

## Hypothesis

1k **geometry-only** OpenMathReasoning traces (12k–28k chars) can shift accuracy on the full-public geometry dev set (~133 ids). High signal density (~1000:133 train:eval). If this fails, narrow-domain SFT is likely not the path.

**Context:** [sft-003](sft-003-openmath-1k.md) mixed 1k was regressive on holdout (−2.66 pp, trace mean 14.1k→10.1k). sft-005 tests whether **geometry-only** targeting lifts the geometry slice without claiming overall lift.

## Setup

| Field | Value |
|-------|--------|
| **Corpus** | `data/sft_corpus_openmath_geo_1k.jsonl` — build via `notebooks/sft_data_prep.ipynb` §10.3 |
| **Manifest** | `data/sft_corpus_openmath_geo_1k_manifest.json` (`corpus_id: sft-005`) |
| **Train** | `notebooks/sft_train.ipynb` — `RUN_NAME=openmath_geo_1k`, same LoRA config as sft-003 |
| **Primary eval** | `notebooks/sft_eval.ipynb` — `EVAL_SLICE=geometry`, `SFT_RUN_NAME=openmath_geo_1k` |
| **Eval set** | `data/eval/geometry_dev.jsonl` — 133 rows (44 MCQ, 89 free-form) |
| **Guardrail eval** | `EVAL_SLICE=holdout` on `holdout_20p.jsonl` (225 rows) |
| Dataset | `nvidia/OpenMathReasoning` `cot`; geometry keyword filter; pass_rate ∈ [0.05, 0.70] |
| Trace band | 12,000–28,000 chars (above sft-003 8k floor) |
| Model | `Qwen/Qwen3-4B-Thinking-2507` + LoRA |
| Prompt / decoding | `multi_blank`, `MAX_TOKENS=16384`, same as pub-002 |

## Baseline anchors (geometry)

| Anchor | Eval set | N | Geometry acc | Notes |
|--------|----------|--:|-------------:|-------|
| pub-002 | baseline topic classifier | 115 | **50.4%** | [`baseline-public-16k.md`](../../analysis/baseline-public-16k.md) |
| pub-002 | `geometry_dev.jsonl` (AM-Qwen regex) | 133 | **53.38%** | Scored from `data/full_public_16k.jsonl` filtered to manifest ids |

### pub-002 on `geometry_dev.jsonl` (primary eval baseline)

Source: `data/full_public_16k.jsonl` (pub-002, 16k tokens, same prompting as sft-005 eval). All 133 manifest ids present.

| Split | Correct | N | Accuracy |
|-------|--------:|--:|---------:|
| MCQ | 30 | 44 | **68.18%** |
| Free-form | 41 | 89 | **46.07%** |
| **Overall** | 71 | 133 | **53.38%** |

The topic-classifier geometry slice (115 rows, 50.4%) and `geometry_dev.jsonl` (133 rows, 53.38%) use different id sets — compare sft-005 against the **geometry_dev** row above.

```bash
python -c "
import json
from pathlib import Path
REPO = Path('.')
manifest = json.load(open(REPO / 'data/eval/watch_manifest.json'))
geo_ids = set(manifest['watch']['geometry']['ids'])
rows = [json.loads(l) for l in open(REPO / 'data/full_public_16k.jsonl')]
sub = [r for r in rows if r['id'] in geo_ids]
acc = sum(r['correct'] for r in sub) / len(sub) * 100
print(f'pub-002 geometry_dev: {sum(r[\"correct\"] for r in sub)}/{len(sub)} = {acc:.2f}%')
"
```

## Results

**Eval:** `notebooks/sft_eval.ipynb` — `EVAL_SLICE=geometry`, `openmath_geo_1k/final_adapter`, step 0, Colab A100 bf16, 2026-05-28.  
**Artifacts:** Drive `results/sft_eval/openmath_geo_1k/eval_0.json(l)`

### Primary — geometry_dev (`EVAL_SLICE=geometry`)

| Split | Correct | N | Accuracy | Δ vs pub-002 geo |
|-------|--------:|--:|---------:|-----------------:|
| MCQ | 33 | 44 | **75.00%** | **+6.82 pp** |
| Free-form | 40 | 89 | **44.94%** | −1.13 pp |
| **Overall** | 73 | 133 | **54.89%** | **+1.51 pp** |

| Metric | Value | Notes |
|--------|------:|-------|
| Multi-blank (2+ `[ANS]`) | 47.50% (19/40) | |
| Single-blank | 42.86% (21/49) | |
| MCQ `\boxed{}` emission | 90.91% | |
| Mean response length | 12,765 chars | +41.8% vs 9k baseline ref in notebook |

**Verdict (primary):** **Flat** (+1.51 pp overall; gate: −2 to +5 pp). MCQ lift (+6.82 pp) offset by free-form regression (−1.13 pp). Not a clear win; not a fail.

### Watch subsets (geometry_dev ∩ watch ids)

Only ids present in `geometry_dev.jsonl` are scored when `EVAL_SLICE=geometry`.

| Watch | Correct | N | Accuracy | Notes |
|-------|--------:|--:|---------:|-------|
| Geometry (primary) | 73 | 133 | **54.89%** | Same as overall |
| Q4 long (≥435 chars) | 1 | 3 | 33.33% | 3/30 q4_long ids in geometry_dev |
| Multi-blank ≥3 | 0 | 1 | 0.00% | 1/20 mb≥3 ids in geometry_dev |

### Guardrail — holdout_20p (`EVAL_SLICE=holdout`)

**Eval:** `notebooks/sft_eval.ipynb` — `EVAL_SLICE=holdout`, same checkpoint, Colab A100 bf16, 2026-05-28.  
**Artifacts:** Drive `results/sft_eval/openmath_geo_1k/eval_0.json(l)` (overwrites geometry run unless renamed on Drive)

| Split | Correct | N | Accuracy | Δ vs [sft-eval-001](sft-eval-001-baseline-holdout-20p.md) |
|-------|--------:|--:|---------:|----------------------------------------------------------:|
| MCQ | 58 | 75 | **77.33%** | **0.00 pp** |
| Free-form | 87 | 150 | **58.00%** | **0.00 pp** |
| **Overall** | 145 | 225 | **64.44%** | **0.00 pp** |

| Metric | Value | Notes |
|--------|------:|-------|
| Multi-blank (2+ `[ANS]`) | 52.44% (43/82) | 0.00 pp vs sft-eval-001 |
| Single-blank | 64.71% (44/68) | 0.00 pp vs sft-eval-001 |
| MCQ `\boxed{}` emission | 96.00% | +5.33 pp vs sft-eval-001 (90.67%) |
| Mean response length | 10,096 chars | +12.2% vs 9k ref; **−28.5%** vs sft-eval-001 (14,124) |

### Watch sets (holdout_20p)

| Watch | Correct | N | Accuracy | Δ vs sft-eval-001 |
|-------|--------:|--:|---------:|--------------------:|
| Geometry (holdout ∩ geo ids) | 12 | 21 | **57.14%** | — |
| Q4 long (≥435 chars) | 12 | 30 | **40.00%** | −3.33 pp |
| Multi-blank ≥3 | 8 | 20 | **40.00%** | 0.00 pp |

**Verdict (guardrail):** **Pass** MCQ gate (0.00 pp vs anchor; fail threshold −3 pp). **Flat** overall — **identical** 145/225 to base model ([sft-eval-001](sft-eval-001-baseline-holdout-20p.md)) and [sft-002a](sft-002a-openr1-1k.md); unlike [sft-003](sft-003-openmath-1k.md) (−2.66 pp). Shorter traces than base without accuracy loss.

### Combined go/no-go

| Gate | Result |
|------|--------|
| Primary geometry (+1.51 pp, flat band) | **Flat** |
| Holdout MCQ (≥ −3 pp vs sft-eval-001) | **Pass** (0.00 pp) |
| Holdout overall regression | **None** (matches 64.44%) |

**Overall:** Narrow-domain geo SFT gives a small geometry slice lift with **no holdout harm** — inconclusive for scaling (not a +5 pp win), but **not** a fail. Next: 2 epochs or 2k geo rows if continuing geo SFT; rename Drive `eval_0.*` per slice if keeping both artifacts.

## Decision gates

| Outcome | Geometry Δ vs pub-002 on `geometry_dev` | Interpretation |
|---------|----------------------------------------:|----------------|
| **Win** | ≥ +5 pp | Domain SFT works — scale to multi-topic targeted mix |
| **Flat** | −2 to +5 pp | Inconclusive — try 2 epochs or 2k geo rows |
| **Fail** | ≤ −3 pp geo **or** holdout MCQ −3 pp | Narrow SFT hurts or doesn't transfer — deprioritize SFT |

## Artifacts

- Corpus: `notebooks/sft_data_prep.ipynb` §10.3 (`RUN_OPENMATH_GEO_1K=True`)
- Geometry eval: `python scripts/build_eval_geometry_set.py --force`
- Checkpoint: `checkpoints/openmath_geo_1k/final_adapter`
- Results: `results/sft_eval/openmath_geo_1k/eval_*.json(l)`

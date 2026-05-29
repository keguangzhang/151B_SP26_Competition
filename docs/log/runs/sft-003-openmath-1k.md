# sft-003 — OpenMathReasoning 1k probe (Fallback A)

**Date:** 2026-05-27  
**Registry:** [sft-003](../experiments.md#sft-003) · **Plan:** [Fallback A: sft-003](../../sft/pipeline.md#fallback-a-sft-003--nvidiaopenmathreasonin)  
**Status:** done — **regressive** (−2.66 pp vs base model; do not scale)

## Setup

| Field | Value |
|-------|--------|
| **Training** | `notebooks/sft_train.ipynb` — `RUN_NAME=openmath_1k` |
| Corpus | `data/sft_corpus_openmath_1k.jsonl` (1,000 rows, `nvidia/OpenMathReasoning` `cot` split) |
| Base model | `Qwen/Qwen3-4B-Thinking-2507` |
| LoRA | r=32, α=64, dropout=0.05; targets q/k/v/o + MLP; **bf16 LoRA** |
| Train | `LEARNING_RATE=1e-5`, `PER_DEVICE_BATCH=1`, `GRAD_ACCUM=16`, `MAX_SEQ_LENGTH=8192`, **`NUM_TRAIN_EPOCHS=1`** |
| Checkpoint | `checkpoints/openmath_1k/final_adapter` (Drive: `MyDrive/CSE151B/checkpoints/openmath_1k/final_adapter`) |
| **Eval** | `notebooks/sft_eval.ipynb` — `EVAL_STEP=0`, `PROMPT_VARIANT=multi_blank`, `MAX_TOKENS=16384` |
| Eval set | `data/eval/holdout_20p.jsonl` — **225 rows** (75 MCQ, 150 free-form), seed 42 |
| Inference | vLLM bf16 + LoRA on A100; decoding `temperature=0.6`, `top_p=0.95`, `top_k=20` |

## Results (holdout_20p)

| Split | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| MCQ | 57 | 75 | **76.00%** |
| Free-form | 82 | 150 | **54.67%** |
| Multi-blank (2+ `[ANS]`) | 41 | 82 | **50.00%** |
| Single-blank | 41 | 68 | **60.29%** |
| **Overall** | 139 | 225 | **61.78%** |

### Watch sets (`data/eval/watch_manifest.json`)

| Watch | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| Q4 long (≥435 chars) | 12 | 30 | **40.00%** |
| Multi-blank ≥3 | 8 | 20 | **40.00%** |

### SFT health metrics

| Metric | Value |
|--------|------:|
| MCQ `\boxed{Letter}` emission | **98.67%** |
| Mean response length | **10,135** chars (**+12.6%** vs `BASELINE_MEAN_RESPONSE_LEN=9000`) |

## Comparison (decision gate)

| Anchor | Eval | N | Overall | MCQ | FF | Multi-blank | Mean len |
|--------|------|--:|--------:|----:|---:|------------:|---------:|
| [sft-eval-001](sft-eval-001-baseline-holdout-20p.md) — base | holdout_20p | 225 | 64.44% | 77.33% | 58.00% | 52.44% | 14,124 chars |
| [sft-002a](sft-002a-openr1-1k.md) — OpenR1 1k | holdout_20p | 225 | 64.44% | 77.33% | 58.00% | 53.66% | 9,595 chars |
| **sft-003** (this run) — OpenMath 1k | holdout_20p | 225 | **61.78%** | **76.00%** | **54.67%** | **50.00%** | **10,135** chars |

| Δ vs base model (sft-eval-001) | Overall | MCQ | FF | Multi-blank | Single-blank | Q4 long | MB≥3 | Mean len |
|-------------------------------|--------:|----:|---:|------------:|-------------:|--------:|-----:|---------:|
| sft-003 LoRA | **−2.66 pp** | −1.33 pp | −3.33 pp | −2.44 pp | −2.95 pp | −3.33 pp | 0.00 pp | −28.2% (10.1k vs 14.1k) |

| Δ vs sft-002a (OpenR1 1k) | Overall | MCQ | FF | Multi-blank |
|--------------------------|--------:|----:|---:|------------:|
| sft-003 LoRA | **−2.66 pp** | −1.33 pp | −3.33 pp | −3.66 pp |

## Takeaway

OpenMathReasoning 1k × 1 epoch is **regressive** vs both the base model and the OpenR1 run: overall **61.78%** vs **64.44%** for both prior anchors (−2.66 pp). Every split declined. Format health is good (MCQ emission 98.67%), but response length dropped from base 14.1k → 10.1k chars (−28%), suggesting the model learned shorter trace patterns from competition-math training data. This exceeds the −20% warning threshold for trace-style compression.

**Decision:** Both sft-002a and sft-003 are flat/regressive. Per pipeline, this triggers **Fallback B (sft-004 — s1K-1.1)** or pivoting to inference-side gains (SC K=5, stronger prompts).

**Recommended next steps** (per pipeline §fallback):

1. **sft-004** (`simplescaling/s1K-1.1`, 1000 curated traces) — last SFT fallback before pivoting strategy.
2. **SC K=5 on base model** via `holdout_10p` (cheap; already shown +1.78 pp in dev-012).
3. Consider whether trace-length collapse is addressable with data filtering (longer traces from OpenMath, length-conditioned sampling).

## Artifacts (Colab Drive)

| Path | Role |
|------|------|
| `checkpoints/openmath_1k/final_adapter` | LoRA weights |
| `results/sft_eval/openmath_1k/eval_0.responses.jsonl` | Generation checkpoint |
| `results/sft_eval/openmath_1k/eval_0.jsonl` | Judged rows |
| `results/sft_eval/openmath_1k/eval_0.json` | Summary metrics |

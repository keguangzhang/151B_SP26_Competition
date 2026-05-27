# sft-002a — OpenR1 1k probe (QLoRA eval on holdout_20p)

**Date:** 2026-05-26  
**Registry:** [sft-002a](../experiments.md#sft-002a) · **Plan:** [sft-002a](../../sft/pipeline.md#phase-1-1k-probe-sft-002a)  
**Status:** done — **flat** vs inference-only anchors (no scale-up to sft-002b on this checkpoint)

## Setup

| Field | Value |
|-------|--------|
| **Training** | `notebooks/sft_train.ipynb` — `RUN_NAME=openr1_1k`, `SMOKE_MODE=False` |
| Corpus | `data/sft_corpus_openr1_1k.jsonl` (1,000 rows, `open-r1/OpenR1-Math-220k`, seed 42) |
| Base model | `Qwen/Qwen3-4B-Thinking-2507` |
| LoRA | r=32, α=64, dropout=0.05; targets q/k/v/o + MLP; **bf16 LoRA** ([D009](../decisions.md#d009--bf16-lora-replaces-qlora-for-sft-001-a100-only)) |
| Train | `LEARNING_RATE=1e-5`, `PER_DEVICE_BATCH=1`, `GRAD_ACCUM=16`, `MAX_SEQ_LENGTH=8192`, **`NUM_TRAIN_EPOCHS=1`** (~63 steps; pipeline had planned 2 epochs) |
| Checkpoint | `checkpoints/openr1_1k/final_adapter` (Drive: `MyDrive/CSE151B/checkpoints/openr1_1k/final_adapter`) |
| **Eval** | `notebooks/sft_eval.ipynb` — `EVAL_STEP=0`, `PROMPT_VARIANT=multi_blank`, `MAX_TOKENS=16384` |
| Eval set | `data/eval/holdout.jsonl` (`holdout_20p`) — **225 rows** (75 MCQ, 150 free-form), seed 42 |
| Inference | vLLM bf16 + LoRA on A100; decoding `temperature=0.6`, `top_p=0.95`, `top_k=20` |

## Results (holdout_20p)

| Split | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| MCQ | 58 | 75 | **77.33%** |
| Free-form | 87 | 150 | **58.00%** |
| Multi-blank (2+ `[ANS]`) | 44 | 82 | **53.66%** |
| Single-blank | 43 | 68 | **63.24%** |
| **Overall** | 145 | 225 | **64.44%** |

### Watch sets (`data/eval/watch_manifest.json`)

| Watch | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| Q4 long (≥435 chars) | 12 | 30 | **40.00%** |
| Multi-blank ≥3 | 8 | 20 | **40.00%** |

### SFT health metrics

| Metric | Value |
|--------|------:|
| MCQ `\boxed{Letter}` emission | **94.67%** |
| Mean response length | **9,595** chars (**+6.6%** vs `BASELINE_MEAN_RESPONSE_LEN=9000`) |

## Comparison (decision gate)

| Anchor | Eval | N | Overall | MCQ | FF | Notes |
|--------|------|--:|--------:|----:|---:|-------|
| pub-002 (shipped) | full public | 1126 | 61.90% | 72.00% | 56.86% | Different set — directional only |
| [dev-008](dev-008-multi-blank-16k.md) | holdout_10p | 112 | **65.18%** | 78.38% | 58.67% | Same prompt stack, **no LoRA** (smaller slice) |
| [dev-012-sc5](dev-012-sc5.md) | holdout_10p | 112 | 66.96% | 78.38% | 61.33% | Inference ceiling (SC K=5) |
| [sft-eval-001](sft-eval-001-baseline-holdout-20p.md) | holdout_20p | 225 | **64.44%** | 77.33% | 58.00% | **Base model** A/B on same 225 rows |
| **sft-002a** (this run) | holdout_20p | 225 | **64.44%** | 77.33% | 58.00% | LoRA on `final_adapter` |

| Δ vs dev-008 (inference-only multi_blank, 10% slice) | Overall | MCQ | FF | Multi-blank |
|-----------------------------------------------------|--------:|----:|---:|------------:|
| sft-002a @ holdout_20p | **−0.74 pp** | −1.05 pp | −0.67 pp | **+3.66 pp** (53.66% vs 50.00%; different N: 82 vs 38) |

| Δ vs [sft-eval-001](sft-eval-001-baseline-holdout-20p.md) (base model, **same** 225 rows) | Overall | MCQ | FF | Multi-blank | Single-blank | Q4 long | Mean len |
|-------------------------------------------------------------------------------------------|--------:|----:|---:|------------:|-------------:|--------:|----------:|
| sft-002a LoRA | **0.00 pp** | 0.00 pp | 0.00 pp | **+1.22 pp** | **−1.47 pp** | **−3.33 pp** | **−32%** (9,595 vs 14,124 chars) |

**A/B (holdout_20p):** 145/225 correct on both runs — **no net lift** from OpenR1 1k LoRA; sub-slice error swap only.

**Projection check:** [dev-007](dev-007-max-tokens-16k.md) on the same 225-row fraction with baseline prompts scored **60.00%** overall; dev-008 added **~+4.5 pp** with multi_blank on 112 rows → **~64–65%** expected without SFT. Observed **64.44%** is in the **flat (−1 to +1 pp)** band in [pipeline §decision gate](../../sft/pipeline.md#phase-1-1k-probe-sft-002a).

**Stop rules:** none triggered — MCQ holdout **77.33%** (>72% pub-002 floor); FF and length **up**; `\boxed{}` emission high.

## Artifacts (Colab Drive)

| Path | Role |
|------|------|
| `checkpoints/openr1_1k/final_adapter` | LoRA weights |
| `results/sft_eval/openr1_1k/eval_0.responses.jsonl` | Generation checkpoint |
| `results/sft_eval/openr1_1k/eval_0.jsonl` | Judged rows (`id`, `is_mcq`, `gold`, `response`, `correct`) |
| `results/sft_eval/openr1_1k/eval_0.json` | Summary metrics (`eval_record` dict) |

## Takeaway

OpenR1 1k × 1 epoch **does not show a clear holdout lift** over the multi_blank + 16k inference stack: overall **64.44%** sits ~**0.7 pp below** dev-008 and ~at the **dev-007 + multi_blank projection**. On the same 225-row holdout, [sft-eval-001](sft-eval-001-baseline-holdout-20p.md) (base model) also scores **64.44%** — **0.00 pp** vs this LoRA run (145/225 correct both ways); LoRA shortens traces (~9.6k vs ~14.1k chars) with sub-slice error swap only. Training looks **healthy** (no format collapse, MCQ emission strong). Target weak slices (**Q4 long**, **multi-blank ≥3**) remain at **40%** on the adapter eval (base Q4: 43.33%).

**Recommended next steps** (per pipeline):

1. ~~**A/B:** base model vs LoRA on **same** `holdout_20p`~~ — **done:** [sft-eval-001](sft-eval-001-baseline-holdout-20p.md) ties at **64.44%** (0.00 pp); confirms flat is not slice noise.
2. **sft-003** (s1K-1.1) or re-run with **2 epochs** / mid-training checkpoint (this eval used `final_adapter` after **1 epoch only**).
3. Optional: **SC K=5** on base or adapter via `holdout_10p` (cheap check per pipeline modest-win path).
4. **Do not** scale to sft-002b (5k) on this result alone.

**Decision:** [D010](../decisions.md#d010--sft-002a-openr1-1k-flat--do-not-scale-to-5k-yet)

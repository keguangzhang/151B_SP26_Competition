# sft-eval-001 — Base model on holdout_20p (SFT eval anchor)

**Date:** 2026-05-27  
**Registry:** [sft-eval-001](../experiments.md#sft-eval-001) · **Related:** [sft-002a](sft-002a-openr1-1k.md) A/B control  
**Status:** done — inference-only anchor on frozen 20% holdout

## Setup

| Field | Value |
|-------|--------|
| **Eval** | `notebooks/sft_eval.ipynb` — `LORA_PATH=""`, `SFT_RUN_NAME=baseline`, `EVAL_STEP=0` |
| Eval set | `data/eval/holdout_20p.jsonl` — **225 rows** (75 MCQ, 150 free-form), seed 42 |
| Model | `Qwen/Qwen3-4B-Thinking-2507` (no LoRA) |
| Prompt / decoding | `PROMPT_VARIANT=multi_blank`, `MAX_TOKENS=16384`, `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Inference | vLLM bf16 on A100 (same stack as [sft-002a](sft-002a-openr1-1k.md)) |
| `BASELINE_MEAN_RESPONSE_LEN` | 9000 (length delta reference only) |

## Results (holdout_20p)

| Split | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| MCQ | 58 | 75 | **77.33%** |
| Free-form | 87 | 150 | **58.00%** |
| Multi-blank (2+ `[ANS]`) | 43 | 82 | **52.44%** |
| Single-blank | 44 | 68 | **64.71%** |
| **Overall** | 145 | 225 | **64.44%** |

### Watch sets (`data/eval/watch_manifest.json`)

| Watch | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| Q4 long (≥435 chars) | 13 | 30 | **43.33%** |
| Multi-blank ≥3 | 8 | 20 | **40.00%** |

### SFT health metrics

| Metric | Value |
|--------|------:|
| MCQ `\boxed{Letter}` emission | **90.67%** |
| Mean response length | **14,124** chars (**+56.9%** vs `BASELINE_MEAN_RESPONSE_LEN=9000`) |

## Comparison

| Anchor | Eval | N | Overall | MCQ | FF | Multi-blank | Notes |
|--------|------|--:|--------:|----:|---:|------------:|-------|
| [dev-008](dev-008-multi-blank-16k.md) | holdout_10p | 112 | **65.18%** | 78.38% | 58.67% | 50.00% (38) | Same prompt; smaller slice |
| **sft-eval-001** (this run) | holdout_20p | 225 | **64.44%** | 77.33% | 58.00% | 52.44% (82) | Base model |
| [sft-002a](sft-002a-openr1-1k.md) | holdout_20p | 225 | **64.44%** | 77.33% | 58.00% | 53.66% (82) | OpenR1 1k LoRA |

| Δ sft-002a LoRA vs this run (same 225 rows) | Overall | MCQ | FF | Multi-blank | Single-blank | Q4 long |
|---------------------------------------------|--------:|----:|---:|------------:|-------------:|--------:|
| | **0.00 pp** | 0.00 pp | 0.00 pp | **+1.22 pp** | **−1.47 pp** | **−3.33 pp** |

**Item-level:** 145/225 correct on both runs — **identical** headline accuracy; LoRA shifts errors between single-blank (−1) and multi-blank (+1) with no net gain. Mean trace length **−32%** (14,124 → 9,595 chars) while MCQ `\boxed{}` emission **+4.0 pp**.

## Artifacts (Colab Drive)

| Path | Role |
|------|------|
| `results/sft_eval/baseline/eval_0.responses.jsonl` | Generation checkpoint |
| `results/sft_eval/baseline/eval_0.jsonl` | Judged rows |
| `results/sft_eval/baseline/eval_0.json` | Summary metrics (`eval_record`) |

## Takeaway

Registers the **holdout_20p** inference-only anchor requested after [sft-002a](sft-002a-openr1-1k.md): **64.44% overall** matches the OpenR1 1k adapter on the **same** 225 ids — flat A/B, not slice noise. Sub-slice tradeoffs (multi-blank up, single-blank and Q4-long down) and much shorter generations suggest the adapter changes trace style without improving graded accuracy on this eval. Confirms [D010](../decisions.md#d010--sft-002a-openr1-1k-flat--do-not-scale-to-5k-yet): do not scale to 5k on this checkpoint; try epoch-2 / mid-checkpoint eval or [sft-003](../../sft/pipeline.md#sft-003--s1k-11-fallback-if-sft-002a-flat).

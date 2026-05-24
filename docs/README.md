# Documentation index

Start here for project status. Deep dives live in subfolders; **do not duplicate metrics** outside [`log/experiments.md`](log/experiments.md).

## Current best (inference baseline)

| Setting | Value |
|--------|--------|
| Model | `Qwen/Qwen3-4B-Thinking-2507` (INT8, vLLM) |
| Eval | `data/public.jsonl` — 1,126 rows |
| `max_tokens` | **8192** |
| Decoding | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| **Overall** | **52.66%** (593 / 1,126) |
| MCQ | **50.40%** (189 / 375) |
| Free-form | **53.79%** (404 / 751) |

Registry row: [`pub-001`](log/experiments.md#pub-001). Analysis: [`analysis/baseline-public-8k.md`](analysis/baseline-public-8k.md).

## Active work

- **Inference (highest priority):** `max_tokens=16384` full-public run (`pub-002`) — dev slice validated at 60% overall / 70.7% MCQ ([dev-007](log/runs/dev-007-max-tokens-16k.md)); 32k ablation flat ([dev-009](log/runs/dev-009-max-tokens-32k.md)). See [`roadmap.md`](roadmap.md) §1.1.
- **SFT:** Numina-only QLoRA first run — see [`sft/pipeline.md`](sft/pipeline.md), data prep in `notebooks/sft_data_prep.ipynb`.
- **Analysis revised 2026-05-23:** token truncation (not format) is dominant MCQ failure. See [`analysis/baseline-public-8k.md`](analysis/baseline-public-8k.md).

## Quick links

| Need | Doc |
|------|-----|
| Experiment registry + scores | [`log/experiments.md`](log/experiments.md) |
| Per-run notebook notes | [`log/runs/`](log/runs/) |
| Why we chose X | [`log/decisions.md`](log/decisions.md) |
| Final report skeleton | [`report-outline.md`](report-outline.md) |
| Ideas & priorities (no numbers) | [`roadmap.md`](roadmap.md) |
| 8k public error analysis | [`analysis/baseline-public-8k.md`](analysis/baseline-public-8k.md) |
| SFT plan / data spec / QA | [`sft/`](sft/) |
| Numina clean corpus audit (2026-05-21) | [`sft/numina-clean-audit.md`](sft/numina-clean-audit.md) |
| Colab + vLLM install / env | [`infra/vllm-colab-l4.md`](infra/vllm-colab-l4.md) |
| vLLM `LLM(...)` profiles (L4 vs A100) | [`infra/vllm-inference-config.md`](infra/vllm-inference-config.md) |
| Competition constraints | [`reference/constraints.md`](reference/constraints.md) |

## Layout

```
docs/
  README.md              ← this file (dashboard)
  report-outline.md
  roadmap.md
  log/
    experiments.md       ← master run table
    decisions.md
    runs/                ← dated experiment notes
  analysis/
  sft/
  infra/
  reference/
```

## Conventions

- **Run IDs:** `dev-*` (fast slice), `pub-*` (full public), `priv-*` (submission), `sft-*` (training).
- **Status:** `planned` → `running` → `done` | `rejected` | `shipped` (in baseline/submission path).
- **New experiment:** add a row to `log/experiments.md`, then add or extend a note under `log/runs/`.
- **New decision:** append to `log/decisions.md` (ADR-style); link from `roadmap.md` or `sft/pipeline.md`.

## Legacy paths

Files at the old flat names (`improvement-directions.md`, `tests.md`, etc.) are stub redirects only.

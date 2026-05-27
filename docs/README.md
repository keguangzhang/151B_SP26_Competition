# Documentation index

Start here for project status. Deep dives live in subfolders; **do not duplicate metrics** outside [`log/experiments.md`](log/experiments.md).

## Current best (inference baseline)

| Setting | Value |
|--------|--------|
| Model | `Qwen/Qwen3-4B-Thinking-2507` (bfloat16, vLLM) |
| Eval | `data/public.jsonl` — 1,126 rows |
| `max_tokens` | **16384** |
| Decoding | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| **Overall** | **61.90%** (697 / 1,126) |
| MCQ | **72.00%** (270 / 375) |
| Free-form | **56.86%** (427 / 751) |

Registry row: [`pub-002`](log/experiments.md#pub-002). Analysis: [`analysis/baseline-public-16k.md`](analysis/baseline-public-16k.md).

## Active work

- **Inference:** pub-002 (16k + adaptive multi-blank prompt) shipped at 61.90% overall / 72.00% MCQ. Reasoning errors now dominate wrong MCQ (51.4%); truncation mostly solved.
- **SFT:** [sft-002a](log/runs/sft-002a-openr1-1k.md) OpenR1 1k probe **flat** on holdout_20p (64.44%); do not scale to 5k yet — see [`sft/pipeline.md`](sft/pipeline.md).
- **Analysis 2026-05-24:** 16k failure-mode shift — truncation 84% → 41% of wrong MCQ; reasoning errors now 51.4%. See [`analysis/baseline-public-16k.md`](analysis/baseline-public-16k.md).

## Quick links

| Need | Doc |
|------|-----|
| Experiment registry + scores | [`log/experiments.md`](log/experiments.md) |
| Per-run notebook notes | [`log/runs/`](log/runs/) |
| Why we chose X | [`log/decisions.md`](log/decisions.md) |
| Final report skeleton | [`report-outline.md`](report-outline.md) |
| Ideas & priorities (no numbers) | [`roadmap.md`](roadmap.md) |
| External-evidence technique survey (2026-05-24) | [`research/2026-05-24-improvement-techniques-survey.md`](research/2026-05-24-improvement-techniques-survey.md) |
| 8k public error analysis | [`analysis/baseline-public-8k.md`](analysis/baseline-public-8k.md) |
| 16k public error analysis | [`analysis/baseline-public-16k.md`](analysis/baseline-public-16k.md) |
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
  research/             ← external-evidence technique surveys (dated)
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

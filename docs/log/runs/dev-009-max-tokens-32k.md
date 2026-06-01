# dev-009 — `max_tokens` 32768 (32k) ablation

**Date:** 2026-05-24  
**Registry:** [dev-009](../experiments.md#dev-009) · **Roadmap:** [§1.1](../../roadmap.md#11-lift-max_tokens-to-16384--next--highest-priority)  
**Status:** done (rejected — no lift vs 16k on same slice)

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/dev.jsonl` — **10%** stratified, seed 42 (**112 rows**: 37 MCQ, 75 free-form) |
| Change | **`max_tokens` 16384 → 32768** only (same prompt as [dev-008](dev-008-multi-blank-16k.md)) |
| Prompt / decoding | `PROMPT_VARIANT="multi_blank"`; `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Model | `Qwen/Qwen3-4B-Thinking-2507`, bf16, `max_model_len=32768` — [vLLM A100 profile](../../infra/vllm-inference-config.md#notebooksdevipynb--a100-optimized-load-7) |
| Notebook | `notebooks/dev.ipynb` — `MAX_TOKENS=32768`, `PROMPT_VARIANT="multi_blank"`, `SMOKE_TEST=False` |

**Note:** With `max_model_len=32768`, effective generation budget is `32768 − prompt_tokens` (not a full 32k on top of the prompt). This run tests whether consuming nearly the full context window helps beyond the validated 16k cap.

## Results

| Split | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| MCQ | 29 | 37 | **78.38%** |
| Free-form | 43 | 75 | **57.33%** |
| Multi-blank | 18 | 38 | **47.37%** |
| Single-blank | 25 | 37 | **67.57%** |
| **Overall** | 72 | 112 | **64.29%** |

## Comparison (same 10% dev slice, multi_blank)

| Metric | [dev-008](dev-008-multi-blank-16k.md) (16k) | **dev-009 (32k)** | Δ |
|--------|---------------------------------------------:|------------------:|--:|
| MCQ | 78.38% | **78.38%** | 0.00 pp |
| Free-form | 58.67% | **57.33%** | −1.34 pp |
| Multi-blank | 50.00% | **47.37%** | −2.63 pp |
| Single-blank | 67.57% | **67.57%** | 0.00 pp |
| Overall | 65.18% | **64.29%** | **−0.89 pp** |

MCQ and single-blank are identical; free-form and multi-blank are slightly worse at 32k. No evidence that raising the cap beyond 16k helps on this slice — likely most gains from §1.1 were already captured at 16k.

## Artifacts

- `results/dev_results_multi_blank_32k.jsonl`
- Colab Drive: `MyDrive/CSE151B/results/dev_results_multi_blank_32k.jsonl`

## Follow-up

- Keep **16k** as the inference generation cap for pub-002 / submission path ([D008](../decisions.md#d008--32k-max_tokens-rejected-stay-at-16k)).
- Do not raise `max_model_len` further unless a new failure mode (mid-think truncation at 16k) is observed on full public.

## Takeaway

32k `max_tokens` on A100 (`max_model_len=32768`) does **not** beat 16k on the same 10% dev slice with multi_blank prompts. Overall −0.9 pp vs dev-008; MCQ flat. §1.1 ceiling appears reached at 16k for this stack.

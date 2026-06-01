# dev-012-sc5 — K=5 self-consistency (multi_blank, 16k)

**Date:** 2026-05-26  
**Registry:** [dev-012-sc5](../experiments.md#dev-012-sc5)  
**Status:** done (10% holdout; +1.78 pp overall vs dev-010-bf; multi-blank gains all the lift)

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/eval/holdout_10p.jsonl` — **10%** stratified, seed 42 (**112 rows**: 37 MCQ, 75 free-form) |
| Change | **`SELF_CONSISTENCY=True`, `SC_K=5`** — `SamplingParams(n=5)` → extract → canonicalize → majority vote → tiebreak (anchor = sample 0) |
| Prompt / decoding | `PROMPT_VARIANT="multi_blank"`; `max_tokens=16384`; `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Model | `Qwen/Qwen3-4B-Thinking-2507`, bf16 — A100 |
| Notebook | `notebooks/dev.ipynb` — `SELF_CONSISTENCY=True`, `SC_K=5`, `SMOKE_TEST=False` |
| Runtime | ~45 min (5× inference vs K=1 baseline) |

## Results

| Split | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| MCQ | 29 | 37 | **78.38%** |
| Free-form | 46 | 75 | **61.33%** |
| Multi-blank | 21 | 38 | **55.26%** |
| Single-blank | 25 | 37 | **67.57%** |
| **Overall** | 75 | 112 | **66.96%** |

All 112 items have exactly K=5 traces (confirmed from `sc_traces.jsonl`).

## Comparison

| Metric | [dev-010-bf](dev-010-bf-budget-forcing.md) (baseline) | **dev-012-sc5** | Δ |
|--------|------------------------------------------------------:|----------------:|--:|
| MCQ | 78.38% | **78.38%** | 0.00 pp |
| Free-form | 58.67% | **61.33%** | **+2.67 pp** |
| Multi-blank | 50.00% | **55.26%** | **+5.26 pp** |
| Single-blank | 67.57% | **67.57%** | 0.00 pp |
| **Overall** | 65.18% | **66.96%** | **+1.78 pp** |

SC gain is entirely on **multi-blank**: +2 items (19→21/38). Single-blank and MCQ are flat. MCQ also received K=5 samples with majority vote on the extracted letter — no MCQ/FF split in generation — but produced zero lift.

## Artifacts

- `data/dev_results_multi_blank_16k_sc5.jsonl` — final judged results
- `data/dev_results_multi_blank_16k_sc5.responses.jsonl` — checkpoint (K responses per item collapsed to winner)
- `data/dev_results_multi_blank_16k_sc5.sc_traces.jsonl` — raw 5 traces per item

## Takeaway

K=5 SC on top of the multi_blank prompt gives a real but narrow lift (+1.78 pp overall, driven by multi-blank +5.26 pp). Plausible mechanism: multi-blank items require matching multiple `\boxed{}` values; a wrong order or extra comma in one trace gets outvoted. Single-blank and MCQ see no gain — single-blank already votes well at K=1, MCQ is structurally letter-selection (no multi-element canonicalization).

**Cost is 5× inference time (~45 min vs ~9 min).** Marginal rate: ~+1.78 pp for 5× cost. Compare to §1.3 multi_blank prompt which gave ~+4.5 pp for free (no extra inference). SC is a valid ingredient but expensive; prioritize SFT before scaling K for the full private run. MCQ SC (majority vote on extracted letter) ran but gave zero lift — MCQ errors are likely deterministic wrong-letter rather than vote-recoverable variance.

## Follow-up

- **SC on full public/private:** 5× cost is ~3.75 h on A100 for 1126 items (vs ~45 min). Viable if accuracy gap vs SFT baseline justifies it.
- **Larger K (K=8, K=16):** diminishing returns expected; multi-blank vote quality saturates once most traces agree. Not worth running until SFT sets a new floor.
- **SC + SFT:** if SFT lifts the per-trace quality, SC vote variance drops and majority becomes more reliable. Natural stack after SFT-001.
- **MCQ SC:** currently greedy. Sampling MCQ at K=5 might help if MCQ errors are vote-recoverable (low priority — MCQ at 78% is already strong).

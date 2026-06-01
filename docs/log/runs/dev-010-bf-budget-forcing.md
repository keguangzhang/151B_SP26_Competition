# dev-010-bf — Budget forcing (FF-biased, baseline 16k)

**Date:** 2026-05-25  
**Registry:** [dev-010-bf](../experiments.md#dev-010-bf) · **Survey:** [§A budget forcing](../../research/2026-05-24-improvement-techniques-survey.md#a-budget-forcing-s1-style-test-time-scaling)  
**Status:** done (10% holdout; no lift vs [dev-008](dev-008-multi-blank-16k.md) on same slice)

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/eval/holdout_10p.jsonl` — **10%** stratified, seed 42 (**112 rows**: 37 MCQ, 75 free-form) |
| Change | **`BUDGET_FORCING=True`** — wave-batched s1-style `Wait` on early thinking close; **FF-biased** (`BUDGET_FORCE_MIN_THINK_CHARS_MCQ=0` disables MCQ injection) |
| Prompt / decoding | `PROMPT_VARIANT="baseline"`; `max_tokens=16384`; `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| BF knobs | `max_iter=2`, `wait=" Wait"`, `min_think_ff=10**9` (force on every FF close attempt), `context_cap=14000` |
| Model | `Qwen/Qwen3-4B-Thinking-2507`, bf16 — [A100 profile](../../infra/vllm-inference-config.md) |
| Notebook | `notebooks/dev.ipynb` — `RUN_ID=dev-010-bf`, `SMOKE_TEST=False` |

## Results

| Split | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| MCQ | 29 | 37 | **78.38%** |
| Free-form | 44 | 75 | **58.67%** |
| Multi-blank | 19 | 38 | **50.00%** |
| Single-blank | 25 | 37 | **67.57%** |
| **Overall** | 73 | 112 | **65.18%** |

### Budget-forcing diagnostics (§8)

| Slice | Items forced | Rate | Mean `Wait` inj/item |
|-------|-------------:|-----:|---------------------:|
| MCQ | 0 / 37 | 0.0% | 0.00 |
| **Free-form** | **72 / 75** | **96.0%** | **1.92** |

All 75 FF traces were below `min_think_ff` at close attempts (`75/75` under `10**9` char threshold). Thinking-length p50 on FF: **7,886** chars (p25 3,547 / p75 17,587).

## Comparison

| Metric | †10% baseline 16k | [dev-008](dev-008-multi-blank-16k.md) multi_blank | **dev-010-bf** | Δ vs † | Δ vs dev-008 |
|--------|------------------:|--------------------------------------------------:|---------------:|-------:|-------------:|
| MCQ | ~75.68% | 78.38% | **78.38%** | — | 0.00 pp |
| Free-form | ~53.33% | 58.67% | **58.67%** | **+5.34 pp** | 0.00 pp |
| Multi-blank | — | 50.00% | **50.00%** | — | 0.00 pp |
| Overall | ~60.71% | 65.18% | **65.18%** | **+4.47 pp** | 0.00 pp |

†Unregistered Colab run: same 112-row slice, `PROMPT_VARIANT="baseline"`, `MAX_TOKENS=16384`, no budget forcing (notebook footnote in dev-008).

On this slice, **FF-biased budget forcing + baseline prompt matches dev-008 multi_blank on every reported bucket** (including multi-blank 19/38). Does not add lift beyond the §1.3 prompt change already validated; still supports that extended FF reasoning (via `Wait`) can recover ~+5 pp FF vs plain baseline 16k.

## Artifacts

- `results/dev_results_baseline_16k_bf.jsonl`
- Checkpoint: `results/dev_results_baseline_16k_bf.responses.jsonl`
- Colab Drive: `MyDrive/CSE151B/results/` (same filenames)

## Why BF appears inert on this model (post-hoc, vs s1 paper)

Config matches [s1 §3](https://arxiv.org/abs/2501.19393) intent — Wait injection fires on natural `</think>` (not on truncation). `min_think_ff=10**9` is deliberate: force on every close so we measure BF's full effect, not a thresholded subset. 96% fire rate confirms the mechanism ran. Lowering `min_think_ff` would only target the lower-quartile (shortest, most confident) traces — those are the ones least likely to flip, so a smaller threshold is **not** a meaningful follow-up here.

Lift absent despite correct firing — hypotheses ranked by likelihood:

1. **s1 fine-tunes the base model on Wait-conditioned traces (s1K, ~1k examples), then applies BF.** s1-32B learned to *use* the Wait token as a revise-signal during SFT. **Qwen3-4B-Thinking-2507 was RL-tuned with its own thinking distribution and has never seen `" Wait"` as a continuation cue.** Out-of-distribution injection → model resumes thinking, re-derives, converges to the same `\boxed{}`. The paper's +7 pp AIME24 (50→57) and +27% MATH numbers assume the SFT half of the recipe — we ran only the inference half.
2. **Scale.** Paper headline numbers are on 32B. Self-correction-via-Wait needs spare capacity to attack a problem along a *different* path on retry. A 4B model often has one mode of attack; first pass and second pass converge.
3. **Calibrated `</think>`.** Post-trained thinking models emit close-tag when reasoning is genuinely done. s1's base closes were less reliable, so Wait usefully prods premature closes. Forcing Wait on a calibrated stop ≈ no-op.
4. **Sample size + domain.** s1's AIME24 N=30; ±2 problems = 7 pp. Our FF N=75 didn't move, but item-level diffs vs dev-008 not yet checked — net-zero could mask compensating flips. Also: CSE 151B FF failures include multi-blank format mismatches (judger contiguous-group rule) that Wait cannot fix.

**Repositioning:** treat §A as **gated on SFT**, not a drop-in inference trick on this model. Order of operations should be SFT first (with Wait-style revision traces in the training set), then BF as an inference-time multiplier — not BF alone on an off-the-shelf Qwen3-Thinking checkpoint.

## Follow-up

- **Item-level diff vs dev-008:** confirm whether BF flips any individual items (net-zero accuracy can hide compensating swaps). If identical per-id, Wait is a genuine no-op on this model.
- **SFT-gated retry:** revisit BF only **after** an SFT cycle that includes Wait-style backtracking traces (e.g. distil s1K-style continuations, or mine `</think> Wait` revisions from a stronger thinking model). Drop standalone BF from public-eval candidates until then.
- **Stack (low priority):** BF + `multi_blank` prompt — only worth running if item-level diff above shows BF is doing *something* under the baseline prompt.
- **MCQ:** current guard (`min_think_mcq=0`) intentionally skips MCQ; pub-002 wrong-MCQ mode is "think finished, wrong letter". Skip MCQ-BF ablation under same SFT-gated reasoning.

## Takeaway

On 10% holdout, BF matches dev-008 multi_blank exactly (FF 58.67%, multi-blank 50.00%, overall 65.18%) despite 96% fire rate and mean 1.92 `Wait` injections per FF item. Mechanism ran as the s1 paper intends, but **the paper's lift comes from SFT-on-Wait-traces + BF, not BF alone.** Qwen3-4B-Thinking-2507 has not seen `" Wait"` as a revise-cue during training — it just resumes and reconverges. **Reposition §A as SFT-dependent; do not promote standalone BF to public/private eval. Re-evaluate after the SFT phase if Wait-style traces are included in the dataset.**

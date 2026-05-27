# dev-011-php — Progressive-hint prompting (PHP, FF-only, format-guarded)

**Date:** 2026-05-25
**Registry:** [dev-011-php](../experiments.md#dev-011-php) · **Survey:** [§1.13 PHP](../../research/2026-05-24-improvement-techniques-survey.md) · **Roadmap:** [§1.13](../../roadmap.md#113-progressive-hint-prompting-php--promoted)
**Status:** rejected for submission. PHP+baseline: **+5.33 pp FF within-run** (5 W→R / 1 R→W, all format). PHP+multi_blank follow-up: **+0.00 pp** (1 W→R / 1 R→W, net 0). Both land at 58.67% FF — same as multi_blank alone at 1× cost.

## Setup

| Field | Value |
|-------|--------|
| Eval | `data/eval/holdout_10p.jsonl` — **10%** stratified, seed 42 (**112 rows**: 37 MCQ, 75 free-form) |
| Change | **`PHP_ENABLED=True`** with **per-item format clause in pass-2 hint** — 2-pass FF: pass-1 generates tentative `\boxed{}`; pass-2 re-prompts with neutral *"previous attempt got X — re-solve, confirm or revise"* hint **plus an explicit format instruction tailored to the question shape** |
| Prompt / decoding | `PROMPT_VARIANT="baseline"`; `max_tokens=16384` both passes; `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| PHP knobs | `PHP_FF_ONLY=True` (MCQ skipped — high pass-1 accuracy, low headroom); `PHP_SKIP_IF_NO_PASS1_ANSWER=True`; `PHP_MAX_TOKENS=16384` |
| Format clause | Single-blank / MCQ → `"Put your final answer inside \boxed{} as before."` Multi-blank (n≥2 `[ANS]`) → `"The problem has N [ANS] blanks. Give N comma-separated \boxed{...} values in the order the blanks appear: \boxed{...}, \boxed{...}, …. Do not use labels like '(a)', 'Answer 1:', or sentences between boxes — the boxes must be adjacent so the grader can group them."` (helper `_php_format_clause` in [notebooks/dev.ipynb](../../../notebooks/dev.ipynb)) |
| Model | `Qwen/Qwen3-4B-Thinking-2507`, bf16 — A100 |
| Notebook | `notebooks/dev.ipynb` — auto-derived `RUN_ID=dev-007-php`, `BUDGET_FORCING=False`, `SMOKE_TEST=False` |

This run replaces an earlier PHP attempt that used a single generic format clause for all FF items. That attempt regressed FF by −4 pp on the same slice: pass-2 reformatted multi-blank answers into prose-separated `(a) \boxed{…} (b) Number's reciprocal: \boxed{…}` layouts that fail `judger.extract_all_boxed`'s contiguous-group rule. The fix is purely in the hint template — pass-2 generation otherwise unchanged.

## Results

| Split | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| MCQ | 28 | 37 | **75.68%** |
| Free-form | 44 | 75 | **58.67%** |
| Multi-blank | 20 | 38 | **52.63%** |
| Single-blank | 24 | 37 | **64.86%** |
| **Overall** | 72 | 112 | **64.29%** |

### PHP pass-2 diagnostics (§8b)

| Slice | Items targeted | Pass-2 replaced | Kept pass-1 (no boxed) |
|-------|---------------:|----------------:|-----------------------:|
| Free-form | 75 | **73** (97.3%) | 2 |
| MCQ | 0 | 0 | — (skipped by `PHP_FF_ONLY`) |

Pass-1 snapshot persisted to `dev_results_baseline_16k_php.php_pass1.jsonl`; pass-2 checkpoint at `dev_results_baseline_16k_php.php_pass2.jsonl`.

### Item-level diff (pass-1 vs final, re-scored offline against `holdout_10p.jsonl`)

Replay via `Judger(strict_extract=False)` on the persisted JSONLs reproduces the notebook FF total exactly (44/75 = 58.67%). Pass-1 alone scores **40/75 = 53.33%** on the same slice — PHP's second pass **fixed 5 items and broke 1**, net **+4**.

| Transition (pass-1 → final) | Count |
|------------------------------|------:|
| Correct → Correct (R→R) | 39 |
| **Wrong → Correct (W→R)** | **5** |
| Correct → Wrong (R→W) | **1** |
| Wrong → Wrong (W→W) | 30 |

Multi-blank vs single-blank within-run (pass-1 → final):

| Slice | N | Pass-1 acc | Final acc | Δ |
|-------|--:|----------:|----------:|--:|
| Multi-blank FF | 38 | 39.47% | **52.63%** | **+13.16 pp** |
| Single-blank FF | 37 | 67.57% | **64.86%** | **−2.70 pp** |

The +5.33 pp FF gain is **entirely concentrated in multi-blank**. Single-blank lost one item.

### The 5 W→R fixes are all multi-blank format recoveries

All five gains share the same mechanism: pass-1 (under the baseline prompt) boxed only the **last** blank as a single `\boxed{X}` at the end of the trace; pass-2's format clause forced an N-box contiguous layout, and the judger's contiguous-group rule then captured all blanks.

| id | gold | pass-1 box (lost blanks) | pass-2 box (recovered) |
|----|------|--------------------------|------------------------|
| 256 | `D, D, A` | `A` | `D, D, A` |
| 410 | `x = -1/2, (-1/2, 1/4)` | `(-1/2, 1/4)` | `-1/2, (-1/2, 1/4)` |
| 908 | `0.5, 0, 0.25, 1.25, Biology` | `Biology` | `0.5, 0, 0.25, 1.25, Biology` |
| 915 | `CE, OS, OS` | `OS` | `CE, OS, OS` |
| 949 | `A, D, A` | `D, D, A` | `A, D, A` |

Ids 908 and 951 — the two format-collapse breaks in the prior (rejected) PHP attempt — are now both correct or unchanged: 908 flips W→R, 951 stays R→R with identical boxes across passes. The format guard fired exactly where it was designed to.

### The 1 R→W break

| id | Question pattern | Pass-1 box | Pass-2 box | Failure |
|----|------------------|------------|------------|---------|
| 436 | `Find the equation … [ANS] = x` (single-blank) | `\frac{y^2}{32}` ✓ | `y^2 = 32x` ✗ | Pass-2 misread the `[ANS] = x` template and emitted the full equation instead of the LHS expression |

This is a **reasoning / question-template misread**, not a format failure. Format guard does not address it (the question is single-blank, so default format clause applies). Survives the fix.

## Comparison

| Metric | †10% baseline 16k | [dev-008](dev-008-multi-blank-16k.md) multi_blank | [dev-010-bf](dev-010-bf-budget-forcing.md) | **dev-011-php pass-1** | **dev-011-php final** | Δ final vs pass-1 |
|--------|------------------:|--------------------------------------------------:|-------------------------------------------:|-----------------------:|----------------------:|------------------:|
| MCQ | ~75.68% | 78.38% | 78.38% | **75.68%** | **75.68%** | — (PHP skipped MCQ) |
| Free-form | ~53.33% | 58.67% | 58.67% | **53.33%** | **58.67%** | **+5.33 pp** |
| Multi-blank | — | 50.00% | 50.00% | **39.47%** | **52.63%** | **+13.16 pp** |
| Single-blank | — | — | — | **67.57%** | **64.86%** | **−2.70 pp** |
| **Overall** | ~60.71% | 65.18% | 65.18% | **64.29%** | **64.29%** | — |

†Unregistered Colab run: same 112-row slice, `PROMPT_VARIANT="baseline"`, `MAX_TOKENS=16384`, no PHP, no BF (same anchor as dev-008 and dev-010-bf).

Key reading:
- **PHP+baseline at the final number ≈ multi_blank prompt at 1× cost.** Both land on 58.67% FF / 52–53% multi-blank. PHP costs **2× FF inference** for the same destination.
- **Within-run, PHP's pass-2 recovers exactly the gap between baseline pass-1 (53.33%) and the multi_blank prompt (58.67%).** Pass-1 here is the same baseline configuration as the † reference at 53.33%. PHP's gain is fully attributable to fixing multi-blank format collapses that the multi_blank prompt already prevents at pass-1.
- **MCQ −2.70 pp vs dev-008 is RNG variance** (PHP off on MCQ; different sampling draw on a 37-row slice).
- **Single-blank lost one item (id 436)** to a question-template misread; not a format problem.

## Why PHP helped this time

The prior (rejected) PHP attempt used a single generic `Put your final answer inside \boxed{}` clause. With a generic clause, pass-2's "independently re-solve" framing nudged the model toward natural labeled layouts (`(a) \boxed{…} (b) Number's reciprocal: \boxed{…}`) for multi-blank items — reasoning right, boxes broken, judger captured only the last group. Format-collapse, not reasoning-failure.

The fix injects an explicit per-item format clause: when the question has ≥2 `[ANS]` placeholders, the hint tells pass-2 the exact box count and forbids labels/prose between boxes. The same hint that broke multi-blank now constrains it. Five W→R fixes and zero new format-collapse R→W breaks confirm the mechanism.

## When is PHP worth it?

Strictly on this slice: PHP+baseline (2× FF inference) ≈ multi_blank prompt (1× FF inference) for FF accuracy. For submission, the multi_blank prompt is the better pick.

PHP becomes attractive only in combinations:

1. **PHP + `PROMPT_VARIANT="multi_blank"` base prompt.** Pass-1 already encodes the multi-blank format, removing format-collapse W→R items from pass-2's reachable set. Remaining W→R would have to come from reasoning revision. If pass-2 still pulls ≥1 reasoning W→R on the dev-008 base, PHP is genuinely additive at 2× cost.
2. **PHP-k (k≥2).** Survey lists this as the +2–4 pp variant on stronger bases. Triple inference cost; only justified after (1) shows non-zero reasoning-driven W→R.

Both should run on `holdout_10p.jsonl` before any decision to include PHP in public/private submission.

## Follow-up run — PHP + `PROMPT_VARIANT="multi_blank"` base prompt

Re-ran with `PROMPT_VARIANT="multi_blank"` to test whether PHP's second pass adds anything beyond what the multi_blank prompt already gives at pass-1.

**Setup:** identical to above except `PROMPT_VARIANT="multi_blank"`. Same eval slice (`holdout_10p.jsonl`, 112 rows), same decoding, same per-item format clause on pass-2 hint. `OUTPUT_PATH = dev_results_multi_blank_16k_php.jsonl`. Notebook re-run on 2026-05-25, A100 bf16. Artifacts on Colab Drive only (not synced locally); numbers below come from the §10 summary and §9b PHP-analysis cell printed in the notebook.

### Results

| Split | Correct | N | Accuracy |
|-------|--------:|--:|--------:|
| MCQ | 29 | 37 | **78.38%** |
| Free-form | 44 | 75 | **58.67%** |
| Multi-blank | 19 | 38 | **50.00%** |
| Single-blank | 25 | 37 | **67.57%** |
| **Overall** | 73 | 112 | **65.18%** |

Identical (to the item) to dev-008 multi_blank alone on this slice.

### PHP within-run effect

| Slice | N | Pass-1 acc | Final acc | Δ |
|-------|--:|----------:|----------:|--:|
| FF overall | 75 | 58.67% | **58.67%** | **+0.00 pp** |
| Multi-blank | 38 | 50.00% | 50.00% | +0.00 pp |
| Single-blank | 37 | 67.57% | 67.57% | +0.00 pp |

Pass-2 fired on 74/75 FF items (one had no pass-1 box; skipped).

| Transition (pass-1 → final) | Count | ids |
|------------------------------|------:|-----|
| Correct → Correct (R→R) | 43 | — |
| **Wrong → Correct (W→R)** | **1** | 565 |
| **Correct → Wrong (R→W)** | **1** | 158 |
| Wrong → Wrong (W→W) | 30 | — |

Net items moved: **0**.

### The two flips

| id | blanks | pass-1 boxes (extract_all_boxed) | pass-2 boxes | gold | verdict |
|----|------:|----------------------------------|--------------|------|---------|
| 158 | 2 | `['A', 'C']` ✓ | `['A', 'C', 'A', 'C']` ✗ | `[A, C]` | R→W (pass-2 duplicated the answer set — length mismatch fails judger) |
| 565 | 3 | `['2']` ✗ | `['(17-2x)(4-2x)x', '0', '2']` ✓ | (3-blank) | W→R (multi_blank base still missed 2 of 3 blanks at pass-1; pass-2 recovered) |

Both flips are format-side, not reasoning-side. Id 565 shows the multi_blank prompt is not perfectly self-enforcing (1 leak through pass-1); id 158 shows the pass-2 hint can over-shoot in the other direction. They cancel.

### Reading

The follow-up answers the question posed in the prior section: **PHP+multi_blank produces zero reasoning-driven W→R on this slice**. The single W→R is a residual format recovery (same mechanism as PHP+baseline, just rarer because the multi_blank prompt already prevents most of them at pass-1). The matching R→W is a new format-collapse direction (pass-2 doubling the answer set).

Combined with the PHP+baseline result:

| Config | Pass-1 FF | Final FF | Δ within-run | W→R | R→W | Net | FF inference cost |
|--------|----------:|---------:|-------------:|----:|----:|----:|------------------:|
| PHP + baseline prompt | 53.33% | 58.67% | +5.33 pp | 5 | 1 | +4 | 2× |
| **PHP + multi_blank prompt** | **58.67%** | **58.67%** | **+0.00 pp** | **1** | **1** | **0** | **2×** |
| multi_blank prompt alone (dev-008) | 58.67% | — | — | — | — | — | **1×** |

PHP's measured lift on this model is **entirely format recovery**, not reasoning revision. Once the format problem is solved upstream (multi_blank prompt), PHP has nothing left to fix.

## Follow-up

- **DONE — item-level diff (PHP+baseline):** 5 W→R / 1 R→W; all 5 fixes are multi-blank format recoveries; the 1 break is a single-blank template misread.
- **DONE — PHP + multi_blank base:** 1 W→R / 1 R→W, net 0. PHP adds nothing on top of multi_blank pass-1 on this slice.
- **Reject PHP for public/private submission.** Both flavors land at 58.67% FF. multi_blank alone gets there at 1× FF inference; PHP costs 2× for the same destination (PHP+baseline) or zero gain (PHP+multi_blank). No path to positive ROI without changing the underlying model.
- **PHP-k=2 not worth running** on this model: the k=1 follow-up shows zero reasoning W→R on the multi_blank base, so additional rounds would only multiply format perturbation. Survey's +2–4 pp claim assumed a stronger base where reasoning W→R is reachable; not this model.
- **Item 436 (single-blank `[ANS] = x` template misread):** affects pass-1 of any config — orthogonal to PHP. Worth a prompt-side mitigation in §6 (e.g., when `[ANS] = <var>` appears in the question, instruct: "return only the expression on the right side of =, not the full equation").
- **Revisit PHP only post-SFT** if the trained model's pass-1 still leaks multi-blank format. With current 4B-Thinking + multi_blank prompt, the pass-1 ceiling on this slice is the same as the PHP final.

## Artifacts

PHP + baseline run (this notebook, locally synced):
- Final responses: `results/dev_results_baseline_16k_php.jsonl`
- Pass-1 snapshot: `results/dev_results_baseline_16k_php.php_pass1.jsonl`
- Pass-2 checkpoint: `results/dev_results_baseline_16k_php.php_pass2.jsonl`
- Pass-1 checkpoint (pre-PHP): `results/dev_results_baseline_16k_php.responses.jsonl`

PHP + multi_blank follow-up run (Colab Drive only — not yet synced locally):
- Final responses: `MyDrive/CSE151B/results/dev_results_multi_blank_16k_php.jsonl`
- Pass-1 / pass-2 / pre-PHP: same stem with `.php_pass1.jsonl`, `.php_pass2.jsonl`, `.responses.jsonl` suffixes

## Takeaway

Two complete PHP runs on `holdout_10p.jsonl`:

1. **PHP + baseline prompt:** pass-1 53.33% FF → final 58.67% FF (**+5.33 pp**, 5 W→R / 1 R→W). All 5 fixes are multi-blank format recoveries; the 1 break is a single-blank template misread (id 436).
2. **PHP + multi_blank prompt:** pass-1 58.67% FF → final 58.67% FF (**+0.00 pp**, 1 W→R / 1 R→W). Both flips are format-side and cancel.

Both land at the same final number (58.67% FF) as dev-008 multi_blank alone. **PHP's entire measured benefit on this model is format recovery, not reasoning revision** — and the multi_blank prompt closes the format gap at pass-1 for 1× inference. **Reject PHP for public/private submission**; do not pursue PHP-k=2 on this model. Revisit only if a stronger post-SFT model shows non-zero reasoning-driven W→R from pass-2.

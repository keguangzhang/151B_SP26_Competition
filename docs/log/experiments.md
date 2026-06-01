# Experiment registry

Master table of inference and training runs. **Detailed notes:** [`runs/`](runs/). **Decisions:** [`decisions.md`](decisions.md).

### Eval holdout (`data/eval/holdout.jsonl`)

Stratified holdout from public, seed 42 — **112 rows** at 10% or **225 rows** at 20% (canonical for SFT). Build: `scripts/build_eval_holdout.py`. Older runs logged as `data/dev.jsonl` (same sampling logic, renamed path).

Watch sets for SFT monitoring: `data/eval/watch_q4_long.jsonl` (30), `data/eval/watch_multi_blank_ge3.jsonl` (20) — see `data/eval/watch_manifest.json`.

| ID | Date | Eval set | N | Change (one line) | MCQ | Free-form | Overall | Δ overall | Artifacts | Status | Notes |
|----|------|----------|---|-------------------|-----|-----------|---------|-----------|-----------|--------|-------|
| [dev-001](runs/dev-001-baseline.md) | — | `dev.jsonl` | 112 | Starter prompt + decoding (`max_tokens` implicit 4k path) | 29.73% | 52.00% | 44.64% | — | `results/dev_results.responses.jsonl` | baseline | |
| [dev-002](runs/dev-002-mcq-prompt-temp.md) | — | `dev.jsonl` | 112 | §1.3 MCQ format clause + `temp=0.2` MCQ-only | 27.03% | 54.67% | 45.54% | +0.90 pp | — | rejected | Flat on dev; see [D002](decisions.md#d002) |
| [dev-003](runs/dev-003-max-tokens-8k.md) | — | `dev.jsonl` | 112 | `max_tokens` 4096 → **8192** | 54.05% | 54.67% | 54.46% | **+9.82 pp** | — | **shipped** | Confirmed on public → pub-001 |
| [dev-004](runs/dev-004-guided-decoding-10pct.md) | — | `dev.jsonl` | 112 | §1.1 MCQ tail regex + 8k | 51.35% | 54.67% | 53.57% | −0.89 pp | — | done | No lift at n=37 MCQ |
| [dev-005](runs/dev-005-guided-decoding-20pct.md) | — | `dev.jsonl` | 225 | Same as dev-004, **20%** dev slice | 53.33% | 52.00% | 52.44% | — | — | done | Still flat vs pub-001 MCQ |
| [dev-006](runs/dev-006-concise-prompt.md) | 2026-05-23 | `dev.jsonl` | 225 | §1.2 "concise" system prompt (non-repetitive, commit once identified) | 48.00% | 54.67% | 52.44% | −0.2 pp | `results/dev_results_concise.jsonl` | rejected | No MCQ gain; truncation is structural, not prompt-addressable |
| [dev-007](runs/dev-007-max-tokens-16k.md) | 2026-05-23 | `dev.jsonl` | 225 | `max_tokens` 8192 → **16384**, baseline prompts (20% slice) | **70.67%** | 54.67% | **60.00%** | **+7.56 pp** vs dev-006 | `results/dev_results_baseline_16k.jsonl` | done | Validates §1.1; run pub-002 on full public |
| [dev-008-baseline-smoke](runs/dev-008-baseline-smoke.md) | 2026-05-24 | `dev.jsonl` smoke | 20 | §1.3 smoke — **baseline** prompts, multi-blank FF only (16k) | — | **30.00%** | **30.00%** | — | `results/dev_results_baseline_16k_smoke.jsonl` | done | A/B control for §1.3; 6/20 multi-blank |
| [dev-008-smoke](runs/dev-008-smoke.md) | 2026-05-24 | `dev.jsonl` smoke | 20 | §1.3 — **multi_blank** prompt (`\\boxed{a}, \\boxed{b}` judger-compatible) | — | **40.00%** | **40.00%** | **+10 pp** vs baseline smoke | `results/dev_results_multi_blank_16k_smoke.jsonl` | smoke done | 8/20; → dev-008 |
| [dev-008](runs/dev-008-multi-blank-16k.md) | 2026-05-24 | `dev.jsonl` | 112 | §1.3 **multi_blank** + 16k (10% dev) | **78.38%** | **58.67%** | **65.18%** | **+4.5 pp** vs 10% baseline 16k† | `results/dev_results_multi_blank_16k.jsonl` | done | Multi-blank **50%** (19/38); pub pending |
| [dev-009](runs/dev-009-max-tokens-32k.md) | 2026-05-24 | `dev.jsonl` | 112 | **32k** `max_tokens` ablation (multi_blank, same slice as dev-008) | **78.38%** | 57.33% | 64.29% | **−0.89 pp** vs dev-008 | `results/dev_results_multi_blank_32k.jsonl` | rejected | No lift vs 16k; MCQ flat, FF −1.3 pp |
| [dev-010-bf](runs/dev-010-bf-budget-forcing.md) | 2026-05-25 | `holdout_10p.jsonl` | 112 | §A **budget forcing** (FF `Wait`; MCQ guard=0), baseline 16k | **78.38%** | **58.67%** | **65.18%** | **0 pp** vs dev-008; **+4.5 pp** vs †10% baseline 16k | `results/dev_results_baseline_16k_bf.jsonl` | done | FF 72/75 forced (96%); flat vs multi_blank on same slice |
| [dev-011-php](runs/dev-011-php.md) | 2026-05-25 | `holdout_10p.jsonl` | 112 | §1.13 **progressive-hint prompting** with per-item format clause (FF 2-pass, k=1; MCQ skipped), baseline 16k | 75.68% | **58.67%** (pass-1: 53.33%) | 64.29% (pass-1: 60.71%) | **+5.33 pp FF** vs pass-1 same run; ties dev-008 FF at 2× cost | `results/dev_results_baseline_16k_php{,.php_pass1,.php_pass2}.jsonl` | rejected | Item-diff: 5 W→R, 1 R→W; all 5 fixes are multi-blank format recoveries (multi-blank +13.16 pp); break (id 436) is single-blank template misread; superseded by dev-011-php-mb |
| [dev-011-php-mb](runs/dev-011-php.md#follow-up-run--php--prompt_variantmulti_blank-base-prompt) | 2026-05-25 | `holdout_10p.jsonl` | 112 | §1.13 PHP follow-up: **`PROMPT_VARIANT="multi_blank"` base** + per-item format clause on pass-2 (FF 2-pass, k=1; MCQ skipped), 16k | 78.38% | **58.67%** (pass-1: 58.67%) | 65.18% (pass-1: 65.18%) | **+0.00 pp FF** vs pass-1; identical to dev-008 multi_blank alone at 2× cost | `results/dev_results_multi_blank_16k_php{,.php_pass1,.php_pass2}.jsonl` (Drive only) | rejected | Item-diff: 1 W→R (id 565, format recovery), 1 R→W (id 158, pass-2 doubled answer set), net 0; confirms PHP's entire measured lift is format recovery, not reasoning — drop from public/private candidates |
| [dev-012-sc5](runs/dev-012-sc5.md) | 2026-05-26 | `holdout_10p.jsonl` | 112 | **K=5 self-consistency** majority vote; `multi_blank` prompt, 16k; ~45 min (~5× cost) | **78.38%** | **61.33%** | **66.96%** | **+1.78 pp** vs dev-010-bf | `data/dev_results_multi_blank_16k_sc5{,.responses,.sc_traces}.jsonl` | done | Multi-blank +5.26 pp (19→21/38); single-blank +0 pp; MCQ +0 pp; all lift from multi-blank vote agreement |
| [dev-013-verify](runs/dev-013-verify-holdout-20p.md) | 2026-05-28 | `holdout_20p.jsonl` | 225 | **`verify_prompt`** — verify-before-box + multi_blank format, 16k | **78.67%** | **57.33%** | **64.44%** | **0.00 pp** vs [sft-eval-001](runs/sft-eval-001-baseline-holdout-20p.md) | `results/dev_results_verify_prompt_16k.jsonl` | rejected | 145/225 identical to `multi_blank` anchor; +1 MCQ / −1 FF / −1 multi-blank → [D011](decisions.md#d011--verify_prompt-rejected-on-holdout_20p) |
| [dev-014-precision](runs/dev-014-precision-holdout-20p.md) | 2026-05-31 | `holdout_20p.jsonl` | 225 | **`precision`** — exact-form / no-round + grader box hygiene + multi_blank layout, 16k | **76.00%** | **61.33%** | **66.22%** | **+1.78 pp** vs [sft-eval-001](runs/sft-eval-001-baseline-holdout-20p.md) | `results/dev_results_precision_16k.jsonl` | done | FF +3.33 pp (single-blank +4.41 pp); MCQ −1.33 pp; +4 items vs anchor; public confirm pending |

### Full public (`data/public.jsonl`)

| ID | Date | Eval set | N | Change (one line) | MCQ | Free-form | Overall | Artifacts | Status | Notes |
|----|------|----------|---|-------------------|-----|-----------|---------|-----------|--------|-------|
| [pub-001](runs/pub-001-full-public-8k.md) | — | `public.jsonl` | 1126 | 8k tokens, starter prompts (current baseline) | 50.40% | 53.79% | **52.66%** | `data/full_public_8k*.jsonl` | **shipped** | [`analysis/baseline-public-8k.md`](../analysis/baseline-public-8k.md) |

| [pub-002](runs/pub-002-full-public-16k.md) | 2026-05-24 | `public.jsonl` | 1126 | 16k tokens, adaptive prompts (§1.3 multi-blank for 2+ `[ANS]`), **bf16 on A100** (was L4 INT8 in pub-001) | 72.00% | 56.86% | **61.90%** | `data/full_public_16k*.jsonl` | **shipped** | [`analysis/baseline-public-16k.md`](../analysis/baseline-public-16k.md) — Δ vs pub-001 conflates 3 changes: tokens, prompt, precision+hardware |
| pub-003 | 2026-05-29 | `public.jsonl` | 1126 | 32k tokens, same adaptive prompts as pub-002 | 81.07% | 58.19% | **65.81%** | `data/full_public_32k*.jsonl` | done | [`analysis/baseline-public-32k.md`](../analysis/baseline-public-32k.md); submission path for priv-001 |

### SFT / submission

| ID | Date | Eval set | N | Change (one line) | MCQ | Free-form | Overall | Artifacts | Status | Notes |
|----|------|----------|---|-------------------|-----|-----------|---------|-----------|--------|-------|
| [sft-eval-001](runs/sft-eval-001-baseline-holdout-20p.md) | 2026-05-27 | `holdout_20p.jsonl` | 225 | Base model (no LoRA); `sft_eval.ipynb`, `multi_blank` 16k — holdout anchor | **77.33%** | **58.00%** | **64.44%** | — | Drive: `results/sft_eval/baseline/eval_0.*` | done | A/B control for sft-002a; MB 52.44%, Q4 43.33%; mean len 14.1k chars |
| [sft-002a](runs/sft-002a-openr1-1k.md) | 2026-05-26 | `holdout_20p.jsonl` | 225 | OpenR1 1k bf16 LoRA (`openr1_1k` × 1 epoch); eval `final_adapter`, `multi_blank` 16k | **77.33%** | **58.00%** | **64.44%** | **0.00 pp** vs [sft-eval-001](runs/sft-eval-001-baseline-holdout-20p.md); −0.7 pp vs dev-008 (10% slice) | Drive: `results/sft_eval/openr1_1k/eval_0.*` | **flat** | Watch: Q4 40%, MB≥3 40%; MB 53.66%; shorter traces vs base → [D010](decisions.md#d010--sft-002a-openr1-1k-flat--do-not-scale-to-5k-yet) |
| [sft-003](runs/sft-003-openmath-1k.md) | 2026-05-27 | `holdout_20p.jsonl` | 225 | OpenMathReasoning 1k bf16 LoRA (`openmath_1k` × 1 epoch); eval `final_adapter`, `multi_blank` 16k | **76.00%** | **54.67%** | **61.78%** | **−2.66 pp** vs [sft-eval-001](runs/sft-eval-001-baseline-holdout-20p.md); **−2.66 pp** vs [sft-002a](runs/sft-002a-openr1-1k.md) | Drive: `results/sft_eval/openmath_1k/eval_0.*` | **regressive** | Watch: Q4 40%, MB≥3 40%; MB 50.00%; mean len 10.1k chars (−28% vs base 14.1k); worse than both base model and OpenR1 run |
| [sft-005](runs/sft-005-openmath-geo-1k.md) | 2026-05-28 | `geometry_dev` + `holdout_20p` | 133 / 225 | OpenMathReasoning **geometry-only** 1k (`openmath_geo_1k` × 1 epoch); 12k–28k traces | **77.33%** | **58.00%** | **64.44%** | **0.00 pp** vs [sft-eval-001](runs/sft-eval-001-baseline-holdout-20p.md); geo **+1.51 pp** vs pub-002 (53.38%) | Drive: `results/sft_eval/openmath_geo_1k/eval_0.*` | **flat** | Holdout: 145/225 = base; geo 73/133 (+1.51 pp); MCQ +6.82 pp geo / 0 pp holdout; mean len 10.1k holdout (−28% vs base 14.1k) |
| [sft-006](runs/sft-006-openmath-seq-1k.md) | 2026-05-28 | `sequences_dev` + `holdout_20p` | 74 / 225 | OpenMathReasoning **sequences-only** 1k (`openmath_seq_1k` × 1 epoch); eval `final_adapter`, `multi_blank` 16k | seq **73.33%** / hold **72.00%** | seq **35.71%** / hold **54.67%** | seq **66.22%** / hold **60.44%** | seq **+10.81 pp** vs pub-002; hold **−4.00 pp** vs [sft-eval-001](runs/sft-eval-001-baseline-holdout-20p.md) | Drive: `eval_sequences_0.*`, `eval_holdout_0.*` | **mixed** | Primary win; holdout MCQ −5.33 pp (fail gate); mean len 20.7k seq / 10.0k holdout |
| [sft-007](runs/sft-007-openmath-weak-5k.md) | 2026-05-30 | `holdout_20p` + `geometry_dev` + `prob_stats_dev` | 225 / 133 / 205 | OpenMath **weak-topic + anchor** ~5k (`openmath_sft007_5k`); gentle LoRA; §8 eval in `sft_train.ipynb` | — | — | — | TBD | `scripts/build_sft_corpus_sft007.py`; Drive: `checkpoints/openmath_sft007_5k/` | **planned** | Corpus: topic_classify geo/prob/trig + anchor; gates 64.44% / 77% MCQ |
| sft-001 | — | — | — | Numina-only QLoRA (planned) | — | — | — | TBD | planned | [`sft/pipeline.md`](../sft/pipeline.md) |
| sft-prep-001 | 2026-05-21 | — | 23,089 ready | Numina clean Step 2 + §5.2 audit | — | — | — | `data/sft_sources/numina_cot_clean_*` | done | [`sft/numina-clean-audit.md`](../sft/numina-clean-audit.md) |
| sft-prep-002 | 2026-05-22 | — | 15,000 | Step 5 corpus mix (drop 426, 3× weak, seed 42) | — | — | — | `data/sft_corpus.jsonl`, `data/sft_corpus_manifest.json` | done | `scripts/build_sft_corpus.py base` |
| sft-prep-003 | 2026-05-24 | — | 18,000 | v2 supplements: 1.5k long-trace + 1.5k multi-blank → `sft_corpus_v2` | — | — | — | `data/sft_corpus_v2.jsonl`, `data/sft_corpus_v2_manifest.json`, `data/sft_sources/numina_{long_trace,multi_blank_synth}.jsonl` | done | `build_sft_corpus.py supplements`; interim long p95 4119 |
| sft-prep-003b | 2026-05-24 | — | 3,000 long ready | §5.3 heap long pass + rebuild long-trace | — | — | — | `numina_cot_clean_ready_long.jsonl`, refreshed `numina_long_trace.jsonl` | done | `build_numina_ready_long.py`; long-trace p95 trace_chars 5594 (51 rows ≥6000) |
| [priv-001](runs/priv-001-submission-32k.md) | 2026-05-30 | `private.jsonl` (LB ~30%) | 943 | pub-003 path, 32k, `submission_32k.csv` | — | — | **48.00%**† | `results/submission_32k.csv` | done | [`analysis/private-submission-32k-priv-001.md`](../analysis/private-submission-32k-priv-001.md); `notebooks/submission_analysis.ipynb` |

† Interim leaderboard unified accuracy (~30% of private until finals).

---

## How to add a run

1. Pick the next ID (`dev-008`, `pub-002`, …).
2. Add a row above with metrics and artifact paths.
3. Create `log/runs/<id>-<short-slug>.md` with setup, commands, failures, and takeaway.
4. If the run changes strategy, add an entry to [`decisions.md`](decisions.md).
5. Update [`README.md`](../README.md) **Current best** only when something is **shipped**.

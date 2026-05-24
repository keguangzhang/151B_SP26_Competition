# Experiment registry

Master table of inference and training runs. **Detailed notes:** [`runs/`](runs/). **Decisions:** [`decisions.md`](decisions.md).

### Dev slice (`data/dev.jsonl`)

Stratified dev slice, seed 42 — **112 rows** at 10% (`DEV_FRACTION=0.10`) or **225 rows** at 20%. See `notebooks/dev.ipynb`.

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

### Full public (`data/public.jsonl`)

| ID | Date | Eval set | N | Change (one line) | MCQ | Free-form | Overall | Artifacts | Status | Notes |
|----|------|----------|---|-------------------|-----|-----------|---------|-----------|--------|-------|
| [pub-001](runs/pub-001-full-public-8k.md) | — | `public.jsonl` | 1126 | 8k tokens, starter prompts (current baseline) | 50.40% | 53.79% | **52.66%** | `data/full_public_8k*.jsonl` | **shipped** | [`analysis/baseline-public-8k.md`](../analysis/baseline-public-8k.md) |

### SFT / submission

| ID | Date | Eval set | N | Change (one line) | MCQ | Free-form | Overall | Artifacts | Status | Notes |
|----|------|----------|---|-------------------|-----|-----------|---------|-----------|--------|-------|
| sft-001 | — | — | — | Numina-only QLoRA (planned) | — | — | — | TBD | planned | [`sft/pipeline.md`](../sft/pipeline.md) |
| sft-prep-001 | 2026-05-21 | — | 23,089 ready | Numina clean Step 2 + §5.2 audit | — | — | — | `data/sft_sources/numina_cot_clean_*` | done | [`sft/numina-clean-audit.md`](../sft/numina-clean-audit.md) |
| sft-prep-002 | 2026-05-22 | — | 15,000 | Step 5 corpus mix (drop 426, 3× weak, seed 42) | — | — | — | `data/sft_corpus.jsonl`, `data/sft_corpus_manifest.json` | done | `scripts/build_sft_corpus.py` |
| priv-001 | — | `private.jsonl` | — | Leaderboard submission | — | — | — | `results/submission.csv` | planned | `notebooks/submission.ipynb` |

---

## How to add a run

1. Pick the next ID (`dev-008`, `pub-002`, …).
2. Add a row above with metrics and artifact paths.
3. Create `log/runs/<id>-<short-slug>.md` with setup, commands, failures, and takeaway.
4. If the run changes strategy, add an entry to [`decisions.md`](decisions.md).
5. Update [`README.md`](../README.md) **Current best** only when something is **shipped**.

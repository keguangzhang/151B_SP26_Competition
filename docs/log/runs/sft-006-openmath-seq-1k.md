# sft-006 — OpenMathReasoning sequences-only 1k (narrow-domain SFT probe)

**Date:** 2026-05-28  
**Registry:** [sft-006](../experiments.md#sft-006)  
**Status:** done — sequences primary + holdout guardrail

## Hypothesis

1k **sequences/recurrences-only** OpenMathReasoning traces (12k–28k chars) can shift accuracy on the full-public sequences dev set (~74 ids). High signal density (~1000:74 train:eval). Mirrors [sft-005](sft-005-openmath-geo-1k.md) geometry probe on a second weak topic.

**Context:** sft-005 gave +1.51 pp on geometry with flat holdout. sft-006 tests whether narrow-domain SFT transfers on an MCQ-heavy slice (60/74 MCQ).

## Setup

| Field | Value |
|-------|--------|
| **Corpus** | `data/sft_corpus_openmath_seq_1k.jsonl` — build via `notebooks/sft_data_prep.ipynb` §10.4 |
| **Manifest** | `data/sft_corpus_openmath_seq_1k_manifest.json` (`corpus_id: sft-006`) |
| **Train** | `notebooks/sft_train.ipynb` — `RUN_NAME=openmath_seq_1k`, same LoRA config as sft-005 |
| **Primary eval** | `notebooks/sft_eval.ipynb` — `EVAL_SLICE=sequences`, `SFT_RUN_NAME=openmath_seq_1k` |
| **Eval set** | `data/eval/sequences_dev.jsonl` — 74 rows (60 MCQ, 14 free-form) |
| **Guardrail eval** | `EVAL_SLICE=holdout` on `holdout_20p.jsonl` (225 rows) |
| Dataset | `nvidia/OpenMathReasoning` `cot`; OpenMath sequence keyword filter; pass_rate ∈ [0.05, 0.70] |
| Pool gate | ≥ 3,000 qualified rows before sampling 1k |
| Trace band | 12,000–28,000 chars |
| Model | `Qwen/Qwen3-4B-Thinking-2507` + LoRA |
| Prompt / decoding | `multi_blank`, `MAX_TOKENS=16384`, same as pub-002 |

## Baseline anchors (sequences)

| Anchor | Eval set | N | Sequences acc | Notes |
|--------|----------|--:|--------------:|-------|
| pub-002 | `sequences_dev.jsonl` (OpenMath regex) | 74 | **55.41%** | Scored from `data/full_public_16k.jsonl` filtered to manifest ids |

### pub-002 on `sequences_dev.jsonl` (primary eval baseline)

Source: `data/full_public_16k.jsonl` (pub-002, 16k tokens, same prompting as sft-006 eval). All 74 manifest ids present.

| Split | Correct | N | Accuracy |
|-------|--------:|--:|---------:|
| MCQ | 38 | 60 | **63.33%** |
| Free-form | 3 | 14 | **21.43%** |
| **Overall** | 41 | 74 | **55.41%** |

Filter: `is_openmath_sequence_question()` in `scripts/sft_prompt.py` (shared with corpus build).

```bash
python scripts/build_eval_sequences_set.py --force
```

## Results

**Eval:** `notebooks/sft_eval.ipynb` — `EVAL_SLICE=sequences`, `openmath_seq_1k/final_adapter`, step 0, Colab A100 bf16, 2026-05-28.  
**Artifacts:** Drive `results/sft_eval/openmath_seq_1k/eval_sequences_0.json(l)`, `eval_sequences_0.responses.jsonl`

### Primary — sequences_dev (`EVAL_SLICE=sequences`)

| Split | Correct | N | Accuracy | Δ vs pub-002 seq |
|-------|--------:|--:|---------:|-----------------:|
| MCQ | 44 | 60 | **73.33%** | **+10.00 pp** |
| Free-form | 5 | 14 | **35.71%** | **+14.28 pp** |
| **Overall** | 49 | 74 | **66.22%** | **+10.81 pp** |

| Metric | Value | Notes |
|--------|------:|-------|
| Multi-blank (2+ `[ANS]`) | 30.00% (3/10) | |
| Single-blank | 50.00% (2/4) | |
| MCQ `\boxed{}` emission | 95.00% | |
| Mean response length | 20,710 chars | +130.1% vs 9k baseline ref in notebook |

**Verdict (primary):** **Win** (+10.81 pp overall; gate: ≥ +5 pp). Largest narrow-domain SFT lift so far; exceeds sft-005 geometry (+1.51 pp).

### Watch subsets (sequences_dev ∩ watch ids)

Only ids present in the eval slice are scored.

| Watch | Correct | N | Accuracy | Notes |
|-------|--------:|--:|---------:|-------|
| Sequences (primary) | 49 | 74 | **66.22%** | Same as overall |
| Geometry (overlap) | 6 | 7 | **85.71%** | 7/133 geometry ids in sequences_dev |
| Q4 long (≥435 chars) | 0 | 0 | — | No q4_long ids in this slice |
| Multi-blank ≥3 | 0 | 1 | **0.00%** | 1/20 mb≥3 ids in sequences_dev |

### Guardrail — holdout_20p (`EVAL_SLICE=holdout`)

**Eval:** `notebooks/sft_eval.ipynb` — `EVAL_SLICE=holdout`, same checkpoint, Colab A100 bf16, 2026-05-28.  
**Artifacts:** Drive `results/sft_eval/openmath_seq_1k/eval_holdout_0.json(l)`, `eval_holdout_0.responses.jsonl`

| Split | Correct | N | Accuracy | Δ vs [sft-eval-001](sft-eval-001-baseline-holdout-20p.md) |
|-------|--------:|--:|---------:|----------------------------------------------------------:|
| MCQ | 54 | 75 | **72.00%** | **−5.33 pp** |
| Free-form | 82 | 150 | **54.67%** | **−3.33 pp** |
| **Overall** | 136 | 225 | **60.44%** | **−4.00 pp** |

| Metric | Value | Notes |
|--------|------:|-------|
| Multi-blank (2+ `[ANS]`) | 48.78% (40/82) | −3.66 pp vs sft-eval-001 |
| Single-blank | 61.76% (42/68) | −2.95 pp vs sft-eval-001 |
| MCQ `\boxed{}` emission | 96.00% | +5.33 pp vs sft-eval-001 (90.67%) |
| Mean response length | 10,022 chars | +11.4% vs 9k ref; **−29.1%** vs sft-eval-001 (14,124) |

### Watch sets (holdout_20p)

| Watch | Correct | N | Accuracy | Δ vs sft-eval-001 |
|-------|--------:|--:|---------:|--------------------:|
| Geometry (holdout ∩ geo ids) | 11 | 21 | **52.38%** | — |
| Sequences (holdout ∩ seq ids) | 9 | 13 | **69.23%** | — |
| Q4 long (≥435 chars) | 12 | 30 | **40.00%** | −3.33 pp |
| Multi-blank ≥3 | 8 | 20 | **40.00%** | 0.00 pp |

**Verdict (guardrail):** **Fail** MCQ gate (−5.33 pp vs anchor; threshold ≥ −3 pp). Overall **−4.00 pp** — worse than [sft-003](sft-003-openmath-1k.md) (−2.66 pp) on same holdout. Shorter traces than base without accuracy gain on general holdout.

### Combined go/no-go

| Gate | Result |
|------|--------|
| Primary sequences (+10.81 pp) | **Win** |
| Holdout MCQ (≥ −3 pp vs sft-eval-001) | **Fail** (−5.33 pp) |
| Holdout overall regression | **Yes** (−4.00 pp) |

**Overall:** Strong **in-slice** transfer on `sequences_dev` (+10.81 pp) but **holdout regression** — do not deploy this checkpoint for full public/private without mitigation. Per decision gates, treat as **fail** for scaling; consider 2-epoch ablation only if isolating seq-slice lift, or blend / multi-topic mix before another holdout run.

## Decision gates

| Outcome | Sequences Δ vs pub-002 on `sequences_dev` | Interpretation |
|---------|------------------------------------------:|----------------|
| **Win** | ≥ +5 pp | Domain SFT works — scale to multi-topic targeted mix |
| **Flat** | −2 to +5 pp | Inconclusive — try 2 epochs or 2k seq rows |
| **Fail** | ≤ −3 pp seq **or** holdout MCQ −3 pp | Narrow SFT hurts or doesn't transfer — deprioritize SFT |

**Observed:** +10.81 pp primary (**win**) + holdout MCQ −5.33 pp (**fail**) → **mixed / do not scale** as-is.

## Artifacts

- Corpus: `notebooks/sft_data_prep.ipynb` §10.4 (`RUN_OPENMATH_SEQ_1K=True`)
- Sequences eval: `python scripts/build_eval_sequences_set.py --force`
- Checkpoint: `checkpoints/openmath_seq_1k/final_adapter`
- Results: `results/sft_eval/openmath_seq_1k/eval_sequences_0.json(l)`, `eval_holdout_0.json(l)`

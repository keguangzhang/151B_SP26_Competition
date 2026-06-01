# sft-007 — OpenMath weak-topic SFT with general anchor + gentle LoRA

**Date:** 2026-05-30
**Status:** design approved, pending implementation plan
**Owner:** a1yin
**Run ID:** `sft-007`

## Goal

One-shot SFT run, ~1 day budget, to produce the **first holdout-non-regressing LoRA adapter**
and lift unified accuracy on the weak topics from the 16k/32k baseline analysis (geometry,
probability/stats, trigonometry). Ship to `private.jsonl` only if it clears the base model's
holdout accuracy without regression; otherwise fall back to the shipped base-model submission
(pub-002, 61.90%).

## What the prior SFT runs actually showed

Empirical results on `holdout_20p` (225 rows; base = 64.44%, base mean trace = **14,124 chars**):

| Source | Holdout acc | Δ vs base | Holdout trace len | On-topic |
|---|---|---|---|---|
| OpenR1 1k (sft-002a) | 64.44% | 0.00pp | 9,595 (−32%) | — |
| OpenMath gen 1k (sft-003) | 61.78% | −2.66pp | ~10.1k (−28%) | — |
| OpenMath geo 1k (sft-005) | 64.44% | 0.00pp | ~10.1k (−28%) | geo +1.5pp |
| OpenMath seq 1k (sft-006) | 60.44% | −4.00pp | 10,022 (−29%) | seq dev **+10.8pp** |

Three facts drive this design:

1. **Holdout deltas are within noise.** At p≈0.64, n=225, the 95% CI half-width is ~±6.3pp.
   −2.66, −4.00, and 0.00 are statistically indistinguishable from each other and from zero.
   There is **no robust evidence** that any 1k SFT run regresses holdout — or that OpenR1 is
   safer than OpenMath. Holdout's job here is a **non-regression guard**, not a fine-grained
   ranking signal.

2. **The only above-noise signal is the on-topic gain.** sft-006 produced **+10.8pp on a
   dedicated sequences dev set**. Targeted weak-topic SFT moves the targeted topic; that is the
   lever this run pulls (now aimed at geometry / prob-stats / trig instead of sequences).

3. **Trace-length collapse is a universal side-effect, not a cause.** *Every* SFT run shrinks
   holdout traces ~28–32% vs the 14.1k base — including the harmless OpenR1 run, which collapses
   the **most** (−32%) yet loses zero accuracy. Trace length does **not** predict accuracy and
   **must not** be a ship gate. *(An earlier draft of this spec wrongly claimed OpenR1 "grew"
   trace length — a misread of `+6.6% vs an arbitrary 9000-char notebook constant`, not vs the
   true 14.1k base. Corrected.)* The `MAX_SEQ_LENGTH=8192` filter is also not a factor — measured
   drop rate at 8192 is ~0% for these corpora.

**Single-source decision (all OpenMath, no OpenR1).** With the trace-length theory retracted, the
reasons to pay for OpenR1 are gone: (a) holdout safety is indistinguishable between sources, and
(b) trace quality is equal — **all three corpora are 100% correctness-aligned** (final `\boxed`
== gold answer in 1000/1000 rows each; the pass-rate band selects hard problems but every kept
trace reaches the right answer). So the corpus is **single-source OpenMath**, which is coherent in
style and maximizes weak-topic relevance. The anti-forgetting role OpenR1 would have played is
filled by a **general / strong-topic OpenMath slice** instead.

## Design

### 1. Corpus (~5k rows), all OpenMath, rebuilt with the current `topic_classify.py`

The current `topic_classify.py` is more accurate than the simpler classifier used to build the
original geo/seq slices, so all weak-topic slices are rebuilt with it.

| Slice | Target rows | Role |
|---|---|---|
| Probability/stats | ~1.3k | weak topic — largest bucket (49.8%, n=205) → most rows |
| Geometry | ~1.2k | weak topic (52.8%, n=108) — re-extracted with accurate classifier |
| Trigonometry | ~1.0k | weak topic (53.9%, n=65) — rare; backfill if short |
| General / strong-topic | ~1.5k | breadth anchor — guards against forgetting integration/algebra/MCQ |

Sequences is **dropped** — fine at 16k/32k per the baseline analysis.

**The general/strong-topic slice is the anti-regression anchor** (it replaces the OpenR1 anchor
from earlier drafts). It keeps the model exposed to broad/strong topics so a corpus of only-hard
weak topics doesn't erode the strengths.

#### Build method — single streaming pass

prob/stats and trig have no ready pools and are rare in OpenMath (geo/seq each needed a
300–600k-row filtered scan). Do **one** streaming pass over OpenMathReasoning:

1. For each candidate row, classify the question with `topic_classify.py`.
2. Route to the geo / prob-stats / trig bucket (and optionally collect strong-topic rows for the
   breadth slice in the same pass).
3. Apply the quality filters used by the prior OpenMath builds:
   - trace-char band (default ~12k–28k; **trig only** may widen down to ~8k to fill quota),
   - pass-rate band (`min_pass_rate≈0.05`, `max_pass_rate≈0.7`),
   - correctness alignment (final `\boxed` == gold answer — already 100% in prior builds),
   - decontamination against `public.jsonl` question keys.
4. Stop when the weak-topic buckets reach target (or input exhausted).

The breadth slice may instead be sampled directly from the existing
`data/sft_sources/openmath_reasoning_ready.jsonl` pool (already on disk, topic-mixed, no scan) —
whichever the rebuild produces. Record the actual per-slice split in the corpus manifest.

**Trig backfill rule:** if the scan cannot fill ~1k trig rows, backfill the deficit from
prob/stats + geometry (and/or widen the trig trace-char floor to ~8k). Record in the manifest.

#### Corpus format (unchanged from prior runs)

- Rows are full long CoT: `<think> … </think>` then final `\boxed{}`.
- All free-form (the weak topics are free-form in the source). MCQ coverage is **not** addressed
  in this run (out of scope — see below); gentle LoRA is relied on to keep MCQ from regressing.
- Same row schema and `thinking_template` as the existing OpenMath corpora.

### 2. Training config — gentle LoRA (minimize forgetting / overfit)

Dampen the update by **magnitude** (low lr/α) rather than by freezing a subspace, so the MLP
retains capacity to learn the weak-topic reasoning while the base model's broad capability is
disturbed as little as possible. This directly reduces the regression risk seen in the
larger-lr/α prior runs.

| Param | Current (sft-006) | sft-007 | Rationale |
|---|---|---|---|
| `LEARNING_RATE` | 1e-5 | **5e-6** | halve step size → less drift from base |
| `LORA_ALPHA` | 64 (α/r=2) | **32 (α/r=1)** | weaker adapter contribution |
| `LORA_DROPOUT` | 0.05 | **0.1** | mild regularization |
| `LORA_R` | 32 | 32 (keep) | capacity for topical learning |
| `LORA_TARGETS` | all 7 | **all 7 (keep)** | MLP needed to learn new topics; soften via lr/α, not by dropping modules |
| `MAX_SEQ_LENGTH` | 8192 | **16384** | cheap insurance (measured ~0% drop today) |
| `NUM_TRAIN_EPOCHS` | 1 | 1 | unchanged |

**Rejected alternative — attention-only modules (`q/k/v/o`):** preserves base by freezing the MLP
knowledge stores, but (a) weak at teaching new math (geometry/trig procedures live in the MLP) and
(b) gives up topical-learning capacity for a forgetting-defense the lr/α dampening already
provides. Soft-all-7 serves both the goal (topical learning) and the constraint (minimal drift).

All other training-stack settings unchanged from `notebooks/sft_train.ipynb`:
`assistant_only_loss=True`, `packing=False`, bf16, cosine schedule, warmup 3%,
`per_device_batch=1`, `grad_accum=16`, A100 only.

### 3. Decision gate — eval on `holdout_20p` (225 rows) vs base 64.44%

**Accuracy-only gate.** Trace length is recorded as a descriptive metric but is **not** a gate
(every adapter collapses it ~30% regardless of quality).

Ship-or-kill — both must hold to ship to private:

- **Overall ≥ 64.44%** — non-regression vs base ([sft-eval-001](../../log/runs/sft-eval-001-baseline-holdout-20p.md)). Because holdout deltas are noise-level at n=225, treat "within ~−1pp" as effectively flat/acceptable; reject only a clear drop.
- **MCQ ≥ 77%** — no MCQ erosion (sft-006 lost −5.33pp here; the prime regression watch).

Upside signals (not gates, but the reason to run): **`geometry_dev` and a prob/stats dev slice
should beat base** — this is the only above-noise signal we expect to move. Also report the
`watch_q4_long` and `watch_multi_blank_ge3` watch sets.

**Decision:**
- **Gate passes AND a weak-topic dev slice improves** → run private submission via
  `notebooks/submission.ipynb` with the adapter.
- **Gate passes but no topical upside** → judgment call; default to base submission (no reason to
  ship an adapter that only matches base on a one-shot).
- **Gate fails** → fall back to the shipped base-model submission (pub-002, 61.90%).

### 4. Sequence (fits one day)

1. **Build corpus** — single classified scan (~1–2 hr, long pole) → ~5k corpus + manifest.
2. **Train** — A100, 1 epoch (~1–2 hr); save `final_adapter` + checkpoints.
3. **Eval** — `final_adapter` on `holdout_20p` + geometry/prob-stats dev slices (~45 min).
4. **Decide** against the gate → submit private or fall back.

## Scope / out of scope

- **In scope:** the three weak-topic slices + general anchor slice (all OpenMath), gentle-LoRA
  config, one training run, accuracy-gated decision, private submission or fallback.
- **Out of scope (explicitly deferred):**
  - **MCQ coverage in the corpus.** The 0%-MCQ-vs-33%-target gap is real, but it tracks the
    OpenMath source more than coverage (OpenR1 was also 0% MCQ yet held MCQ), and adding MCQ
    training data requires rejection-sampling traces on non-holdout public (hours of generation)
    or synthetic MCQ construction — both blow the one-day budget. This run relies on gentle LoRA
    to hold MCQ; if MCQ still regresses, MCQ coverage via self-distillation on non-holdout public
    is the first follow-up.
  - Dedicated prob/stats HF scale-up beyond what the single pass yields.
  - Any change to the inference stack (stays pub-002: `multi_blank`, 16k, temp 0.6).

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Trig slice underfills | medium | backfill rule (prob/stats + geo, or widen trace floor to 8k) |
| General/MCQ regression from OpenMath distribution shift | medium | general anchor slice + gentle LoRA; accuracy gate kills it; fall back to base |
| No topical upside (gain stays within noise) | medium | dev slices are the read; if flat, ship base — no loss vs current 61.90% |
| Scan overruns the day | low–medium | cap input rows; weak-topic quota lets the scan stop early |
| A100 unavailable | low | abort session per pipeline.md (no L4 / 4-bit fallback) |

## Artifacts produced

- `data/sft_corpus_sft007_*.jsonl` + `..._manifest.json` (final per-slice split recorded).
- Rebuilt weak-topic ready pools / slices under `data/sft_sources/`.
- Adapter under Drive `checkpoints/sft-007/final_adapter`.
- Eval outputs under `results/sft_eval/sft007/`.
- Registry row `sft-007` in `docs/log/experiments.md` + run note under `docs/log/runs/`.
- Decision entry in `docs/log/decisions.md` (ship vs fall back).

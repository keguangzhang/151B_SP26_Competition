# SFT pipeline plan

Plan for supervised fine-tuning `Qwen/Qwen3-4B-Thinking-2507` to improve unified accuracy on `public.jsonl` and `private.jsonl`.

**2026-05-26 pivot:** primary plan is now a small-corpus SFT on **pre-distilled DeepSeek-R1 traces** (`open-r1/OpenR1-Math-220k`), starting with a 1k probe. The previously planned Numina-only sft-001 (corpus `data/sft_corpus_v2.jsonl`, 18k rows, prepared) is retained as a fallback if the distilled-trace path does not lift.

The guiding principle is unchanged: train on fewer high-quality long reasoning traces before adding noisy coverage. The pivot is motivated by:

- **Inference ceiling confirmed.** dev-010-bf (budget forcing), dev-011-php (progressive hints), and dev-012-sc5 (self-consistency K=5) all converge at **~65–67% on `holdout_10p`** (≈ +1 to +2 pp full public projection over pub-002). Pure-inference scaffolds are saturated on this base.
- **Distilled traces beat human/synthetic Numina traces per dollar.** `OpenR1-Math-220k` config `default` is ~94k DeepSeek-R1 traces pre-filtered for correctness against gold answers — same teacher source the entire R1-Distill student series is trained on. Cost: $0.
- **LoRA capacity argument.** LoRA r=32 on a 4B model saturates near ~2–3k diverse reasoning examples (LIMO at 817, s1 at 1000). A small probe is the right first move; large corpora waste compute and risk overfit.
- **Course rules confirmed.** Base is locked to `Qwen/Qwen3-4B-Thinking-2507`; training on offline traces is permitted; inference at submission stays model-only with no external calls.

## Environment assumptions

- **Compute:** single GPU on Colab. **A100 only (40 GB or 80 GB).** Training is bf16 LoRA — no 4-bit fallback, no L4 path. If A100 is unavailable, abort the session.
- **Persistence:** Google Drive at `/content/drive/MyDrive/CSE151B/` mirrors the repo layout used by `notebooks/dev.ipynb` and `notebooks/submission.ipynb`. `data/eval/` (holdout + watch sets), checkpoints, eval outputs, and corpus manifests live there so disconnects are recoverable.
- **Workflow:** notebook-driven: `notebooks/sft_data_prep.ipynb`, `notebooks/sft_train.ipynb`, `notebooks/sft_eval.ipynb`, `notebooks/submission.ipynb`.
- **Submission path:** final private inference remains model-only generation. No tool-augmented inference, no external APIs, no calculator loop.

---

## Goal

Lift unified accuracy from the current **61.90%** baseline (72.00% MCQ / 56.86% free-form, [pub-002](../log/experiments.md#pub-002)) without collapsing Qwen3-Thinking's reasoning style or regressing MCQ.

**Target weaknesses** (from [`baseline-public-16k.md`](../analysis/baseline-public-16k.md), filtered to slices with tight enough N to trust):

| Slice | N | Acc | Headroom |
|---|--:|---:|---|
| Q4 question length (≥435 chars) | 281 | 43.8% | largest single addressable bucket |
| Multi-blank ≥6 blanks | 63 | ~34% | sharpest per-slice gap |
| Probability / stats | 205 | 49.8% | largest named weak topic (`weighted_v1`) |
| Geometry | 108 | 52.8% | clear weakness below 61.9% overall |
| MCQ "think finished, wrong boxed" | 54 | reasoning failure | method-agnostic — needs better reasoning, not better prompts |

Topic taxonomy: `scripts/topic_classify.py` (`weighted_v1`). Residual `other` is 14.8% of rows (167 / 1126), not the old 51.6% catch-all. Small-n topics (limits, complex analysis, derivatives n ≤ 21) still have wide CIs.

First checkpoint target:

- **Primary:** no MCQ regression beyond 3 pp; no free-form regression beyond noise.
- **Secondary:** overall gain on `holdout_20p`; per-slice movement on Q4 long-context and multi-blank ≥3 watch sets.
- **Stretch:** +3 to +8 pp overall on full public after selecting the best checkpoint.

The first run answers one question cleanly: **does training on R1-distilled traces lift accuracy on this base without trace collapse or MCQ regression?**

### Expected outcome distribution

Anchored on the R1-Distill student series (DeepSeek-R1-Distill-Qwen-1.5B/7B/14B all gained 10–20 pp on AIME from R1 distillation) and LIMO/s1 evidence that small curated corpora suffice on already-strong bases.

| Outcome | Probability | Overall Δ vs pub-002 |
|---|---|---|
| Strong win | ~25% | +5 to +10 pp |
| Modest win | ~40% | +2 to +5 pp |
| Flat | ~20% | −1 to +2 pp |
| Regression | ~15% | −2 to −5 pp — keep pub-002, run recovery |

Probabilities are slightly more optimistic than the earlier Numina-only estimate because R1 is the teacher the entire R1-Distill series scaled from, and the corpus is correctness-filtered upstream rather than relying on Numina cleanup.

### Non-goals

- No RL / GRPO / DPO until SFT has a clean baseline.
- No full-parameter fine-tuning.
- No training on `public.jsonl`; it stays evaluation-only.
- No mixed-source corpus until the distilled-trace run gives a trusted reference point.
- No synthetic filler traces.
- No API teacher distillation (e.g., OpenRouter R1 rollout) until a pre-distilled-dataset attempt is exhausted — same trace source, no cost.

---

## Why change the plan

**Decision records:** [D004 — Numina-only first SFT run](../log/decisions.md#d004) (superseded as primary), [D009 — bf16 LoRA over QLoRA](../log/decisions.md#d009), and new pivot rationale below.

The earlier "clean Numina-first" plan was correct given the information at the time: course rules were uncertain on external trace sources, and the Numina cleanup work was already partially done. After:

1. course staff confirming offline trace-based training is permitted;
2. three independent inference scaffolds (BF / PHP / SC) all plateauing at ~65–67% on `holdout_10p`, exhausting the pure-inference path;
3. surveying available pre-distilled R1 trace datasets;

the highest-EV next move is a small-corpus SFT on R1-distilled traces. The Numina sft_corpus_v2 (18k, fully prepared) is retained as a fallback rather than discarded — the prep work is not wasted, it's a known-good safety floor.

---

## Source policy

| Source | Decision | Reason |
| --- | --- | --- |
| `open-r1/OpenR1-Math-220k` (config `default`, ~94k judger-correct R1 traces) | **Primary for sft-002** | DeepSeek-R1 traces directly, pre-filtered for correctness, native `<think>...</think>` format, $0 cost |
| `nvidia/OpenMathReasoning` (`cot` split, 3.2M R1+QwQ traces, AoPS/AIME/Olympiad) | **Fallback A — sft-003** | Competition-math problems; traces up to 76k chars (no 10k cap); `pass_rate_72b_tir` difficulty filter; hardest-problem coverage that OpenR1 lacks |
| `simplescaling/s1K-1.1` (1000 curated R1+Gemini-Thinking traces) | **Fallback B — sft-004** | Demoted: max trace ~2.4k chars — shorter than OpenR1; try only if OpenMathReasoning also flat |
| `data/sft_corpus_v2.jsonl` (Numina, 18k, prepared) | **Fallback C — sft-001** | Already built and validated; safety floor if all distilled-trace sources fail |
| `bespokelabs/Bespoke-Stratos-17k` (R1 math+code) | Defer | Multi-domain mix; subsample if all three above plateau |
| `MATH train` original solutions | Exclude | Too short, not Thinking-style |
| `AGIEval-Math`, `GaoKao-MCQ` | Exclude | Synthetic / Chinese |
| Baseline self-distillation (MCQ) | Defer to recovery | Only if MCQ regresses post-SFT |
| OpenRouter R1 API distillation | Defer | Same trace source as OpenR1-Math-220k at $200–400; skip unless pre-distilled options exhausted |

---

## Primary plan: sft-002 — OpenR1-Math-220k

### Phase 1: 1k probe (sft-002a)

**Goal:** answer "do R1-distilled traces lift `holdout_20p` on this base?" in one A100 session.

**Data prep** (`notebooks/sft_data_prep.ipynb`, new section):

```python
from datasets import load_dataset

ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")

def pick_correct_generation(ex):
    for gen, ok in zip(ex["generations"], ex["correctness_math_verify"]):
        if ok and len(gen) <= 10_000:
            return gen
    return None

filtered = ds.filter(lambda ex: pick_correct_generation(ex) is not None)
sample = filtered.shuffle(seed=42).select(range(1000))

def to_sft_row(ex):
    trace = pick_correct_generation(ex)
    if "<think>" not in trace:
        trace = f"<think>{trace}</think>"
    return {"prompt": ex["problem"], "response": trace}

sft_rows = sample.map(to_sft_row, remove_columns=sample.column_names)
sft_rows.to_json("data/sft_corpus_openr1_1k.jsonl")
```

Filter sequence:
1. Load `open-r1/OpenR1-Math-220k` config `default` from HuggingFace.
2. Keep rows where at least one entry in `correctness_math_verify` is `True`.
3. Pick the first correct generation per row.
4. Length cap: `len(generation) <= 10_000` chars (4B student cannot stably reproduce 16k+ traces).
5. Decontam against `public.jsonl` and `private.jsonl` problem text (substring match on first 200 chars).
6. Random subsample 1000 with `seed=42`.
7. Format each row to `{"prompt": problem, "response": "<think>...reasoning...</think>\n\n<final answer with \\boxed{}>"}` matching [D005](../log/decisions.md#d005--sft-assistant-schema-explicit-redacted_thinking-wrapper).
8. Write `data/sft_corpus_openr1_1k.jsonl` and `data/sft_corpus_openr1_1k_manifest.json`.

**Manifest fields:** source dataset version, config name, sample seed, filter retention rate, length distribution (p25 / p50 / p95 of `len(response)` and `template_tokens`), public/private decontam hits, `thinking_template: "explicit_redacted_thinking"`.

**Spot check:** 10 rows by hand — verify R1 trace style is Qwen3-Thinking-compatible (long, exploratory, not Numina-template-shaped); `<think>...</think>` wrapping correct; `\boxed{}` present after closing tag; no CJK residue; no figure-dependent rows ("in the diagram", Asymptote blocks).

**Training:**

- Same config as the earlier Numina plan (bf16 LoRA r=32, see [Training config](#training-config) below) with two changes:
  - **2 epochs** (1k corpus permits more epochs without overfit; previously 18k corpus capped at 1).
  - **Checkpoint every 200 steps** (corpus much smaller; ~125 update steps per epoch at mb=1 / accum=16).
- Eval on `holdout_20p` at end of each epoch + watch sets every 200 steps.

**Compute estimate per probe attempt:** ~1.5–2h on A100.

| Stage | Time |
|---|---|
| Data prep + spot check | 30 min CPU |
| Smoke train (50 steps on 200-row subset) + 10-sample inspection | ~15 min |
| Full SFT 1k × 2 epochs (~250 update steps) | ~30–50 min |
| Eval on `holdout_20p` (225 rows × 16k tokens) | ~30 min |
| Eval on watch sets (50 rows total) | ~10 min |
| **Total** | **~1.5–2 hr A100** |

**Decision gate after sft-002a** — compare best checkpoint's `holdout_20p` numbers against three anchors:

| Anchor | Overall | MCQ | FF | Source |
|---|---|---|---|---|
| pub-002 baseline (full public) | 61.90% | 72.00% | 56.86% | shipped |
| dev-008 multi_blank 16k (10% holdout) | 65.18% | 78.38% | 58.67% | best inference-only config |
| dev-012-sc5 K=5 (10% holdout) | 66.96% | 78.38% | 61.33% | inference ceiling |

| Outcome | Δ overall on `holdout_20p` vs dev-008 baseline projection | Action |
|---|---|---|
| **Strong** | ≥ +3 pp | Scale to sft-002b (5k stratified) |
| **Modest** | +1 to +3 pp | Run SC K=5 on top of checkpoint on `holdout_10p`; if combined ≥ +3 pp → scale to sft-002b; else iterate sample seed or filter |
| **Flat** | −1 to +1 pp | Switch to sft-003 (`s1K-1.1`) |
| **Regression** | < −1 pp | Halt; diagnose format / training config; consider sft-001 (Numina) fallback |

Stop rules (apply during training, halt immediately if any trigger):

- MCQ holdout accuracy drops more than 3 pp from pub-002 baseline (72%).
- FF holdout accuracy drops more than 2 pp from base **and** mean response length drops.
- Mean response length drops more than 20% vs baseline.
- `\boxed{}` emission rate drops below baseline.
- `<think>` opener/closer emission rate drops below 95% (schema breakage).
- Two consecutive eval points show no improvement and qualitative samples look shorter.

### Phase 2: 5k stratified scale-up (sft-002b)

**Trigger:** sft-002a passes the strong-win or modest-win gate.

**Sampling strategy** — target weak slices identified in pub-002 analysis:

| Stratum | Rows | Filter |
|---|--:|---|
| Long-trace | ~1500 | OpenR1 problem text ≥ 435 chars (matches Q4 weak slice); fallback to longest correct generations |
| Multi-answer | ~1500 | Rows with ≥2 `\boxed{}` in correct generation OR problem text with multiple sub-questions |
| Geometry-flavored | ~1000 | Keyword match: `triangle`, `polygon`, `angle`, `circle`, `quadrilateral`, `parallelogram` — exclude figure-dependent rows |
| Random fill | ~1000 | Uniform from remaining filtered pool |
| **Total** | **5000** | Stratified shuffle, seed 42 |

Reject diagram-dependent rows (Asymptote code blocks, "in the figure shown", "in the diagram") — text-only model has no figure access.

**Training:** same config as sft-002a, **2 epochs**, checkpoint every 500 steps.

**Compute estimate:** ~3–4h A100.

**Decision gate:** ship if `holdout_20p` lift ≥ +3 pp overall AND no slice regression > 3 pp (MCQ, FF, Q4-long, multi-blank ≥3); else iterate or fall back.

---

## Fallback A: sft-003 — `nvidia/OpenMathReasoning`

**Trigger:** sft-002a flat (confirmed: [sft-002a](../log/runs/sft-002a-openr1-1k.md) 0.00 pp vs base).

**Why this dataset:** 3.2M CoT solutions from DeepSeek-R1 + QwQ-32B on AoPS/AIME/Olympiad competition problems. Key advantage over OpenR1: traces up to 76k chars with no artificial 10k cap — covers the long-tail distribution the base model produces at inference (p90 ~35k chars on pub-002). `pass_rate_72b_tir` field allows explicit difficulty targeting of the weak slices (Q4 long, multi-blank ≥3). OpenR1's 10k cap excluded the top 10% hardest problems entirely; this corpus has them.

**Data prep:**

```python
from datasets import load_dataset

# Use CoT split only — TIR requires tool calls not available at submission inference
ds = load_dataset("nvidia/OpenMathReasoning", "cot", split="train")

# Verify field names on first load — likely: problem/question, generated_solution, pass_rate_72b_tir
# print(ds.features)

MIN_CHARS = 8_000    # floor: target traces above OpenR1's hard cap
MAX_CHARS = 28_000   # ceiling: ~7k tokens response budget within 8192 seq length
MAX_PASS_RATE = 0.7  # exclude trivial problems (high pass rate = easy)
MIN_PASS_RATE = 0.05 # exclude unsolvable problems

def is_valid(ex):
    resp = ex.get("generated_solution", "")
    pr = ex.get("pass_rate_72b_tir", None)
    if pr is None or not (MIN_PASS_RATE <= pr <= MAX_PASS_RATE):
        return False
    if not (MIN_CHARS <= len(resp) <= MAX_CHARS):
        return False
    return True

filtered = ds.filter(is_valid)
sample = filtered.shuffle(seed=42).select(range(1000))

def to_sft_row(ex):
    resp = ex["generated_solution"]
    if "<think>" not in resp:
        resp = f"<think>{resp}</think>"
    return {"prompt": ex["problem"], "response": resp}

sft_rows = sample.map(to_sft_row, remove_columns=sample.column_names)
sft_rows.to_json("data/sft_corpus_openmath_1k.jsonl")
```

Filter sequence:
1. Load `nvidia/OpenMathReasoning` `cot` split from HuggingFace.
2. `pass_rate_72b_tir` in `[0.05, 0.70]` — hard but solvable problems only.
3. Length: `8_000 ≤ len(generated_solution) ≤ 28_000` chars — targets long-trace hard problems, stays within 8192 token training budget.
4. Decontam against `public.jsonl` and `private.jsonl` (substring match, first 200 chars).
5. Shuffle seed 42, select 1000.
6. Wrap in D005 `<think>...</think>` format if not already present.
7. Write `data/sft_corpus_openmath_1k.jsonl` + manifest.

**No MCQ needed** — MCQ already at 77%; this corpus is free-form only (format restrictions exclude MCQ by design). This is fine: weak slices are Q4 long and multi-blank ≥3, both free-form.

**Training:** same SFT config, 2 epochs. Note: longer traces mean longer sequences — confirm A100 memory budget holds with a smoke run before full training.

**Decision gate:** same anchors and outcome table as sft-002a.

---

## Fallback B: sft-004 — `s1K-1.1`

**Trigger:** both sft-002a and sft-003 flat.

**Note:** demoted from original Fallback A position. Max trace ~2.4k chars — shorter than OpenR1's 5.7k mean. Try only to isolate whether curation quality (not trace length) is the bottleneck.

**Data prep:**

```python
ds = load_dataset("simplescaling/s1K-1.1", split="train")
def to_sft_row(ex):
    trace = ex["deepseek_thinking_trajectory"]
    final = ex["deepseek_attempt"]
    return {"prompt": ex["question"], "response": f"<think>{trace}</think>\n\n{final}"}
```

Same decontam, D005 wrapper. Write `data/sft_corpus_s1k.jsonl`.

**Training:** same SFT config, 2 epochs.

**Decision gate:** same anchors and outcome table as sft-002a.

---

## Fallback C: sft-001 — Numina `sft_corpus_v2`

**Trigger:** sft-002a, sft-003, and sft-004 all flat or regressive.

**Status:** corpus fully prepared. `data/sft_corpus_v2.jsonl` (18k rows: 15k Numina base + 1.5k long-trace + 1.5k multi-blank synthetic). Built by `scripts/build_sft_corpus.py` per [D004](../log/decisions.md#d004); audit in [`numina-clean-audit.md`](numina-clean-audit.md).

**Training:** same SFT config, **1 epoch** (larger corpus). Compute: ~3–4h A100 training + ~2h full-public eval = ~6–8h A100 per attempt.

**Decision gate:** ship if full-public lift ≥ +2 pp AND no slice regression > 3 pp; else halt SFT work entirely and route to root-cause diagnosis (likely Qwen3 thinking schema, completion masking, or trace style mismatch).

---

## Data prep plan (general)

Use `notebooks/sft_data_prep.ipynb` as the single orchestration notebook. Each corpus (OpenR1, s1K-1.1, Numina) gets its own section under a parent header.

### Step 1: verify Qwen3 thinking template

**Status: done** — see [D005](../log/decisions.md#d005--sft-assistant-schema-explicit-redacted_thinking-wrapper). Explicit `<think>...</think>` wrapper required; reasoning inside, `\boxed{...}` after closing tag. Record `thinking_template: "explicit_redacted_thinking"` in every corpus manifest.

### Step 2: format-conform from source

For each source dataset, identify the trace field(s) and re-wrap into `{"prompt": ..., "response": "<think>{reasoning}</think>\n\n{final}"}`. R1-derived datasets typically include the `<think>` wrapping natively; verify per dataset before assuming.

### Step 3: quality filters

Accept a row only if all checks pass:

- English-only question and trace (CJK reject filter from existing Numina prep applies to all sources).
- `len(response) <= 10_000` chars (relaxed from earlier `trace_chars <= 12_000` after multi-source survey).
- `template_tokens <= 7_900` for 8k sequence-length training.
- Final answer contains parseable `\boxed{...}`.
- No `public.jsonl` or `private.jsonl` problem text overlap (substring decontam).
- For OpenR1: at least one entry in `correctness_math_verify` is `True`.

### Step 4: spot check

Sample 10 accepted rows per source. Confirm:

- trace is exploratory long-CoT, not templated answer stub;
- `<think>...</think>` wraps reasoning correctly;
- `\boxed{...}` appears after closing tag;
- no figure-dependent prompts ("in the diagram", "see figure", Asymptote blocks);
- no CJK residue.

If more than 1 row fails spot check, fix filters and rebuild.

### Step 5: manifest

Per corpus, record:

- source file path / HuggingFace dataset id + config + revision
- source row count, accepted row count, reject reasons + counts
- sample seed, target size, final row count
- length distribution: p25 / p50 / p95 of `len(response)` and `template_tokens`
- decontamination hits against `public.jsonl` / `private.jsonl`
- `thinking_template` decision
- timestamp + git commit

---

## Training config

A100-only single-GPU bf16 LoRA. No 4-bit path. [D009](../log/decisions.md#d009--bf16-lora-replaces-qlora-for-sft-001-a100-only).

- **Framework:** `trl.SFTTrainer + peft` (no `bitsandbytes`).
- **Precision:** bf16 base + bf16 LoRA + bf16 compute on A100. No quantization. Memory at 8k seq, mb=1, grad-ckpt: ~22–30 GB — fits A100 40 GB; A100 80 GB allows mb=2.
- **LoRA:** rank 32, alpha 64, dropout 0.05.
- **Targets:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Loss:** completion-only; mask prompt tokens.
- **Optimizer:** `adamw_torch_fused`.
- **Learning rate:** `1e-5`.
- **Schedule:** cosine decay, 3% warmup.
- **Sequence length:** 8192. A100 only — if Colab assigns L4, abort the session, do not downgrade.
- **Batch:** micro-batch 1, grad accum 16. On 80 GB A100, try micro-batch 2 and grad accum 8.
- **Epochs:**
  - sft-002a (1k corpus): **2 epochs**
  - sft-002b (5k corpus): **2 epochs**
  - sft-003 (1k s1K-1.1): **2 epochs**
  - sft-001 (18k Numina, fallback): **1 epoch**
- **Checkpointing:**
  - sft-002a / sft-003: every 200 steps to Drive
  - sft-002b: every 500 steps
  - sft-001 fallback: every 500 steps
- **Eval cadence:** end of each epoch on `holdout_20p` + watch sets every checkpoint.
- **Resume:** auto-detect latest checkpoint in the run directory and resume from it.

### Smoke run

Before each full training:

1. Train 50 steps on a 200-row subset of the target corpus.
2. Confirm loss decreases monotonically.
3. Generate 10 sample responses from a frozen dev slice.
4. Confirm responses still contain reasoning and valid `\boxed{...}` answers.

If smoke outputs become short answer-only completions, stop and inspect the response schema before full training.

---

## Eval harness

Use `notebooks/sft_eval.ipynb`. The eval path must match the final submission path except for loading a LoRA adapter.

Required eval sets:

- **Holdout (canonical for SFT):** `data/eval/holdout.jsonl` (alias `holdout_20p.jsonl`), 225 rows. Rebuild: `python scripts/build_eval_holdout.py --fraction 0.20 --seed 42`.
- **Watch Q4 long:** `data/eval/watch_q4_long.jsonl` — 30 frozen rows (`question` length ≥ 435 chars; 3 MCQ / 27 free-form). IDs in `data/eval/watch_manifest.json` → `watch.q4_long.ids`.
- **Watch multi-blank ≥3:** `data/eval/watch_multi_blank_ge3.jsonl` — 20 frozen rows (≥3 `[ANS]`). IDs → `watch.multi_blank_ge3.ids`. Rebuild watch: `python scripts/build_eval_watch_sets.py`.
- **Full public:** full `data/public.jsonl` after selecting a candidate checkpoint.
- **Optional stress slice:** 20–50 hand-picked weak-topic / multi-blank examples for qualitative inspection only.

Metrics per checkpoint:

| Metric | Why |
| --- | --- |
| Unified accuracy | Main score proxy. |
| MCQ accuracy | **Hard regression watch** — FF-heavy mix likely to pressure MCQ. |
| Free-form accuracy | Primary expected gain area. |
| Q4 long-context accuracy | Direct watch on long-trace lift. |
| Multi-blank ≥3 accuracy | Direct watch on multi-blank lift. |
| `\boxed{}` emission rate | Format regression detector. |
| Multi-blank `\boxed{}` count vs gold blanks | Multi-blank format integrity. |
| Mean / median response length | Trace-collapse detector. |
| `<think>` / `</think>` tag emission rate | Thinking-schema integrity detector. |
| Empty or answer-only response rate | Thinking-style failure detector. |

Stop rules: see [sft-002a Decision gate](#phase-1-1k-probe-sft-002a).

Checkpoint selection:

1. Pick best holdout checkpoint that does not violate stop rules.
2. Re-evaluate on full `public.jsonl`.
3. **Decision rule:** ship only if overall public lift ≥ +2 pp **and** no slice regression > 3 pp (MCQ, FF, Q4-long, multi-blank ≥3). Otherwise keep pub-002 baseline and route to next fallback.

---

## Stacking SC K=5 on top of SFT

[dev-012-sc5](../log/runs/dev-012-sc5.md) showed self-consistency K=5 lifts holdout by +1.78 pp at 5× inference cost on the inference-only stack. The lift was concentrated entirely in multi-blank (+5.26 pp); MCQ vote gave zero.

**Hypothesis (from dev-012 takeaway):** SFT improves per-trace quality → SC vote becomes more reliable on items that still split → multi-blank lift could climb beyond the current +5.26 pp.

**Plan:** after selecting the best SFT checkpoint, re-run SC K=5 against it on `holdout_10p` (cheaper than `holdout_20p` for SC since 5× cost), compare delta to dev-012-sc5. If SC+SFT lift exceeds SC alone by ≥ 1 pp, include SC in the submission inference path. Otherwise ship single-pass inference.

---

## Recovery / next options

Only start these after the current attempt has full eval results.

### If free-form improves but MCQ stays flat / regresses

Build a small MCQ self-distillation source from external MCQ-style problems, not `public.jsonl`:

- use English-only external problems;
- expand to 10 options only with type-coherent distractors;
- generate k baseline traces per problem;
- keep only traces whose boxed letter matches gold;
- reject short or answer-only traces;
- cap this source at 10–20% of the next corpus.

Do not revive the current `AGIEval/GaoKao` synthetic traces.

### If trace length collapses

Do not add more data. Fix the assistant schema first:

- verify `<think>` target format in tokenized output;
- inspect whether completion-only masking includes assistant tokens correctly;
- lower learning rate to `5e-6`;
- reduce epoch fraction;
- train on fewer, longer rows.

### If weak topics remain bad

Use OpenR1-Math-220k as a **problem source** for stratified resampling:

- sample weak-topic problems (geometry keyword, Q4-length, multi-answer);
- keep their correct R1 traces;
- compose a corpus weighted toward the weak slices.

This is essentially sft-002b but with higher slice weights.

### If both distilled-trace runs (sft-002, sft-003) are neutral or negative

Two paths:

1. **Run sft-001 fallback** — Numina sft_corpus_v2, already prepared. Lower expected ceiling (+2 to +4 pp) but a known-good safety net.
2. **Halt SFT work** until root cause is known. Likely causes: wrong Qwen3 thinking schema, completion masking bug, training targets not matching inference prompt, answer normalization bug.

### If everything plateaus

Move to Tier 3 (RL — GRPO/DPO). Compute is expensive on A100; revisit only after the SFT ceiling on this base is established.

---

## Distribution risks

Ranked by current likelihood given the OpenR1-primary plan.

1. **R1 trace style differs from Qwen3-Thinking's calibrated thinking.** Most insidious failure: model still reasons but in an R1 rhythm that subtly hurts on competition-style problems. Mitigation: explicit D005 `<think>` wrapping; track tag emission rates; conservative LR (`1e-5`); 2-epoch cap on small corpora.
2. **Trace length mismatch.** OpenR1's 10k char cap (sft-002a) was confirmed to pull base model output from 14k → 9.6k mean with zero accuracy gain — the model learned a shorter distribution, not better reasoning. sft-003 (OpenMathReasoning) deliberately targets 8k–28k char traces to match the inference distribution. Risk is the reverse: very long training traces may hit the 8192 token sequence budget. Mitigation: 28k char cap ≈ 7k tokens leaves ~1k token headroom for question + template; confirm with smoke run; eval mean response length vs base.
3. **MCQ regression from FF-heavy mix.** OpenR1 is overwhelmingly free-form math (AIME / MATH / competition style). MCQ at 72% is the slice with most to lose. Mitigation: hard MCQ floor stop rule (−3 pp from pub-002); no MCQ data in sft-002; recovery phase has MCQ self-distillation if needed.
4. **Trace-style collapse.** Short or wrongly formatted targets can kill thinking behavior. Mitigation: `len(response) ≥ ~2000` floor on filtered rows; validate `<think>` schema before training; smoke run before full training.
5. **Format mismatch between training and inference.** Training rows must match final `build_prompt(...)` and `apply_chat_template(...)` path. Mitigation: tokenized sample inspection before training; build through same path used by pub-002.
6. **OpenR1 problem distribution skew.** AIME-heavy, possibly less geometry / multi-blank than competition. Mitigation: stratified sampling in sft-002b prioritizes weak slices.
7. **Figure-dependent problems.** OpenR1 contains some "see diagram" / Asymptote-code problems. Mitigation: keyword reject filter.
8. **Public/private leakage.** Public remains eval-only. Mitigation: substring decontam against `public.jsonl` and `private.jsonl` problem text during corpus build.

---

## Compute budget

Single A100-40GB Colab session per probe attempt.

### sft-002a (OpenR1 1k probe)

| Stage | Time |
|---|---|
| Data prep + spot check | 30 min CPU |
| Smoke train + 10-sample inspection | ~15 min |
| Full SFT 1k × 2 epochs (~250 update steps) | ~30–50 min |
| Holdout eval (225 rows × 16k tokens) | ~30 min |
| Watch-set evals (50 rows) | ~10 min |
| **Total** | **~1.5–2 hr A100** |

### sft-002b (OpenR1 5k stratified)

| Stage | Time |
|---|---|
| Stratified resample + spot check | 30 min CPU |
| Smoke train + inspection | ~15 min |
| Full SFT 5k × 2 epochs (~625 update steps) | ~2–3 hr |
| Holdout + watch evals | ~40 min |
| Full public eval (1126 rows × 16k) | ~2 hr |
| **Total** | **~5–6 hr A100** |

### sft-003 (s1K-1.1 fallback)

Same as sft-002a: **~1.5–2 hr A100**.

### sft-001 (Numina fallback)

| Stage | Time |
|---|---|
| Full SFT 18k × 1 epoch (~1.1k update steps) | ~2–3 hr |
| Mid-training dev evals (3–4 × ~10 min) | ~40 min |
| Best-checkpoint full public eval | ~2 hr |
| **Total** | **~5–6 hr A100** |

### Total budget (all paths combined)

- sft-002a probe: ~2 hr
- if sft-002a passes → sft-002b scale: ~6 hr
- if sft-002a fails → sft-003 alt: ~2 hr
- if both fail → sft-001 fallback: ~6 hr
- SFT+SC stacking eval: ~2 hr
- Final submission inference on private (1126+ rows): ~3 hr

**Worst-case total: ~21 hr A100 across 3–4 Colab sessions.** Best-case (sft-002a strong): ~10 hr A100 across 2 sessions.

## Colab failure-mode guardrails

- **Session disconnect mid-training:** checkpoints and eval JSONs go to Drive; rerun notebook from top and resume latest checkpoint.
- **A100 unavailable:** abort the session. Do not attempt L4 — bf16 LoRA at 8k seq will OOM. Reconnect later.
- **Drive I/O bottleneck:** checkpoint every 500 steps on large corpora, 200 steps on small.
- **OOM:** drop rows above token cap; if needed lower sequence length only for smoke runs.
- **HF Hub rate limits:** set `HF_TOKEN` in Colab secrets before pulling models/datasets.
- **Bad smoke generations:** stop immediately; do not launch full training.

---

## Execution checklist

### Pre-flight (carried over from prior plan)

- [x] Create `notebooks/sft_eval.ipynb` from `dev.ipynb` with LoRA adapter loading and extra metrics.
- [x] Freeze `data/eval/holdout.jsonl` on Drive (`CSE151B/data/eval/holdout.jsonl`).
- [x] Verify Qwen3-Thinking assistant schema with `apply_chat_template` (D005).
- [x] Record schema decision in `notebooks/sft_data_prep.ipynb`.
- [x] Eval watch sets: Q4 long (30) + multi-blank ≥3 (20) — `scripts/build_eval_watch_sets.py`, `data/eval/watch_manifest.json`.

### sft-002a — OpenR1 1k probe (PRIMARY)

- [ ] Add OpenR1 data-prep section to `notebooks/sft_data_prep.ipynb`.
- [ ] Load `open-r1/OpenR1-Math-220k` config `default`; verify schema (`generations`, `correctness_math_verify`, `problem`, `solution`).
- [ ] Apply correctness + length + decontam filters.
- [ ] Random sample 1000 rows, seed 42.
- [ ] Format to D005 `<think>...</think>` wrapper.
- [ ] Write `data/sft_corpus_openr1_1k.jsonl` + manifest.
- [ ] Spot check 10 rows manually.
- [ ] Smoke train 50 steps on 200-row subset; inspect 10 generations.
- [ ] Full SFT 1k × 2 epochs (~250 update steps), checkpoint every 200 steps.
- [x] Eval on `holdout_20p` + watch sets per epoch — **64.44% overall** (`final_adapter`, 1 epoch); see [sft-002a](../log/runs/sft-002a-openr1-1k.md).
- [x] Apply stop rules; pick best checkpoint — no triggers; only `final_adapter` @ 1 epoch evaluated.
- [x] Decision gate: **flat** → do not scale to sft-002b yet ([D010](../log/decisions.md#d010--sft-002a-openr1-1k-flat--do-not-scale-to-5k-yet)).
- [x] Log to `docs/log/runs/sft-002a-openr1-1k.md`; row in `docs/log/experiments.md`.

### sft-002b — OpenR1 5k stratified (if sft-002a passes)

- [ ] Stratified sampling: 1500 long + 1500 multi-answer + 1000 geometry + 1000 random.
- [ ] Reject diagram-dependent rows.
- [ ] Write `data/sft_corpus_openr1_5k.jsonl` + manifest.
- [ ] Smoke train; full train 2 epochs; eval; decide.
- [ ] Log to `docs/log/runs/sft-002b-openr1-5k.md`.

### sft-003 — OpenMathReasoning probe (Fallback A, triggered by sft-002a flat)

- [ ] Load `nvidia/OpenMathReasoning` `cot` split; verify field names (`problem`, `generated_solution`, `pass_rate_72b_tir`).
- [ ] Apply filters: `pass_rate_72b_tir` in [0.05, 0.70]; `8_000 ≤ len(generated_solution) ≤ 28_000`; decontam vs `public.jsonl` + `private.jsonl`.
- [ ] Shuffle seed 42, select 1000; format to D005 wrapper.
- [ ] Write `data/sft_corpus_openmath_1k.jsonl` + manifest (include length distribution p25/p50/p95, pass_rate distribution, filter retention rate).
- [ ] Spot check 10 rows: confirm long exploratory trace, `<think>` wrap, `\boxed{}` after close, no CJK, no figure-dependent.
- [ ] Smoke train 50 steps; confirm traces not truncated in tokenized output; inspect 10 generations.
- [ ] Full SFT 2 epochs; eval on `holdout_20p` + watch sets; apply decision gate.
- [ ] Log to `docs/log/runs/sft-003-openmath-1k.md`.

### sft-004 — s1K-1.1 (Fallback B, if sft-003 also flat)

- [ ] Load `simplescaling/s1K-1.1`; verify trace field names.
- [ ] Format to D005 wrapper.
- [ ] Write `data/sft_corpus_s1k.jsonl` + manifest.
- [ ] Smoke train; full train 2 epochs; eval; decide.
- [ ] Log to `docs/log/runs/sft-004-s1k.md`.

### sft-001 — Numina fallback (if both distilled options fail)

- [x] Corpus built: `data/sft_corpus_v2.jsonl` (18k); manifest `data/sft_corpus_v2_manifest.json`.
- [x] Spot check 30 supplement rows.
- [ ] Smoke train; full train 1 epoch; eval; decide.
- [ ] Log to `docs/log/runs/sft-001-numina-v2.md`.

### Post-training (any successful SFT)

- [ ] Evaluate best checkpoint on `holdout_20p`.
- [ ] Evaluate best checkpoint on full `public.jsonl`.
- [ ] Compare against pub-002 baseline + dev-008/dev-012-sc5 anchors.
- [ ] Run SC K=5 on best checkpoint via `holdout_10p`; compare delta vs dev-012-sc5.
- [ ] Decide: submit SFT-only / SFT+SC / keep pub-002 baseline / route to recovery.
- [ ] If submitting SFT, update `notebooks/submission.ipynb` to load the chosen LoRA adapter or merged model.
- [ ] Verify private CSV still uses `id,response` with full traces.

---

## Open questions

- Best `OpenR1-Math-220k` field for picking the "best" correct generation per row when multiple exist (currently picking first; could pick shortest or longest).
- Whether the OpenR1 traces preserve their `<think>...</think>` wrapper as a literal string or need re-construction from chat-format messages.
- Whether `unsloth` supports `Qwen/Qwen3-4B-Thinking-2507` cleanly in the Colab image.
- Whether course submission expects merged weights or permits loading a LoRA adapter in the notebook.
- How much A100 time is realistically available for full runs across all phases.
- Whether MCQ recovery should use self-distillation, manually rebuilt AGIEval English rows, or another cleaner MCQ source.

# SFT pipeline plan

Plan for supervised fine-tuning `Qwen/Qwen3-4B-Thinking-2507` to improve unified accuracy on `public.jsonl` and `private.jsonl`. This version replaces the earlier mixed-source recipe with a simpler, safer **clean Numina-first** pipeline.

The guiding principle is: train on fewer high-quality long reasoning traces before adding noisy coverage. A bad source can teach the model the wrong trace style, so `AGIEval/GaoKao` and original `MATH train` traces are excluded from the first run.

## Environment assumptions

- **Compute:** single GPU on Colab. **A100 only (40 GB or 80 GB).** Training is bf16 LoRA — no 4-bit fallback, no L4 path. If A100 is unavailable, abort the session.
- **Persistence:** Google Drive at `/content/drive/MyDrive/CSE151B/` mirrors the repo layout used by `notebooks/dev.ipynb` and `notebooks/submission.ipynb`. `data/eval/` (holdout + watch sets), checkpoints, eval outputs, and corpus manifests live there so disconnects are recoverable.
- **Workflow:** keep the notebook workflow: `notebooks/sft_data_prep.ipynb`, `notebooks/sft_train.ipynb`, `notebooks/sft_eval.ipynb`, and `notebooks/submission.ipynb`.
- **Submission path:** final private inference remains model-only generation. No tool-augmented inference, no external APIs, no calculator loop.

---

## Goal

Lift unified accuracy from the current **61.90%** baseline (72.00% MCQ / 56.86% free-form, [pub-002](../log/experiments.md#pub-002)) without collapsing Qwen3-Thinking's reasoning style or regressing MCQ.

**Target weaknesses** (from [`baseline-public-16k.md`](../analysis/baseline-public-16k.md), filtered to slices with tight enough N to trust):

| Slice | N | Acc | Headroom |
|---|--:|---:|---|
| Q4 question length (≥435 chars) | 281 | 43.8% | largest single addressable bucket |
| Multi-blank ≥6 blanks | 63 | ~34% | sharpest per-slice gap |
| Geometry | 115 | 50.4% | only topic with clean weakness signal |
| MCQ "think finished, wrong boxed" | 54 | reasoning failure | method-agnostic — needs better reasoning, not better prompts |

Topic-level targeting beyond geometry is not statistically reliable: "Other" (581 / 1126 rows) is a heterogeneous catch-all, and small-n named topics (n ≤ 25) have 95% CI bands wider than their gap to overall accuracy.

First checkpoint target:

- **Primary:** no MCQ regression beyond 3 pp on dev; no free-form regression beyond noise.
- **Secondary:** overall gain on the frozen dev slice; per-slice movement on Q4 long-context and multi-blank ≥3 slices.
- **Stretch:** +2 to +4 pp overall on full public after selecting the best checkpoint.

The first run does not need to solve every failure mode. It needs to answer one question cleanly: **does targeted Numina SFT improve the long-context and multi-blank slices without trace collapse or MCQ regression?**

### Expected outcome distribution

Gut estimates anchored on similar LoRA-on-thinking-model results. The Qwen3-4B-Thinking base is already RL-tuned for reasoning, so SFT has less low-hanging fruit than for a vanilla base.

| Outcome | Probability | Overall Δ vs pub-002 |
|---|---|---|
| Strong win | ~15% | +5 to +8 pp |
| Modest win | ~35% | +2 to +4 pp |
| Flat (within noise) | ~30% | −1 to +2 pp |
| Regression | ~20% | −2 to −5 pp — keep pub-002, run recovery |

### Non-goals

- No RL / GRPO / DPO until SFT has a clean baseline.
- No full-parameter fine-tuning.
- No training on `public.jsonl`; it stays evaluation-only.
- No mixed-source corpus until the Numina-only run gives a trusted reference point.
- No synthetic filler traces.

---

## Why change the plan

**Decision record:** [D004 — Numina-only first SFT run](../log/decisions.md#d004). Defect audit: [`data-issues.md`](data-issues.md).

Summary: mixed corpus had synthetic AGIEval/GaoKao traces, short MATH solutions, and noisy MCQ expansion — unsafe for a first Thinking-style SFT. First run is Numina-only after cleanup; add sources only after eval trusts the baseline.

---

## Source policy

| Source | First-run decision | Reason |
| --- | --- | --- |
| `NuminaMath-CoT` (current 15k corpus) | **Primary base** | Real long math traces; already cleaned, schema-validated, decontam'd. |
| `NuminaMath-CoT` long-trace subset (`trace_chars ≥ 6000`) | **Targeted supplement (~1.5k)** | Directly attacks Q4 long-context weakness; sample from existing `numina_cot_clean_ready.jsonl`. |
| Synthetic multi-blank from Numina | **Targeted supplement (~1.5k)** | Numina problems with sequential `\boxed{a}, \boxed{b}` answers, or 2–3 related sub-questions composed into one multi-`[ANS]` prompt. Directly attacks the ≥6-blank weakness. |
| `MATH train` original solutions | **Exclude from first run** | Solutions are too short; useful as problem source later, not as trace target now. |
| `AGIEval-Math` | **Exclude from first run** | Current prepared responses are synthetic no-reasoning templates. |
| `GaoKao-MCQ` | **Exclude from first run** | Mostly Chinese and tied to the same synthetic-response path. |
| `OpenR1 / DeepSeek distill` | **Defer** | Possible hard-tail supplement later, but license and style compatibility are unresolved. |
| Baseline self-distillation (MCQ) | **Defer to recovery phase** | Use pub-002 correct MCQ traces only if sft-001 regresses MCQ. MCQ is already 72%; don't touch in first run. |

### First corpus target

Existing build:

```text
data/sft_sources/numina_cot_clean_ready.jsonl  ← 23,089 rows (clean Numina)
data/sft_sources/numina_cot_clean_stats.json
data/sft_sources/numina_cot_clean_rejects.jsonl
data/sft_corpus.jsonl                          ← 15,000 rows (current)
data/sft_corpus_manifest.json
```

**Revised target for sft-001:** ~18k rows total = current 15k Numina base + ~3k targeted supplement (long-trace + synthetic multi-blank). Build supplements as separate JSONL artifacts and concatenate at corpus-build time so source contribution is auditable.

```text
data/sft_sources/numina_long_trace.jsonl        ← new (~1.5k, trace_chars ≥ 6000)
data/sft_sources/numina_multi_blank_synth.jsonl ← new (~1.5k synthetic multi-blank)
data/sft_corpus_v2.jsonl                        ← 18k merged
data/sft_corpus_v2_manifest.json
```

A100 40 GB capacity holds 18k easily at 8k seq. Do not backfill from weak sources.

---

## Data prep plan

Use `notebooks/sft_data_prep.ipynb` as the single orchestration notebook. Rebuild the Numina artifact instead of relying on the current `numina_cot_ready.jsonl`.

### Step 1: verify Qwen3 thinking template

**Notebook:** one cell in `notebooks/dev.ipynb` (after §6 prompt construction; before vLLM load). Run it after §5–§6 so `MODEL_ID`, `build_prompt`, and `free_sample` are defined.

Before rebuilding rows, run a small tokenizer cell with:

- one free-form prompt
- one assistant response with plain reasoning
- one assistant response wrapped in `<think>...</think>`

Inspect `tokenizer.apply_chat_template(...)` output and decide the target assistant schema. This gates every response string. If the model's normal inference path emits `<think>...</think>`, training targets should match that schema.

**Decision (2026-05-21):** explicit wrapper — reasoning inside `<think>...</think>`, `\boxed{...}` after the closing tag. Plain assistant text produces an empty thinking block and puts CoT outside the tags (see `data/qwen_thinking_trace.txt`, [D005](../log/decisions.md#d005--sft-assistant-schema-explicit-redacted_thinking-wrapper)). Record `thinking_template: "explicit_redacted_thinking"` in `data/sft_corpus_manifest.json` at corpus build time.

### Step 2: rebuild Numina rows

For each Numina candidate:

1. Load problem, solution trace, and gold answer.
2. Reject any question or response with CJK characters.
3. Detect inline MCQ patterns such as `(A) ... (B) ... (C) ...`.
4. For the first run, reject inline MCQ rows unless parsing into separate `options` and a gold letter is reliable.
5. Normalize the final answer line to the competition format.
6. Render the prompt through the exact `build_prompt(...)` and `apply_chat_template(...)` path used by eval/submission.
7. Count templated tokens and reject rows above the training max length.
8. Write accepted and rejected rows with explicit reject reasons.

Recommended first-run rule: **reject inline MCQ rather than attempting clever parsing**. Numina-only free-form reasoning is the cleanest baseline. MCQ repair can come later.

### Step 3: quality filters

Accept a row only if all checks pass:

- English-only question and trace.
- Genuine multi-step reasoning, not a templated answer stub.
- `trace_chars >= 2,000`.
- `trace_chars <= 12,000` unless tokenization proves it still fits comfortably.
- `template_tokens <= 7,900` for an 8k sequence run.
- Final line contains parseable `\boxed{...}`.
- No obvious `public.jsonl` or `private.jsonl` question overlap.
- Source metadata survives: `source`, `source_id`, `task_type`, `topic`, `answer`, `trace_chars`, `template_tokens`, `reject_reason` where applicable.

### Step 4: human spot check

Before training, sample 50 accepted rows and 50 rejected rows:

- accepted rows should look like real math traces
- no CJK residue
- no hidden MCQ rows mislabeled as free-form
- final answer line matches expected format
- no ultra-short official-solution style traces

If more than 2 accepted rows fail the spot check, fix filters and rebuild.

**2026-05-21 audit:** Full re-audit of 23,089 ready rows in `notebooks/sft_data_prep.ipynb` §5.2; 20-row manual spot-check passed. Recorded in [`numina-clean-audit.md`](numina-clean-audit.md). Structural checks 0 failures; 10 rows `trace_chars < 500`; 416 letter-final MCQ-style; `NUMINA_MAX_READY=25k` left ~76k qualifying rows untokenized.

**2026-05-22 Step 5:** `scripts/build_sft_corpus.py` → 15,000-row `data/sft_corpus.jsonl` (dropped 426, 3× weak-topic weight, seed 42). Manifest: `data/sft_corpus_manifest.json`.

### Step 5: build targeted supplement (new — 2026-05-24)

Two new builders extend `scripts/build_sft_corpus.py`:

**5a. Long-trace supplement (`numina_long_trace.jsonl`, ~1.5k rows)**

- Filter `numina_cot_clean_ready.jsonl` to `trace_chars ∈ [6000, 12000]`.
- Bias toward geometry by keyword match (`triangle`, `polygon`, `angle`, `circle`, `quadrilateral`, `parallelogram`, etc.) — target ~30% geometry-flavored.
- Reject diagram-dependent rows (Asymptote code blocks, "in the figure shown", "in the diagram"). Text-only model has no figure access.
- Decontam against `public.jsonl` / `private.jsonl` problem text as in Step 3.

**5b. Synthetic multi-blank supplement (`numina_multi_blank_synth.jsonl`, ~1.5k rows)**

Two construction modes — both produce the inference-time `[ANS]` placeholder format:

1. **Native multi-answer Numina rows** — where one Numina problem has multiple sub-answers in its trace (e.g., "find a, b, c such that..."), reformat the prompt to use `[ANS]` placeholders and the trace to end with `\boxed{a}, \boxed{b}, \boxed{c}`.
2. **Composed sub-questions** — combine 2–3 related sub-questions from the same Numina problem (or paired problems with shared setup) into a single prompt; concatenate traces; emit comma-separated `\boxed{}`. Cap composed prompts at 4 blanks for first run; 6+ blanks deferred.

Must use the **exact** multi-blank prompt template from `notebooks/full_public.ipynb` so training rows match inference rows byte-for-byte after `apply_chat_template`.

**5c. Merge step**

```python
sft_corpus_v2 = shuffle(sft_corpus + numina_long_trace + numina_multi_blank_synth, seed=42)
```

Record per-source contribution in `data/sft_corpus_v2_manifest.json`: row counts, trace_chars distribution, blank-count distribution, geometry-keyword hit rate, source hashes.

**Spot check:** sample 30 supplement rows manually — 15 long-trace, 15 multi-blank. Verify trace style is Qwen3-Thinking-compatible (long, exploratory, not Numina-template-shaped), verify multi-blank rows have the right `\boxed{a}, \boxed{b}` rhythm.

### Step 6: final corpus

For sft-001, `data/sft_corpus_v2.jsonl` is the merged shuffle from Step 5c.

The manifest must record:

- source file path
- source file hash
- total candidates
- accepted count
- reject counts by reason
- sample seed
- target size
- final row count
- mean / median / p95 `trace_chars`
- mean / median / p95 `template_tokens`
- thinking-template decision

---

## Training config

A100-only single-GPU bf16 LoRA. No 4-bit path. Keep the first run conservative; the goal is signal, not maximum adapter capacity. Decision record: [D009](../log/decisions.md#d009--bf16-lora-replaces-qlora-for-sft-001-a100-only).

- **Framework:** `trl.SFTTrainer + peft` (no `bitsandbytes`).
- **Precision:** **bf16 base + bf16 LoRA + bf16 compute on A100.** No quantization. Memory budget at 8k seq, mb=1, grad-ckpt: ~22–30 GB — fits A100 40 GB; A100 80 GB allows mb=2.
- **LoRA:** rank 32, alpha 64, dropout 0.05.
- **Targets:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Loss:** completion-only; mask prompt tokens.
- **Optimizer:** `adamw_torch_fused`.
- **Learning rate:** `1e-5`.
- **Schedule:** cosine decay, 3% warmup.
- **Sequence length:** 8192. A100 only — if Colab assigns L4, abort the session, do not downgrade.
- **Batch:** micro-batch 1, grad accum 16. On 80 GB A100, try micro-batch 2 and grad accum 8.
- **Epochs:** 1. Do not run a second epoch until eval proves no trace collapse.
- **Checkpointing:** every 500 steps to Drive.
- **Eval cadence:** every 500-1000 steps, depending on runtime.
- **Resume:** auto-detect latest checkpoint in the run directory and resume from it.

### Smoke run

Before full training:

1. Train 50-100 steps on 200-500 rows.
2. Confirm loss decreases.
3. Generate 10 sample responses from the frozen dev slice.
4. Confirm responses still contain reasoning and valid `\boxed{...}` answers.
5. Run one dev eval if runtime allows.

If smoke outputs become short answer-only completions, stop and inspect the response schema before full training.

---

## Eval harness

Use `notebooks/sft_eval.ipynb`. The eval path must match the final submission path except for loading a LoRA adapter.

Required eval sets:

- **Eval holdout:** `data/eval/holdout.jsonl`, 225 rows. Rebuild: `python scripts/build_eval_holdout.py --fraction 0.20 --seed 42`.
- **Watch Q4 long:** `data/eval/watch_q4_long.jsonl` — **30** frozen rows (`question` length ≥ 435 chars; 3 MCQ / 27 free-form). IDs in `data/eval/watch_manifest.json` → `watch.q4_long.ids`.
- **Watch multi-blank ≥3:** `data/eval/watch_multi_blank_ge3.jsonl` — **20** frozen rows (≥3 `[ANS]`). IDs → `watch.multi_blank_ge3.ids`. Rebuild watch: `python scripts/build_eval_watch_sets.py`.
- **Full public holdout:** full `data/public.jsonl` after selecting a candidate checkpoint.
- **Optional stress slice:** 20-50 hand-picked weak-topic / multi-blank examples for qualitative inspection only.

Metrics per checkpoint:

| Metric | Why |
| --- | --- |
| Unified accuracy | Main score proxy. |
| MCQ accuracy | **Hard regression watch** — Numina-heavy mix likely to pressure MCQ. |
| Free-form accuracy | Primary expected gain area. |
| Q4 long-context accuracy | Direct watch on long-trace supplement. |
| Multi-blank ≥3 accuracy | Direct watch on multi-blank supplement. |
| `\boxed{}` emission rate | Format regression detector. |
| Multi-blank `\boxed{}` count vs gold blanks | Multi-blank format integrity. |
| Mean / median response length | Trace-collapse detector. |
| `<think>` / `</think>` tag emission rate | Thinking-schema integrity detector. |
| Empty or answer-only response rate | Thinking-style failure detector. |

Stop rules:

- **MCQ dev accuracy drops more than 3 pp from pub-002 baseline (72%).** Halt and triage; Numina mix is destabilizing MCQ format.
- Free-form dev accuracy drops more than 2 pp from base and response length also drops.
- Mean response length drops more than 20% vs the baseline eval run.
- `\boxed{}` emission rate drops below baseline, OR multi-blank rows emit fewer `\boxed{}` than gold blanks at >2× the pub-002 rate.
- `<think>` opener/closer emission rate drops below 95% (schema breakage).
- Two consecutive eval points show no improvement and qualitative samples look shorter.
- Any checkpoint emits mostly answer-only completions.

Checkpoint selection:

1. Pick best dev checkpoint that does not violate stop rules.
2. Re-evaluate that checkpoint on full public.
3. **Decision rule:** ship only if overall public lift ≥ +2 pp **and** no slice regression > 3 pp (MCQ, FF, Q4-long, multi-blank ≥3). Otherwise keep pub-002 baseline for submission and route to recovery phase.

---

## Recovery phase after Numina-only

Only start this after the Numina-only run has full eval results.

### If free-form improves but MCQ stays flat

Build a small MCQ self-distillation source from external MCQ-style problems, not `public.jsonl`:

- use English-only external problems
- expand to 10 options only with type-coherent distractors
- generate k baseline traces per problem
- keep only traces whose boxed letter matches gold
- reject short or answer-only traces
- cap this source at 10-20% of the next corpus

Do not revive the current `AGIEval/GaoKao` synthetic traces.

### If trace length collapses

Do not add more data. Fix the assistant schema first:

- verify `<think>` target format
- inspect whether completion-only masking includes assistant tokens correctly
- lower learning rate to `5e-6`
- reduce epoch fraction
- train on fewer, longer Numina rows

### If weak topics remain bad

Use `MATH train` and Numina as **problem sources** for baseline self-distillation:

- sample weak-topic problems
- generate multiple baseline traces
- keep only correct traces
- repair only final formatting
- reject short traces

Do not train on original `MATH train` official solutions unless running a tiny controlled ablation.

### If Numina-only is neutral or negative

Stop SFT work until root cause is known. Likely causes:

- wrong Qwen3 thinking schema
- completion masking bug
- training targets not matching inference prompt
- answer normalization bug
- too many low-quality Numina rows slipping through

---

## Distribution risks

Ranked by current likelihood given the targeted supplement + pub-002 floor.

1. **MCQ regression from FF-heavy mix (most likely failure).** Numina + multi-blank synthetic supplement are both free-form. MCQ at 72% is the slice with most to lose. Mitigation: hard MCQ floor stop rule (−3 pp from pub-002); no MCQ data in sft-001; recovery phase has MCQ self-distillation if needed.
2. **Trace-style perturbation.** Most insidious failure — model still reasons but in a Numina rhythm that subtly hurts. Hard to detect without qualitative inspection. Mitigation: 50-row spot check at best dev checkpoint; track `<think>` tag emission rate; conservative LR (`1e-5`).
3. **Multi-blank format breakage from synthetic supplement.** If synthetic multi-blank rows don't match the inference template exactly, model learns a new format that disagrees with eval. Mitigation: build supplement rows through the **exact** `build_prompt + apply_chat_template` path used by pub-002; spot check tokenized output.
4. **Trace-style collapse.** Short or wrongly formatted targets can kill thinking behavior. Mitigation: keep `trace_chars ≥ 2000` floor; exclude short MATH solutions; validate `<think>` schema.
5. **Format mismatch.** Training rows must match final `build_prompt(...)` and `apply_chat_template(...)` path. Mitigation: tokenized sample inspection before training.
6. **Numina distribution skew.** Olympiad-flavored proofs may not transfer to competition-style problems. Mitigation: targeted supplement biases toward long-context and multi-blank — the actually weak slices.
7. **Inline MCQ contamination.** Numina embeds some answer choices in problem text. Mitigation: reject inline MCQ for first run.
8. **Public/private leakage.** Public remains eval-only. Mitigation: decontam against public/private question text before corpus write — extend to supplement builders.

---

## Compute budget

Single A100-40GB Colab session.

| Stage | Time |
|---|---|
| Targeted supplement build + spot check | 1–2 hr CPU |
| Smoke train (200–500 rows, 50–100 steps) + 10-sample inspection | ~30 min |
| Full SFT — 18k rows × 1 epoch × 8k seq, micro-batch 1 × accum 16 (~1.1k steps) | ~2–3 hr |
| Mid-training dev evals (3–4 × ~10 min on 225-row dev) | ~40 min |
| Best-checkpoint full public eval (1126 rows at 16k tokens) | ~2 hr |
| **Total per attempt** | **~6–8 hr A100** |

Budget for 2 attempts (sft-001 + one recovery cycle): **~14–16 hr A100**. Anything beyond that and the experiment has overrun its return-on-compute window.

## Colab failure-mode guardrails

- **Session disconnect mid-training:** checkpoints and eval JSONs go to Drive; rerun notebook from top and resume latest checkpoint.
- **A100 unavailable:** abort the session. Do not attempt L4 — bf16 LoRA at 8k seq will OOM. Reconnect later.
- **Drive I/O bottleneck:** checkpoint every 500 steps, not every 100.
- **OOM:** drop rows above token cap; if needed lower sequence length only for smoke runs.
- **HF Hub rate limits:** set `HF_TOKEN` in Colab secrets before pulling models/datasets.
- **Bad smoke generations:** stop immediately; do not launch full training.

---

## Execution checklist

### Pre-flight

- [x] Create `notebooks/sft_eval.ipynb` from `dev.ipynb` with LoRA adapter loading and extra metrics.
- [x] Freeze `data/eval/holdout.jsonl` on Drive (`CSE151B/data/eval/holdout.jsonl`).
- [x] Verify Qwen3-Thinking assistant schema with `apply_chat_template`.
- [x] Record the schema decision in `notebooks/sft_data_prep.ipynb`.

### Data prep

- [x] Add CJK reject filter to Numina prep.
- [x] Add inline MCQ detector to Numina prep.
- [x] For first run, reject inline MCQ rows.
- [x] Normalize final answer lines to competition `\boxed{...}` format.
- [x] Tokenize with the exact eval/submission chat-template path.
- [x] Reject rows with `template_tokens > 7900`.
- [x] Reject rows with `trace_chars < 2000`.
- [x] Write `data/sft_sources/numina_cot_clean_ready.jsonl` (23,089 rows; see [`numina-clean-audit.md`](numina-clean-audit.md)).
- [x] Write `data/sft_sources/numina_cot_clean_stats.json`.
- [x] Write `data/sft_sources/numina_cot_clean_rejects.jsonl` (1,918 sampled tokenize rejects).
- [x] Spot-check + §5.2 bulk audit (20 manual OK; full audit in [`numina-clean-audit.md`](numina-clean-audit.md)). Optional: 50+50 manual still useful.
- [x] Build `data/sft_corpus.jsonl` from clean Numina only (15k rows; `scripts/build_sft_corpus.py`).
- [x] Write `data/sft_corpus_manifest.json`.

### Targeted supplement (new — sft-001 prep)

- [x] Build `data/sft_sources/numina_cot_clean_ready_long.jsonl` (3,000 rows; §5.3 heap top-25k raw length outside first-pass `source_id`s; only 53 wrapped `trace_chars ≥ 6000`).
- [x] Rebuild `data/sft_sources/numina_long_trace.jsonl` from long pool (`long-trace --ready-path …_long.jsonl`; p95 trace_chars 5594).
- [x] Re-merge `data/sft_corpus_v2.jsonl` (18,000 rows).
- [x] Build `data/sft_sources/numina_multi_blank_synth.jsonl` (1,500 rows; `multi-blank` subcommand). pub-002 multi-blank prompt; composed 2–4 parts (native single-boxed after Numina prep).
- [x] Decontam supplements against `public.jsonl` / `private.jsonl` problem text.
- [x] Merge → `data/sft_corpus_v2.jsonl` (18,000); `data/sft_corpus_v2_manifest.json` with per-source row counts and distributions.
- [x] Manual spot check 30 supplement rows (15 long-trace, 15 multi-blank) in `notebooks/sft_data_prep.ipynb` — passed.
- [x] Eval watch sets: Q4 long (30 rows) + multi-blank ≥3 (20 rows) — `scripts/build_eval_watch_sets.py`, `data/eval/watch_manifest.json`.

### Training

- [x] Add `notebooks/sft_train.ipynb` (bf16 LoRA smoke + full run, A100 only, Drive checkpoints).
- [ ] Smoke-test training for 50-100 steps on 200-500 row subset (must include long-trace + multi-blank rows).
- [ ] Inspect 10 generated dev-slice samples — verify thinking schema intact, multi-blank format intact, MCQ letter format intact.
- [ ] Run a quick dev eval if runtime allows.
- [ ] Full A100 bf16 LoRA run for 1 epoch on `sft_corpus_v2.jsonl`.
- [ ] Save checkpoints every 500 steps.
- [ ] Evaluate every 500-1000 steps with all new metrics (MCQ floor, Q4-long, multi-blank ≥3, `<think>` tag rate).
- [ ] Apply stop rules and select best checkpoint.

### Post-training

- [ ] Evaluate best checkpoint on frozen dev slice.
- [ ] Evaluate best checkpoint on full `public.jsonl`.
- [ ] Compare against baseline by overall, MCQ, free-form, multi-blank, response length, and boxed emission.
- [ ] Decide: submit SFT checkpoint, keep baseline, or run recovery phase.
- [ ] If submitting SFT, update `notebooks/submission.ipynb` to load the chosen LoRA adapter or merged model.
- [ ] Verify private CSV still uses `id,response` with full traces.

---

## Open questions

- Exact Qwen3-Thinking target schema for assistant responses.
- Whether `unsloth` supports `Qwen/Qwen3-4B-Thinking-2507` cleanly in the Colab image.
- Whether course submission expects merged weights or permits loading a LoRA adapter in the notebook.
- How much A100 time is realistically available for full runs.
- Whether MCQ recovery should use self-distillation, manually rebuilt AGIEval English rows, or another cleaner MCQ source.


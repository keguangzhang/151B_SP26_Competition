# SFT pipeline plan

Plan for supervised fine-tuning `Qwen/Qwen3-4B-Thinking-2507` to improve unified accuracy on `public.jsonl` and `private.jsonl`. This version replaces the earlier mixed-source recipe with a simpler, safer **clean Numina-first** pipeline.

The guiding principle is: train on fewer high-quality long reasoning traces before adding noisy coverage. A bad source can teach the model the wrong trace style, so `AGIEval/GaoKao` and original `MATH train` traces are excluded from the first run.

## Environment assumptions

- **Compute:** single GPU on Colab. Target an A100 session (40 GB; 80 GB if available). L4 fallback is possible but should be treated as a smaller smoke or ablation run.
- **Persistence:** Google Drive at `/content/drive/MyDrive/CSE151B/` mirrors the repo layout used by `notebooks/dev.ipynb` and `notebooks/submission.ipynb`. `data/dev.jsonl`, checkpoints, eval outputs, and corpus manifests live there so disconnects are recoverable.
- **Workflow:** keep the notebook workflow: `notebooks/sft_data_prep.ipynb`, `notebooks/sft_train.ipynb`, `notebooks/sft_eval.ipynb`, and `notebooks/submission.ipynb`.
- **Submission path:** final private inference remains model-only generation. No tool-augmented inference, no external APIs, no calculator loop.

---

## Goal

Lift unified accuracy from the current **52.66%** baseline (50.40% MCQ / 53.79% free-form) without collapsing Qwen3-Thinking's reasoning style.

First checkpoint target:

- **Primary:** no free-form regression beyond noise.
- **Secondary:** overall gain on the frozen dev slice.
- **Stretch:** +3 to +5 pp overall on full public after selecting the best checkpoint.

The first run does not need to solve every failure mode. It needs to answer one question cleanly: **does high-quality Numina SFT improve the base model without trace collapse?**

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
| `NuminaMath-CoT` | **Use as primary source** | Large, real math reasoning traces, already long enough for thinking-style SFT. |
| `MATH train` original solutions | **Exclude from first run** | Solutions are too short; useful as problem source later, not as trace target now. |
| `AGIEval-Math` | **Exclude from first run** | Current prepared responses are synthetic no-reasoning templates. |
| `GaoKao-MCQ` | **Exclude from first run** | Mostly Chinese and tied to the same synthetic-response path. |
| `OpenR1 / DeepSeek distill` | **Defer** | Possible hard-tail supplement later, but license and style compatibility are unresolved. |
| Baseline self-distillation | **Defer to recovery phase** | Better after measuring whether Numina-only leaves format or MCQ gaps. |

### First corpus target

Build:

```text
data/sft_sources/numina_cot_clean_ready.jsonl
data/sft_sources/numina_cot_clean_stats.json
data/sft_sources/numina_cot_clean_rejects.jsonl
data/sft_corpus.jsonl
data/sft_corpus_manifest.json
```

Target size:

- **A100 40 GB:** 12k-18k examples after filters.
- **A100 80 GB:** 18k-25k examples after filters.
- **L4 fallback:** 3k-8k examples, treated as a smoke run only.

Do not backfill missing rows from weak sources. If Numina filtering leaves fewer rows than expected, train the smaller clean corpus.

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

### Step 5: final corpus

For the first run, `data/sft_corpus.jsonl` is just a shuffled sample from `numina_cot_clean_ready.jsonl`.

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

A100-tuned single-GPU QLoRA. Keep the first run conservative; the goal is signal, not maximum adapter capacity.

- **Framework:** `unsloth` if smoke-tested with `Qwen/Qwen3-4B-Thinking-2507`; fallback to `trl.SFTTrainer + peft + bitsandbytes`.
- **Quantization:** 4-bit base, NF4, double quantization.
- **LoRA:** rank 32, alpha 64, dropout 0.05.
- **Targets:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Loss:** completion-only; mask prompt tokens.
- **Optimizer:** paged AdamW 8-bit.
- **Learning rate:** `1e-5`.
- **Schedule:** cosine decay, 3% warmup.
- **Precision:** bf16 on A100.
- **Sequence length:** 8192 on A100; 4096 only for L4 smoke runs.
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

- **Frozen dev slice:** `data/dev.jsonl`, 225 rows.
- **Full public holdout:** full `data/public.jsonl` after selecting a candidate checkpoint.
- **Optional stress slice:** 20-50 hand-picked weak-topic / multi-blank examples for qualitative inspection only.

Metrics per checkpoint:

| Metric | Why |
| --- | --- |
| Unified accuracy | Main score proxy. |
| MCQ accuracy | Watch whether Numina-only accidentally helps or hurts MCQ. |
| Free-form accuracy | Primary expected gain area. |
| Multi-blank free-form accuracy | Known weak sub-bucket. |
| `\boxed{}` emission rate | Format regression detector. |
| Mean / median response length | Trace-collapse detector. |
| Empty or answer-only response rate | Thinking-style failure detector. |
| Per-topic accuracy where labels exist | Regression watch. |

Stop rules:

- Free-form dev accuracy drops more than 2 pp from base and response length also drops.
- Mean response length drops more than 20% vs the baseline eval run.
- `\boxed{}` emission rate drops below baseline.
- Two consecutive eval points show no improvement and qualitative samples look shorter.
- Any checkpoint emits mostly answer-only completions.

Checkpoint selection:

1. Pick best dev checkpoint that does not violate stop rules.
2. Re-evaluate that checkpoint on full public.
3. If full-public gain is under 2 pp or bucket regressions are severe, keep baseline for submission and treat SFT as failed experiment.

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

1. **Trace-style collapse.** Short or wrongly formatted targets can kill thinking behavior. Mitigation: exclude short MATH traces, validate `<think>` schema, track response length.
2. **Format mismatch.** Training rows must match final `build_prompt(...)` and `apply_chat_template(...)` path. Mitigation: tokenized sample inspection before training.
3. **Numina distribution skew.** Numina may overrepresent certain olympiad styles and underrepresent competition MCQ. Mitigation: first run is a baseline; add targeted data only after eval shows the gap.
4. **Inline MCQ contamination.** Numina embeds some answer choices in problem text. Mitigation: reject inline MCQ for first run.
5. **Public/private leakage.** Public remains eval-only. Mitigation: decontam against public/private question text before corpus write.

---

## Colab failure-mode guardrails

- **Session disconnect mid-training:** checkpoints and eval JSONs go to Drive; rerun notebook from top and resume latest checkpoint.
- **A100 unavailable:** run only smoke or small ablation on L4. Do not compare L4 4096-token results directly with A100 8192-token results.
- **Drive I/O bottleneck:** checkpoint every 500 steps, not every 100.
- **OOM:** drop rows above token cap; if needed lower sequence length only for smoke runs.
- **HF Hub rate limits:** set `HF_TOKEN` in Colab secrets before pulling models/datasets.
- **Bad smoke generations:** stop immediately; do not launch full training.

---

## Execution checklist

### Pre-flight

- [x] Create `notebooks/sft_eval.ipynb` from `dev.ipynb` with LoRA adapter loading and extra metrics.
- [x] Freeze `data/dev.jsonl` on Drive (`CSE151B/data/dev.jsonl`).
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

### Training

- [x] Add `notebooks/sft_train.ipynb` (QLoRA smoke + full run, Drive checkpoints).
- [ ] Smoke-test training for 50-100 steps.
- [ ] Inspect 10 generated dev-slice samples.
- [ ] Run a quick dev eval if runtime allows.
- [ ] Full A100 QLoRA run for 1 epoch.
- [ ] Save checkpoints every 500 steps.
- [ ] Evaluate every 500-1000 steps.
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


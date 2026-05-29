# Decision log

Append-only record of choices that matter for the final report. Template: **Context → Options → Decision → Rationale → Consequences**.

---

## D001 — 8k `max_tokens` as inference baseline

**Date:** (early dev phase)

**Context:** MCQ accuracy on dev/public was dominated by truncation at `max_tokens=4096`. Full-public analysis showed most wrong MCQ at 4k hit the cap.

**Options:** (A) keep 4k (B) raise to 8192 (C) raise to 16384 MCQ-only

**Decision:** **B** — `max_tokens=8192` for all items; treat as shipped baseline.

**Rationale:** Measured **+9.82 pp overall** on dev ([dev-003](runs/dev-003-max-tokens-8k.md)); **+11.9 pp overall** on full public ([pub-001](runs/pub-001-full-public-8k.md)), **+24.3 pp MCQ**.

**Consequences:** Remaining MCQ bottleneck is **format** (`\boxed{Letter}`), not truncation. §1.2 in [roadmap.md](../roadmap.md) marked done. Optional 16k ablation is low priority.

**Experiment:** [dev-003](experiments.md#dev-003), [pub-001](experiments.md#pub-001)

---

## D002 — MCQ prompt + low temperature (§1.3) not adopted

**Date:** (dev ablation)

**Context:** Hypothesis that stronger `\boxed{}` clause and MCQ-only `temperature=0.2` would improve format compliance.

**Options:** (A) ship §1.3 on 4k baseline (B) reject (C) retest on 8k without 1500-token cap

**Decision:** **B** for now; **C** remains open on [roadmap.md](../roadmap.md) §1.3.

**Rationale:** On 112-row dev, overall **+0.90 pp** (noise); MCQ **−2.70 pp** ([dev-002](runs/dev-002-mcq-prompt-temp.md)). Tested when truncation still dominated.

**Consequences:** Do not treat §1.3 as a win without re-measuring on 8k baseline.

**Experiment:** [dev-002](experiments.md#dev-002)

---

## D003 — Guided decoding on MCQ tail (§1.1) inconclusive on dev

**Date:** (dev ablations)

**Context:** pub-001 shows 88.3% MCQ accuracy when `\boxed{Letter}` is emitted, but only ~55% emission rate. vLLM `StructuredOutputsParams` regex on MCQ tail was tried.

**Options:** (A) ship guided decoding (B) reject (C) full `public.jsonl` eval before deciding

**Decision:** **C** — not shipped; full-public run still needed.

**Rationale:** dev-004 (n=37 MCQ) **−0.89 pp** vs 8k dev; dev-005 (n=75 MCQ) ~53% vs 50.4% public — no +10–20 pp signal on dev.

**Consequences:** See [dev-004](runs/dev-004-guided-decoding-10pct.md), [dev-005](runs/dev-005-guided-decoding-20pct.md). Consider two-pass or tail-only constraint variants per [roadmap.md](../roadmap.md) §1.1.

**Experiment:** [dev-004](experiments.md#dev-004), [dev-005](experiments.md#dev-005)

---

## D004 — Numina-only first SFT run (exclude mixed corpus)

**Date:** 2026-05-21

**Context:** Prepared mixed sources (`AGIEval`, `GaoKao`, short `MATH train`, `Numina`) had blockers documented in [sft/data-issues.md](../sft/data-issues.md).

**Options:** (A) train on mixed `sft_sources` as prepared (B) **Numina-only** after cleanup (C) defer SFT

**Decision:** **B** — single-source Numina QLoRA; exclude synthetic AGIEval/GaoKao and short MATH traces from run 1.

**Rationale:** Synthetic AGIEval responses teach wrong trace style; GaoKao is largely Chinese; MATH solutions are too short for Thinking-style SFT; Numina is closest to desired long CoT after cleanup.

**Consequences:** First checkpoint judged on no free-form regression + dev/public eval per [sft/pipeline.md](../sft/pipeline.md). Mixed sources only after Numina baseline is trusted.

**Details:** [sft/pipeline.md](../sft/pipeline.md) § "Why change the plan"; audit [sft/data-issues.md](../sft/data-issues.md).

---

## D006 — §1.2 concise-prompt rejected

**Date:** 2026-05-23

**Context:** 84% of wrong MCQ truncate mid-think at 8k tokens. Hypothesis: if the model reasons more concisely, more responses finish within budget and emit an answer.

**Options:** (A) add "non-repetitive, commit once identified" to system prompt (B) keep baseline prompts (C) combine with 16k tokens

**Decision:** **B** — §1.2 rejected; keep baseline prompts.

**Rationale:** dev-006 (n=225, 20% dev slice) showed MCQ 48.00% vs 50.40% pub-001 baseline — flat to slightly worse. Concise instruction cannot overcome structural truncation; the model hits the 8k cap because the problems require long chains, not because it loops. Risk noted in roadmap materialized.

**Consequences:** §1.1 (16k tokens, pub-002) remains the highest-priority fix. No prompt variant shipped.

**Experiment:** [dev-006](runs/dev-006-concise-prompt.md)

---

## D007 — 16k `max_tokens` validated on dev; proceed to pub-002

**Date:** 2026-05-23

**Context:** Truncation drives 44% of MCQ failures at 8k ([analysis](../analysis/baseline-public-8k.md)). §1.2 concise prompting rejected ([D006](#d006--12-concise-prompt-rejected)); §1.1 doubles the generation budget.

**Options:** (A) run full public at 16k (pub-002) (B) stay at 8k (C) MCQ-only 16k

**Decision:** **A** — run pub-002 on full `public.jsonl` before shipping; do not change shipped baseline until confirmed.

**Rationale:** dev-007 (n=225, 20% slice): MCQ **70.67%** (+20.3 pp vs pub-001), overall **60.00%** (+7.6 pp vs 8k dev slice). Free-form flat (54.67%) — lift is almost entirely MCQ, matching truncation hypothesis.

**Consequences:** §1.1 dev gate passed. Update inference baseline only after pub-002. Artifact naming: `dev_results_{variant}_{Nk}k.jsonl`.

**Experiment:** [dev-007](runs/dev-007-max-tokens-16k.md)

---

## D008 — 32k `max_tokens` rejected; stay at 16k

**Date:** 2026-05-24

**Context:** After dev-007/dev-008 validated 16k on A100, checked whether `MAX_TOKENS=32768` with `max_model_len=32768` unlocks further gains on the same 10% dev slice ([dev-008](runs/dev-008-multi-blank-16k.md) setup).

**Options:** (A) adopt 32k for pub-002 (B) keep 16k (C) raise `max_model_len` beyond 32k for true 32k+ generation headroom

**Decision:** **B** — keep 16k generation cap; do not pursue 32k or higher `max_model_len` without new truncation evidence.

**Rationale:** dev-009 (n=112, multi_blank): overall **64.29%** vs dev-008 **65.18%** (−0.89 pp). MCQ identical (78.38%); free-form slightly worse (57.33% vs 58.67%). No MCQ truncation benefit beyond 16k on this slice.

**Consequences:** pub-002 and submission path stay at `max_tokens=16384`. Option C deferred unless full-public analysis shows residual 16k truncation.

**Experiment:** [dev-009](runs/dev-009-max-tokens-32k.md)

---

## D005 — SFT assistant schema: explicit `<think>` wrapper

**Date:** 2026-05-21

**Context:** Step 1 in [sft/pipeline.md](../sft/pipeline.md) — compare `apply_chat_template` for plain vs tagged assistant turns on `Qwen/Qwen3-4B-Thinking-2507`. Artifact: `data/qwen_thinking_trace.txt` (from `notebooks/dev.ipynb`).

**Options:** (A) store plain CoT in the assistant `content` field (B) wrap reasoning in `<think>...</think>` and keep `\boxed{...}` after the closing tag

**Decision:** **B**

**Rationale:** Inference ends with an open `<think>` block (model fills it, then emits the answer after `</think>`). With **plain** assistant text, the template inserts an **empty** `</think>` pair and places all reasoning **outside** the thinking block (137 tokens, wrong layout). With **explicit** tags, reasoning is inside the block and `\boxed{5/8}` is outside — same token count, correct layout. `judger.py` already grades from text after the last `</think>`.

**Consequences:** Numina (and any SFT) `response` strings must use the explicit wrapper; do not rely on plain CoT in `render_training_messages`. Record `thinking_template: "explicit_redacted_thinking"` in `data/sft_corpus_manifest.json` when the corpus is built. Unblocks M3 / trace generation in [sft/data-spec.md](../sft/data-spec.md).

---

## D009 — bf16 LoRA replaces QLoRA for sft-001 (A100-only)

**Date:** 2026-05-24

**Context:** [sft/pipeline.md](../sft/pipeline.md) Training config originally specified QLoRA 4-bit NF4 + paged AdamW 8-bit (via `bitsandbytes`). Memory check on A100 40 GB shows Qwen3-4B (~8 GB bf16) + LoRA r=32 + Adam states + activations at 8k seq, mb=1, grad-ckpt fits in ~22–30 GB — comfortably within 40 GB.

**Options:** (A) keep QLoRA 4-bit + paged AdamW 8-bit (B) **bf16 LoRA + `adamw_torch_fused`, A100-only, no L4 fallback** (C) flag-toggled both paths

**Decision:** **B** — drop `bitsandbytes` entirely; bf16 base + bf16 LoRA + bf16 compute on A100. No 4-bit path, no L4 path. If Colab assigns L4, abort the session.

**Rationale:** A100 has native bf16 tensor cores — no dequant overhead, ~1.5–2× faster wall-clock than QLoRA at the same config. Avoids 4-bit dequantization quality loss (small but non-zero on math reasoning). Simpler dep tree: no `bitsandbytes` install, no NF4 / double-quant flags, no `BitsAndBytesConfig`. Course compute (Colab Pro+) reliably provides A100; the L4 fallback was a hedge for compute we don't need.

**Consequences:**
- `notebooks/sft_train.ipynb`: removed `BitsAndBytesConfig`, swapped optimizer to `adamw_torch_fused`, dropped `bitsandbytes` from `%pip`.
- `pipeline.md` Training config + Compute budget + Colab guardrails updated.
- Quality comparison against any future QLoRA ablation must control for this difference.
- On 80 GB A100, mb=2 / accum=8 is now feasible; on 40 GB stay at mb=1 / accum=16.

**Details:** [sft/pipeline.md](../sft/pipeline.md) §"Training config".

---

## D010 — sft-002a OpenR1 1k flat — do not scale to 5k yet

**Date:** 2026-05-26

**Context:** First probe of R1-distilled traces ([sft-002a](runs/sft-002a-openr1-1k.md)): 1k OpenR1-Math rows, bf16 LoRA r=32, 1 epoch (~63 steps), eval on frozen `holdout_20p` (225 rows) with shipped inference stack (`multi_blank`, 16k).

**Options:** (A) scale to sft-002b (5k stratified) (B) treat as flat and pivot to sft-003 / iterate training (C) run SC K=5 on adapter before abandoning

**Decision:** **B** for scale-up; **C** optional before pivot.

**Rationale:** Overall **64.44%** on holdout_20p is **~0.7 pp below** [dev-008](runs/dev-008-multi-blank-16k.md) (65.18% on 112 rows, same prompt, no LoRA) and matches the **~64–65%** projection from dev-007 + multi_blank on 20% — i.e. **flat** per [pipeline decision gate](../sft/pipeline.md#phase-1-1k-probe-sft-002a). No stop-rule violations (MCQ 77.33%, length +6.6%, boxed emission 94.67%), but watch slices **Q4 long** and **multi-blank ≥3** stayed at **40%**. Training ran **1 epoch** (pipeline had planned 2); `final_adapter` may not be the best checkpoint.

**Consequences:** Do not open sft-002b on this result alone. Base-vs-LoRA A/B on holdout_20p: [sft-eval-001](runs/sft-eval-001-baseline-holdout-20p.md) **0.00 pp** vs sft-002a (same 225 ids). Next: epoch-2 or `checkpoint-*` eval; if still flat → sft-003 or Numina fallback (sft-001). Optional SC K=5 on `holdout_10p` per modest-win path.

**Experiment:** [sft-002a](experiments.md#sft-002a)

---

## D011 — `verify_prompt` rejected on holdout_20p

**Date:** 2026-05-28

**Context:** Hypothesis that a light “verify before boxing” nudge (substitute back, units/range/signs, independent check) on top of shipped `multi_blank` format would catch arithmetic/sign errors without the token cost of strategy enumeration or PHP.

**Options:** (A) ship `verify_prompt` on public/private (B) reject; keep `multi_blank` (C) MCQ-only verify clause

**Decision:** **B** — do not ship; keep **`multi_blank`** for inference.

**Rationale:** [dev-013-verify](runs/dev-013-verify-holdout-20p.md) on frozen `holdout_20p` (225 rows): **64.44% overall**, **0.00 pp** vs [sft-eval-001](runs/sft-eval-001-baseline-holdout-20p.md) (same 145/225 correct). Sub-slices: MCQ **+1.34 pp**, FF **−0.67 pp**, multi-blank **−1.22 pp** — net zero with item-level tradeoffs only.

**Consequences:** Drop verification-forcing from submission path. Format instructions (`\boxed{}` layout) remain the proven prompt lever; reasoning nudges on a thinking model are low ROI at 4B scale.

**Experiment:** [dev-013-verify](experiments.md#dev-013-verify)

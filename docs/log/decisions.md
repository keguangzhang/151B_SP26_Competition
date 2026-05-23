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

## D005 — SFT assistant schema: explicit `<think>` wrapper

**Date:** 2026-05-21

**Context:** Step 1 in [sft/pipeline.md](../sft/pipeline.md) — compare `apply_chat_template` for plain vs tagged assistant turns on `Qwen/Qwen3-4B-Thinking-2507`. Artifact: `data/qwen_thinking_trace.txt` (from `notebooks/dev.ipynb`).

**Options:** (A) store plain CoT in the assistant `content` field (B) wrap reasoning in `<think>...</think>` and keep `\boxed{...}` after the closing tag

**Decision:** **B**

**Rationale:** Inference ends with an open `<think>` block (model fills it, then emits the answer after `</think>`). With **plain** assistant text, the template inserts an **empty** `</think>` pair and places all reasoning **outside** the thinking block (137 tokens, wrong layout). With **explicit** tags, reasoning is inside the block and `\boxed{5/8}` is outside — same token count, correct layout. `judger.py` already grades from text after the last `</think>`.

**Consequences:** Numina (and any SFT) `response` strings must use the explicit wrapper; do not rely on plain CoT in `render_training_messages`. Record `thinking_template: "explicit_redacted_thinking"` in `data/sft_corpus_manifest.json` when the corpus is built. Unblocks M3 / trace generation in [sft/data-spec.md](../sft/data-spec.md).

---

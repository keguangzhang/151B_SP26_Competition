# Roadmap — improvement directions

Forward-looking ideas and priorities. **Measured results live in** [`log/experiments.md`](log/experiments.md); **decisions in** [`log/decisions.md`](log/decisions.md).

## Constraints

See [`reference/constraints.md`](reference/constraints.md). Current shipped baseline: **52.66% overall** (50.40% MCQ / 53.79% free-form) at `max_tokens=8192` — [pub-001](log/experiments.md#pub-001).

**Dominant MCQ bottleneck (8k):** format — 87% of wrong MCQ lack `\boxed{Letter}`; when emitted, ~88.3% correct. See [`analysis/baseline-public-8k.md`](analysis/baseline-public-8k.md).

---

## Tier 1 — Inference-time fixes (no training)

### 1.1 Constrained decoding for MCQ — **next / inconclusive**

vLLM guided/structured generation on MCQ tail only. Two-pass variant if single-pass regex fails.

- **Expected:** +10–20 pp MCQ (+4–7 pp overall) if emission rate lifts.
- **Status:** tried on dev — [dev-004](log/runs/dev-004-guided-decoding-10pct.md), [dev-005](log/runs/dev-005-guided-decoding-20pct.md); [D003](log/decisions.md#d003). **Need full public run.**

### 1.2 Lift `max_tokens` — **DONE → baseline**

8192 shipped ([D001](log/decisions.md#d001)). Optional: 16,384 MCQ-only — low priority.

### 1.3 MCQ-specific prompt + lower temperature — **re-scope**

Tested flat on 4k dev ([dev-002](log/runs/dev-002-mcq-prompt-temp.md), [D002](log/decisions.md#d002)). Retest on **8k** without 1500-token cap; isolate lower temperature.

### 1.4 Self-consistency / majority vote

k=5–8 samples; vote letters / canonicalized boxed answers. **Expected:** +3–7 pp overall. **Status:** open.

### 1.5 Multi-blank free-form structure

Explicit per-blank `\boxed{}` prompt. **Expected:** +3–6 pp free-form. **Status:** open.

### 1.6 Length-aware free-form stop

Soft commit / truncate after first complete `\boxed{}`. **Expected:** +1–3 pp free-form. **Status:** open.

---

## Tier 2 — Supervised fine-tuning

Active plan: [`sft/pipeline.md`](sft/pipeline.md). First run: Numina-only QLoRA — [D004](log/decisions.md#d004).

### 2.1 Data sources

Numina primary; defer AGIEval/GaoKao/short MATH; MCQ-heavy mix later if Numina run succeeds.

### 2.2 QLoRA on single GPU

`unsloth` / `trl` + `peft`; eval each epoch on held-out public slice.

### 2.3 Avoid free-form regression

Long-CoT mix, `loss_on_response_only`, early stop on FF plateau.

### 2.4 Format-only mini-SFT

If §1.1 leaves gap after inference experiments.

---

## Tier 3 — Reinforcement learning

GRPO/DPO with `judger` reward — after SFT plateaus. See original tier detail in git history of `improvement-directions.md` if needed.

---

## Tier 4 — Targeted weaknesses

Topic gaps from [pub-001 analysis](analysis/baseline-public-8k.md): number theory (~30%), sequences (~35%), geometry (~36%). Few-shot routing or topic-weighted SFT mix.

| Topic | Count | Baseline acc (8k) |
|-------|------:|------------------:|
| Number theory | 23 | **30.4%** |
| Sequences / recurrences | 75 | **34.7%** |
| Geometry | 115 | **35.6%** |
| Integration | 55 | **74.5%** |

---

## Tier 5 — Deployment / capacity

### 5.1 BF16 instead of INT8

**Expected:** +1–3 pp if VRAM allows. **Status:** open.

### 5.2 Speculative decoding

Throughput for self-consistency only.

---

## Suggested execution order

| Order | Direction | Cost | Status / experiment |
|-------|-----------|-----:|---------------------|
| — | §1.2 `max_tokens` → 8192 | — | ✅ [pub-001](log/experiments.md#pub-001) |
| 1 | §1.1 Guided decoding (full public) | hours | [D003](log/decisions.md#d003) — pub run TBD |
| 2 | §1.5 Multi-blank prompt | hours | open |
| 3 | §1.3 MCQ format on 8k baseline | hours | [dev-002](log/experiments.md#dev-002) — re-scope |
| 4 | §1.2b 16k MCQ-only | minutes | open |
| 5 | §1.4 Self-consistency k=5 | day | open |
| 6 | §1.6 Length-aware FF stop | hours | open |
| 7 | §5.1 INT8 → BF16 | hour | open |
| 8 | §2 Numina QLoRA SFT | 1–2 days | [sft-001](log/experiments.md) planned |
| 9 | §2.4 Format mini-SFT | half day | conditional |
| 10 | §4 Topic few-shot / weighted mix | day | open |
| 11 | §3 GRPO | 2–4 days | open |

Re-measure on full `public.jsonl` after each shipped inference change; dev Δ &lt; ~3 pp is noise at n=112.

---

## Open questions

- DeepSeek-R1 distill license for competition use.
- `guided_decoding` on Colab L4 image ([`infra/vllm-colab-l4.md`](infra/vllm-colab-l4.md)).
- Private MCQ option-count distribution vs public.
- GPU-hour budget for RL feasibility.

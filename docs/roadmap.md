# Roadmap — improvement directions

Forward-looking ideas and priorities. **Measured results live in** [`log/experiments.md`](log/experiments.md); **decisions in** [`log/decisions.md`](log/decisions.md).

## Constraints

See [`reference/constraints.md`](reference/constraints.md). Current shipped baseline: **52.66% overall** (50.40% MCQ / 53.79% free-form) at `max_tokens=8192` — [pub-001](log/experiments.md#pub-001).

**Dominant MCQ bottleneck (8k): token truncation** — 84.4% of wrong MCQ were cut off mid-think (no `</think>`), leaving no answer to format. Only 2.7% of wrong MCQ finished thinking but failed to emit `\boxed{Letter}`. Finished responses score **69.5%** vs **3.8%** for truncated. See [`analysis/baseline-public-8k.md`](analysis/baseline-public-8k.md) (2026-05-23 revision).

| Group | N | Accuracy |
|-------|--:|--------:|
| Think finished (`</think>` present) | 838 | **69.5%** |
| Truncated mid-think (no `</think>`) | 288 | **3.8%** |
| MCQ truncation rate | 165/375 | **44.0%** |

---

## Tier 1 — Inference-time fixes (no training)

### 1.1 Lift `max_tokens` to 16,384 — **NEXT / highest priority**

44% of MCQ responses truncate before writing any answer at 8k tokens. Doubling the budget directly unlocks answers the model has already reasoned toward.

- **Expected:** MCQ truncation 44% → ~10–15%; MCQ accuracy ~70–78%; overall ~62–68%.
- **Ceiling:** finished-response accuracy is ~69.5% overall; more tokens gets more responses there.
- **Status:** open — run `pub-002`.

### 1.2 Thinking-efficiency prompting — **next if 16k not feasible**

If GPU VRAM forces staying at 8k, prompt the model to reason more concisely (e.g., `/no_think` or a "be concise" system prompt directive). Goal: reduce median think-trace length so more responses finish within budget.

- **Expected:** modest gain; less reliable than more tokens. Estimate +3–8 pp MCQ.
- **Risk:** concise reasoning may reduce accuracy for hard items that need long chains.
- **Status:** done — [dev-006](log/runs/dev-006-concise-prompt.md) rejected; MCQ 48.00% vs 50.40% baseline (flat/worse). Truncation is structural; prompt can't fix it.

### 1.3 Multi-blank free-form structure — **high value, independent**

Explicit per-blank `\boxed{}` prompt: "Answer 1: `\boxed{...}`, Answer 2: `\boxed{...}`". Currently 47.8% vs 63% for single-blank.

- **Expected:** +3–6 pp free-form. **Status:** open.

### 1.4 Self-consistency / majority vote

k=5–8 samples; vote letters / canonicalized boxed answers. Most useful after truncation is resolved (truncated samples vote randomly).

- **Expected:** +3–7 pp overall on finished responses. **Status:** open — defer until §1.1 done.

### 1.5 MCQ-specific prompt + lower temperature — **low priority**

Retest on 8k baseline ([dev-002](log/runs/dev-002-mcq-prompt-temp.md), [D002](log/decisions.md#d002)). Low priority: format is only 2.7% of wrong MCQ; prompt changes won't move truncation.

### 1.6 Constrained decoding for MCQ — **deprioritized**

vLLM guided/structured generation on MCQ tail. Previously ranked #1 based on misdiagnosis of format compliance as the bottleneck. Updated analysis: only 5 responses (2.7% of wrong MCQ) finish thinking without a boxed letter — constrained decoding addresses a tiny slice.

- **Status:** [dev-004](log/runs/dev-004-guided-decoding-10pct.md), [dev-005](log/runs/dev-005-guided-decoding-20pct.md); [D003](log/decisions.md#d003). Deprioritized — revisit only after §1.1.

### 1.7 Length-aware free-form stop

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

> Priority order revised 2026-05-23: token truncation (not format) is the dominant failure. See [`analysis/baseline-public-8k.md`](analysis/baseline-public-8k.md).

| Order | Direction | Expected gain | Cost | Status |
|-------|-----------|--------------|-----:|--------|
| — | 8k baseline | — | — | ✅ [pub-001](log/experiments.md#pub-001) |
| 1 | §1.1 `max_tokens` → 16,384 | +10–16 pp overall | hours | open — **pub-002** |
| 2 | §1.3 Multi-blank free-form prompt | +3–6 pp FF | hours | open |
| 3 | §1.2 Thinking-efficiency prompt (if 16k constrained) | +3–8 pp MCQ | hours | ~~rejected~~ [dev-006](log/runs/dev-006-concise-prompt.md) |
| 4 | §1.4 Self-consistency k=5 | +3–7 pp overall | day | open — after §1.1 |
| 5 | §5.1 INT8 → BF16 | +1–3 pp | hour | open |
| 6 | §2 Numina QLoRA SFT | unknown | 1–2 days | [sft-001](log/experiments.md) planned |
| 7 | §2.4 Format mini-SFT | conditional | half day | conditional |
| 8 | §1.7 Length-aware FF stop | +1–3 pp FF | hours | open |
| 9 | §4 Topic few-shot / weighted mix | unknown | day | open |
| 10 | §1.6 Guided decoding | +0–2 pp MCQ | hours | deprioritized |
| 11 | §3 GRPO | unknown | 2–4 days | open |

Re-measure on full `public.jsonl` after each shipped inference change; dev Δ &lt; ~3 pp is noise at n=112.

---

## Open questions

- DeepSeek-R1 distill license for competition use.
- `guided_decoding` on Colab L4 image ([`infra/vllm-colab-l4.md`](infra/vllm-colab-l4.md)).
- Private MCQ option-count distribution vs public.
- GPU-hour budget for RL feasibility.

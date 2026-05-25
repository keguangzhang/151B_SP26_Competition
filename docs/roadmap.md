# Roadmap — improvement directions

Forward-looking ideas and priorities. **Measured results live in** [`log/experiments.md`](log/experiments.md); **decisions in** [`log/decisions.md`](log/decisions.md).

## Constraints

See [`reference/constraints.md`](reference/constraints.md). Current shipped baseline: **61.90% overall** (72.00% MCQ / 56.86% free-form) at `max_tokens=16384` with adaptive multi-blank prompt — [pub-002](log/experiments.md#pub-002).

**Dominant failure mode (16k): MCQ reasoning errors.** Truncation is mostly solved (6.9% of all responses, down from 25.6% at 8k). The wrong-MCQ distribution has inverted: **51.4%** of wrong MCQ are now "think finished, wrong `\boxed{Letter}`" — pure reasoning failures. Truncation accounts for only 41.0% of wrong MCQ (down from 84.4% at 8k). See [`analysis/baseline-public-16k.md`](analysis/baseline-public-16k.md).

| Group | N | Accuracy |
|-------|--:|--------:|
| Think finished (`</think>` present) | 1,048 | **66.3%** |
| Truncated mid-think (no `</think>`) | 78 | **2.6%** |
| MCQ truncation rate | 44/375 | **11.7%** |

Secondary bottlenecks ranked by leverage (only slices with N large enough to trust):

1. **Long questions** (Q4, ≥435 chars, n=281) score **43.8%** — 35 pp below short questions. Tightest signal.
2. **High-blank-count free-form** (≥6 blanks, n=63) score **~34%** — multi-blank prompt is already applied; gap is reasoning.
3. **Geometry** (n=115) at **50.4%** — only topic with clean weakness signal (95% CI ±9.1). "Other" (581 rows) is a heterogeneous catch-all and uninterpretable; small-n topics (number theory, limits, derivatives, linear algebra at n=12–23) have CI bands wider than their gap to overall — ignore them as targets.
4. **MCQ reasoning errors** — 51.4% of wrong MCQ at 16k are "think finished, wrong `\boxed{Letter}`" (n=54). Method-agnostic finding: inference tricks can't fix these; needs SFT.

---

## Tier 1 — Inference-time fixes (no training)

### 1.1 Lift `max_tokens` to 16,384 — ✅ shipped

Doubled token budget unlocked +9.24 pp overall (+21.6 pp MCQ). Truncation dropped from 25.6% to 6.9%. See [pub-002](log/experiments.md#pub-002), [dev-007](log/runs/dev-007-max-tokens-16k.md), [dev-008](log/runs/dev-008-multi-blank-16k.md). 32k ablation rejected — [dev-009](log/runs/dev-009-max-tokens-32k.md): no lift, −0.89 pp overall.

### 1.2 Thinking-efficiency prompting — rejected

[dev-006](log/runs/dev-006-concise-prompt.md): MCQ 48.00% vs 50.40% baseline. Truncation was structural; prompt cannot fix it. Now superseded by §1.1.

### 1.3 Multi-blank free-form structure — ✅ shipped

Adaptive prompt (`\boxed{a}, \boxed{b}, …` for 2+ `[ANS]` questions) shipped in pub-002. Multi-blank residual gap (≥6 blanks at 34.0%) is now a reasoning problem, not a format problem.

### 1.4 Self-consistency / majority vote — **promoted**

k=5–8 samples; vote on letters / canonicalized boxed answers. Now that truncation is solved, samples reach the answer reliably and reasoning variance is the dominant noise source — exactly what self-consistency exists to suppress.

- **Expected:** +3–7 pp overall, larger on MCQ where reasoning errors now dominate.
- **Cost:** 5× inference compute on the full eval set.
- **Status:** open — strongest pure-inference lever post-pub-002. Consider running on MCQ-only first to bound cost.

### 1.5 MCQ-specific prompt + lower temperature — low priority

Format compliance is no longer the bottleneck. Only 7.6% of wrong MCQ at 16k are "finished, no `\boxed{Letter}`". Re-test only if a structured-output route is attempted alongside.

### 1.6 Constrained decoding for MCQ — deprioritized

[dev-004](log/runs/dev-004-guided-decoding-10pct.md), [dev-005](log/runs/dev-005-guided-decoding-20pct.md), [D003](log/decisions.md#d003). Addresses a small slice (7.6% of wrong MCQ at 16k). Skip.

### 1.7 Length-aware free-form stop

Soft commit / truncate after first complete `\boxed{}`. **Expected:** +1–3 pp free-form. **Status:** open.

### 1.8 Re-audit truncation at 16k — ✅ done (folded into [`baseline-public-16k.md`](analysis/baseline-public-16k.md))

Residual truncation is 6.9% overall (11.7% MCQ, 4.5% FF). Confirms §1.9 / §1.10 / §1.12 have shrinking returns; deprioritize all three relative to reasoning-quality interventions.

### 1.9 Multi-blank-specific token budget — deprioritized

Truncation rate on free-form is now only 4.5%. The multi-blank weakness at ≥6 blanks (34.0%) is dominated by reasoning, not budget — the multi-blank §6 data shows wrong responses average 18,151 chars, well within 16k budget. Expected lift ≤1–2 pp; skip unless §1.4 / §2 plateau.

### 1.10 Retry-on-truncation — deprioritized

Residual truncation tail is 78 rows (6.9%). Even perfect recovery caps the lift at ~2 pp overall, and retries risk re-truncating. **Status:** open but low priority.

### 1.11 Few-shot exemplar for multi-blank prompt

The adaptive multi-blank prompt is shipped and works. An exemplar might lock in shape on long-blank-count items (≥6 blanks at 34.0%). **Expected:** +1–3 pp on the multi-blank slice (small overall lift). **Status:** open — cheap, run before §1.4 if time permits.

### 1.12 Budget-forced thinking cap — deprioritized

Inject `</think>` at a soft threshold. With truncation at 6.9%, the addressable slice is small. Skip unless §1.4 plateaus.

### 1.13 Progressive-hint prompting (PHP) — **promoted**

Two-pass: run #1 produces tentative answer; run #2 re-prompts with "your previous attempt was X, verify or revise." Directly attacks the new dominant failure (reasoning errors that survive a single pass).

- **Expected:** +2–4 pp overall; 2× compute (vs 5× for §1.4 self-consistency).
- **Risk:** pass 2 may anchor on a wrong pass-1 answer — use neutral re-prompt.
- **Status:** open — cheapest reasoning-error lever after SFT.

---

## Tier 2 — Supervised fine-tuning — **next highest-leverage**

Active plan: [`sft/pipeline.md`](sft/pipeline.md). Numina corpus prepared ([`sft-prep-002`](log/experiments.md#sft-prep-002): 15,000 rows, mix locked).

### 2.1 Why now

MCQ reasoning errors (51.4% of wrong MCQ) and long-question failures (Q4 at 43.8%) are both reasoning-quality problems that inference tricks cannot move. SFT on Numina hard problems targets both simultaneously:

- Hard MCQ on geometry / sequences / "other" — 22–25% error rate on finished-boxed.
- Long-context problems — Q4 mean question length 498 chars.
- High-blank-count free-form — same reasoning-chain quality gap.

### 2.2 Data sources

Numina primary ([`sft/numina-clean-audit.md`](sft/numina-clean-audit.md)). Defer AGIEval / GaoKao. If sft-001 succeeds, consider MCQ-heavy supplement in sft-002.

### 2.3 QLoRA on single GPU

`unsloth` / `trl` + `peft`; eval each epoch on held-out public slice. See [`sft/pipeline.md`](sft/pipeline.md).

### 2.4 Avoid free-form regression

Long-CoT mix, `loss_on_response_only`, early stop on FF plateau. FF baseline 56.86% is the floor — don't ship regressions there.

### 2.5 Format-only mini-SFT

Skip unless §2 (full SFT) leaves a residual format gap.

---

## Tier 3 — Reinforcement learning

GRPO/DPO with `judger` reward — after SFT plateaus. See original tier detail in git history of `improvement-directions.md` if needed.

---

## Tier 4 — Targeted weaknesses (16k baselines)

Topic counts and 95% Wald CIs from `data/full_public_16k_topics.json`. **Caveat:** "Other" is 51.6% of the dataset (581 / 1126) — a heterogeneous catch-all, not a coherent topic. Most named topics have n ≤ 25 with CI bands wider than the gap to the overall mean (61.9%); their accuracy numbers are noise.

| Topic | n | Acc | 95% CI ± | Signal |
|-------|--:|----:|---------:|--------|
| other | 581 | 62.1% | ±3.9 | uninterpretable — catch-all |
| polynomials/algebra | 146 | 63.7% | ±7.8 | near overall — no gap |
| **geometry** | 115 | **50.4%** | ±9.1 | **real weakness — only topic with clean signal** |
| probability/stats | 82 | 58.5% | ±10.7 | within noise |
| sequences/recurrences | 75 | 58.7% | ±11.1 | within noise |
| integration | 55 | 87.3% | ±8.8 | real strength |
| linear algebra | 23 | 60.9% | ±19.9 | noise |
| number theory | 23 | 56.5% | ±20.3 | noise |
| limits | 14 | 57.1% | ±25.9 | noise |
| derivatives | 12 | 83.3% | ±21.1 | noise |

**Actionable signal for SFT mix** (slices with tight enough N to trust):

| Slice | n | Acc | Source |
|---|--:|---:|---|
| Q4 question length (≥435 chars) | 281 | **43.8%** | §4 of 16k analysis |
| Multi-blank ≥6 blanks | 63 | **~34%** | §3 of 16k analysis |
| Geometry | 115 | **50.4%** | this table |
| MCQ "think finished, wrong boxed" | 54 | reasoning failures | §1 of 16k analysis |

Strategy: oversample **long-context** and **multi-blank** problems in the Numina mix; weight geometry; ignore the small-n topic differentials and "Other" entirely.

---

## Tier 5 — Deployment / capacity

### 5.1 BF16 instead of INT8 — ✅ shipped (in pub-002)

pub-002 already runs `dtype="bfloat16"` on A100 (see `notebooks/full_public.ipynb` and the vLLM init log). The starter L4 INT8 path was only used for pub-001. The `pub-001 → pub-002` delta therefore conflates three changes: 8k→16k tokens, starter→adaptive multi-blank prompt, and L4 INT8 → A100 bf16. dev-007 (bf16, 16k, baseline prompts, 20% dev) hit 60.00% overall, so most of the +9.24 pp lift is attributable to tokens, but precision contribution is not separately measured.

### 5.2 Speculative decoding

Throughput for self-consistency only.

---

## Suggested execution order

> Priority order revised 2026-05-24 after pub-002. Truncation is solved; reasoning errors and long-question / high-blank failures are the new bottlenecks. See [`analysis/baseline-public-16k.md`](analysis/baseline-public-16k.md).

| Order | Direction | Expected gain | Cost | Status |
|-------|-----------|--------------|-----:|--------|
| — | 8k baseline | — | — | ✅ [pub-001](log/experiments.md#pub-001) — 52.66% |
| — | 16k + adaptive multi-blank | +9.24 pp | done | ✅ [pub-002](log/experiments.md#pub-002) — **61.90%** |
| **1** | **§2 Numina QLoRA SFT → sft-001** | **unknown (target +3–8 pp)** | **1–2 days** | **next** — corpus ready, attacks dominant reasoning bottleneck |
| 2 | §1.13 Progressive-hint prompting (2-pass) | +2–4 pp | 2× compute (~4 hr Colab) | open — cheapest reasoning-error lever |
| 3 | §1.4 Self-consistency k=5 | +3–7 pp claimed; lift constrained — MCQ already 72%, FF voting needs symbolic canonicalization | 5× compute (~10 hr Colab) | open but **questionable ROI** at current MCQ ceiling |
| 4 | §1.7 Length-aware FF stop | +1–3 pp FF | hours | open |
| 5 | §1.11 Multi-blank few-shot exemplar | ≤1 pp overall (analysis says gap is reasoning, not format) | hour | low-priority — likely noise |
| 6 | §4 Topic few-shot / weighted SFT mix | unknown | day | open — fold into sft-002 |
| 7 | §1.9 Multi-blank 24k routing | ≤2 pp | hours | deprioritized — truncation tail too small |
| 8 | §1.10 Retry-on-truncation | ≤2 pp | hours | deprioritized |
| 9 | §1.12 Budget-forced thinking cap | ≤2 pp | hours | deprioritized |
| 10 | §1.6 Guided decoding | ≤2 pp MCQ | hours | deprioritized |
| 11 | §3 GRPO | unknown | 2–4 days | open — after SFT plateaus |

> §5.1 (BF16) is already shipped in pub-002 — removed from order.

Re-measure on full `public.jsonl` after each shipped change; dev Δ < ~3 pp is noise at n=112.

---

## Open questions

- DeepSeek-R1 distill license for competition use.
- `guided_decoding` on Colab L4 image ([`infra/vllm-colab-l4.md`](infra/vllm-colab-l4.md)).
- Private MCQ option-count distribution vs public.
- GPU-hour budget for RL feasibility.
- Self-consistency vote scheme for free-form (canonicalize numeric / symbolic answers before voting).

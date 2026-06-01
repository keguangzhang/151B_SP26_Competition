# Improvement techniques survey — 2026-05-24

External-evidence survey of techniques to lift unified accuracy on `Qwen/Qwen3-4B-Thinking-2507` for the CSE 151B competition. Complements (does not replace) [`../roadmap.md`](../roadmap.md). All techniques are filtered through the competition rules: **single ~4B model, no tool-augmented inference, no external APIs at submission time**.

Baseline: [pub-002](../log/experiments.md#pub-002) — 61.90% overall (72.00% MCQ / 56.86% FF) at `max_tokens=16384`. Dominant failure: MCQ reasoning errors (51.4% of wrong MCQ are "think finished, wrong `\boxed{Letter}`"); secondary: Q4 long-questions (43.8%) and ≥6-blank free-form (~34%).

---

## Why this survey exists

The roadmap's top-ranked lever is `sft-001` (Numina LoRA SFT), with `pipeline.md` projecting EV ≈ +1–2 pp and a 20% regression tail. That ranking was set when the realistic alternatives were §1.4 self-consistency (5× inference, doc itself flags "questionable ROI") and §1.13 PHP (+2–4 pp, 2× inference). This survey pulls in 2025–2026 techniques that didn't exist (or weren't on the roadmap) at the time pub-002 shipped, then re-ranks.

---

## New techniques considered

### A. Budget forcing (s1-style test-time scaling)

**Source:** [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) — Stanford, ICML 2025.

**Mechanism:** when the model emits `</think>` early, suppress the token and append `"Wait"` to force continued reasoning. Inverse — also supports forcing termination at a budget cap. No fine-tuning required.

**Headline result:** Qwen2.5-32B-Instruct on AIME24 lifted from 50% → 57% (+7 pp) with budget forcing alone after a tiny 1k SFT. The lift came from the decoding hack, not the SFT.

**Relevance:** the residual failure mode after pub-002 is exactly "think finished, wrong answer." Budget forcing literally targets that case — give the model another pass at its own reasoning before committing to `\boxed{}`.

| | |
|---|---|
| Training cost | 0 |
| Inference cost | ~1.3× (only kicks in when `</think>` fires early) |
| Realistic gain | +3–7 pp |
| Risk | Forcing past 16k budget can re-introduce truncation. Cap forcing iterations (e.g., at most 2) and stop if context > 14k. |
| Composes with | PHP (§1.13), CISC (C), per-class decoding (F) |

**Measured (2026-05-25, dev slice):** [dev-010-bf](../log/runs/dev-010-bf-budget-forcing.md) — FF **58.67%** on 112-row `holdout_10p` (+5.3 pp vs baseline 16k without BF); **72/75** FF items forced (96%). **Flat vs dev-008 multi_blank** on the same rows — not independent of §1.3 on dev.

### B. Distillation from a stronger thinking-style teacher

**Sources:**
- [NemoSkills AIMO2 winner blog](https://blogs.nvidia.com/blog/reasoning-ai-math-olympiad/) — distilled DeepSeek-R1 + QwQ-32B traces into Qwen2.5-14B-Base, scored 34/50.
- [Phi-4-Mini-Reasoning](https://arxiv.org/abs/2504.21233) — 3.8B model beat DeepSeek-R1-Distill-Qwen-7B by distilling from larger reasoners.
- [DeepSeek-R1 paper](https://arxiv.org/html/2501.12948v1) — confirms simple SFT from R1 traces transfers reasoning to smaller bases.

**Mechanism:** generate long `<think>` traces on Numina (or public-style synthetic) problems using a much larger thinking model — **Qwen3-30B-Thinking** (same family, same schema, ideal) or **DeepSeek-R1**. SFT 4B on those traces instead of human Numina solutions. Schema-matched teacher >> generic short human solutions.

**Why it dominates Numina-only SFT:**
1. Human Numina solutions are short, terse, and don't match Qwen3-Thinking's long `<think>` distribution. Training on them risks **trace collapse** (the exact risk pipeline.md §2.4 calls out).
2. Larger thinking models produce traces in the same style the base already knows — preserves rather than overwrites the RL-tuned reasoning.
3. AIMO2 evidence: this exact pattern was the winning recipe at a comparable scale.

| | |
|---|---|
| Training cost | Same as planned sft-001 (LoRA, A100). **Plus** teacher inference: ~12–24 hr on rented A100, or free via Together/OpenRouter/Groq for R1, or self-host Qwen3-30B-Thinking on a single A100 80GB. |
| Inference cost | 1× (one-time training cost) |
| Realistic gain | +4–10 pp |
| Risk | Teacher traces may exceed 16k. Filter at corpus build. License: DeepSeek-R1 is MIT; Qwen3-30B is Apache-2.0 — both safe for course use. |
| Composes with | All inference tricks (A, C, D, F) ride on top |

**Pivot proposal:** replace the Numina-only corpus in sft-001 with a teacher-distilled corpus on the same problem set. Keep the training pipeline identical. The corpus build (`notebooks/sft_data_prep.ipynb`) is the only thing that changes.

### C. CISC — Confidence-Informed Self-Consistency

**Source:** [Confidence Improves Self-Consistency in LLMs (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.1030.pdf).

**Mechanism:** instead of uniform majority vote, weight each sample by model confidence (token-level log-prob average, or verbal "how confident are you" prompt). Reaches the accuracy of k=10 vanilla SC at k=3.

**Why this matters:** roadmap §1.4 self-consistency was flagged "questionable ROI" because 5× inference cost is steep on a private set. CISC cuts that to 3× while reaching the same answer quality.

| | |
|---|---|
| Training cost | 0 |
| Inference cost | 3× (vs 5× for vanilla SC) |
| Realistic gain | +2–5 pp (mostly MCQ; FF requires symbolic canonicalization to vote on) |
| Risk | Token-prob confidence may be poorly calibrated on a thinking model that's already RL-tuned. Verbal-confidence variant is more robust but adds tokens. |
| Composes with | A, B, D, F |

### D. DPO on wrong-rollout pairs (post-SFT preference learning)

**Source:** [Phi-4-Mini-Reasoning](https://arxiv.org/abs/2504.21233) — uses rejected SFT-time rollouts to build preference pairs for DPO, after an initial SFT pass.

**Mechanism:** after sft-001 (or the distilled variant in B) lands, sample k=4 generations per training prompt. For each prompt, if at least one sample is correct (judged by `judger.py` against gold) and at least one is wrong, form a (correct, wrong) preference pair. Train DPO on those pairs.

**Why now:** "free" data — the SFT model already exists and its mistakes are the most informative training signal we have. No new annotation, no new corpus, no teacher needed.

| | |
|---|---|
| Training cost | ~4 hr rollout generation + ~6 hr DPO LoRA on A100 |
| Inference cost | 1× (training-time gain) |
| Realistic gain | +1–3 pp on top of SFT, larger if SFT win was modest |
| Risk | DPO can over-shrink the policy. Use a low β (~0.05) and keep a held-out eval. |
| Composes with | Stacks on top of SFT or B's distilled variant |

### E. Small outcome verifier / re-ranker

**Sources:**
- [Small Language Models Need Strong Verifiers to Self-Correct](https://arxiv.org/pdf/2404.17140) — small models benefit disproportionately from a separate verifier.
- [PROVE](https://arxiv.org/abs/2410.12608) — verifier-based re-ranking lifts Qwen2-0.5B GSM8K from 49% → 54%, Llama-3.2-1B from 66% → 73%.

**Mechanism:** train a small classifier (Qwen3-0.6B head, or even a logistic-reg over trace features) to predict P(correct | trace, question). Use it to re-rank self-consistency samples or budget-forcing branches. **Note:** PROVE's program-execution variant is ruled out (Python at inference = tool use), but the verifier-as-classifier variant is rules-compliant.

**Training data is free:** the pub-002 run already produced 1,126 traces with gold labels. That's enough to train a small verifier.

| | |
|---|---|
| Training cost | ~2–4 hr to train Qwen3-0.6B classifier |
| Inference cost | Small (verifier is 0.6B; runs once per candidate) |
| Realistic gain | +2–4 pp when stacked on self-consistency / budget forcing |
| Risk | Verifier overfits to pub-002 distribution; private may differ. Mitigate by training on multiple sampled traces per problem, not just pub-002. |
| Composes with | A, B, C — multiplicative on top of any candidate-generation method |

### F. Per-class sampling / decoding routing

**Mechanism:** classify the incoming question (MCQ vs FF, multi-blank count, topic, length quartile) and route to a tuned (temperature, top_p, max_tokens, budget-force aggressiveness) tuple per class.

**Why:** pub-002 uses one decoding profile for everything. Q4 long questions and ≥6-blank FF benefit from longer budget + more aggressive forcing; MCQ benefits from lower temp.

| | |
|---|---|
| Training cost | 0 (analysis only, ~1 day) |
| Inference cost | ~1× average (some classes use less, some more) |
| Realistic gain | +1–3 pp, mostly from MCQ stability and FF long-tail |
| Risk | Per-class profiles overfit to public distribution. Use coarse classes only. |
| Composes with | All others |

---

## Techniques ruled out by competition rules

| Technique | Why ruled out |
|---|---|
| **PROVE (program-as-verifier)** with Python execution | Python execution at inference = tool-augmented inference |
| **SC-TIR / TIR** (AIMO1 Numina recipe) | Tool-integrated reasoning explicitly forbidden |
| **GPT-4 / o1 / Claude as inference-time verifier** | External API at inference forbidden |
| **RAG over external math knowledge base via API** | External API; local-only retrieval would be borderline-OK but adds complexity for likely <2 pp |
| **MCTS over reasoning steps** | Not forbidden, but practically requires multiple model calls per step; cost prohibitive at 4B+ |
| **Full-parameter fine-tuning** | Not forbidden by competition; ruled out by our own pipeline.md §2 ("No full-parameter fine-tuning") for stability/compute reasons |

---

## Re-ranked execution order (post-survey)

Ordering combines EV, compute cost, composition value (does it stack with later items?), and time-to-result. Compare against [`../roadmap.md`](../roadmap.md) "Suggested execution order".

| Rank | Lever | Train cost | Inf cost | EV gain | Composes |
|---|---|---|---|---|---|
| **1** | **A — Budget forcing (`Wait` injection)** | 0 | ~1.3× | +3–7 pp | with everything |
| **2** | **B — Distill from Qwen3-30B-Thinking or R1 → 4B SFT** | 1–2 days + teacher inference | 1× | +4–10 pp | one-time base improvement |
| 3 | §1.13 PHP (existing roadmap item) | 0 | 2× | +2–4 pp | with A, C |
| 4 | C — CISC k=3 weighted self-consistency | 0 | 3× | +2–5 pp | with A, B, E |
| 5 | D — DPO on wrong rollouts | ~10 hr | 1× | +1–3 pp | only after SFT |
| 6 | Original sft-001 (Numina only, as planned) | 1–2 days | 1× | +1–2 pp EV | superseded by B |
| 7 | E — Small outcome verifier re-ranker | hours | small | +2–4 pp | on top of any sampler |
| 8 | F — Per-class decoding routing | day | ~1× | +1–3 pp | with everything |

**Concrete recommendation:**

1. **Run A first.** One day of work, no training. If it lands a +3–5 pp lift, the baseline jumps to ~65–67% and resets the case for everything downstream.
2. **Pivot sft-001 to option B.** Same training budget, much stronger teacher. Numina-only is keeping us safe; teacher distillation is the bet-the-house move with AIMO2 precedent.
3. **Stack PHP + CISC k=3 at submission.** Inference cost ≈ 2× × 3× = 6× on private; if A + B compounded to ~68%, this could push to 73–75%.
4. **D and E** are follow-ups once the above plateau.

---

## Why this changes the SFT-first ranking

The roadmap put SFT at #1 partly on two arguments that this survey weakens:

1. *"Only training touches reasoning quality."* — Budget forcing (A) is a counter-example: a pure decoding hack with +7pp empirical lift on a comparable thinking model, directly attacking our dominant failure mode.
2. *"Inference tricks compound multiplicatively on cost."* — CISC (C) cuts that compounding from 5× to 3×, making inference-side gains cost-competitive again.

The SFT-first ranking is still defensible *if we replace Numina with distilled teacher traces (B)*. Numina-only SFT against a 4B thinking base has a +1–2 pp EV and a 20% regression tail — option B has materially better upside with the same compute envelope and a real precedent (AIMO2).

---

## Open questions for follow-up research

- ~~**Budget forcing on `Qwen3-4B-Thinking-2507` specifically**~~ — **Answered (dev slice):** [dev-010-bf](../log/runs/dev-010-bf-budget-forcing.md) on `holdout_10p` (112 rows): FF **58.67%** (+5.3 pp vs unregistered baseline 16k), **96%** of FF items got ≥1 `Wait` injection; **0 pp** vs [dev-008](../log/runs/dev-008-multi-blank-16k.md) multi_blank on the same slice. Public A/B vs pub-002 still open.
- **Teacher choice for option B** — Qwen3-30B-Thinking (same family, same template) vs DeepSeek-R1 (stronger reasoning, different style). One should dominate; needs a small ablation on 500 Numina problems.
- **Verifier training distribution for option E** — pub-002 has 1,126 traces. Is that enough? Worth sampling 4 generations per problem on a 2k Numina slice to expand the verifier corpus.
- **Per-class profile design for F** — needs an MCQ/FF × length-quartile × multi-blank-count grid analysis on the 16k baseline failure data.

---

## Sources

- [s1: Simple test-time scaling — Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393)
- [Phi-4-Mini-Reasoning — Microsoft, 2025](https://arxiv.org/abs/2504.21233)
- [NemoSkills AIMO2 winner — NVIDIA blog](https://blogs.nvidia.com/blog/reasoning-ai-math-olympiad/)
- [HuggingFace AIMO Progress Prize 1 blog (Numina TIR)](https://github.com/huggingface/blog/blob/main/winning-aimo-progress-prize.md)
- [Confidence-Informed Self-Consistency (CISC) — ACL Findings 2025](https://aclanthology.org/2025.findings-acl.1030.pdf)
- [PROVE: Programs as Verifiers — 2024](https://arxiv.org/abs/2410.12608)
- [S2R: Self-verify and Self-correct — ACL 2025](https://aclanthology.org/2025.acl-long.1104.pdf)
- [Small Language Models Need Strong Verifiers — 2024](https://arxiv.org/pdf/2404.17140)
- [DeepSeek-R1 paper — 2025](https://arxiv.org/html/2501.12948v1)
- [LoRA Without Regret — Thinking Machines Lab](https://thinkingmachines.ai/blog/lora/)

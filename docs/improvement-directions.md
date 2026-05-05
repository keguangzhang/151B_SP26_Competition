# Improvement directions

Possible directions for improving baseline accuracy on the CSE 151B competition.

## Constraints (from competition rules)

- **Model is fixed:** `Qwen/Qwen3-4B-Thinking-2507`. May be fine-tuned, but cannot be swapped.
- **No external tools at inference:** no code interpreter, no calculator, no API calls, no retrieval against external services. Generation must be self-contained.
- **Allowed levers:** prompt engineering, decoding strategies, supervised fine-tuning (LoRA/QLoRA/full), reinforcement learning (GRPO/DPO/RLOO/outcome-reward), self-consistency, model-internal verification (re-prompting same model).
- **Metric:** unified accuracy = correct / total across MCQ + free-form. Equal weight per question.

The current baseline (`max_tokens=8192`, `Qwen3-4B-Thinking-2507`, INT8, 1,126 examples on `public.jsonl`) lands **50.40% MCQ** and **53.79% free-form**, **52.66% overall**. Both buckets matter at roughly equal weight, but **MCQ format compliance** is now the dominant bottleneck — 87% of wrong MCQ responses lack `\boxed{Letter}`.

---

## Measured results so far

### `data/dev.jsonl` ablations (112 rows)

See `docs/tests.md` for full tables.

| Change | MCQ Δ | Free-form Δ | Overall Δ |
|--------|------:|------------:|----------:|
| §1.3 prompt + temp tweak (MCQ-only) | −2.70 pp | +2.67 pp | +0.90 pp |
| §1.2 `max_tokens` 4096 → 8192 | **+24.32 pp** | +2.67 pp | **+9.82 pp** |

### Full `public.jsonl` baseline (1,126 rows, 8k tokens) — **current baseline**

| Split | Count | Accuracy |
|-------|------:|---------:|
| Overall | 1,126 | **52.66%** |
| MCQ | 375 | **50.40%** |
| Free-form | 751 | **53.79%** |

Key findings from full-set analysis (`docs/baseline-public-analysis.md`):
- **Format, not truncation, is now the MCQ bottleneck.** Only 6/186 wrong MCQ responses (~3%) hit the 8k cap. 87% of wrong MCQ responses have no `\boxed{Letter}`.
- **When the model emits `\boxed{Letter}`, accuracy is 88.3%.** Lifting emission rate is the highest-value MCQ lever.
- **Multi-blank free-form lags 16 pp** (46.7% vs 63.0% single-blank).
- **Wrong free-form traces are 2.3× longer** (14,094 vs 6,204 chars mean).

This flips the priority ordering: §1.1 (guided decoding / format enforcement) is now the **top** MCQ lever, not §1.2 which is already baked into the baseline.

---

## Tier 1 — Inference-time fixes (no training, ship today)

These should be tried first. They are cheap, reversible, and the analysis already points to large headroom.

### 1.1 Constrained decoding for MCQ

vLLM supports guided/structured generation (`guided_choice=["A","B",...,"J"]`, `guided_regex`, or `outlines` integration). For MCQ items, force the *final emission* to be a single letter from the option set.

Two implementations:

- **Single-pass with stop sequence:** allow thinking, then require model to commit via a structured suffix like `\nFinal answer: \boxed{X}` where `X` is constrained to the legal letters. vLLM's `guided_decoding` can enforce the regex on the tail.
- **Two-pass:** (a) run thinking pass with a budget (e.g. 3,500 tokens); (b) feed truncated trace + "Based on your analysis, the answer is `\boxed{`" and use guided choice to pick a letter. This guarantees a clean letter even when the trace is messy.

The full-public baseline shows **88.3% accuracy when the model emits `\boxed{Letter}`** (205 items emitted; 181 correct). Current emission rate: 205/375 = **54.7%**. The 170 non-emitting items score only **4.7%** (fallback letter guesses). Lifting emission rate from 55% → 95% moves MCQ from 50% toward the high 80s — the single highest-value MCQ lever remaining.

**Expected lift:** +10 to +20 pp on MCQ subset (≈ +4 to +7pp overall), starting from the 8k baseline.

**Risks:** guided decoding can interact with thinking-style models if applied to the entire sequence. Apply it only to the tail, not the reasoning trace.

### 1.2 Lift / repartition `max_tokens` — **DONE, now the baseline**

`max_tokens=8192` is already the current baseline. Key confirmed facts:
- Full `public.jsonl` (1,126 rows): **+11.9 pp overall vs 4k run**, **+24.3 pp MCQ**.
- Only **6/186** wrong MCQ responses (~3%) are near the 8k cap — truncation is no longer a meaningful failure mode.

**Remaining experiment:** push to **16,384** (MCQ-only) to find the diminishing-returns knee. Low priority since format is now the bottleneck, but worth a quick run to rule out residual truncation on long MCQ stems.

### 1.3 MCQ-specific prompt + lower temperature — **TESTED, ≈ flat on dev slice**

The starter uses one prompt for MCQ vs free-form already. Strengthen the MCQ prompt:

- **Hard format clause:** "After you finish reasoning, your response MUST end with a single line containing only `\boxed{X}` where X is one of the option letters."
- **Bound the reasoning:** "Limit your reasoning to at most ~1500 tokens before committing to an answer."
- **Lower temperature** for MCQ: `temperature=0.2`, `top_p=0.9`. MCQ is closed-form — exploration buys nothing past the first plausible chain.

**Measured (dev slice, 112 rows):** stronger `\boxed{}` final-line clause + ~1500-token reasoning hint + MCQ-only `temperature=0.2`, `top_p=0.9`. Result: MCQ −2.70 pp, free-form +2.67 pp, **overall +0.90 pp** — essentially noise at this sample size.

**Interpretation (updated):** tested on the 4k baseline where truncation dominated, so the format clause had nothing to fix. Now that 8k is the baseline and format is the primary failure mode, retesting the format clause *without* the 1500-token reasoning cap is worthwhile. The lower temperature sub-change is also worth isolating — MCQ is closed-form and temperature=0.2 shouldn't hurt.

### 1.4 Self-consistency / majority vote

Sample `k=5–8` completions per question, majority-vote the final boxed answer.

- For MCQ: vote over extracted letter. With 88% per-sample correctness *given a boxed letter*, voting compounds reliability.
- For free-form: vote over `simplify`-canonicalized boxed expressions (use `sympy` at scoring time only — judger code is allowed, just not at inference inside the model).

Cost: linear in `k`. With INT8 + vLLM the 4B model is fast enough that `k=5` is realistic for ~1,100 prompts. Watch the GPU-hour budget.

**Expected lift:** +3–7pp overall (well-documented pattern; smaller relative gains with thinking models because each chain is already long).

### 1.5 Multi-blank free-form structure

424 multi-blank items score **16 pp lower** than single-blank (46.7% vs 63.0%). Fix the prompt:

- Detect multi-blank by counting `____` / "(1)", "(2)" markers in the question (or by gold answer length when training).
- Prompt: "Provide one `\boxed{...}` per blank, in order. Number them: `\boxed{ans1}, \boxed{ans2}, ...`."
- Update post-processing to map all boxed answers, in order, to the gold list.

**Expected lift:** +3–6pp on free-form (lifts overall by ~2–4pp given 67% free-form share).

### 1.6 Length-aware free-form stop

Wrong free-form runs are **2.3× longer** than correct ones (14,094 vs 6,204 chars). Soft fix: instruct the model to commit once it has a clean derivation, and add stop-sequence after first complete `\boxed{...}\n\n` *only after some minimum length*. Implement as post-hoc truncation if needed.

**Expected lift:** +1–3pp free-form. Lower priority — fixing it fully needs better reasoning, not better stops.

---

## Tier 2 — Supervised fine-tuning (medium cost, large lift)

Once Tier 1 is exhausted, SFT on math-reasoning data is the highest-yield next step. The base model is already a *Thinking* variant, so SFT should preserve thinking traces, not flatten them.

### 2.1 Data sources (publicly available)

All math-only, freely available, decent-quality CoT or final-answer data:

- **Numina-Math** (1M problems with verified solutions; AI-MO).
- **OpenMathInstruct-2** (NVIDIA; ~14M GSM8K/MATH-style with chains).
- **DeepSeek-Math RL data** / **DeepSeek-R1-distill traces** (if license allows for competition use — verify).
- **MetaMathQA**, **MathInstruct**, **OpenR1-Math** distillation set.
- **OlympiadBench**, **MATH** train split, **AIME 1983–2023** (small but very high signal).
- **Multiple-choice-specific:** **ARC-Challenge** (general MCQ format), **AGIEval** math MCQs, **GaoKao** math MCQs — directly addresses the MCQ format gap.

Curate ~50–200k examples weighted toward MCQ (since that is the bottleneck) and toward formats matching `public.jsonl` (single-blank, multi-blank with `\boxed{}` per blank, MCQ with `\boxed{Letter}`).

**Critical:** rewrite training targets so the *final answer format matches the inference prompt exactly* — single `\boxed{Letter}` for MCQ, one `\boxed{...}` per blank for multi-blank. Format mismatch in SFT data is the single biggest waste of compute.

### 2.2 Method: QLoRA on a single A100/L4

For Qwen3-4B-Thinking, QLoRA (4-bit base, LoRA on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, rank 32–64) is the obvious entry point.

- Frameworks: `unsloth` (fastest), `axolotl`, or `trl.SFTTrainer` + `peft`.
- 1–2 epochs over 50k examples on a single A100 ≈ 6–12 GPU-hours.
- Eval after each epoch on a held-out slice of `public.jsonl` to catch overfitting.

### 2.3 Avoiding regression on free-form

Thinking models can collapse if SFT data has shorter or formulaic chains. Mitigations:

- Mix in long-CoT examples (DeepSeek-R1 distill, OpenR1) so trace length distribution stays similar.
- Train with `loss_on_response_only=True` — do not waste loss mass on the prompt.
- Hold out 5–10% as eval; stop when free-form accuracy on `public.jsonl` plateaus.

**Expected lift:** +5–15pp overall depending on data quality and overfit discipline. Larger lift on MCQ than free-form since the model already handles free-form reasonably well.

### 2.4 Format-only SFT (lightweight alternative)

If full SFT is too expensive, do a **tiny format-correction SFT**: take the baseline's own outputs that are *almost* right (correct reasoning, wrong format), rewrite final lines to canonical format, train on ~5–10k of those. Cheap, narrow, and directly attacks the MCQ-extractability problem from §1.1 even when guided decoding isn't available.

**Expected lift:** +3–6pp MCQ at low cost.

---

## Tier 3 — Reinforcement learning (highest ceiling, highest cost)

Outcome-reward RL is the canonical step beyond SFT for math reasoning, and competition rules explicitly allow it. Use only after SFT is plateaued — RL on top of a weak SFT base wastes compute.

### 3.1 GRPO with the existing judger as the reward

`judger.auto_judge` already gives a binary correctness signal. That is exactly what GRPO needs: per-sample reward ∈ {0, 1}.

- Sample G=8 completions per prompt, normalize advantages within group, update LoRA adapters.
- Reward: `judger` correctness. Add a small format-bonus (`+0.1` if a `\boxed{}` is emitted) to teach the model to commit before running out of budget.
- Penalty: `-0.1` for hitting `max_tokens` without a boxed answer.

Frameworks: `trl.GRPOTrainer` (HF), `verl` (ByteDance, faster on long traces), or `OpenRLHF`. `verl` is the right choice if traces are long.

**Cost:** GRPO on 4B with G=8 and 5–10k unique problems is a 1–3 day single-GPU job. With multi-GPU it scales linearly.

**Expected lift:** +5–15pp overall on top of SFT. DeepSeek-R1-Zero-style results suggest large gains specifically for MCQ-style commitment behavior.

### 3.2 DPO as a lighter alternative

If GRPO infrastructure is too heavy: collect pairs `(correct, incorrect)` from baseline + SFT model on a held-out problem set, train DPO. Less powerful than GRPO but a single training run, no online sampling, much simpler.

### 3.3 Reward design pitfalls

- **Length-hacking:** if reward correlates with verbosity, model rambles. Add a length penalty if traces drift longer.
- **Format-hacking:** if format-bonus is too large, model emits boxed garbage to claim it. Keep format bonus < 0.2 of correctness reward.
- **MCQ shortcut:** model may guess letters since random ≈ 20% on 5-option / 10% on 10-option. Use a min-correctness threshold or filter prompts where the SFT model can't reach >0 correctness.

---

## Tier 4 — Targeted weaknesses (orthogonal)

Per-topic accuracy from the 8k baseline (`docs/baseline-public-analysis.md`):

| Topic | Count | Baseline acc (8k) | Note |
|-------|------:|------------------:|------|
| Number theory | 23 | **30.4%** | weakest |
| Sequences / recurrences | 75 | **34.7%** | weak |
| Geometry | 115 | **35.6%** | weak |
| Limits | 14 | 35.7% | small n |
| Probability / stats | 82 | 50.0% | mid |
| Linear algebra | 23 | 52.2% | mid |
| Derivatives | 12 | 58.3% | mid |
| Polynomials / algebra | 146 | 59.6% | solid |
| Integration | 55 | **74.5%** | standout |

Two approaches:

- **Few-shot exemplars:** detect topic via lightweight keyword matcher; route to a topic-specific 1–3-shot prompt with worked solutions in the same `\boxed{}` format. Particularly useful for number theory, sequences, and geometry where format conventions matter.
- **Topic-weighted SFT mix:** oversample number theory, sequences/recurrences, and geometry in the SFT data (e.g., 3× weight). Even modest absolute gains in dead zones compound because they start so low.

**Expected lift:** +1–3pp overall, mainly through weak-topic recovery.

---

## Tier 5 — Deployment / capacity (small lift, may matter at the margin)

### 5.1 BF16 instead of INT8

INT8 BNB compresses the 4B model further. If GPU memory permits BF16 (≈ 8 GB weights), accuracy gain is typically +1–3pp at zero training cost. Worth checking once Tier 1 is shipped.

### 5.2 Speculative decoding

If wall-clock matters for self-consistency (k=8 samples), spec decoding with a tiny draft model can roughly halve latency. Pure throughput optimization; no accuracy effect.

---

## Suggested execution order (revised for 8k baseline)

| Order | Direction | Cost | ΔAcc (measured / expected) | Status |
|-------|-----------|-----:|---------------------------:|:------|
| — | §1.2 `max_tokens` 4096 → 8192 | — | **+9.82 pp overall, +24.32 pp MCQ** | ✅ done — now the baseline |
| 1 | §1.1 Guided decoding for MCQ tail | hours | +10–20 pp MCQ expected (+4–7pp overall) | **next** |
| 2 | §1.5 Multi-blank free-form prompt | hours | +3–6pp free-form expected | open |
| 3 | §1.3 MCQ format clause *without* 1500-token cap + lower temp | hours | unknown; retesting on 8k baseline | re-scope |
| 4 | §1.2b Push `max_tokens` to 16,384 (MCQ-only) | minutes | small; find diminishing-returns knee | low priority |
| 5 | §1.4 Self-consistency k=5 | day (compute) | +3–7pp expected | open |
| 6 | §1.6 Length-aware free-form stop | hours | +1–3pp expected | open |
| 7 | §5.1 Switch INT8 → BF16 if RAM allows | hour | +1–3pp expected | open |
| 8 | §2.1–§2.3 QLoRA SFT on 50k Numina+R1-distill+MCQ mix | 1–2 days | +5–15pp expected | training tier |
| 9 | §2.4 Format-correction mini-SFT (only if §1.1 still leaves gap) | half day | +3–6pp MCQ expected | conditional |
| 10 | §4 Topic-weighted few-shot for number theory/sequences/geometry | day | +1–3pp expected | orthogonal |
| 11 | §3.1 GRPO with judger reward, format bonus | 2–4 days | +5–15pp expected | RL tier |

**Why this ordering:** §1.1 (guided decoding) is now the top priority because format emission is the dominant MCQ failure mode at 8k tokens — 170/375 MCQ items never emit `\boxed{Letter}` and score ~5%. §1.5 is second because the 16pp multi-blank gap is cheap to attack via prompt alone. §1.3 is worth retesting now that it can actually address the real failure mode, but expectations are moderate. Re-measure on full `public.jsonl` after each step; dev-slice Δ < ~3 pp is too noisy to trust.

---

## Open questions / dependencies to confirm

- License compatibility of DeepSeek-R1 distillation traces with competition rules (assumed OK; verify).
- Whether `vllm` build supports `guided_decoding` on the deployed environment (Colab L4 image — confirm).
- Whether the private test set's MCQ option count distribution matches public's (336/375 are 10-option) — affects how aggressively we constrain decoding.
- GPU-hour budget for the team — sets the ceiling on §3 (RL) feasibility.

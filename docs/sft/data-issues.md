# SFT data prep — known issues

Audit of the three prepared sources under `data/sft_sources/` and the prep code in
`notebooks/sft_data_prep.ipynb`, against the guardrails in [`docs/sft/pipeline.md`](pipeline.md).
Found 2026-05-21. Issues are ordered worst-first. None of these should reach
`data/sft_corpus.jsonl` unfixed.

| # | Source | Severity | One-line |
|---|--------|----------|----------|
| 1 | agieval_mcq | Blocker | Synthetic responses contain no reasoning |
| 2 | agieval_mcq | Blocker | Distractor pool is type-incoherent |
| 3 | agieval_mcq | Major | 345/825 questions are Chinese; no language filter |
| 4 | numina_cot | Major | MCQ problems mislabeled and trained as free-form |
| 5 | numina_cot | Minor | ~150 rows carry Chinese reasoning traces |
| 6 | math_train | Major | Traces too short — short-CoT, trace-collapse risk |
| 7 | all sources | Blocker | No `<think>` blocks — untested against the Thinking model template |

---

## 1. agieval_mcq — responses are synthetic, no reasoning

`synthesize_mcq_response()` (cell 20) returns a fixed template for **every** row:

```
I'll solve this step by step and compare the answer choices.

Working through the problem, I eliminate inconsistent options
and verify the remaining candidate against the constraints.

The correct choice is {letter}.
\boxed{{letter}}
```

All 825 rows use this. `mean_trace_chars` = 219; every row is in the `<=500`
trace bucket.

This does not merely fail the `> 2k char` trace filter (`[pipeline.md](pipeline.md)` risk #1,
§204) — it actively trains the model to emit a `\boxed{letter}` with **zero
reasoning**. That is the exact MCQ format-collapse behavior the SFT run is meant
to cure. As prepared, this source is worse than omitting it.

**Fix:** discard the synthetic responses. Generate real traces by running the
baseline model on each (question + expanded options) under `SYSTEM_PROMPT_MCQ`,
then keep only rows where the model's boxed letter matches the gold letter.

## 2. agieval_mcq — distractor pool is type-incoherent

`expand_mcq_options()` pads 4-option problems to 10 by sampling from
`distractor_pools[subset]`, which is built by flattening **every** non-correct
option across **all** problems in that subset:

```python
distractor_pools = {
    subset: [opt for item in items
                 for i, opt in enumerate(item["options"])
                 if i != item["correct_idx"]]
    for subset, items in by_subset.items()
}
```

`[pipeline.md](pipeline.md)` §62 specified "distractor sampling from **sibling problems**".
The implementation reads "sibling" as "any problem in the subset", so a
set-theory question receives options like `8`, `16`, `$S_n = 2n^2 - 8n$`. The
synthesized 10-option set is type-incoherent and trivially solvable by
elimination — it trains no real discrimination.

**Fix:** restrict the distractor pool to problems of the same type (match on
inferred topic, or on answer-value shape — all-numeric vs. all-set vs.
all-expression). Reject a row if too few same-type distractors exist rather than
padding with garbage.

## 3. agieval_mcq — Chinese questions, no language filter

345 of 825 ready rows (the `gaokao` subset, 351 rows) have questions in Chinese
mixed with LaTeX. `response` fields contain no CJK only because the responses are
synthetic (issue #1).

The prep code is aware of Chinese input — `parse_hails_query_question()` has an
explicit `问题：` / `选项：` / `答案：` branch — but Chinese was never treated as a
problem. `public.jsonl` / `private.jsonl` inputs are all English, so Chinese rows
teach MCQ format under an input distribution the test set never presents.
Qwen3-Thinking is multilingual so this will not crash, but it is mismatched
signal and the topic-distribution sanity check (§139) does not catch language
skew.

**Fix:** drop the `gaokao` subset, or English-only filter the source. Folds
naturally into the issue #1 rebuild — regenerate from English rows only.

## 4. numina_cot — MCQ problems mislabeled as free-form

`prepare_numina_row()` always calls `render_training_messages(problem, None, ...)`
— `options` is hard-coded `None`, so every row is written `task_type: "freeform"`
and rendered under `SYSTEM_PROMPT_MATH`.

NuminaMath embeds answer choices **inside the problem text**
(e.g. `... (A) 3/2 (B) ... `) with a value, not a letter, as the gold answer.
Such rows are MCQ in form but get trained as free-form. This is inconsistent with
how `public.jsonl` presents MCQ (a separate `options` list, letter answer) and
muddies the free-form bucket.

**Fix:** add a detector for inline `(A) (B) (C) ...` option blocks. Either route
matched rows through the MCQ path (parse options out, letter answer) or reject
them. Do not silently file them as free-form.

## 5. numina_cot — Chinese reasoning traces leak through

21 ready rows have Chinese in the question and ~150 have Chinese in the
`response` (the `cn_k12` subset is 2009 rows; most are already English-translated,
but a residue carries Chinese reasoning). Same input/trace-language mismatch as
issue #3, smaller scale.

**Fix:** add a CJK-character reject to `numina_qualifies()`.

## 6. math_train — traces are too short (short-CoT)

`math_train` is clean on every other axis: no CJK, real solutions,
decontaminated, correct `\boxed{}` final lines. But the traces are short:

- `mean_trace_chars` = 523
- trace buckets: `<=500`: 4706, `<=1000`: 1954 — ~89% of 7493 rows under 1000 chars

These are terse official MATH reference solutions, not long thinking traces.
`[pipeline.md](pipeline.md)` risk #1 (§204) says "filter **all** training data to traces
> 2k chars"; `run_math_prep()` applies no length filter, and the source table
(§63) also omitted one. At 30% mix weight this is a real trace-style-collapse
risk — the failure mode the plan explicitly warns about.

**Fix:** decide what `math_train` is for. If it is a trace source, length-filter
hard (this removes most of it). If it is a format/weak-topic source, accept the
short traces but cap its mix weight well below 30% and document the decision.

## 7. All sources — no `<think>` blocks; untested against the Thinking template

`render_training_messages()` places plain reasoning text directly as the
assistant turn. `Qwen/Qwen3-4B-Thinking-2507` is a thinking model — at inference
`notebooks/dev.ipynb` generates `<think>...</think>` traces. None of the prepared
training targets contain `<think>` blocks.

SFT on no-think assistant turns risks teaching the model to stop thinking
entirely, which would be catastrophic for a thinking model and could waste the
whole run. `[pipeline.md](pipeline.md)` requires matching the `apply_chat_template` path
(§128) but never addresses the `<think>` block question.

**Fix (do before any data rebuild):** verify what
`tokenizer.apply_chat_template(...)` produces for `Qwen3-4B-Thinking-2507` on a
3-message list with a plain-text assistant turn. If the template expects a
`<think>` block, every source's response construction must change — so settle
this first, before regenerating agieval or remixing.

---

## Suggested order of work

1. **Issue #7** — resolve the `<think>` template question. It changes how every
   source builds its `response`, so nothing else should be rebuilt until it is
   answered.
2. **Issues #1–#3** — rebuild `agieval_mcq`: English-only, real baseline-traced
   responses, same-type distractor pools. Keep `expand_mcq_options`' permutation
   logic; replace `synthesize_mcq_response` and the distractor pooling.
3. **Issues #4–#5** — add an inline-MCQ detector and a CJK reject to the
   NuminaMath path; re-run `run_numina_prep`.
4. **Issue #6** — decide `math_train`'s role and either length-filter it or
   down-weight it in the mixer; record the decision in `[pipeline.md](pipeline.md)`.

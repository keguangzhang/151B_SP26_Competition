# Private submission analysis — priv-002 (`precision_xml_verify`, 32k)

Post-upload review of the second leaderboard submission: **38.5% unified accuracy** on the interim private holdout (~30% of `private.jsonl` until finals). Responses in `results/submission_precision_xml_verify_32k.csv`. This run **regressed −9.5 pp vs the priv-001 32k baseline** (48.0%) despite the same variant scoring the **best-ever holdout** number (68.0%). Baseline reference: **priv-001** ([`private-submission-32k-priv-001.md`](private-submission-32k-priv-001.md)).

> **Correction (2026-05-31):** an earlier draft of this doc claimed the regression was caused by *format pollution* (rule-echo injecting boxes, ~80 corrupted MCQ). **That was wrong** — it came from flawed QA heuristics (a too-narrow `[A-H]` letter regex that mis-flagged 10-option `I`/`J` answers, and comma-token counting that mis-flagged tuple/interval/list answers). Verified against `data/private.jsonl` metadata, **format compliance is ~100% and is NOT the cause.** The regression is answer-content/reasoning quality. This doc reflects the corrected finding.

---

## Setup

| Setting | Value |
|--------|--------|
| Run ID | **priv-002** |
| Model | `Qwen/Qwen3-4B-Thinking-2507` (bf16, vLLM) |
| Max generation tokens | **32,768** (same as priv-001) |
| Prompting | **`precision_xml_verify`** — XML-scaffolded system prompt (`<instructions>` role+verify+concise, `<response_format>` declarative `<rule>` list), per-type XML user body. Bundles the `precision_v2` decimal clause (rejected on holdout as D013), a verify clause (`verify_prompt`, rejected as dev-013), and grader box-hygiene rules |
| Private eval | `data/private.jsonl` — **943** rows (300 MCQ, 643 free-form) |
| Submission artifact | `results/submission_precision_xml_verify_32k.csv` |
| Leaderboard score | **38.5%** unified accuracy (interim subset; course reports ~30% until finals) |
| Holdout validation | `data/dev_results_precision_xml_verify_16k.jsonl` (16k, n=225) — **68.0%** |

---

## Headline comparison

| Metric | priv-002 (private LB) | priv-001 (private LB) | Δ |
|--------|----------------------:|----------------------:|--:|
| **Overall** | **38.5%** | **48.0%** | **−9.5 pp** |

| Anchor | Eval | Overall | Note |
|--------|------|--------:|------|
| priv-002 holdout (`precision_xml_verify`, 16k) | holdout_20p (n=225) | **68.0%** | best-ever holdout — **misled the ship decision** |
| priv-001 holdout-equiv (`precision` v1, 16k) | holdout_20p (n=225) | 66.2% | shipped-best anchor |
| **priv-002 private LB** | private (~30%) | **38.5%** | **−29.5 pp vs its own holdout** |
| priv-001 private LB | private (~30%) | 48.0% | −18 pp vs public 65.8% |

**The holdout said this was the best variant; the leaderboard says it was the worst.** Holdout `precision_xml_verify` beat `precision` v1 by +1.78 pp (68.0% vs 66.2%), entirely on an **MCQ swing to 80.0%** (60/75). But MCQ at n=75 is pure noise: the no-verify `precision_xml` variant scored **60.0% MCQ on the identical 75 items** — a 20 pp swing from sampling alone. The headline holdout lift was a lucky MCQ draw, not a real gain.

---

## Format health — clean, NOT the cause

Verified against `private.jsonl` (per-question option count and `[ANS]` blank count), counting actual `\boxed{}` in the grader's last contiguous group ([`judger.py:431`](../../judger.py)):

| Check | priv-002 | priv-001 |
|-------|---------:|---------:|
| Box-count == expected (blanks / 1 for MCQ) | **940 / 943 (99.7%)** | high |
| MCQ extracts a single valid option letter (A–J) | **300 / 300 (100%)** | 97.7% |
| FF single-blank box count OK | ~99% | ~100% |
| FF multi-blank box count == blanks | ~99% | 93% |
| Single-answer emphasis-duplicate `\boxed{X}\boxed{X}` (grader rejects) | **0** | 0 |
| Truncated (no `</think>`, 0 boxes) | **3 (0.3%)** | 29 (3.1%) |

**The only format defects are 3 truncated responses** (ids 453, 498, 724) — and those need *regeneration*, not text editing.

### Cosmetic-only issues (do not affect grading)

- **Rule-echo (24.1%):** the model parrots the system prompt's `<rule>` / XML text into its output, including literal `\boxed{C}` example tokens. **This does not corrupt grading:** the judger takes only the *last contiguous box group*, and the echoed examples sit behind prose, outside that group. Verified: stripping all echoed scaffolding from every response changes **0 / 943** graded extractions. It is ugly but score-neutral.
- **Tuple/interval/list answers** (e.g. `\boxed{(12.28, 25.12)}`, `\boxed{e^2, -e^2}`) contain internal commas, but the model boxes them correctly (one box per blank). Not a defect.

**Conclusion: there is no format fix that recovers points.** A "reformatted" CSV would be a graded no-op at best, and risks corrupting correct multi-blank/tuple answers at worst.

---

## So what caused the −9.5 pp?

By elimination, the regression is **answer-content / reasoning quality**, not extraction. Format compliance, truncation, and box hygiene are all as good as or better than priv-001. The differences vs the priv-001 baseline path are entirely in **prompt content**:

1. **`precision_v2` decimal clause** — *"use DECIMAL form with ≥10 significant figures … when in doubt, prefer decimal."* This was already **rejected on holdout (D013)**. On private it can push answers to decimal where gold is exact (or to a precision/rounding that misses the grader's tolerance), and encourages decimal scratch values.
2. **Verify clause** (`verify_prompt`) — rejected as dev-013 (0 pp on holdout); adds length and self-revision that can flip correct answers.
3. **"Concise / rigorous" role rewrite** — untested change to reasoning behavior vs the baseline `multi_blank` prompt.

These are **hypotheses for which sub-clause dominates** — they cannot be isolated without per-item private grades. What *is* certain: the lost points are reasoning/value errors on cleanly-formatted, parseable answers.

---

## Why holdout failed to catch it

Holdout (n=225, MCQ n=75, 16k) scored this variant **best** (68.0%) while the leaderboard scored it **worst**. Two reasons:

1. **MCQ at n=75 is noise** — `precision_xml` (no verify) got 60% on the same items; ±13% churn at temp=0.6 swamps any real signal.
2. **Holdout items are public-derived (easier/shorter)**; the private mix is harder, and the decimal/verify clauses do more damage on the harder, more numeric private items. The 16k vs 32k budget difference compounds it.

Holdout (especially with the noisy MCQ slice) **cannot gate a submission-path prompt change.**

---

## Takeaways

1. **Revert the submission path to the priv-001 baseline** (`multi_blank` / `precision` v1, 32k). priv-002 is −9.5 pp on the only metric that counts.
2. **There is no formatting fix to apply** — the submission is ~100% format-clean. The lost points are reasoning/value errors caused by prompt *content*, recoverable only by changing the prompt and re-generating, not by post-processing the CSV.
3. **The `precision_v2` decimal clause and the verify clause were both already rejected on holdout (D013, dev-013).** Bundling rejected components into the submission prompt is what broke it. Do not ship clauses that failed their own A/B.
4. **Gate submission-path prompt changes on full public (`pub-*`, n=1126), not holdout.** This variant scored best on holdout and worst on the leaderboard. Validate any box-emission/precision change there first.
5. **Drop the XML `<rule>` scaffolding regardless.** The 24% echo is score-neutral but pollutes the traces in the graded CSV; a cleaner prompt avoids it.

---

## Artifacts

| Path | Role |
|------|------|
| `results/submission_precision_xml_verify_32k.csv` | Graded submission (full traces) |
| `data/private.jsonl` | Question metadata used to verify format compliance |
| `data/dev_results_precision_xml_verify_16k.jsonl` | Holdout judged results (n=225, 68.0%) |
| `data/dev_results_precision_xml_16k.jsonl` | No-verify ablation (60.0% MCQ — noise band) |
| `data/dev_results_precision_16k.jsonl` | `precision` v1 holdout anchor (66.2%) |
| [`judger.py:431`](../../judger.py) | `extract_all_boxed` last-contiguous-group rule |
| [`private-submission-32k-priv-001.md`](private-submission-32k-priv-001.md) | priv-001 baseline analysis (48%) |

---

## Registry

- Baseline run: [`priv-001`](../log/experiments.md) — 48.0%
- Decision to revert: record as a new ADR in [`decisions.md`](../log/decisions.md)

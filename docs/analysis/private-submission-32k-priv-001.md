# Private submission analysis — priv-001 (32k baseline)

Post-upload review of the first leaderboard submission: **48.0% unified accuracy** on the interim private holdout (~30% of `private.jsonl` until finals). Responses in `results/submission_32k.csv`; QA and gap decomposition in `notebooks/submission_analysis.ipynb` §11. Public reference: **pub-003** ([`baseline-public-32k.md`](baseline-public-32k.md)).

---

## Setup

| Setting | Value |
|--------|--------|
| Run ID | **priv-001** |
| Model | `Qwen/Qwen3-4B-Thinking-2507` (same stack as pub-003) |
| Max generation tokens | **32,768** |
| Prompting | Adaptive multi-blank (MCQ + single-blank baseline; 2+ `[ANS]` → multi-blank format) |
| Private eval | `data/private.jsonl` — **943** rows (300 MCQ, 643 free-form) |
| Submission artifact | `results/submission_32k.csv` |
| Leaderboard score | **48.0%** unified accuracy (interim subset; course reports ~30% until finals) |
| Analysis notebook | `notebooks/submission_analysis.ipynb` |

---

## Headline comparison

| Metric | priv-001 (private LB) | pub-003 (public 32k) | Δ |
|--------|----------------------:|---------------------:|--:|
| **Overall** | **48.0%** | **65.81%** | **−17.8 pp** |
| MCQ | — (no labels) | 81.07% (304/375) | — |
| Free-form | — | 58.19% (437/751) | — |

Private split matches public roughly (32% MCQ vs 33% public). Format compliance on private is high (see below), so the headline gap is **not** explained by missing `\boxed{}` or mass truncation.

**Naive projection:** applying pub-003 MCQ/FF rates to the private split (300 MCQ + 643 FF) yields **65.5%** — close to public overall. Observed **48.0%** is **−17.5 pp** below that naive line, indicating a large **generalization / private-hardness** effect rather than a format or split-mix artifact alone.

---

## Format health (private, no accuracy labels)

Integrity gates passed: **943/943** rows, no empty responses.

| Check | Private | pub-003 public |
|-------|--------:|---------------:|
| Truncated mid-think | **29 (3.1%)** | 11 (1.0%) |
| Misformat (think done, extract ≠ blanks) | **46 (4.9%)** | 26 (2.3%) |
| Clean (finished + format OK) | **868 (92.0%)** | 1,089 (96.7%) |
| MCQ `\boxed{Letter}` | **97.7%** | 97.9% |
| FF extracted count == blanks | **93.0%** | ~97% |
| Mean response length | **19,050 chars** | ~similar order |

Private has **~2× the truncation rate** and **~2× the misformat rate** vs public 32k, but both remain small in absolute terms (75 flagged ids total).

---

## Gap decomposition (format-risk buckets)

Buckets use the same rules as §11 in `submission_analysis.ipynb`: **truncated** (no closing think tag), **misformat** (finished but wrong extract count / no MCQ letter), **clean** (graders can parse answers). Bucket accuracies are **calibrated on pub-003** (`data/full_public_32k.jsonl`):

| Bucket | pub-003 acc | pub n | priv n | priv % |
|--------|----------:|------:|-------:|-------:|
| truncated | 18.2% | 11 | 29 | 3.1% |
| misformat | 7.7% | 26 | 46 | 4.9% |
| clean (all) | 67.7% | 1,089 | 868 | 92.0% |
| clean MCQ | 82.3% | — | 293 | — |
| clean FF | 60.2% | — | 575 | — |

**Expected private accuracy** if public bucket rates apply to private mix:

$$\text{est} = \frac{29 \cdot 0.182 + 46 \cdot 0.077 + 293 \cdot 0.823 + 575 \cdot 0.602}{943} \approx \mathbf{63.2\%}$$

| Step | Accuracy | Δ vs prior |
|------|----------|------------|
| pub-003 public | 65.81% | — |
| Bucket model (private mix × public bucket acc) | 63.24% | −2.6 pp (format-mix drag) |
| **Observed leaderboard** | **48.00%** | **−15.2 pp** (residual) |

**Interpretation:**

1. **Format-mix drag (~2.6 pp):** Slightly more truncation/misformat on private vs public explains only a small slice of the public→private gap.
2. **Residual (~15 pp):** Even after calibrating on public format buckets, private scores **far below** the model — dominated by **reasoning errors on clean, parseable traces**, private difficulty, and **leaderboard subset noise** (interim ~30% sample).
3. **Format is not the lever for +15 pp.** ~92% of private items are clean; fixing all 75 flagged ids could at best recover on the order of a few points, not 15.

---

## Weak slices (private format vs public accuracy)

Topics where public 32k was already weak; private format health adds risk signal:

| Topic | priv n | priv format_ok | priv truncated | pub-003 acc (n) |
|-------|-------:|---------------:|---------------:|----------------:|
| geometry | 92 | 85.9% | **10.9%** | 57.4% (108) |
| probability/stats | 166 | 94.6% | 2.4% | 51.7% (205) |
| sequences/recurrences | 53 | 86.8% | 3.8% | 73.2% (56) |
| polynomials/algebra | 141 | 97.2% | 0.7% | 71.8% (163) |

**Structural slices:**

| Slice | priv n | format_ok | truncated |
|-------|-------:|----------:|----------:|
| FF single-blank | 255 | **100.0%** | 0.0% |
| FF ≥6 blanks | 39 | 97.4% | 0.0% |
| Q4-long (≥435 chars) | 220 | 88.6% | **5.0%** |

**Geometry** stands out: highest truncation share on private (10.9%) and lowest public accuracy among major topics. **Q4-long** items combine lower format_ok (88.6%) with elevated truncation (5.0%) — aligned with public Q4 accuracy (~48.4% at 32k).

---

## Takeaways

1. **48% vs 65.8% public is a real generalization gap**, not a CSV or completeness failure. Submission passed all integrity gates.
2. **Do not chase format for leaderboard recovery.** MCQ boxed emission (97.7%) and FF count_ok (93.0%) are already strong; misformat/truncation explain ~2–3 pp vs public, not ~18 pp.
3. **Reasoning / topic weakness dominates.** Residual −15 pp vs bucket-calibrated expectation matches pub-003’s failure mode on public: wrong values and wrong MCQ letters on **finished** traces. Priority remains **SFT on weak topics** (geometry, probability/stats) per [sft-007](../log/runs/sft-007-openmath-weak-5k.md), not higher `max_tokens` alone.
4. **Interim leaderboard caveat:** 48% is on ~30% of private; finals may shift ± few pp. Re-run §11 after each upload with updated `LEADERBOARD_SCORE`.
5. **Regen candidates:** 75 flagged ids in `results/submission_flags.jsonl` (29 truncated + 46 misformat). Low ROI vs SFT unless re-inference is cheap.

---

## Artifacts

| Path | Role |
|------|------|
| `results/submission_32k.csv` | Graded submission (full traces) |
| `results/submission_flags.jsonl` | Flagged ids (truncated / misformat) |
| `results/submission_analysis_summary.json` | Machine-readable summary (after notebook §12) |
| `notebooks/submission_analysis.ipynb` | QA + §11 leaderboard analysis |
| `data/full_public_32k.jsonl` | pub-003 public scores for calibration |
| [`baseline-public-32k.md`](baseline-public-32k.md) | Public 32k error analysis |

---

## Registry

- Run note: [`log/runs/priv-001-submission-32k.md`](../log/runs/priv-001-submission-32k.md)
- Experiment row: [`priv-001`](../log/experiments.md) in `log/experiments.md`

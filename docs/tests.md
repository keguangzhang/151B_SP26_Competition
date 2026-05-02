# Evaluation notes — dev slice (`data/dev.jsonl`)

Small stratified holdout from `public.jsonl` for fast iteration (see `notebooks/dev.ipynb`).

| Setting | Value |
|--------|--------|
| Fraction per stratum | 10% MCQ + 10% free-form |
| Seed | 42 |
| Rows | 112 total (37 MCQ, 75 free-form) |

---

## Baseline (starter-style prompt + decoding)

Single prompt style for MCQ; `temperature=0.6`, `top_p=0.95` for all rows.

| Split | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| MCQ | 11 | 37 | **29.73%** |
| Free-form | 39 | 75 | **52.00%** |
| **Overall** | **50** | **112** | **44.64%** |

---

## After §1.3 (`docs/improvement-directions.md`)

MCQ: stronger `\boxed{}` final-line clause + ~1500-token reasoning hint; **MCQ-only** decoding `temperature=0.2`, `top_p=0.9`. Free-form unchanged (`0.6` / `0.95`).

| Split | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| MCQ | 10 | 37 | **27.03%** |
| Free-form | 41 | 75 | **54.67%** |
| **Overall** | **51** | **112** | **45.54%** |

### Δ vs baseline (percentage points)

| Split | Change |
|-------|--------|
| MCQ | −2.70 pp |
| Free-form | +2.67 pp |
| Overall | +0.90 pp |

**Takeaway:** Numbers move slightly in opposite directions on MCQ vs free-form; overall is essentially flat at this sample size — consistent with §1.3 being a small, format-focused tweak rather than a guaranteed lift on a 112-row dev set.

---

## After `MAX_TOKENS = 8192` (`notebooks/dev.ipynb`)

Same prompts and decoding as the starter baseline (`temperature=0.6`, `top_p=0.95`); only change is **`max_tokens` doubled** from 4096 → 8192 so long chains can finish before truncation (see `docs/improvement-directions.md` §1.2).

| Split | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| MCQ | 20 | 37 | **54.05%** |
| Free-form | 41 | 75 | **54.67%** |
| **Overall** | **61** | **112** | **54.46%** |

### Δ vs baseline (percentage points)

| Split | Change |
|-------|--------|
| MCQ | +24.32 pp |
| Free-form | +2.67 pp |
| Overall | +9.82 pp |

**Takeaway:** On this dev slice, doubling the generation cap yields a large MCQ gain (consistent with many wrong MCQ runs hitting the 4k cap in the public analysis). Free-form moves modestly; confirm on full `public.jsonl` before treating MCQ lift as stable.

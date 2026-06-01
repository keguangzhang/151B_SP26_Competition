# Numina clean corpus audit

**Date:** 2026-05-21  
**Artifacts:** `data/sft_sources/numina_cot_clean_{ready,stats,rejects}.jsonl`  
**Prep notebook:** `notebooks/sft_data_prep.ipynb` §5.1 (build), §5.2 (re-audit)  
**Pipeline step:** Step 2 complete; Step 4 spot-check partially done (20 manual + full §5.2 audit)

Re-run §5.2 after any filter change or rebuild; this file records the audit from the notebook run with `execution_count: 11`.

---

## Verdict

**Proceed to Step 5 (corpus shuffle / downsample)** with two conscious choices:

1. **`NUMINA_MAX_READY = 25_000` under-sampled tokenization** — only 23,089 of ~99.8k qualifying rows were tokenized. Accept 23k for a fast first SFT, or re-run §5.1 with `NUMINA_MAX_READY = None` and downsample to 12k–25k at Step 5 (~92k projected ready).
2. **416 letter-final rows (1.8%)** — likely MCQ leaks not caught by `(A)(B)(C)` inline detector. Optional filter: reject free-form rows whose final line matches `MCQ_FINAL_RE`.

**Step 4 hard-failures:** §5.2 reports **10** rows with `trace_chars < 500` (pipeline gate: rebuild if > 2). Structural checks (wrapper, CJK, inline MCQ, overlap, token cap) are **0**. Recommend dropping the 10 ultra-short rows at Step 5 or adding a post-wrap `trace_chars >= 2000` reject before rebuild.

---

## Funnel (full scan)

| Stage | Count | % of 859,494 scanned |
|--------|------:|---------------------:|
| Scanned | 859,494 | 100% |
| Qualifying (pre-token) | 99,826 | 11.6% |
| Ready (post wrap + tokenize) | 23,089 | 2.69% |
| Ready / qualifying | — | 23.1% |

**Config:** `NUMINA_MAX_SCAN = None`, `NUMINA_MAX_READY = 25_000`, `RANDOM_SEED = 42`, `thinking_template = explicit_redacted_thinking`.

**Reservoir cap:** 99,826 qualifying rows competed for a 25,000 tokenize slot; ~92.4% of tokenized rows passed → **~92,195** estimated ready if all qualifying rows are tokenized.

### Scan-phase rejects

| Reason | Count | % of scan |
|--------|------:|----------:|
| `short_trace` | 535,762 | 62.3% |
| `dropped_source` | 160,656 | 18.7% |
| `inline_mcq` | 47,723 | 5.6% |
| `missing_boxed_hint` | 13,396 | 1.6% |
| `cjk_text` | 2,124 | 0.2% |
| `missing_boxed` | 1,620 | 0.2% |
| `bad_final_line` | 291 | 0.03% |
| `competition_overlap` | 7 | 0.001% |

**Tokenize-phase rejects** (sampled in `numina_cot_clean_rejects.jsonl`, 1,918 rows): `missing_boxed` 1,620, `bad_final_line` 291, `competition_overlap` 7.

---

## Ready corpus profile (23,089 rows)

| Metric | Value |
|--------|------:|
| `ready_sha256` | `bbdb1c7b29f0536177bf583f6e2e78f1beea3e1d4d4472e21d1d8518949fa343` |
| Mean / median `trace_chars` | 2,507 / 2,395 |
| P5 / P95 `trace_chars` | 2,037 / 3,376 |
| Min / max `trace_chars` | 148 / 9,160 |
| Mean / median `template_tokens` | 1,102 / 1,066 |
| Max `template_tokens` | 3,731 |
| `weak_topic` rows | 8,596 (37.2%) |

### Trace length buckets (from stats)

| Bucket | Rows |
|--------|-----:|
| ≤500 | 10 |
| ≤1000 | 37 |
| ≤2000 | 327 |
| ≤4000 | 22,430 |
| ≤8000 | 282 |
| >8000 | 3 |

### HF source (ready)

| Source | Rows | % |
|--------|-----:|--:|
| olympiads | 17,181 | 74.4% |
| cn_k12 | 3,084 | 13.4% |
| aops_forum | 2,571 | 11.1% |
| math | 102 | 0.4% |
| amc_aime | 66 | 0.3% |
| other | 85 | 0.4% |

### Topic tags (ready)

| Topic | Rows | % |
|-------|-----:|--:|
| olympiads | 10,828 | 46.9% |
| geometry | 5,632 | 24.4% |
| cn_k12 | 1,981 | 8.6% |
| sequence_recurrence | 1,761 | 7.6% |
| aops_forum | 1,524 | 6.6% |
| number_theory | 1,203 | 5.2% |
| other | 160 | 0.7% |

Skew vs `public.jsonl` is expected for Numina-only (light MCQ, heavy olympiad/geometry).

---

## §5.2 re-audit (structural + soft issues)

Recorded from `notebooks/sft_data_prep.ipynb` §5.2 output (`Audit source: in-memory numina_ready`).

### Structural (target: 0)

| Check | Count |
|--------|------:|
| Missing `<think>` wrapper | 0 |
| Bad final-line regex | 0 |
| CJK leak | 0 |
| Inline `(A)…` MCQ leak in question | 0 |
| Competition overlap (public+private keys) | 0 |
| `template_tokens > 7900` | 0 |
| `trace_chars > 12000` | 0 |

### Soft issues (informational)

| Issue | Count | % |
|--------|------:|--:|
| `dangling_latex_before_close` | 14,582 | 63.2% |
| `extra_boxed_in_thinking` | 3,722 | 16.1% |
| `letter_final_mcq_style` (`\boxed{A}`–`\boxed{J}`) | 416 | 1.8% |
| `post_wrap_under_min_trace` (< 2000 chars) | 370 | 1.6% |

### Step 4 gate

- **Hard-failure heuristic:** 10 (all `trace_chars < 500` per stats bucket)
- **Pipeline rule:** rebuild if > 2 → **investigate ultra-short rows**, not a full filter rebuild

### Sample rows (from notebook)

**Letter-final (MCQ-style):**

- `cn_k12:310` → `\boxed{B}` — arithmetic sequence / MCQ stem
- `olympiads:404683` → `\boxed{C}` — sugar-water mixture problem
- `cn_k12:2104` → `\boxed{D}` — pyramid distance MCQ

**Shortest post-wrap (< 2000 chars):**

| chars | tokens | source_id | final |
|------:|-------:|-----------|-------|
| 148 | 329 | aops_forum:138399 | `\boxed{671}` |
| 165 | 208 | aops_forum:383471 | `\boxed{730}` |
| 181 | 236 | aops_forum:116796 | `\boxed{k = 8}` |
| 209 | 205 | aops_forum:283970 | `\boxed{6}` |
| 210 | 284 | cn_k12:807011 | `\boxed{x+y+1=0}` |

**Extra `\boxed` in thinking:**

- `cn_k12:332553` — 4 boxes in thinking, final `\boxed{2800 \text{ yuan}}`
- `olympiads:751075` — 1 box in thinking, final `\boxed{\text{Any multiple of } 4}`

---

## Comparison to legacy `numina_cot`

| | `numina_cot` (old) | `numina_cot_clean` |
|--|-------------------:|-------------------:|
| Ready rows | 23,104 | 23,089 |
| Mean trace chars | 2,484 | 2,507 |
| Thinking wrapper | No | 100% |
| CJK / inline MCQ filters | No | Yes |
| Private decontam | No | Yes |

Row count nearly unchanged because both runs hit the same **25k reservoir cap**; clean run adds schema and safety filters, not volume.

---

## Excluded sources (not in run 1)

| Artifact | Rows | Mean trace | Notes |
|----------|-----:|-----------:|-------|
| `math_train_ready.jsonl` | 7,493 | 523 | Too short; no thinking wrapper — [D004](../log/decisions.md#d004) |
| `agieval_mcq_ready.jsonl` | 825 | 219 | Synthetic short CoT — excluded |

---

## Step 5 complete (2026-05-22)

Built via `scripts/build_sft_corpus.py`:

| Field | Value |
|--------|------:|
| `final_row_count` | 15,000 |
| Filter drops | 10 `trace_chars_below_min`, 416 `letter_final_mcq_style` |
| `weak_topic_fraction` | 0.50 (3× weight at mix) |
| `corpus_sha256` | `4c76557290c9bb836e320fe0839ae9709ec80cec03f2de818089d28f22e87179` |
| Artifacts | `data/sft_corpus.jsonl`, `data/sft_corpus_manifest.json` |

## Recommended next steps

1. **Training:** Smoke QLoRA then full run per `pipeline.md` (`notebooks/sft_train.ipynb` when added).
2. **Optional later:** Re-run §5.1 with `NUMINA_MAX_READY = None` for ~92k ready pool; rebuild corpus at 18k–25k if first SFT shows signal.

---

## Related

- [pipeline.md](pipeline.md) — Steps 2–5
- [data-spec.md](data-spec.md) — Training contract
- [data-issues.md](data-issues.md) — Defects that motivated clean rebuild
- [decisions.md](../log/decisions.md#d004) — Numina-only first run
- [decisions.md](../log/decisions.md#d005) — Explicit thinking wrapper

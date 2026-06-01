# dev-014-precision — item-level error analysis (holdout_20p)

**Date:** 2026-05-31
**Subject run:** [dev-014-precision](../log/runs/dev-014-precision-holdout-20p.md) — 149/225 correct (66.22%)
**Baseline used for diff:** `data/full_public_16k.jsonl` restricted to the 225 holdout_20p ids (143/225 = 63.56%; `sft-eval-001` anchor on Drive showed 145, ~1 pp seed noise)
**Drives:** [`precision_v2`](#fix-summary--prompt-v2) prompt variant in `notebooks/dev.ipynb`

The Drive-side `sft-eval-001` jsonl was not synced at log time, so the official anchor cell in [dev-014](../log/runs/dev-014-precision-holdout-20p.md) item-diff stayed empty. `full_public_16k.jsonl` covers all 225 holdout ids with the same base model + `multi_blank` prompt, and is sufficient for failure-mode diagnosis.

## Item-level diff vs `full_public_16k` baseline (same 225 ids)

| | Overall | MCQ | Multi-blank FF | Single-blank FF |
|---|--------:|----:|---------------:|----------------:|
| baseline | 143/225 (63.56%) | — | — | — |
| dev-014-precision | 149/225 (66.22%) | — | — | — |
| **Δ** | **+6 items** | | | |
| fixed (b✗ → p✓) | **16** | 5 | 6 | 5 |
| broken (b✓ → p✗) | **10** | 5 | 4 | 1 |
| both wrong | 66 | 13 | 33 | 20 |

Net +6 with **26 flipped** — the precision rule is doing real work (especially single-blank, +4 net), but multi-blank and MCQ each surrendered items, so the rule isn't free.

## Failure-mode taxonomy of the 76 wrong items

Extraction follows [`judger.py`](../../judger.py) `extract_all_boxed` exactly (last contiguous `\boxed{}` group; gap regex `[\s,$.;:\-&\\]*`). Reproduces in `python3 -c "from judger import Judger; ..."` against `data/dev_results_precision_16k.jsonl`.

| Bucket | N | Prompt-fixable? | Targeted by `precision_v2` |
|--------|--:|-----------------|----------------------------|
| Truncation — no final `\boxed{}` in response | 14 | Partial | Budget-hint clause (~3–5 items rescued) |
| Multi-blank: box-count mismatch in grader's last contiguous group | 7 | **Yes** | Final-block + separator + multi-select clauses |
| MCQ wrong letter (10-option A–J, real reasoning miss) | 10 | No | — |
| FF count-OK, numeric far (computation error) | 11 | No | — |
| FF count-OK, symbolic form rejected by grader | 9 | **Yes** | Domain-conditional decimal-vs-exact clause |
| FF count-OK, numeric close (under-precision, e.g. `2.04` vs `2.03972`) | 5 | **Yes** | "≥10 sig figs, no rounding" tightening |
| FF count-OK, over-rounded / `100.000`-padded integers | 4 | **Yes** | Integer-format clause |
| FF "other" — wrong slot order, wrong type in slot (letter vs value), etc. | 16 | Partial | Slot-order + letter-only clauses |

**Total: 76.** Prompt-addressable upper bound: ~30 items, of which `precision_v2` clauses target 12–22 with conservative estimate **+8 to +15** on holdout_20p (66.22% → ~69.8–72.9%).

## Failure modes with citations

### 1. Last-block contiguity violations (4–6 items)

The grader keeps only the **last contiguous run** of `\boxed{}`. Two boxes are contiguous iff the text between them matches `r'^[\s,\$\.\;\:\-\&\\]*$'` ([`judger.py:465`](../../judger.py)). When that contiguity breaks, boxes upstream of the break are dropped → count mismatch → instant fail.

- **`\quad` between boxes breaks contiguity** — `id=547` final block was `\boxed{1.6}, \quad \boxed{4}, \quad \boxed{13.3}, \quad \boxed{A}` → grader extracted only `A` (`q`/`u`/`a`/`d` aren't in the gap character class). Gold expected 4 slots → mismatch → wrong.
- **Boxed bullets above the final block extend contiguity backwards** — `id=44` (ans_count=2) ends with `- The value of $\sin(\alpha)$ is $\boxed{-\dfrac{3\sqrt{10}}{10}}$\n\n$$\n\boxed{(-1,-3)}, \boxed{-\dfrac{3\sqrt{10}}{10}}\n$$`. The bullet's box plus the display block's two boxes form one contiguous group → 3 boxes for 2 slots → wrong. Same pattern at `id=250` (5 vs 4).
- **Multi-select MCQ formatted as `A,B`** — `id=545` gold `['…','…','AB']`, model emitted `\boxed{14.026}, \boxed{19.373}, \boxed{A,B}`. Grader's `split_by_comma` exploded `A,B` into two slots → 4 != 3 → wrong.

### 2. Over-rounding / under-precision (4–9 items)

The "≥8 sig figs" rule fires on big intermediates but not short tidy decimals or integers:

- `id=806`: gold `2.03972`, model `\boxed{2.04}` — 3 sig figs.
- `id=895`: gold `25.7736…`, model `\boxed{25.774}` — 5 sig figs.
- `id=20`: 15-box answer, every value padded to `.000` (gold `100`, model `100.000`; gold `60.8`, model `60.800`) — grader's string-strict integer check rejects the trailing zeros.
- `id=775`, `id=811`: 7-digit decimals off the 12-digit gold by >1e-8 rel → outside grader tolerance.

### 3. Symbolic-form rejection on numeric problems (9 items)

The precision prompt encouraged exact form. Grader's `parse_latex` path is fragile on certain forms:

- `id=44` — model `-\dfrac{3\sqrt{10}}{10}` vs gold decimal `-0.948683298050514` — mathematically equal, symbolic parse fails.
- `id=482`, `id=495`, `id=509`, `id=754` — same pattern (closed-form expressions where the gold is a decimal evaluation).

The driver heuristic: **stats / probability / hypothesis tests / CIs / finance / physics / "estimate" / "approximate" questions almost always have decimal golds.** Algebra / geometry / trig identities / combinatorics keep symbolic golds.

### 4. Slot ordering & slot-type mismatch (≥4 items)

- `id=391` — gold order `[110, 0.25, A, A]`, model `[1/4, 110, A, A]`. Swapped values *and* used `1/4` for a `0.25` slot.
- `id=80` — single-blank, gold `A` (option letter expected), model `\boxed{3360}` (the option's content).
- `id=250` — last `[ANS]` expects letter `B`, model wrote `\boxed{Yes}`.
- `id=358` — 3 letters expected; first slot model `A` vs gold `C`.

### 5. Truncation (14 items)

Model hit 16k while still mid-think (e.g. `id=148` at 41k chars, `id=762` at 45k chars, `id=100` 8 slots never reached final block). Genuinely budget-bound for some; over-verification for others. Pure prompt can only nudge.

### 6. MCQ wrong letter (10 items)

Real reasoning failures on 10-option (A–J) MCQs. Not prompt-fixable; needs SFT or self-consistency (see [`dev-012-sc5`](../log/runs/dev-012-sc5.md)).

## Fix summary → `precision_v2` prompt

Added in [`notebooks/dev.ipynb`](../../notebooks/dev.ipynb) cell 12 alongside `precision`. Composition of `_MATH_PRECISION_V2`:

| Clause | Fixes (ids) | Estimated lift |
|--------|-------------|----------------|
| `_FINAL_BLOCK_CLAUSE` — one Final Answer line, no boxed bullets | 44, 250 (+ probable others) | +4 to +6 |
| `_SEPARATOR_CLAUSE` — plain `, ` only; no `\quad`/`\;`/`\,`/`\hspace`/`&` | 547 | +1 to +3 |
| `_MULTI_SELECT_CLAUSE` — `\boxed{AB}` not `\boxed{A,B}` | 545 | +1 to +2 |
| `_INTEGER_FORMAT_CLAUSE` — `\boxed{100}` not `\boxed{100.000}` | 20 (15 boxes) + a few | +1 to +3 |
| `_PRECISION_V2_CLAUSE` — domain-conditional decimal vs exact; ≥10 sig figs | 44, 482, 495, 509, 754, 806, 895 | +3 to +5 |
| `_SLOT_ORDER_CLAUSE` + `_LETTER_ONLY_CLAUSE` — slot order + letter for "which of" | 80, 250, 358, 391 | +2 to +4 |
| `_BUDGET_HINT_CLAUSE` — stop verifying at ~12k chars | 148, 762, 100, … | +3 to +5 |

`_MCQ_PRECISION_V2 = _MCQ_BASELINE` — MCQ wrong-letter losses are reasoning, not prompt-addressable.

**Conservative composite upside: +8 to +15 items on holdout_20p** (66.22% → ~69.8–72.9%), assuming no regression. Some clauses can backfire (longer system prompt may displace reasoning tokens; "prefer decimal" can break the few cases where gold is symbolic). Validate on holdout_20p first vs the dev-014 anchor; promote to full-public only if it holds.

## Method / reproduction

```python
import json, sys; sys.path.insert(0, '.')
from judger import Judger
J = Judger()

p  = {r['id']: r for r in (json.loads(l) for l in open('data/dev_results_precision_16k.jsonl'))}
fp = {r['id']: r for r in (json.loads(l) for l in open('data/full_public_16k.jsonl'))}
ho = {json.loads(l)['id']: json.loads(l) for l in open('data/eval/holdout_20p.jsonl')}

def grader_boxes(text):
    raw = J.extract_boxed_answer(text)
    if not raw or raw == text: return []
    return [s.strip() for s in J.split_by_comma(raw) if s.strip()]
```

Bucket walk:
1. For each wrong id, compute `grader_boxes(response)`.
2. No boxes → `truncated_no_box`.
3. MCQ + boxes → `mcq_wrong_letter`.
4. FF + `len(boxes) != ans_count(q)` → `box_count_mismatch`.
5. FF + count matches → `ff_count_ok`, then sub-bucket by element-wise close/far/symbolic/integer-pad diagnosis.

Counts above are reproducible from the local `data/dev_results_precision_16k.jsonl` (Colab Drive synced copy).

## Follow-up

- Run `PROMPT_VARIANT="precision_v2"` on `holdout_20p` at 16k; A/B against dev-014 anchor.
- If holdout shows ≥ +4 items net with no regression bucket > 2 items, promote to full-public.
- Truncation (14 items) is the biggest remaining bucket. Out of scope for prompt — consider raising `max_tokens` to 24k or coupling with budget-forcing ([dev-010](../log/runs/dev-010-bf-budget-forcing.md)).
- MCQ wrong-letter (10 items) is reasoning — track via SC ([dev-012-sc5](../log/runs/dev-012-sc5.md)) and SFT.

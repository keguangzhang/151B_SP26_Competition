"""Inject a `precision` PROMPT_VARIANT into notebooks/dev.ipynb (exact-form /
no-round system prompt, built on the multi_blank layout). Idempotent."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "dev.ipynb"
nb = json.load(open(NB))

# raw string -> cell text keeps `\\boxed` (two backslashes => runtime \boxed)
INSERT = r'''# ── Precision / exact-form prompts (grader is 1e-8 rel-tol on 6dp gold) ───────
# Analysis (scripts/measure_precision_strict.py): 67/751 public FF fail ONLY
# because the model rounds (e.g. 62.78 vs gold 62.777778). The grader accepts
# exact forms via its symbolic path regardless of decimal places, so prefer
# fractions/radicals; if a decimal is unavoidable, >=8 sig figs and never round.
# Built on the multi_blank layout (keeps judger's contiguous-\boxed group rule).
_PRECISION_CLAUSE = (
    "Report each answer in EXACT form whenever possible — fractions, radicals, or "
    "symbolic constants (e.g. \\boxed{565/9} or \\boxed{3\\sqrt{10}/10}), NOT a rounded "
    "decimal. Use a decimal only when no exact form exists, and then give at least "
    "8 significant figures and do not round."
)

_MATH_PRECISION = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    f"{_PRECISION_CLAUSE} "
    "For problems with multiple [ANS] placeholders, put each sub-answer in its own "
    "\\boxed{}, separated by commas, in the order the blanks appear "
    "(e.g. \\boxed{3}, \\boxed{7}). Do not use labels like 'Answer 1:' between boxes, "
    "and do not restate the boxed answers elsewhere. "
    "For single-answer problems, use one \\boxed{}."
)

_MCQ_PRECISION = _MCQ_BASELINE

'''

DICT_OLD = '    "verify_prompt": (_MATH_VERIFY_PROMPT, _MCQ_VERIFY_PROMPT),\n}'
DICT_NEW = ('    "verify_prompt": (_MATH_VERIFY_PROMPT, _MCQ_VERIFY_PROMPT),\n'
            '    "precision":      (_MATH_PRECISION, _MCQ_PRECISION),\n}')

HINT_OLD = '_MULTI_BLANK_USER_HINT = frozenset({"multi_blank", "verify_prompt"})'
HINT_NEW = '_MULTI_BLANK_USER_HINT = frozenset({"multi_blank", "verify_prompt", "precision"})'

SELECT_MARK = '# ── Select active prompts based on PROMPT_VARIANT'

DOC_OLD = '#   "verify_prompt"  — verification forcing + multi_blank format (dev-013)\n'
DOC_NEW = (DOC_OLD +
           '#   "precision"        — exact-form / no-round + multi_blank (grader 1e-8 fix)\n')

patched_prompt = patched_doc = False
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "_PROMPTS = {" in src and "_MATH_MULTI_BLANK" in src:
        if "_MATH_PRECISION" in src:
            print("prompt cell already patched — skipping")
        else:
            assert HINT_OLD in src and DICT_OLD in src and SELECT_MARK in src, "anchors not found"
            src = src.replace(HINT_OLD, HINT_NEW)
            src = src.replace(DICT_OLD, DICT_NEW)
            src = src.replace(SELECT_MARK, INSERT + SELECT_MARK)
            compile(src, "<dev_prompt_cell>", "exec")   # catch f-string/brace errors
            cell["source"] = src.splitlines(keepends=True)
            patched_prompt = True
    if "PROMPT_VARIANT controls which system prompt set" in src and DOC_OLD in src:
        if '"precision"        — exact-form' not in src:
            cell["source"] = src.replace(DOC_OLD, DOC_NEW).splitlines(keepends=True)
            patched_doc = True

if patched_prompt or patched_doc:
    json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
    print(f"patched dev.ipynb (prompt={patched_prompt}, doc={patched_doc})")
else:
    print("no changes written")

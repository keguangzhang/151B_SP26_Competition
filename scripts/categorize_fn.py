"""Categorize free-form grading failures by ROOT CAUSE using the grader's own
extraction. Fast: no equivalence sympy, just count what the grader sees.

  - ALIGNMENT: grader-extracted box count != gold count -> auto-fail regardless
               of values. Recoverable by emitting exactly N boxes (post-process
               or prompt). The cheapest possible win.
  - VALUE    : count matches but strict judge still fails -> precision/rounding
               or symbolic-form mismatch. Needs precision prompt / canonical form.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judger import Judger
j = Judger(strict_extract=False)

items = {str(r["id"]): r for r in (json.loads(l) for l in open(ROOT/"data"/"public.jsonl") if l.strip())}
resp = {str(r["id"]): r["response"] for r in (json.loads(l) for l in open(ROOT/"data"/"full_public_32k.responses.jsonl") if l.strip())}

ff = [i for i in items if i in resp and not items[i].get("options")]
align = value = ok = noextract = 0
align_ids = []
for i in ff:
    it = items[i]; r = resp[i]
    gold = it["answer"]; gold = gold if isinstance(gold, list) else [gold]
    try:
        ep = j.extract_ans(r)
        ep = j.split_by_comma(ep) if ep else []
    except Exception:
        ep = []
    try:
        correct = bool(j.auto_judge(pred=r, gold=gold, options=[[]]*len(gold)))
    except Exception:
        correct = False
    if correct:
        ok += 1
    elif not ep:
        noextract += 1
    elif len(ep) != len(gold):
        align += 1
        if len(align_ids) < 10: align_ids.append((i, f"got {len(ep)} boxes, gold {len(gold)}"))
    else:
        value += 1

n = len(ff)
print(f"free-form public n={n}")
print(f"  graded CORRECT        : {ok}  ({ok/n*100:.1f}%)")
print(f"  ALIGNMENT fail (count): {align}  ({align/n*100:.1f}%)  <- emit exactly N boxes => recover")
print(f"  VALUE fail (prec/form): {value}  ({value/n*100:.1f}%)  <- precision prompt / canonical form")
print(f"  NO boxes extracted    : {noextract}  ({noextract/n*100:.1f}%)")
print(f"\n  sample alignment fails: {align_ids}")

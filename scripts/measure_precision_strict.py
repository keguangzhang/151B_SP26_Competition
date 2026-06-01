"""Airtight precision-recoverable count + examples.

An item is 'pure precision' ONLY if: grader extracts the right count, strict
auto_judge fails, and for EVERY position the model's number equals the gold
value rounded to the number of decimals the model actually printed (i.e. the
model output the correct value, just truncated). This excludes 'close but
genuinely different' answers.

Also reports a 1e-3 relative-tolerance count for comparison, and prints samples.
"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judger import Judger
import sympy as sp
from sympy.parsing.latex import parse_latex
j = Judger(strict_extract=False)

items = {str(r["id"]): r for r in (json.loads(l) for l in open(ROOT/"data"/"public.jsonl") if l.strip())}
resp  = {str(r["id"]): r["response"] for r in (json.loads(l) for l in open(ROOT/"data"/"full_public_32k.responses.jsonl") if l.strip())}

NUM = re.compile(r'^-?\d*\.?\d+$')
def decimals(s):
    s = s.strip()
    return len(s.split('.')[1]) if '.' in s and NUM.match(s) else 0
def to_num(s):
    try: return float(sp.N(parse_latex(str(s))))
    except Exception:
        try: return float(sp.N(sp.sympify(str(s).replace('^','**'))))
        except Exception: return None

def round_match(pred, gold):
    """True iff pred == gold rounded to pred's printed decimals (pure truncation)."""
    p, g = to_num(pred), to_num(gold)
    if p is None or g is None: return False
    d = decimals(pred.strip())
    if d == 0:  # integer-ish pred: allow round-to-int only if gold rounds to it
        return round(g) == p and abs(g - p) < 0.5
    return abs(round(g, d) - p) < 0.5 * 10**(-d) + 1e-9
def rel3(pred, gold):
    p, g = to_num(pred), to_num(gold)
    return p is not None and g is not None and abs(p-g) <= 1e-3*max(1.0,abs(g))

ff = [i for i in items if i in resp and not items[i].get("options")]
strict = pure = rel3c = 0
examples = []
for i in ff:
    it = items[i]; r = resp[i]
    gold = it["answer"]; gold = gold if isinstance(gold, list) else [gold]
    try: s = bool(j.auto_judge(pred=r, gold=gold, options=[[]]*len(gold)))
    except Exception: s = False
    strict += s
    if s: continue
    try:
        pred = [j.norm_ans_str(x) for x in j.split_by_comma(j.extract_ans(r))]
        gnorm = [j.norm_ans_str(x) for x in gold]
    except Exception:
        continue
    if len(pred) != len(gnorm): continue
    if all(round_match(p, g) for p, g in zip(pred, gnorm)):
        pure += 1
        if len(examples) < 18: examples.append((i, list(zip(pred, gnorm))))
    if all(rel3(p, g) for p, g in zip(pred, gnorm)):
        rel3c += 1

n = len(ff)
print(f"free-form public n={n}, strict correct {strict} ({strict/n*100:.1f}%)")
print(f"  PURE-PRECISION (pred == gold truncated to shown digits): {pure}  (+{pure/1126*100:.1f} pp)")
print(f"  (rel-tol 1e-3 count, looser): {rel3c}")
print(f"\n  examples (id: [(pred, gold), ...]):")
for i, pairs in examples:
    show = "; ".join(f"{p} vs {g}" for p, g in pairs[:5])
    print(f"    {i}: {show}")

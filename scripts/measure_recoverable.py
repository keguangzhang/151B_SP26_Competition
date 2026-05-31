"""Honest recoverable estimate under the grader's REAL rules (exact count +
positional match). Two clean, defensible buckets:

  A. PRECISION: grader extracts the right count, positions line up, strict fails
     but each pred[i] is numerically ~= gold[i] (rel tol 1e-2). Recoverable by a
     'do not round, >=6 sig figs' inference prompt. Order-preserving, safe.
  B. [ANS]-vs-gold count divergence: how often question [ANS] count != len(gold)
     (why blind post-processing is unsafe on unlabeled private).
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

def to_num(s):
    try: return float(sp.N(parse_latex(str(s))))
    except Exception:
        try: return float(sp.N(sp.sympify(str(s).replace('^','**'))))
        except Exception: return None

def near(a, b):
    na, nb = to_num(a), to_num(b)
    if na is None or nb is None: return False
    return abs(na - nb) <= 1e-2 * max(1.0, abs(nb))

ff = [i for i in items if i in resp and not items[i].get("options")]
strict = prec_recoverable = count_div = 0
for i in ff:
    it = items[i]; r = resp[i]
    gold = it["answer"]; gold = gold if isinstance(gold, list) else [gold]
    if it["question"].count("[ANS]") not in (0, len(gold)):
        count_div += 1
    try:
        s = bool(j.auto_judge(pred=r, gold=gold, options=[[]]*len(gold)))
    except Exception:
        s = False
    strict += s
    if s: continue
    try:
        pred = j.split_by_comma(j.extract_ans(r))
        pred = [j.norm_ans_str(x) for x in pred]
        gnorm = [j.norm_ans_str(x) for x in gold]
    except Exception:
        continue
    if len(pred) != len(gnorm):
        continue                       # count/order problem, NOT precision
    if all(near(p, g) for p, g in zip(pred, gnorm)):
        prec_recoverable += 1

n = len(ff)
print(f"free-form public n={n}")
print(f"  strict correct            : {strict} ({strict/n*100:.1f}%)")
print(f"  PRECISION-recoverable      : {prec_recoverable}  (+{prec_recoverable/1126*100:.1f} pp overall)")
print(f"     = right count+order, fails only on rounding -> fixable by precision prompt")
print(f"  [ANS]-count != gold-count : {count_div} items ({count_div/n*100:.1f}%)  <- why blind post-proc is unsafe")

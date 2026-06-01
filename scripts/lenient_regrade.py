"""How many 'wrong'/'never-solved' items are actually CORRECT but misjudged?

Re-grade the 32k full-public responses with a LENIENT equivalence checker that
accepts answers judger.py rejects on formatting:
  - numeric equality (pi/symbolic forms evaluated to float, tol)
  - sympy symbolic equality (commutative products, rearrangement)
  - tuple split across multiple boxes (id1057 failure mode)
  - order-insensitive multi-blank matching

Compares strict judger.py vs lenient on ALL public, and isolates the
false-negative rate (judger-wrong but lenient-correct). That delta = points
recoverable by FORMAT/EXTRACTION alone, zero capability change.
"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judger import Judger
import sympy as sp
from sympy.parsing.latex import parse_latex
judger = Judger(strict_extract=False)

LABELS = ROOT/"data"/"public.jsonl"
RESP = ROOT/"data"/"full_public_32k.responses.jsonl"

def load(p): return [json.loads(l) for l in open(p) if l.strip()]

def all_boxes(t):
    out, i = [], 0
    while True:
        j = t.find(r"\boxed{", i)
        if j < 0: break
        k = j + 7; depth = 1; buf = []
        while k < len(t) and depth:
            c = t[k]
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0: break
            buf.append(c); k += 1
        out.append(''.join(buf)); i = k + 1
    return out

def to_num(s):
    s = s.strip().strip('$').replace('\\%','').replace('%','')
    try: return float(sp.N(parse_latex(s)))
    except Exception:
        try: return float(sp.N(sp.sympify(s.replace('^','**'))))
        except Exception: return None

def num_eq(a, b):
    na, nb = to_num(a), to_num(b)
    if na is None or nb is None: return False
    return abs(na - nb) <= 1e-3 * max(1.0, abs(nb))

def sym_eq(a, b):
    try: return sp.simplify(parse_latex(a) - parse_latex(b)) == 0
    except Exception: return False

def norm(s):
    s = s.strip().strip('$').replace(' ', '').replace('\\,', '')
    s = s.replace('\\left','').replace('\\right','').replace('\\dfrac','\\frac').replace('\\tfrac','\\frac')
    return s.lower()

def one_match(gold, boxes):
    g = str(gold)
    for b in boxes:
        if norm(g) == norm(b): return True
        if num_eq(g, b): return True
        if sym_eq(g, b): return True
    # tuple gold vs split boxes:  (a,b) == box 'a', box 'b'
    gm = re.fullmatch(r'\(?\s*([^,]+)\s*,\s*([^,)]+)\s*\)?', g)
    if gm:
        parts = [gm.group(1), gm.group(2)]
        if all(any(num_eq(p, b) or norm(p) == norm(b) for b in boxes) for p in parts):
            return True
    return False

def lenient_ff(gold, resp):
    golds = gold if isinstance(gold, list) else [gold]
    boxes = all_boxes(resp)
    if not boxes: return False
    return all(one_match(g, boxes) for g in golds)

def extract_letter(t):
    m = re.search(r"\\boxed\{([A-Za-z])\}", t)
    if m: return m.group(1).upper()
    ms = re.findall(r"\b([A-Z])\b", t.upper()); return ms[-1] if ms else ""

def strict_score(item, resp):
    if item.get("options"):
        return extract_letter(resp) == str(item["answer"]).strip().upper()
    g = item["answer"]; g = g if isinstance(g, list) else [g]
    try: return bool(judger.auto_judge(pred=resp, gold=g, options=[[]]*len(g)))
    except Exception: return False

items = {str(r["id"]): r for r in load(LABELS)}
resp = {str(r["id"]): r["response"] for r in load(RESP)}

strict_c = lenient_c = both = fn = 0
fn_examples = []
for i, it in items.items():
    if i not in resp: continue
    r = resp[i]
    s = strict_score(it, r)
    if it.get("options"):
        l = extract_letter(r) == str(it["answer"]).strip().upper()
    else:
        l = lenient_ff(it["answer"], r)
    strict_c += s; lenient_c += l
    if s and l: both += 1
    if (not s) and l:
        fn += 1
        if len(fn_examples) < 12:
            fn_examples.append((i, it["answer"], all_boxes(r)[-3:]))

N = sum(1 for i in items if i in resp)
print(f"public N={N}")
print(f"  strict judger.py correct : {strict_c}  ({strict_c/N*100:.1f}%)")
print(f"  lenient correct          : {lenient_c}  ({lenient_c/N*100:.1f}%)")
print(f"  FALSE NEGATIVES (judger wrong, lenient right): {fn}  ({fn/N*100:.1f} pp recoverable)")
print(f"\n  sample false-negatives (gold  ->  model's last boxes):")
for i, g, b in fn_examples:
    print(f"    id={i}: gold={g}  boxes={b}")

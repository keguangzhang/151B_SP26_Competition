"""Does sft-007 bail early on HARDER problems?

Difficulty proxy = pub-002 (16k) trace length: harder problem -> longer baseline
reasoning. Bin items by that, then per bin compare:
  - accuracy: pub-002 (re-judged) vs sft-007 (stored)
  - compression: sft_len / pub_len  (is it cutting most where it can least afford?)

If sft accuracy falls off a cliff in the top difficulty bins relative to pub,
that's a private-specific risk (private is harder). If the acc gap is flat
across bins, public predicts private fine.
"""
import json, re, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judger import Judger
judger = Judger(strict_extract=False)

SLICES = {"geometry": ROOT/"data"/"eval"/"geometry_dev.jsonl",
          "prob_stats": ROOT/"data"/"eval"/"prob_stats_dev.jsonl"}
SFT = {"geometry": ROOT/"data"/"eval_geometry_0.jsonl",
       "prob_stats": ROOT/"data"/"eval_prob_stats_0.jsonl"}
PUB = ROOT/"data"/"full_public_16k.responses.jsonl"

def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def extract_letter(t):
    m=re.search(r"\\boxed\{([A-Za-z])\}",t)
    if m: return m.group(1).upper()
    ms=re.findall(r"\b([A-Z])\b",t.upper()); return ms[-1] if ms else ""
def score(item,resp):
    if item.get("options"):
        return extract_letter(resp)==str(item["answer"]).strip().upper()
    g=item["answer"]; g=g if isinstance(g,list) else [g]
    try: return bool(judger.auto_judge(pred=resp,gold=g,options=[[]]*len(g)))
    except Exception: return False
def to_bool(v): return str(v).lower()=="true" if isinstance(v,str) else bool(v)

pub_resp={str(r["id"]):r["response"] for r in load(PUB)}

# pool both slices
rows=[]
for name in SLICES:
    items={str(r["id"]):r for r in load(SLICES[name])}
    sft={str(r["id"]):r for r in load(SFT[name])}
    for i in sft:
        if i not in pub_resp or i not in items: continue
        rows.append({
            "slice":name,"id":i,
            "pub_len":len(pub_resp[i]),"sft_len":len(sft[i]["response"]),
            "pub_c":score(items[i],pub_resp[i]),
            "sft_c":to_bool(sft[i]["correct"]),
            "is_mcq":bool(items[i].get("options")),
        })

rows.sort(key=lambda r:r["pub_len"])
n=len(rows); q=n//4
bins=[rows[:q],rows[q:2*q],rows[2*q:3*q],rows[3*q:]]
labels=["Q1 easiest","Q2","Q3","Q4 HARDEST"]

print(f"pooled n={n}  (difficulty = pub-002 trace length)\n")
print(f"{'bin':<12}{'n':>4}{'pub_len':>9}{'pub_acc':>9}{'sft_acc':>9}{'Δacc':>8}{'compress':>10}")
for lab,b in zip(labels,bins):
    pa=sum(r['pub_c'] for r in b)/len(b)*100
    sa=sum(r['sft_c'] for r in b)/len(b)*100
    pl=statistics.mean(r['pub_len'] for r in b)
    comp=statistics.mean(r['sft_len']/r['pub_len'] for r in b)
    print(f"{lab:<12}{len(b):>4}{pl:>9,.0f}{pa:>8.1f}%{sa:>8.1f}%{sa-pa:>+7.1f}{comp:>9.2f}x")

# regression concentration: where do the pub✓→sft✗ live?
reg=[r for r in rows if r['pub_c'] and not r['sft_c']]
imp=[r for r in rows if r['sft_c'] and not r['pub_c']]
def quart(r):
    idx=rows.index(r); return min(idx//q,3)
from collections import Counter
print(f"\nregressions (pub✓→sft✗) n={len(reg)} by difficulty quartile: "
      f"{dict(sorted(Counter(quart(r) for r in reg).items()))}")
print(f"improvements (pub✗→sft✓) n={len(imp)} by difficulty quartile: "
      f"{dict(sorted(Counter(quart(r) for r in imp).items()))}")
print("(quartile 0=easiest .. 3=hardest)")

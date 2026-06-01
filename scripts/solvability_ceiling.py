"""Size the solvability ceiling from samples already on disk.

Pool the 3 full-public greedy samples (8k/16k/32k) per item, score each with
the project judger, and bucket every public item:
  - never  (0/3 correct)  -> hard ceiling; STaR/RL can't touch w/o better model
  - unstable(1-2/3)        -> the ONLY surface STaR/RL can recover
  - reliable(3/3)          -> already solved every time

CAVEAT: 8k/16k/32k are the same greedy decode at different length budgets =
highly CORRELATED, not independent temp samples. So 'never-in-3' OVERSTATES
truly-unsolvable; real pass@8 w/ temperature would solve some of them. This is
a conservative read + the exact argument for collecting more diverse samples.
"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judger import Judger
judger = Judger(strict_extract=False)

LABELS = ROOT/"data"/"public.jsonl"
SAMPLES = [ROOT/"data"/f"full_public_{b}.responses.jsonl" for b in ("8k","16k","32k")]

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

items={str(r["id"]):r for r in load(LABELS)}
samples=[{str(r["id"]):r["response"] for r in load(p)} for p in SAMPLES]

rows=[]
for i,item in items.items():
    n_correct=sum(score(item,s[i]) for s in samples if i in s)
    n_have=sum(1 for s in samples if i in s)
    rows.append({"id":i,"is_mcq":bool(item.get("options")),
                 "n_correct":n_correct,"n_have":n_have})

N=len(rows)
never=[r for r in rows if r["n_correct"]==0]
unstable=[r for r in rows if 0<r["n_correct"]<r["n_have"]]
reliable=[r for r in rows if r["n_correct"]==r["n_have"] and r["n_have"]>0]

def pct(x): return f"{len(x)/N*100:.1f}%"
print(f"pooled samples/item = up to 3 (8k/16k/32k, correlated)\n public items N={N}\n")
print(f"  NEVER solved (0/3) : {len(never):>4}  ({pct(never)})   <- ceiling: pass@1 max = {100-len(never)/N*100:.1f}%")
print(f"  UNSTABLE (1-2/3)   : {len(unstable):>4}  ({pct(unstable)})   <- STaR/RL addressable surface")
print(f"  RELIABLE (3/3)     : {len(reliable):>4}  ({pct(reliable)})   <- already solved")
print()
# split by mcq/ff
for tag,pred in [("MCQ",lambda r:r["is_mcq"]),("free-form",lambda r:not r["is_mcq"])]:
    sub=[r for r in rows if pred(r)]
    nv=[r for r in sub if r["n_correct"]==0]
    un=[r for r in sub if 0<r["n_correct"]<r["n_have"]]
    rl=[r for r in sub if r["n_correct"]==r["n_have"]]
    print(f"  {tag:<10} n={len(sub):>4}  never={len(nv)} ({len(nv)/len(sub)*100:.0f}%)  "
          f"unstable={len(un)} ({len(un)/len(sub)*100:.0f}%)  reliable={len(rl)} ({len(rl)/len(sub)*100:.0f}%)")

# pass@k curve: how many NEW items each added budget unlocks (is more sampling worth it?)
cum=set()
print("\n marginal items unlocked as we add budgets (proxy for pass@k slope):")
for bi,b in enumerate(("8k","16k","32k")):
    s=samples[bi]
    solved={i for i,it in items.items() if i in s and score(it,s[i])}
    new=solved-cum; cum|=solved
    print(f"  +{b:<4}: solves {len(solved):>4}, NEW unlocked {len(new):>4}, cumulative ever-solved {len(cum):>4} ({len(cum)/N*100:.1f}%)")

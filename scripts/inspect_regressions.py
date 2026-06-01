"""For the regressed ids (pub correct, sft wrong), show length + final boxes
to classify each as collapse-caused vs parse-noise vs genuine new error."""
import json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

SLICES = {"geometry": ROOT/"data"/"eval"/"geometry_dev.jsonl",
          "prob_stats": ROOT/"data"/"eval"/"prob_stats_dev.jsonl"}
SFT = {"geometry": ROOT/"data"/"eval_geometry_0.jsonl",
       "prob_stats": ROOT/"data"/"eval_prob_stats_0.jsonl"}
PUB = ROOT/"data"/"full_public_16k.responses.jsonl"
REG = {"geometry": ['330','420','565','636','653','952','1025','1057','1060'],
       "prob_stats": ['97','394','491','564','905']}

def load(p): return [json.loads(l) for l in open(p) if l.strip()]
pub = {str(r["id"]): r["response"] for r in load(PUB)}

def boxes(t):
    return re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", t)

for name in SLICES:
    items = {str(r["id"]): r for r in load(SLICES[name])}
    sft = {str(r["id"]): r for r in load(SFT[name])}
    print(f"\n{'='*70}\n{name} regressions\n{'='*70}")
    for i in REG[name]:
        s = sft[i]; sresp = s["response"]; presp = pub[i]
        ratio = len(sresp)/len(presp)
        print(f"\nid={i}  gold={items[i]['answer']}")
        print(f"  len sft={len(sresp):,} pub={len(presp):,}  ({ratio:.2f}x)")
        print(f"  sft boxes: {boxes(sresp)[-4:]}")
        print(f"  pub boxes: {boxes(presp)[-4:]}")

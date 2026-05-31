"""Re-score pub-002 (16k) with the SAME judger path as sft_eval, then diff
against sft-007 per item. Separates real regressions from judge/parse noise.

Scoring mirrors notebooks/sft_eval.ipynb cell 17:
  - MCQ: extract_letter == gold
  - free-form: judger.auto_judge(pred, gold_list, options=[[]]*len)
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judger import Judger

judger = Judger(strict_extract=False)

SLICES = {
    "geometry":  ROOT/"data"/"eval"/"geometry_dev.jsonl",
    "prob_stats":ROOT/"data"/"eval"/"prob_stats_dev.jsonl",
}
SFT = {
    "geometry":  ROOT/"data"/"eval_geometry_0.jsonl",
    "prob_stats":ROOT/"data"/"eval_prob_stats_0.jsonl",
}
PUB = ROOT/"data"/"full_public_16k.responses.jsonl"

def load(p): return [json.loads(l) for l in open(p) if l.strip()]

def extract_letter(text):
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m: return m.group(1).upper()
    ms = re.findall(r"\b([A-Z])\b", text.upper())
    return ms[-1] if ms else ""

def score(item, response):
    is_mcq = bool(item.get("options"))
    gold = item["answer"]
    if is_mcq:
        return extract_letter(response) == str(gold).strip().upper()
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return bool(judger.auto_judge(pred=response, gold=gold_list,
                                      options=[[]]*len(gold_list)))
    except Exception:
        return False

pub_resp = {str(r["id"]): r["response"] for r in load(PUB)}

for name in SLICES:
    items = {str(r["id"]): r for r in load(SLICES[name])}
    sft = {str(r["id"]): r for r in load(SFT[name])}
    ids = [i for i in sft if i in pub_resp and i in items]

    pub_c, sft_c = {}, {}
    for i in ids:
        pub_c[i] = score(items[i], pub_resp[i])
        sft_c[i] = (str(sft[i]["correct"]).lower() == "true"
                    if isinstance(sft[i]["correct"], str) else bool(sft[i]["correct"]))

    both = sum(pub_c[i] and sft_c[i] for i in ids)
    neither = sum((not pub_c[i]) and (not sft_c[i]) for i in ids)
    pub_only = [i for i in ids if pub_c[i] and not sft_c[i]]   # REGRESSED
    sft_only = [i for i in ids if sft_c[i] and not pub_c[i]]   # IMPROVED
    print(f"\n{'='*60}\n{name}  (n={len(ids)})  re-judged pub-002 vs sft-007")
    print(f"  pub-002 acc (re-judged here): {sum(pub_c.values())}/{len(ids)} = {sum(pub_c.values())/len(ids)*100:.1f}%")
    print(f"  sft-007 acc (stored):         {sum(sft_c.values())}/{len(ids)} = {sum(sft_c.values())/len(ids)*100:.1f}%")
    print(f"  both correct={both}  both wrong={neither}")
    print(f"  REGRESSED (pub✓ sft✗)={len(pub_only)}: {pub_only}")
    print(f"  IMPROVED  (pub✗ sft✓)={len(sft_only)}: {sft_only}")
    print(f"  net = {len(sft_only)-len(pub_only):+d} questions")

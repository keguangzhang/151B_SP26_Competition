"""Qualitative look at collapse candidates + boxed-answer emission on wrong free-form."""
import json, re
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
SFT = {"geometry": DATA/"eval_geometry_0.jsonl", "prob_stats": DATA/"eval_prob_stats_0.jsonl"}
PUB = DATA/"full_public_16k.responses.jsonl"

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

pub = {str(r["id"]): r["response"] for r in load(PUB)}
BOXED = re.compile(r"\\boxed\{")

# boxed-emission on WRONG free-form (collapse may mean it never produced an answer)
for name, path in SFT.items():
    rows = load(path)
    wf = [r for r in rows if not r["is_mcq"] and not r["correct"]]
    no_box = [r for r in wf if not BOXED.search(r["response"])]
    print(f"{name}: wrong free-form={len(wf)}, of which NO \\boxed emitted={len(no_box)} "
          f"(ids: {[r['id'] for r in no_box][:15]})")

# dump tail of sft vs pub for two worst candidates
SAMPLES = {"geometry": "44", "prob_stats": "824"}
for name, sid in SAMPLES.items():
    rows = {str(r["id"]): r for r in load(SFT[name])}
    r = rows[sid]
    print(f"\n{'#'*72}\n{name} id={sid}  is_mcq={r['is_mcq']}  gold={r['gold']}  correct={r['correct']}")
    print(f"  sft_len={len(r['response'])}  pub_len={len(pub[sid])}")
    print(f"\n--- SFT-007 trace TAIL (last 1200 chars) ---\n{r['response'][-1200:]}")
    print(f"\n--- PUB-002 trace TAIL (last 900 chars) ---\n{pub[sid][-900:]}")

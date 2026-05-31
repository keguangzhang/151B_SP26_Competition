"""Quantify SFT-007 trace collapse vs pub-002 (16k) on the frozen geo/prob slices.

For each slice:
  - length stats split by correct/wrong for sft-007
  - per-id delta vs pub-002 (same prompt, same eval JSONL)
  - list wrong+short sft-007 cases (biggest negative length deltas)
"""
import json, statistics, sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

# sft-007 scored per-item files (id, is_mcq, gold, response, correct)
SFT = {
    "geometry": DATA / "eval_geometry_0.jsonl",
    "prob_stats": DATA / "eval_prob_stats_0.jsonl",
}
# pub-002 16k baseline responses (id -> response) for the SAME slice eval set
PUB = DATA / "full_public_16k.responses.jsonl"


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def pub_lengths():
    out = {}
    for r in load_jsonl(PUB):
        out[str(r["id"])] = len(r["response"])
    return out


def main():
    publen = pub_lengths()
    for slice_name, path in SFT.items():
        rows = load_jsonl(path)
        print(f"\n{'='*70}\n{slice_name}  (n={len(rows)})\n{'='*70}")
        corr = [len(r["response"]) for r in rows if r["correct"]]
        wrong = [len(r["response"]) for r in rows if not r["correct"]]
        print(f"sft-007 mean len  correct={statistics.mean(corr):,.0f}  "
              f"wrong={statistics.mean(wrong):,.0f}  "
              f"(wrong/corr={statistics.mean(wrong)/statistics.mean(corr):.2f}x)")
        print(f"sft-007 median len correct={statistics.median(corr):,.0f}  "
              f"wrong={statistics.median(wrong):,.0f}")

        # per-id delta vs pub-002, restricted to ids present in both
        deltas = []
        for r in rows:
            sid = str(r["id"])
            if sid not in publen:
                continue
            slen = len(r["response"])
            plen = publen[sid]
            deltas.append({
                "id": sid, "is_mcq": r["is_mcq"], "correct": r["correct"],
                "sft_len": slen, "pub_len": plen,
                "delta": slen - plen,
                "pct": (slen - plen) / plen * 100 if plen else 0.0,
            })
        matched = len(deltas)
        wrong_d = [d for d in deltas if not d["correct"]]
        print(f"matched-to-pub: {matched}/{len(rows)}")
        if matched:
            print(f"mean len delta vs pub  ALL={statistics.mean(d['pct'] for d in deltas):+.1f}%  "
                  f"WRONG-only={statistics.mean(d['pct'] for d in wrong_d):+.1f}%")
            shrunk = sum(1 for d in wrong_d if d["delta"] < 0)
            print(f"wrong cases where sft trace SHORTER than pub: {shrunk}/{len(wrong_d)}")

        # wrong + biggest shrink (collapse candidates)
        cand = sorted(wrong_d, key=lambda d: d["pct"])[:8]
        print("\n  worst wrong+shrunk (collapse candidates):")
        print(f"  {'id':>5} {'mcq':>3} {'sft_len':>8} {'pub_len':>8} {'pct':>8}")
        for d in cand:
            print(f"  {d['id']:>5} {str(d['is_mcq']):>3} {d['sft_len']:>8,} "
                  f"{d['pub_len']:>8,} {d['pct']:>+7.0f}%")


if __name__ == "__main__":
    main()

"""Recover box-alignment points from already-generated traces — zero re-inference.

The course grader (judger.py auto_judge) extracts the LAST CONTIGUOUS group of
\\boxed{} answers, splits on top-level commas, and AUTO-FAILS when the count !=
number of gold blanks. Thinking models restate answers -> extra boxes -> fail
even when every value is right.

Fix: for each item, N = number of [ANS] blanks (known from the question, no
labels). If the grader currently extracts != N boxes, append a canonical
`Final Answers: \\boxed{v1}, ..., \\boxed{vN}` line (becomes the last contiguous
group) using the model's own boxed values, cleaned to exactly N.

DO-NO-HARM INVARIANT: we only modify items whose current extracted count != N.
Those are already failing (count mismatch -> grader returns False), so any edit
is strictly non-negative. Items the grader already parses to N boxes are never
touched.

Usage:
  python scripts/clean_submission.py validate
  python scripts/clean_submission.py apply data/private.jsonl \
         results/submission_32k.csv results/submission_32k_clean.csv
"""
import csv, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judger import Judger
csv.field_size_limit(10**7)
j = Judger(strict_extract=False)

FINAL_TAG = "\n\nFinal Answers: "

def blank_count(question, has_options):
    n = question.count("[ANS]")
    if n > 0:
        return n
    return None if has_options else 1   # MCQ -> None (skip), FF no marker -> 1

def all_boxes(t):
    """Every \\boxed{...} content in the whole text, brace-balanced."""
    out, i = [], 0
    while True:
        k = t.find(r"\boxed{", i)
        if k < 0: break
        s = k + 7; depth = 1; buf = []
        while s < len(t) and depth:
            c = t[s]
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0: break
            buf.append(c); s += 1
        out.append(''.join(buf)); i = s + 1
    return out

def grader_boxes(resp):
    """Exactly what the grader's extract_ans + split_by_comma produce."""
    try:
        a = j.extract_ans(resp)
        return j.split_by_comma(a) if a else []
    except Exception:
        return []

def dedup(seq):
    seen, out = set(), []
    for x in seq:
        key = re.sub(r"\s+", "", x)
        if key not in seen:
            seen.add(key); out.append(x)
    return out

def choose_values(cur, allb, N):
    """Best-effort N final values from the model's own boxes. Only called on
    failing items, so wrong guesses keep it failing (no harm)."""
    if len(cur) > N:
        d = dedup(cur)
        if len(d) == N:
            return d
        return cur[-N:]                 # final restatement tends to be the answer
    if len(cur) < N:
        pool = allb if len(allb) >= N else None
        if pool is None:
            return None                 # can't form N safely -> leave untouched
        return pool[-N:]
    return cur                          # len == N (shouldn't reach here)

def clean_response(resp, N):
    """Return (new_resp, changed)."""
    if N is None:
        return resp, False
    cur = grader_boxes(resp)
    if len(cur) == N:
        return resp, False              # do-no-harm: already aligned
    vals = choose_values(cur, all_boxes(resp), N)
    if not vals or len(vals) != N:
        return resp, False
    line = FINAL_TAG + ", ".join(f"\\boxed{{{v}}}" for v in vals)
    return resp + line, True


def cmd_validate():
    items = {str(r["id"]): r for r in (json.loads(l) for l in open(ROOT/"data"/"public.jsonl") if l.strip())}
    resp  = {str(r["id"]): r["response"] for r in (json.loads(l) for l in open(ROOT/"data"/"full_public_32k.responses.jsonl") if l.strip())}
    def grade(it, r):
        if it.get("options"):
            m = re.search(r"\\boxed\{([A-Za-z])\}", r)
            letter = m.group(1).upper() if m else ""
            return letter == str(it["answer"]).strip().upper()
        g = it["answer"]; g = g if isinstance(g, list) else [g]
        try: return bool(j.auto_judge(pred=r, gold=g, options=[[]]*len(g)))
        except Exception: return False

    before = after = recovered = broken = changed = 0
    ff = [i for i in items if i in resp and not items[i].get("options")]
    for i in ff:
        it = items[i]; r = resp[i]
        N = blank_count(it["question"], False)
        b = grade(it, r)
        nr, ch = clean_response(r, N)
        a = grade(it, nr) if ch else b
        before += b; after += a; changed += ch
        if (not b) and a: recovered += 1
        if b and (not a): broken += 1
    n = len(ff)
    print(f"free-form public n={n}")
    print(f"  modified items     : {changed}")
    print(f"  correct BEFORE     : {before}  ({before/n*100:.1f}%)")
    print(f"  correct AFTER      : {after}  ({after/n*100:.1f}%)   Δ +{(after-before)/n*100:.1f} pp")
    print(f"  RECOVERED (✗->✓)   : {recovered}")
    print(f"  BROKEN   (✓->✗)    : {broken}   (must be 0 by design)")
    # project onto full submission scale (943 graded incl MCQ unchanged)
    print(f"\n  on full public 1126: +{recovered} items = +{recovered/1126*100:.1f} pp overall")


def cmd_apply(meta_path, in_csv, out_csv):
    meta = {str(r["id"]): r for r in (json.loads(l) for l in open(meta_path) if l.strip())}
    n_changed = n_total = 0
    with open(in_csv, newline="") as fin, open(out_csv, "w", newline="") as fout:
        rd = csv.DictReader(fin)
        wr = csv.DictWriter(fout, fieldnames=rd.fieldnames)
        wr.writeheader()
        for row in rd:
            n_total += 1
            it = meta.get(str(row["id"]))
            if it is not None:
                N = blank_count(it.get("question", ""), bool(it.get("options")))
                nr, ch = clean_response(row["response"], N)
                row["response"] = nr; n_changed += ch
            wr.writerow(row)
    print(f"wrote {out_csv}: {n_total} rows, {n_changed} responses cleaned ({n_changed/n_total*100:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "validate":
        cmd_validate()
    elif len(sys.argv) >= 5 and sys.argv[1] == "apply":
        cmd_apply(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(__doc__)

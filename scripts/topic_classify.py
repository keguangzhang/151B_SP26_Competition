"""Weighted regex topic classifier for competition math questions."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CLASSIFIER_VERSION = "weighted_v1"

TOPIC_RULES: list[tuple[str, int, re.Pattern[str]]] = [
    ("linear algebra", 4, re.compile(r"\b(matrix|matrices|determinant|eigen(?:value|vector)s?|vector space|rank|linear transform)\b", re.I)),
    ("complex analysis", 4, re.compile(r"\b(analytic function|complex|real part|imaginary|f\(z\))\b|\\mathrm\{i\}", re.I)),
    ("integration", 4, re.compile(r"\\int|\bint_?\b|\bintegr(?:al|ate|ation|and)\b|\bantiderivative\b|\barea under\b", re.I)),
    ("derivatives", 4, re.compile(r"\bderivative|differentiat|\\frac\{d|\bd/dx\b|\brate of change\b|\btangent line\b|\bcritical point\b", re.I)),
    ("limits", 4, re.compile(r"\blimit\b|lim_|\\lim|\basymptote\b|\bapproaches\b", re.I)),
    ("probability/stats", 4, re.compile(r"\b(probability|expected value|variance|standard deviation|distribution|random variable|random sample|confidence|hypothesis|significance|regression|correlation|permutation|combination|combinatorics|entropy|deck of cards)\b", re.I)),
    ("probability/stats", 2, re.compile(r"\b(sample|population|proportion|median|mean|average|quartile|percentile)\b", re.I)),
    ("geometry", 4, re.compile(r"\b(triangle|circle|angle|polygon|perimeter|volume|surface|congruent|similar|parallelogram|trapezoid|chord|radius|diameter|rectangle|cube|sphere|cylinder|cone|prism|coordinate plane)\b", re.I)),
    ("geometry", 2, re.compile(r"\b(area|square)\b", re.I)),
    ("trigonometry", 4, re.compile(r"\\(?:sin|cos|tan|sec|csc|cot)\b|\b(sine|cosine|tangent|secant|cosecant|cotangent|radian|theta|trig(?:onometric)?)\b", re.I)),
    ("sequences/recurrences", 4, re.compile(r"\b(sequence|recurrence|series|fibonacci|sum of the first)\b", re.I)),
    ("sequences/recurrences", 2, re.compile(r"\b(arithmetic|geometric|terms?)\b", re.I)),
    ("number theory", 4, re.compile(r"\b(prime|divisib|divisor|multiple|modulo|congruence|gcd|lcm|diophantine|remainder)\b|\\lfloor|\\gcd", re.I)),
    ("number theory", 2, re.compile(r"\b(integer|floor|ceil)\b", re.I)),
    ("logs/exponents", 4, re.compile(r"\b(logarithm|log_|ln\b|exponential|exponent|half-life|decay|growth|doubl(?:e|ing))\b|\\log|\\ln", re.I)),
    ("polynomials/algebra", 3, re.compile(r"\b(polynomial|quadratic|cubic|root|zero|factor|expand|simplify|equation|system of equations|inequality|slope|intercept)\b", re.I)),
    ("polynomials/algebra", 1, re.compile(r"\b(expression|linear|solve|evaluate|formula|function)\b", re.I)),
    ("arithmetic/word problems", 3, re.compile(r"\b(fraction|decimal|percent|ratio|dollars?|price|cost|total|nearest|round|hours?|minutes?|years?|paycheck)\b", re.I)),
]


def topic_scores(question: str) -> dict[str, int]:
    scores: dict[str, int] = defaultdict(int)
    for name, weight, pat in TOPIC_RULES:
        if pat.search(question):
            scores[name] += weight
    return dict(scores)


def classify_topic(question: str) -> str:
    scores = topic_scores(question)
    if not scores:
        return "other"
    return max(scores.items(), key=lambda item: item[1])[0]


def topics_cache_path(base_path: Path | str, version: str = CLASSIFIER_VERSION) -> Path:
    base = Path(base_path)
    stem = base.with_suffix("")
    suffix = f"_{version}"
    if stem.name.endswith(suffix):
        stem = Path(str(stem)[: -len(suffix)])
    return stem.parent / f"{stem.name}{suffix}.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def topic_counts(questions: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for q in questions:
        counts[classify_topic(q)] += 1
    return dict(counts)


def aggregate_topics(
    results: list[dict[str, Any]],
    questions_by_id: dict[int, str],
) -> dict[str, dict[str, Any]]:
    topic_total: dict[str, int] = defaultdict(int)
    topic_correct: dict[str, int] = defaultdict(int)
    topic_mcq_t: dict[str, int] = defaultdict(int)
    topic_mcq_c: dict[str, int] = defaultdict(int)

    for r in results:
        qid = r["id"]
        t = classify_topic(questions_by_id[qid])
        topic_total[t] += 1
        topic_correct[t] += int(r.get("correct", False))
        if r.get("is_mcq"):
            topic_mcq_t[t] += 1
            topic_mcq_c[t] += int(r.get("correct", False))

    return {
        t: {
            "n": topic_total[t],
            "correct": topic_correct[t],
            "accuracy": round(topic_correct[t] / topic_total[t] * 100, 2),
            "mcq_n": topic_mcq_t[t],
            "mcq_correct": topic_mcq_c[t],
            "mcq_accuracy": round(topic_mcq_c[t] / topic_mcq_t[t] * 100, 2) if topic_mcq_t[t] else None,
        }
        for t in topic_total
    }


def print_topic_table(topics_agg: dict[str, dict[str, Any]]) -> None:
    print(f"{'Topic':<30} {'N':>6} {'Acc':>8} {'MCQ N':>7} {'MCQ Acc':>9}")
    print("-" * 65)
    for topic, v in sorted(topics_agg.items(), key=lambda kv: -kv[1]["n"]):
        mcq_acc = f"{v['mcq_accuracy']:.1f}%" if v["mcq_accuracy"] is not None else "—"
        print(f"{topic:<30} {v['n']:>6} {v['accuracy']:>7.1f}% {v['mcq_n']:>7} {mcq_acc:>9}")


def build_topics_agg(
    results_path: Path,
    public_path: Path,
    out_path: Path | None = None,
    *,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    out = out_path or topics_cache_path(results_path.with_name(f"{results_path.stem}_topics.json"))
    if out.is_file() and not force:
        with open(out) as f:
            return json.load(f)

    results = read_jsonl(results_path)
    public = read_jsonl(public_path)
    questions_by_id = {r["id"]: r["question"] for r in public}
    topics_agg = aggregate_topics(results, questions_by_id)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(topics_agg, f, indent=2)
        f.write("\n")
    return topics_agg


def label_jsonl(in_path: Path, out_path: Path, *, question_key: str = "question") -> None:
    rows = read_jsonl(in_path)
    for row in rows:
        row["topic"] = classify_topic(row[question_key])
    write_jsonl(out_path, rows)


def cmd_summarize(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    questions = [r[args.question_key] for r in rows]
    counts = topic_counts(questions)
    total = len(questions)
    other = counts.get("other", 0)
    print(f"Classifier: {CLASSIFIER_VERSION}")
    print(f"Rows: {total}   other: {other} ({other / total * 100:.1f}%)")
    print()
    for topic, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {topic:<28} {n:>5}  ({n / total * 100:.1f}%)")


def cmd_aggregate(args: argparse.Namespace) -> None:
    out = args.out or topics_cache_path(args.results.with_name(f"{args.results.stem}_topics.json"))
    if out.is_file() and not args.force:
        print(f"Exists: {out} (use --force to rebuild)")
        with open(out) as f:
            topics_agg = json.load(f)
    else:
        topics_agg = build_topics_agg(args.results, args.public, out, force=True)
        print(f"Wrote {out}")
    print_topic_table(topics_agg)


def cmd_label(args: argparse.Namespace) -> None:
    label_jsonl(args.input, args.out, question_key=args.question_key)
    print(f"Labeled {args.input} → {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("summarize", help="Topic distribution for a JSONL with questions")
    ps.add_argument("input", type=Path, help="Input JSONL (e.g. data/public.jsonl)")
    ps.add_argument("--question-key", default="question", help="Field containing question text")
    ps.set_defaults(func=cmd_summarize)

    pa = sub.add_parser("aggregate", help="Build topic accuracy JSON from results + public labels")
    pa.add_argument("--results", type=Path, required=True, help="Scored results JSONL")
    pa.add_argument("--public", type=Path, default=REPO / "data" / "public.jsonl")
    pa.add_argument("--out", type=Path, default=None, help="Output JSON (default: <results_stem>_topics_<version>.json)")
    pa.add_argument("--force", action="store_true")
    pa.set_defaults(func=cmd_aggregate)

    pl = sub.add_parser("label", help="Add topic field to each row in a JSONL")
    pl.add_argument("input", type=Path)
    pl.add_argument("out", type=Path)
    pl.add_argument("--question-key", default="question")
    pl.set_defaults(func=cmd_label)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

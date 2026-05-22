#!/usr/bin/env python3
"""Step 5: mix numina_cot_clean_ready → data/sft_corpus.jsonl + manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SFT_SOURCES_DIR = REPO / "data" / "sft_sources"
NUMINA_READY = SFT_SOURCES_DIR / "numina_cot_clean_ready.jsonl"
NUMINA_STATS = SFT_SOURCES_DIR / "numina_cot_clean_stats.json"
CORPUS_PATH = REPO / "data" / "sft_corpus.jsonl"
MANIFEST_PATH = REPO / "data" / "sft_corpus_manifest.json"

MCQ_FINAL_RE = re.compile(r"^\\boxed\{[A-J]\}$")
THINKING_TEMPLATE = "explicit_redacted_thinking"
DEFAULT_SEED = 42
DEFAULT_TARGET = 15_000
MIN_TRACE_CHARS = 500
WEAK_TOPIC_WEIGHT = 3.0


def _repo_root() -> Path:
    return REPO


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def last_non_empty_line(text: str) -> str:
    for line in reversed(text.rstrip().splitlines()):
        if line.strip():
            return line.strip()
    return ""


def is_letter_final_mcq(response: str) -> bool:
    return bool(MCQ_FINAL_RE.match(last_non_empty_line(response)))


def filter_ready_rows(
    rows: list[dict[str, Any]],
    *,
    min_trace_chars: int,
    drop_letter_final: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    kept: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    for row in rows:
        if row.get("trace_chars", 0) < min_trace_chars:
            drops["trace_chars_below_min"] += 1
            continue
        if drop_letter_final and is_letter_final_mcq(row["response"]):
            drops["letter_final_mcq_style"] += 1
            continue
        kept.append(row)
    return kept, drops


def weak_topic_weight(row: dict[str, Any]) -> float:
    return WEAK_TOPIC_WEIGHT if row.get("weak_topic") else 1.0


def weighted_sample_no_replace(
    rows: list[dict[str, Any]],
    k: int,
    seed: int,
    weight_fn,
) -> list[dict[str, Any]]:
    """Gumbel-max trick: O(n log n) weighted sample without replacement."""
    rng = random.Random(seed)
    if k >= len(rows):
        out = rows.copy()
        rng.shuffle(out)
        return out
    scored: list[tuple[float, int]] = []
    for i, row in enumerate(rows):
        w = max(weight_fn(row), 1e-9)
        key = math.log(rng.random()) / w
        scored.append((key, i))
    scored.sort(reverse=True)
    picked = [rows[i] for _, i in scored[:k]]
    rng.shuffle(picked)
    return picked


def dist_stats(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    values_sorted = sorted(values)
    p95_idx = max(0, int(0.95 * len(values_sorted)) - 1)
    return {
        "mean": round(statistics.mean(values_sorted), 1),
        "median": round(statistics.median(values_sorted), 1),
        "p95": float(values_sorted[p95_idx]),
    }


def build_manifest(
    *,
    source_path: Path,
    source_sha256: str,
    input_rows: int,
    after_filter_rows: int,
    filter_drops: Counter[str],
    corpus_rows: list[dict[str, Any]],
    target_n: int,
    seed: int,
    min_trace_chars: int,
    drop_letter_final: bool,
    weak_topic_weight: float,
    numina_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    trace_chars = [r["trace_chars"] for r in corpus_rows]
    template_tokens = [r["template_tokens"] for r in corpus_rows]
    topic_counts = Counter(r.get("topic", "unknown") for r in corpus_rows)
    weak_in_corpus = sum(1 for r in corpus_rows if r.get("weak_topic"))

    return {
        "thinking_template": THINKING_TEMPLATE,
        "sources": [
            {
                "path": str(source_path.relative_to(REPO)),
                "sha256": source_sha256,
                "source": "numina_cot_clean",
            }
        ],
        "numina_stats_ready_sha256": (numina_stats or {}).get("ready_sha256"),
        "input_ready_rows": input_rows,
        "after_filter_rows": after_filter_rows,
        "filter_drops": dict(filter_drops),
        "min_trace_chars": min_trace_chars,
        "drop_letter_final_mcq": drop_letter_final,
        "weak_topic_weight": weak_topic_weight,
        "sample_seed": seed,
        "target_n": target_n,
        "final_row_count": len(corpus_rows),
        "weak_topic_rows": weak_in_corpus,
        "weak_topic_fraction": round(weak_in_corpus / max(len(corpus_rows), 1), 4),
        "trace_chars": dist_stats(trace_chars),
        "template_tokens": dist_stats(template_tokens),
        "topic_counts": dict(topic_counts.most_common()),
        "corpus_path": str(CORPUS_PATH.relative_to(REPO)),
        "corpus_sha256": file_sha256(CORPUS_PATH) if CORPUS_PATH.is_file() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-trace-chars", type=int, default=MIN_TRACE_CHARS)
    parser.add_argument(
        "--keep-letter-final",
        action="store_true",
        help="Keep free-form rows with \\boxed{A-J} finals (default: drop)",
    )
    parser.add_argument(
        "--ready-path",
        type=Path,
        default=NUMINA_READY,
    )
    args = parser.parse_args()

    if not args.ready_path.is_file():
        raise SystemExit(f"Missing ready file: {args.ready_path}")

    ready_rows = read_jsonl(args.ready_path)
    source_sha = file_sha256(args.ready_path)
    numina_stats = None
    if NUMINA_STATS.is_file():
        numina_stats = json.loads(NUMINA_STATS.read_text())

    filtered, filter_drops = filter_ready_rows(
        ready_rows,
        min_trace_chars=args.min_trace_chars,
        drop_letter_final=not args.keep_letter_final,
    )

    corpus = weighted_sample_no_replace(
        filtered,
        min(args.target, len(filtered)),
        args.seed,
        weak_topic_weight,
    )

    write_jsonl(CORPUS_PATH, corpus)

    manifest = build_manifest(
        source_path=args.ready_path,
        source_sha256=source_sha,
        input_rows=len(ready_rows),
        after_filter_rows=len(filtered),
        filter_drops=filter_drops,
        corpus_rows=corpus,
        target_n=args.target,
        seed=args.seed,
        min_trace_chars=args.min_trace_chars,
        drop_letter_final=not args.keep_letter_final,
        weak_topic_weight=WEAK_TOPIC_WEIGHT,
        numina_stats=numina_stats,
    )
    manifest["corpus_sha256"] = file_sha256(CORPUS_PATH)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {CORPUS_PATH} ({len(corpus)} rows)")
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

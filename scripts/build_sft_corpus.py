#!/usr/bin/env python3
"""SFT corpus builders: base mix, targeted supplements, v2 merge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.sft_prompt import (  # noqa: E402
    THINKING_CLOSE,
    THINKING_OPEN,
    build_prompt_multi_blank,
    ensure_ans_placeholders,
    extract_all_boxed,
    final_section,
    is_figure_dependent,
    is_geometry_flavored,
    multi_boxed_final_line,
    n_ans_placeholders,
    split_thinking_response,
    wrap_thinking_response,
)

SFT_SOURCES_DIR = REPO / "data" / "sft_sources"
NUMINA_READY = SFT_SOURCES_DIR / "numina_cot_clean_ready.jsonl"
NUMINA_STATS = SFT_SOURCES_DIR / "numina_cot_clean_stats.json"
CORPUS_PATH = REPO / "data" / "sft_corpus.jsonl"
MANIFEST_PATH = REPO / "data" / "sft_corpus_manifest.json"
LONG_TRACE_PATH = SFT_SOURCES_DIR / "numina_long_trace.jsonl"
MULTI_BLANK_PATH = SFT_SOURCES_DIR / "numina_multi_blank_synth.jsonl"
CORPUS_V2_PATH = REPO / "data" / "sft_corpus_v2.jsonl"
MANIFEST_V2_PATH = REPO / "data" / "sft_corpus_v2_manifest.json"
PUBLIC_PATH = REPO / "data" / "public.jsonl"
PRIVATE_PATH = REPO / "data" / "private.jsonl"

MCQ_FINAL_RE = re.compile(r"^\\boxed\{[A-J]\}$")
THINKING_TEMPLATE = "explicit_redacted_thinking"
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_SEED = 42
DEFAULT_TARGET = 15_000
SUPPLEMENT_TARGET = 1_500
MIN_TRACE_CHARS = 500
WEAK_TOPIC_WEIGHT = 3.0
MAX_TEMPLATE_TOKENS = 7900
MAX_TRACE_CHARS = 12_000
LONG_TRACE_MIN = 6000
LONG_TRACE_MAX = 12_000
GEO_TARGET_FRAC = 0.30
MAX_COMPOSED_BLANKS = 4


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


def normalize_question_for_overlap(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("$", "")
    s = re.sub(r"\\(?:mathrm|mathbf|text|textbf)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"[^a-z0-9\\=+\-*/^_{}().,\[\]]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_competition_keys() -> set[str]:
    keys: set[str] = set()
    for path in (PUBLIC_PATH, PRIVATE_PATH):
        if not path.is_file():
            continue
        with open(path) as f:
            for line in f:
                keys.add(normalize_question_for_overlap(json.loads(line)["question"]))
    return keys


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


def weighted_sample_no_replace(
    rows: list[dict[str, Any]],
    k: int,
    seed: int,
    weight_fn: Callable[[dict[str, Any]], float],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if k >= len(rows):
        out = rows.copy()
        rng.shuffle(out)
        return out
    scored: list[tuple[float, int]] = []
    for i, row in enumerate(rows):
        w = max(weight_fn(row), 1e-9)
        scored.append((math.log(rng.random()) / w, i))
    scored.sort(reverse=True)
    picked = [rows[i] for _, i in scored[:k]]
    rng.shuffle(picked)
    return picked


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def count_template_tokens(tokenizer, question: str, options: Optional[list], response: str) -> int:
    system, user = build_prompt_multi_blank(question, options)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def corpus_source_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {r["source_id"] for r in read_jsonl(path)}


def tag_supplement_row(row: dict[str, Any], supplement: str, mode: str) -> dict[str, Any]:
    out = dict(row)
    out["source"] = supplement
    out["supplement_mode"] = mode
    out["thinking_template"] = THINKING_TEMPLATE
    return out


# ── Base corpus (Step 5 original) ───────────────────────────────────────────


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
    corpus_path: Path,
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
        "corpus_path": str(corpus_path.relative_to(REPO)),
        "corpus_sha256": None,
    }


def cmd_base(args: argparse.Namespace) -> None:
    ready_path = args.ready_path
    if not ready_path.is_file():
        raise SystemExit(f"Missing ready file: {ready_path}")

    ready_rows = read_jsonl(ready_path)
    source_sha = file_sha256(ready_path)
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
        source_path=ready_path,
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
        corpus_path=CORPUS_PATH,
    )
    manifest["corpus_sha256"] = file_sha256(CORPUS_PATH)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {CORPUS_PATH} ({len(corpus)} rows)")
    print(f"Wrote {MANIFEST_PATH}")


# ── Long-trace supplement (5a) ──────────────────────────────────────────────


def build_long_trace_supplement(
    ready: list[dict[str, Any]],
    *,
    exclude_ids: set[str],
    target: int,
    seed: int,
    competition_keys: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    for row in ready:
        if row["source_id"] in exclude_ids:
            drops["in_base_corpus"] += 1
            continue
        text = row["question"] + row.get("response", "")
        if is_figure_dependent(text):
            drops["figure_dependent"] += 1
            continue
        if normalize_question_for_overlap(row["question"]) in competition_keys:
            drops["competition_overlap"] += 1
            continue
        if row.get("trace_chars", 0) > LONG_TRACE_MAX:
            drops["trace_chars_above_max"] += 1
            continue
        pool.append(row)

    tier_strict = [r for r in pool if r["trace_chars"] >= LONG_TRACE_MIN]
    pool_sorted = sorted(pool, key=lambda r: r["trace_chars"], reverse=True)
    geo_sorted = [r for r in pool_sorted if is_geometry_flavored(r["question"])]
    non_geo_sorted = [r for r in pool_sorted if not is_geometry_flavored(r["question"])]
    n_geo = min(len(geo_sorted), max(1, int(target * GEO_TARGET_FRAC)))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in geo_sorted:
        if row["source_id"] in seen:
            continue
        selected.append(row)
        seen.add(row["source_id"])
        if len(selected) >= n_geo:
            break
    for row in non_geo_sorted:
        if len(selected) >= target:
            break
        if row["source_id"] in seen:
            continue
        selected.append(row)
        seen.add(row["source_id"])
    for row in geo_sorted:
        if len(selected) >= target:
            break
        if row["source_id"] in seen:
            continue
        selected.append(row)
        seen.add(row["source_id"])
    rng = random.Random(seed)
    rng.shuffle(selected)
    out = [
        tag_supplement_row(r, "numina_long_trace", "long_trace")
        for r in selected
    ]

    geo_hits = sum(1 for r in out if is_geometry_flavored(r["question"]))
    strict_n = sum(1 for r in out if r["trace_chars"] >= LONG_TRACE_MIN)
    meta = {
        "pool_rows": len(pool),
        "tier_strict_rows": len(tier_strict),
        "selected_rows": len(out),
        "strict_threshold_rows": strict_n,
        "geometry_keyword_rows": geo_hits,
        "geometry_keyword_fraction": round(geo_hits / max(len(out), 1), 4),
        "filter_drops": dict(drops),
        "note": (
            f"Only {len(tier_strict)} ready rows have trace_chars >= {LONG_TRACE_MIN}; "
            "remaining slots filled from highest trace_chars below threshold."
            if len(tier_strict) < target
            else None
        ),
    }
    return out, meta


# ── Multi-blank supplement (5b) ───────────────────────────────────────────────


def try_native_multi_blank(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    tail = final_section(row["response"])
    answers = extract_all_boxed(tail)
    if len(answers) < 2 or len(answers) > MAX_COMPOSED_BLANKS:
        return None
    question = ensure_ans_placeholders(row["question"], len(answers))
    reasoning, _ = split_thinking_response(row["response"])
    final_line = multi_boxed_final_line(answers)
    response = wrap_thinking_response(reasoning, final_line)
    out = dict(row)
    out["question"] = question
    out["response"] = response
    out["answer"] = ", ".join(answers)
    out["blank_count"] = len(answers)
    out["trace_chars"] = len(response)
    return out


def try_compose_multi_blank(
    parts: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if len(parts) < 2 or len(parts) > MAX_COMPOSED_BLANKS:
        return None
    total_trace = sum(p.get("trace_chars", 0) for p in parts)
    if total_trace > MAX_TRACE_CHARS:
        return None

    chunks: list[str] = []
    reasonings: list[str] = []
    answers: list[str] = []
    for i, p in enumerate(parts, 1):
        label = chr(96 + i)  # a, b, c, ...
        q = p["question"].strip()
        if "[ANS]" not in q:
            q = f"{q}\n\nAnswer: [ANS]"
        chunks.append(f"({label}) {q}")
        reasoning, final = split_thinking_response(p["response"])
        boxed = extract_all_boxed(final)
        if len(boxed) != 1:
            return None
        reasonings.append(f"### Part ({label})\n{reasoning}".strip())
        answers.append(boxed[0])

    question = "\n\n".join(chunks)
    question = ensure_ans_placeholders(question, len(answers))
    combined_reasoning = "\n\n".join(reasonings)
    final_line = multi_boxed_final_line(answers)
    response = wrap_thinking_response(combined_reasoning, final_line)

    out = {
        "source": "numina_multi_blank_synth",
        "source_id": "+".join(p["source_id"] for p in parts),
        "task_type": "freeform",
        "topic": parts[0].get("topic", "unknown"),
        "hf_source": parts[0].get("hf_source", "unknown"),
        "weak_topic": any(p.get("weak_topic") for p in parts),
        "question": question,
        "options": None,
        "answer": ", ".join(answers),
        "response": response,
        "trace_chars": len(response),
        "blank_count": len(answers),
        "thinking_template": THINKING_TEMPLATE,
        "supplement_mode": "composed",
        "composed_from": [p["source_id"] for p in parts],
    }
    return out


def build_multi_blank_supplement(
    ready: list[dict[str, Any]],
    *,
    exclude_ids: set[str],
    target: int,
    seed: int,
    competition_keys: set[str],
    tokenizer,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    pool: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()

    for row in ready:
        if row["source_id"] in exclude_ids:
            continue
        text = row["question"] + row.get("response", "")
        if is_figure_dependent(text):
            drops["figure_dependent"] += 1
            continue
        if normalize_question_for_overlap(row["question"]) in competition_keys:
            drops["competition_overlap"] += 1
            continue
        if row.get("trace_chars", 0) < 2000:
            drops["short_trace"] += 1
            continue
        pool.append(row)

    native_candidates: list[dict[str, Any]] = []
    for row in pool:
        built = try_native_multi_blank(row)
        if built is not None:
            native_candidates.append(built)

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    rng.shuffle(native_candidates)
    for row in native_candidates:
        if row["source_id"] in used_ids:
            continue
        if tokenizer is not None:
            tokens = count_template_tokens(
                tokenizer, row["question"], row.get("options"), row["response"]
            )
            if tokens > MAX_TEMPLATE_TOKENS:
                drops["too_long_tokens"] += 1
                continue
            row["template_tokens"] = tokens
        selected.append(tag_supplement_row(row, "numina_multi_blank_synth", "native"))
        used_ids.add(row["source_id"])
        if len(selected) >= target:
            break

    if len(selected) < target:
        compose_pool = [
            r
            for r in pool
            if r["source_id"] not in used_ids
            and 2000 <= r.get("trace_chars", 0) <= 4500
        ]
        by_topic: dict[str, list[dict[str, Any]]] = {}
        for r in compose_pool:
            by_topic.setdefault(r.get("topic", "unknown"), []).append(r)
        for rows in by_topic.values():
            rng.shuffle(rows)

        topic_keys = list(by_topic.keys())
        rng.shuffle(topic_keys)
        attempts = 0
        while len(selected) < target and attempts < target * 200:
            attempts += 1
            topic = rng.choice(topic_keys)
            bucket = by_topic[topic]
            if len(bucket) < 2:
                continue
            n_parts = rng.randint(2, min(MAX_COMPOSED_BLANKS, len(bucket)))
            parts = rng.sample(bucket, n_parts)
            if any(p["source_id"] in used_ids for p in parts):
                continue
            built = try_compose_multi_blank(parts)
            if built is None:
                drops["compose_failed"] += 1
                continue
            if tokenizer is not None:
                tokens = count_template_tokens(
                    tokenizer,
                    built["question"],
                    built.get("options"),
                    built["response"],
                )
                if tokens > MAX_TEMPLATE_TOKENS:
                    drops["compose_too_long_tokens"] += 1
                    continue
                built["template_tokens"] = tokens
            selected.append(built)
            for p in parts:
                used_ids.add(p["source_id"])

    mode_counts = Counter(r.get("supplement_mode") for r in selected)
    blank_dist = Counter(r.get("blank_count", 0) for r in selected)
    meta = {
        "pool_rows": len(pool),
        "native_candidates": len(native_candidates),
        "selected_rows": len(selected),
        "mode_counts": dict(mode_counts),
        "blank_count_distribution": dict(sorted(blank_dist.items())),
        "filter_drops": dict(drops),
    }
    return selected[:target], meta


def cmd_long_trace(args: argparse.Namespace) -> None:
    ready = read_jsonl(args.ready_path)
    exclude = corpus_source_ids(CORPUS_PATH)
    keys = load_competition_keys()
    rows, meta = build_long_trace_supplement(
        ready,
        exclude_ids=exclude,
        target=args.target,
        seed=args.seed,
        competition_keys=keys,
    )
    write_jsonl(LONG_TRACE_PATH, rows)
    payload = {
        "artifact": str(LONG_TRACE_PATH.relative_to(REPO)),
        "sha256": file_sha256(LONG_TRACE_PATH),
        "target_n": args.target,
        "sample_seed": args.seed,
        **meta,
        "trace_chars": dist_stats([r["trace_chars"] for r in rows]),
    }
    print(json.dumps(payload, indent=2))
    print(f"Wrote {LONG_TRACE_PATH} ({len(rows)} rows)")


def cmd_multi_blank(args: argparse.Namespace) -> None:
    tokenizer = None if args.skip_tokenize else load_tokenizer()
    if tokenizer is None and not args.skip_tokenize:
        print("Warning: transformers unavailable; skipping template tokenization.")

    ready = read_jsonl(args.ready_path)
    exclude = corpus_source_ids(CORPUS_PATH)
    keys = load_competition_keys()
    rows, meta = build_multi_blank_supplement(
        ready,
        exclude_ids=exclude,
        target=args.target,
        seed=args.seed,
        competition_keys=keys,
        tokenizer=tokenizer,
    )
    write_jsonl(MULTI_BLANK_PATH, rows)
    payload = {
        "artifact": str(MULTI_BLANK_PATH.relative_to(REPO)),
        "sha256": file_sha256(MULTI_BLANK_PATH),
        "target_n": args.target,
        "sample_seed": args.seed,
        "prompt_path": "build_prompt_multi_blank (full_public / pub-002)",
        **meta,
        "trace_chars": dist_stats([r["trace_chars"] for r in rows]),
        "template_tokens": dist_stats(
            [r["template_tokens"] for r in rows if "template_tokens" in r]
        ),
    }
    print(json.dumps(payload, indent=2))
    print(f"Wrote {MULTI_BLANK_PATH} ({len(rows)} rows)")


def cmd_merge_v2(args: argparse.Namespace) -> None:
    for path in (CORPUS_PATH, LONG_TRACE_PATH, MULTI_BLANK_PATH):
        if not path.is_file():
            raise SystemExit(f"Missing {path}")

    base = read_jsonl(CORPUS_PATH)
    long_rows = read_jsonl(LONG_TRACE_PATH)
    multi_rows = read_jsonl(MULTI_BLANK_PATH)
    merged = base + long_rows + multi_rows
    rng = random.Random(args.seed)
    rng.shuffle(merged)
    write_jsonl(CORPUS_V2_PATH, merged)

    blank_counts = [r.get("blank_count", n_ans_placeholders(r["question"])) for r in merged]
    geo_long = sum(1 for r in long_rows if is_geometry_flavored(r["question"]))

    manifest = {
        "thinking_template": THINKING_TEMPLATE,
        "sample_seed": args.seed,
        "final_row_count": len(merged),
        "sources": [
            {
                "path": str(CORPUS_PATH.relative_to(REPO)),
                "sha256": file_sha256(CORPUS_PATH),
                "rows": len(base),
                "role": "base_numina_mix",
            },
            {
                "path": str(LONG_TRACE_PATH.relative_to(REPO)),
                "sha256": file_sha256(LONG_TRACE_PATH),
                "rows": len(long_rows),
                "role": "long_trace_supplement",
                "geometry_keyword_fraction": round(
                    geo_long / max(len(long_rows), 1), 4
                ),
            },
            {
                "path": str(MULTI_BLANK_PATH.relative_to(REPO)),
                "sha256": file_sha256(MULTI_BLANK_PATH),
                "rows": len(multi_rows),
                "role": "multi_blank_supplement",
            },
        ],
        "trace_chars": dist_stats([r["trace_chars"] for r in merged]),
        "template_tokens": dist_stats(
            [r["template_tokens"] for r in merged if "template_tokens" in r]
        ),
        "blank_count_distribution": dict(Counter(blank_counts).most_common()),
        "corpus_path": str(CORPUS_V2_PATH.relative_to(REPO)),
        "corpus_sha256": file_sha256(CORPUS_V2_PATH),
        "base_manifest": (
            json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.is_file() else None
        ),
    }
    MANIFEST_V2_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {CORPUS_V2_PATH} ({len(merged)} rows)")
    print(f"Wrote {MANIFEST_V2_PATH}")


def cmd_all_supplements(args: argparse.Namespace) -> None:
    cmd_long_trace(args)
    cmd_multi_blank(args)
    cmd_merge_v2(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_base = sub.add_parser("base", help="Build 15k base corpus (original Step 5)")
    p_base.add_argument("--target", type=int, default=DEFAULT_TARGET)
    p_base.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_base.add_argument("--min-trace-chars", type=int, default=MIN_TRACE_CHARS)
    p_base.add_argument("--keep-letter-final", action="store_true")
    p_base.add_argument("--ready-path", type=Path, default=NUMINA_READY)
    p_base.set_defaults(func=cmd_base)

    for name, func in (
        ("long-trace", cmd_long_trace),
        ("multi-blank", cmd_multi_blank),
        ("merge-v2", cmd_merge_v2),
        ("supplements", cmd_all_supplements),
    ):
        p = sub.add_parser(name)
        p.add_argument("--target", type=int, default=SUPPLEMENT_TARGET)
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
        p.add_argument("--ready-path", type=Path, default=NUMINA_READY)
        p.add_argument(
            "--skip-tokenize",
            action="store_true",
            help="Skip HF tokenizer pass on multi-blank (faster local smoke)",
        )
        p.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build sft-007 ~5k OpenMath corpus: weak-topic slices + general anchor (topic_classify)."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_sft_corpus import (  # noqa: E402
    file_sha256,
    load_competition_keys,
    read_jsonl,
    write_jsonl,
)
from scripts.openmath_qualify import (  # noqa: E402
    THINKING_TEMPLATE,
    decontam_hit,
    load_decontam_prefixes,
    qualify_openmath_ex,
)
from scripts.topic_classify import classify_topic  # noqa: E402

SFT_SOURCES_DIR = REPO / "data" / "sft_sources"
PUBLIC_PATH = REPO / "data" / "public.jsonl"
PRIVATE_PATH = REPO / "data" / "private.jsonl"
DATASET_ID = "nvidia/OpenMathReasoning"
DATASET_CONFIG = "cot"

WEAK_BUCKETS = ("probability/stats", "geometry", "trigonometry")
ANCHOR_TOPICS = frozenset(
    {
        "integration",
        "derivatives",
        "limits",
        "linear algebra",
        "complex analysis",
        "polynomials/algebra",
        "number theory",
        "logs/exponents",
        "arithmetic/word problems",
        "other",
    }
)
EXCLUDED_TOPICS = frozenset({"sequences/recurrences"} | set(WEAK_BUCKETS))

DEFAULT_QUOTAS = {
    "probability/stats": 1300,
    "geometry": 1200,
    "trigonometry": 1000,
    "anchor": 1500,
}
DEFAULT_MIN_CHARS = 12_000
DEFAULT_MAX_CHARS = 28_000
TRIG_MIN_CHARS = 8_000
MIN_PASS_RATE = 0.05
MAX_PASS_RATE = 0.70
MAX_TEMPLATE_TOKENS = 7900
SAMPLE_SEED = 42


def git_commit_short() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def dist_stats(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"p25": 0.0, "p50": 0.0, "p95": 0.0, "mean": 0.0, "median": 0.0}
    s = sorted(values)

    def pct(p: float) -> float:
        idx = max(0, int(p * len(s)) - 1)
        return float(s[idx])

    return {
        "p25": round(pct(0.25), 1),
        "p50": round(pct(0.50), 1),
        "p95": round(pct(0.95), 1),
        "mean": round(statistics.mean(s), 1),
        "median": round(statistics.median(s), 1),
    }


def route_bucket(topic: str) -> Optional[str]:
    if topic in WEAK_BUCKETS:
        return topic
    if topic in ANCHOR_TOPICS and topic not in EXCLUDED_TOPICS:
        return "anchor"
    return None


def buckets_full(counts: dict[str, int], quotas: dict[str, int]) -> bool:
    for key in WEAK_BUCKETS:
        if counts.get(key, 0) < quotas[key]:
            return False
    if counts.get("anchor", 0) < quotas["anchor"]:
        return False
    return True


def sample_anchor_from_ready(
    ready_path: Path,
    *,
    n: int,
    seed: int,
    min_chars: int,
    max_chars: int,
    competition_keys: set[str],
    decontam_prefixes: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not ready_path.is_file():
        return [], {"anchor_source": "ready_pool", "error": "missing_ready_file"}

    pool: list[dict[str, Any]] = []
    for row in read_jsonl(ready_path):
        topic = classify_topic(row["question"])
        if topic in EXCLUDED_TOPICS or topic in WEAK_BUCKETS:
            continue
        tc = row.get("trace_chars", len(row.get("response", "")))
        if not (min_chars <= tc <= max_chars):
            continue
        if decontam_hit(row["question"], decontam_prefixes, competition_keys):
            continue
        out = dict(row)
        out["topic"] = topic
        out["topic_bucket"] = "anchor"
        out["weak_topic"] = False
        pool.append(out)

    rng = random.Random(seed)
    rng.shuffle(pool)
    picked = pool[:n]
    return picked, {
        "anchor_source": str(ready_path.relative_to(REPO)),
        "anchor_pool_n": len(pool),
        "anchor_sampled_n": len(picked),
    }


def apply_trig_backfill(
    buckets: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
) -> dict[str, Any]:
    meta: dict[str, Any] = {"trig_backfill_applied": False}
    need = quotas["trigonometry"] - len(buckets["trigonometry"])
    if need <= 0:
        return meta

    meta["trig_backfill_applied"] = True
    meta["trig_deficit"] = need
    donors: list[dict[str, Any]] = []
    for name in ("probability/stats", "geometry"):
        excess = len(buckets[name]) - quotas[name]
        if excess > 0:
            donors.extend(buckets[name][quotas[name] : quotas[name] + excess])
    rng = random.Random(SAMPLE_SEED + 7)
    rng.shuffle(donors)
    taken = donors[:need]
    meta["trig_backfill_from"] = len(taken)
    for row in taken:
        row = dict(row)
        row["topic_bucket"] = "trigonometry"
        row["topic"] = "trigonometry"
        row["weak_topic"] = True
        row["trig_backfill"] = True
        buckets["trigonometry"].append(row)
    return meta


def finalize_corpus(
    buckets: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for bucket in (*WEAK_BUCKETS, "anchor"):
        rows = buckets[bucket][: quotas[bucket]]
        for row in rows:
            r = dict(row)
            r["topic_bucket"] = bucket if bucket != "anchor" else r.get("topic", "anchor")
            r["weak_topic"] = bucket in WEAK_BUCKETS
            r["corpus_id"] = "sft-007"
            r["thinking_template"] = THINKING_TEMPLATE
            out.append(r)
        rng.shuffle(rows)
    rng.shuffle(out)
    return out


def run_scan(
    *,
    quotas: dict[str, int],
    max_scan: int,
    min_chars_default: int,
    max_chars: int,
    log_every: int,
    tokenizer,
    competition_keys: set[str],
    decontam_prefixes: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=DATASET_CONFIG, streaming=True)
    buckets: dict[str, list[dict[str, Any]]] = {k: [] for k in (*WEAK_BUCKETS, "anchor")}
    reject_counts: Counter = Counter()
    rows_scanned = 0
    pool_cap = sum(quotas.values()) * 3

    print(
        f"Streaming {DATASET_ID} — max_scan={max_scan:,}, "
        f"quotas={quotas}",
        flush=True,
    )

    for idx, ex in enumerate(ds):
        if rows_scanned >= max_scan:
            print(f"Hit scan cap ({max_scan:,}); stopping.", flush=True)
            break
        if buckets_full({k: len(v) for k, v in buckets.items()}, quotas):
            print(f"All quotas full after {rows_scanned:,} rows; stopping.", flush=True)
            break
        rows_scanned += 1

        problem = (ex.get("problem") or "").strip()
        topic = classify_topic(problem)
        bucket = route_bucket(topic)
        if bucket is None:
            reject_counts["topic_not_routed"] += 1
            continue

        if len(buckets[bucket]) >= quotas[bucket]:
            reject_counts["bucket_full"] += 1
            continue

        min_chars = TRIG_MIN_CHARS if bucket == "trigonometry" else min_chars_default
        row, reject = qualify_openmath_ex(
            ex,
            idx=idx,
            source="openmath_sft007",
            source_id_prefix="om007",
            decontam_prefixes=decontam_prefixes,
            competition_keys=competition_keys,
            min_response_chars=min_chars,
            max_response_chars=max_chars,
            min_pass_rate=MIN_PASS_RATE,
            max_pass_rate=MAX_PASS_RATE,
            tokenizer=tokenizer,
            max_template_tokens=MAX_TEMPLATE_TOKENS,
        )
        if row is not None:
            row["topic"] = topic
            row["topic_bucket"] = bucket
            row["weak_topic"] = bucket in WEAK_BUCKETS
            buckets[bucket].append(row)
            total = sum(len(v) for v in buckets.values())
            if total >= pool_cap:
                print(f"Pool cap {pool_cap} reached; stopping scan.", flush=True)
                break
        elif reject is not None:
            reject_counts[reject.get("reason", "unknown")] += 1

        if rows_scanned % log_every == 0:
            counts = {k: len(v) for k, v in buckets.items()}
            print(f"  scanned={rows_scanned:,} buckets={counts}", flush=True)

    scan_meta = {
        "input_rows_scanned": rows_scanned,
        "reject_counts": dict(reject_counts),
        "bucket_counts_raw": {k: len(v) for k, v in buckets.items()},
    }
    return buckets, scan_meta


def build_corpus(
    *,
    quotas: dict[str, int],
    max_scan: int,
    anchor_ready_path: Optional[Path],
    skip_scan: bool,
    tokenizer,
    corpus_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    competition_keys = load_competition_keys()
    decontam_prefixes = load_decontam_prefixes(PUBLIC_PATH, PRIVATE_PATH)

    scan_meta: dict[str, Any] = {}
    buckets: dict[str, list[dict[str, Any]]] = {k: [] for k in (*WEAK_BUCKETS, "anchor")}

    if not skip_scan:
        buckets, scan_meta = run_scan(
            quotas=quotas,
            max_scan=max_scan,
            min_chars_default=DEFAULT_MIN_CHARS,
            max_chars=DEFAULT_MAX_CHARS,
            log_every=5_000,
            tokenizer=tokenizer,
            competition_keys=competition_keys,
            decontam_prefixes=decontam_prefixes,
        )

    anchor_meta: dict[str, Any] = {}
    anchor_need = quotas["anchor"]
    if len(buckets["anchor"]) < anchor_need and anchor_ready_path:
        n = anchor_need - len(buckets["anchor"])
        sampled, anchor_meta = sample_anchor_from_ready(
            anchor_ready_path,
            n=n,
            seed=SAMPLE_SEED + 3,
            min_chars=DEFAULT_MIN_CHARS,
            max_chars=DEFAULT_MAX_CHARS,
            competition_keys=competition_keys,
            decontam_prefixes=decontam_prefixes,
        )
        buckets["anchor"].extend(sampled)

    backfill_meta = apply_trig_backfill(buckets, quotas)
    corpus = finalize_corpus(buckets, quotas, seed=SAMPLE_SEED)

    SFT_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    for bucket in (*WEAK_BUCKETS, "anchor"):
        slug = bucket.replace("/", "_")
        ready_out = SFT_SOURCES_DIR / f"openmath_sft007_{slug}_ready.jsonl"
        write_jsonl(ready_out, buckets[bucket][: quotas[bucket]])

    write_jsonl(corpus_path, corpus)

    slice_counts = Counter(r["topic_bucket"] for r in corpus)
    trace_chars = [r["trace_chars"] for r in corpus]
    template_tokens = [r.get("template_tokens", 0) for r in corpus]
    pass_rates = [r.get("pass_rate_72b_tir", 0.0) for r in corpus if "pass_rate_72b_tir" in r]

    manifest: dict[str, Any] = {
        "corpus_id": "sft-007",
        "thinking_template": THINKING_TEMPLATE,
        "source_dataset": DATASET_ID,
        "source_config": DATASET_CONFIG,
        "classifier": "weighted_v1",
        "sample_seed": SAMPLE_SEED,
        "target_quotas": quotas,
        "slice_counts": dict(slice_counts),
        "final_row_count": len(corpus),
        "min_response_chars_default": DEFAULT_MIN_CHARS,
        "min_response_chars_trig": TRIG_MIN_CHARS,
        "max_response_chars": DEFAULT_MAX_CHARS,
        "min_pass_rate": MIN_PASS_RATE,
        "max_pass_rate": MAX_PASS_RATE,
        "max_scan": max_scan,
        "trace_chars": dist_stats(trace_chars),
        "template_tokens": dist_stats(template_tokens),
        "pass_rate_distribution": dist_stats(pass_rates),
        "scan": scan_meta,
        "anchor_sampling": anchor_meta,
        "trig_backfill": backfill_meta,
        "corpus_path": str(corpus_path.relative_to(REPO)),
        "corpus_sha256": file_sha256(corpus_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_short(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus-out",
        type=Path,
        default=REPO / "data" / "sft_corpus_sft007_openmath_5k.jsonl",
    )
    p.add_argument(
        "--manifest-out",
        type=Path,
        default=REPO / "data" / "sft_corpus_sft007_openmath_5k_manifest.json",
    )
    p.add_argument("--max-scan", type=int, default=600_000)
    p.add_argument(
        "--anchor-ready",
        type=Path,
        default=SFT_SOURCES_DIR / "openmath_reasoning_ready.jsonl",
    )
    p.add_argument(
        "--skip-scan",
        action="store_true",
        help="Only sample anchor from ready pool (debug)",
    )
    p.add_argument("--no-tokenizer", action="store_true", help="Skip token-length filter")
    args = p.parse_args()

    tokenizer = None
    if not args.no_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Thinking-2507", trust_remote_code=True
        )

    manifest = build_corpus(
        quotas=dict(DEFAULT_QUOTAS),
        max_scan=args.max_scan,
        anchor_ready_path=args.anchor_ready,
        skip_scan=args.skip_scan,
        tokenizer=tokenizer,
        corpus_path=args.corpus_out,
        manifest_path=args.manifest_out,
    )
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {args.corpus_out} ({manifest['final_row_count']} rows)")


if __name__ == "__main__":
    main()

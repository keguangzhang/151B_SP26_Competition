#!/usr/bin/env python3
"""Carve frozen Q4-long and multi-blank≥3 eval watch sets from data/eval/holdout.jsonl."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "data" / "eval"
DEFAULT_HOLDOUT = EVAL_DIR / "holdout.jsonl"
DEFAULT_MANIFEST = EVAL_DIR / "watch_manifest.json"

Q4_MIN_QUESTION_CHARS = 435
Q4_TARGET_N = 30
MULTI_BLANK_MIN_ANS = 3
MULTI_BLANK_TARGET_N = 20
WATCH_SEED = 42


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def n_ans_placeholders(question: str) -> int:
    return question.count("[ANS]")


def is_mcq(row: dict[str, Any]) -> bool:
    return bool(row.get("options"))


def stratified_sample(
    pool: list[dict[str, Any]],
    target_n: int,
    seed: int,
    stratum_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    """Proportional stratified sample; deterministic given seed."""
    if len(pool) <= target_n:
        out = list(pool)
        out.sort(key=lambda r: r.get("id", 0))
        return out

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        strata[stratum_fn(row)].append(row)

    rng = random.Random(seed)
    order = sorted(strata.keys())
    rng.shuffle(order)

    total = len(pool)
    picks: list[dict[str, Any]] = []
    remaining = target_n
    for i, key in enumerate(order):
        items = strata[key]
        if i == len(order) - 1:
            k = remaining
        else:
            k = round(target_n * len(items) / total)
            k = max(1, k) if len(items) >= 1 else 0
            k = min(k, len(items), remaining)
        rng.shuffle(items)
        picks.extend(items[:k])
        remaining -= k

    picks.sort(key=lambda r: r.get("id", 0))
    return picks


def build_eval_watch_sets(
    holdout_path: Path,
    manifest_path: Path,
    q4_target: int,
    mb_target: int,
    seed: int,
) -> dict[str, Any]:
    holdout_rows = read_jsonl(holdout_path)
    by_id = {r["id"]: r for r in holdout_rows}

    q4_pool = [r for r in holdout_rows if len(r["question"]) >= Q4_MIN_QUESTION_CHARS]
    mb_pool = [r for r in holdout_rows if n_ans_placeholders(r["question"]) >= MULTI_BLANK_MIN_ANS]

    q4_rows = stratified_sample(
        q4_pool,
        q4_target,
        seed,
        lambda r: "mcq" if is_mcq(r) else "free_form",
    )
    mb_rows = stratified_sample(
        mb_pool,
        mb_target,
        seed + 1,
        lambda _: "free_form",
    )

    q4_jsonl = EVAL_DIR / "watch_q4_long.jsonl"
    mb_jsonl = EVAL_DIR / "watch_multi_blank_ge3.jsonl"
    write_jsonl(q4_jsonl, q4_rows)
    write_jsonl(mb_jsonl, mb_rows)

    def watch_meta(
        name: str,
        description: str,
        criteria: dict[str, Any],
        target_n: int,
        watch_seed: int,
        pool: list[dict[str, Any]],
        selected: list[dict[str, Any]],
        jsonl_path: Path,
    ) -> dict[str, Any]:
        strata = defaultdict(int)
        for r in selected:
            strata["mcq" if is_mcq(r) else "free_form"] += 1
        return {
            "name": name,
            "description": description,
            "criteria": criteria,
            "target_n": target_n,
            "seed": watch_seed,
            "pool_n": len(pool),
            "n": len(selected),
            "ids": [r["id"] for r in selected],
            "jsonl": str(jsonl_path.relative_to(REPO)),
            "strata": dict(strata),
        }

    manifest: dict[str, Any] = {
        "holdout_jsonl": str(holdout_path.relative_to(REPO)),
        "holdout_n": len(holdout_rows),
        "watch": {
            "q4_long": watch_meta(
                "q4_long",
                "Q4 long-context watch set (question length ≥ 435 chars)",
                {"question_chars_gte": Q4_MIN_QUESTION_CHARS},
                q4_target,
                seed,
                q4_pool,
                q4_rows,
                q4_jsonl,
            ),
            "multi_blank_ge3": watch_meta(
                "multi_blank_ge3",
                "Multi-blank ≥3 watch set",
                {"ans_placeholders_gte": MULTI_BLANK_MIN_ANS},
                mb_target,
                seed + 1,
                mb_pool,
                mb_rows,
                mb_jsonl,
            ),
        },
    }

    for key, w in manifest["watch"].items():
        missing = [i for i in w["ids"] if i not in by_id]
        if missing:
            raise RuntimeError(f"{key}: ids not in holdout: {missing[:5]}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--q4-n", type=int, default=Q4_TARGET_N)
    p.add_argument("--mb-n", type=int, default=MULTI_BLANK_TARGET_N)
    p.add_argument("--seed", type=int, default=WATCH_SEED)
    args = p.parse_args()

    if not args.holdout.is_file():
        raise SystemExit(
            f"Missing {args.holdout}. Run: python scripts/build_eval_holdout.py --fraction 0.20 --seed 42"
        )

    manifest = build_eval_watch_sets(
        args.holdout, args.manifest, args.q4_n, args.mb_n, args.seed
    )
    for name, w in manifest["watch"].items():
        print(
            f"{name}: {w['n']}/{w['pool_n']} rows "
            f"(ids frozen, seed={w['seed']}) strata={w['strata']}"
        )
    print(f"Wrote manifest → {args.manifest}")


if __name__ == "__main__":
    main()

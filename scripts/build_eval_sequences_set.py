#!/usr/bin/env python3
"""Build frozen sequences eval set from data/public.jsonl → data/eval/sequences_dev.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.sft_prompt import is_openmath_sequence_question  # noqa: E402

DEFAULT_PUBLIC = REPO / "data" / "public.jsonl"
EVAL_DIR = REPO / "data" / "eval"
DEFAULT_SEQUENCES_JSONL = EVAL_DIR / "sequences_dev.jsonl"
DEFAULT_MANIFEST = EVAL_DIR / "watch_manifest.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_mcq(row: dict[str, Any]) -> bool:
    return bool(row.get("options"))


def build_eval_sequences_set(
    public_path: Path,
    sequences_jsonl: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    public_rows = read_jsonl(public_path)
    pool = [r for r in public_rows if is_openmath_sequence_question(r["question"])]
    pool.sort(key=lambda r: r.get("id", 0))

    write_jsonl(sequences_jsonl, pool)

    strata: dict[str, int] = defaultdict(int)
    for r in pool:
        strata["mcq" if is_mcq(r) else "free_form"] += 1

    sequences_meta = {
        "name": "sequences",
        "description": "Full-public sequences/recurrences dev set (OpenMath keyword classifier)",
        "criteria": {"openmath_sequence_keywords": True},
        "target_n": len(pool),
        "seed": None,
        "pool_n": len(public_rows),
        "n": len(pool),
        "ids": [r["id"] for r in pool],
        "jsonl": str(sequences_jsonl.relative_to(REPO)),
        "strata": dict(strata),
    }

    if manifest_path.is_file():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"holdout_jsonl": "data/eval/holdout.jsonl", "holdout_n": 0, "watch": {}}

    manifest.setdefault("watch", {})["sequences"] = sequences_meta

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    return {"sequences": sequences_meta, "public_n": len(public_rows)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public", type=Path, default=DEFAULT_PUBLIC, help="Labeled public JSONL")
    p.add_argument("--out", type=Path, default=DEFAULT_SEQUENCES_JSONL, help="Output sequences dev JSONL")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Watch manifest to update")
    p.add_argument("--force", action="store_true", help="Overwrite existing sequences dev file")
    args = p.parse_args()

    if not args.public.is_file():
        raise SystemExit(f"Missing public set: {args.public}")

    if args.out.is_file() and not args.force:
        print(f"Exists: {args.out} (use --force to rebuild)")
        return

    result = build_eval_sequences_set(args.public, args.out, args.manifest)
    seq = result["sequences"]
    print(f"Wrote {seq['n']} rows to {args.out}")
    print(f"  Public total: {result['public_n']}   strata={seq['strata']}")
    print(f"Updated manifest → {args.manifest}")


if __name__ == "__main__":
    main()

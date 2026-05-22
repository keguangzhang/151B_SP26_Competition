#!/usr/bin/env python3
"""Build frozen stratified dev slice from data/public.jsonl → data/dev.jsonl."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC = REPO / "data" / "public.jsonl"
DEFAULT_DEV = REPO / "data" / "dev.jsonl"
DEFAULT_FRACTION = 0.20
DEFAULT_SEED = 42


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dev_jsonl(
    public_path: Path,
    dev_path: Path,
    fraction: float,
    seed: int,
) -> tuple[int, int, int, int, int, int]:
    """Sample fraction from MCQ and free-form strata separately; sort by id."""
    all_rows = read_jsonl(public_path)
    mcq = [r for r in all_rows if r.get("options")]
    free = [r for r in all_rows if not r.get("options")]

    rng = random.Random(seed)

    def sample_frac(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items:
            return []
        k = max(1, int(len(items) * fraction))
        k = min(k, len(items))
        idx = list(range(len(items)))
        rng.shuffle(idx)
        return [items[i] for i in idx[:k]]

    dev_mcq = sample_frac(mcq)
    dev_free = sample_frac(free)
    dev_rows = dev_mcq + dev_free
    dev_rows.sort(key=lambda r: r.get("id", 0))

    write_jsonl(dev_path, dev_rows)
    return len(dev_rows), len(dev_mcq), len(mcq), len(dev_free), len(free), len(all_rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public", type=Path, default=DEFAULT_PUBLIC, help="Labeled public JSONL")
    p.add_argument("--out", type=Path, default=DEFAULT_DEV, help="Output dev JSONL")
    p.add_argument(
        "--fraction",
        type=float,
        default=DEFAULT_FRACTION,
        help="Fraction per stratum (MCQ and free-form); default 0.20 → 225 rows",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--force", action="store_true", help="Overwrite existing dev file")
    args = p.parse_args()

    if not args.public.is_file():
        raise SystemExit(f"Missing public set: {args.public}")

    if args.out.is_file() and not args.force:
        print(f"Exists: {args.out} (use --force to rebuild)")
        return

    n_dev, n_mcq, n_mcq_all, n_free, n_free_all, n_public = build_dev_jsonl(
        args.public, args.out, args.fraction, args.seed
    )
    print(f"Wrote {n_dev} rows to {args.out}")
    print(f"  MCQ in dev: {n_mcq} / {n_mcq_all}   Free-form in dev: {n_free} / {n_free_all}")
    print(f"  Public total: {n_public}")


if __name__ == "__main__":
    main()

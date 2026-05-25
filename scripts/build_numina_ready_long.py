#!/usr/bin/env python3
"""Second-pass Numina prep: long-biased rows outside the 25k reservoir sample."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "sft_data_prep.ipynb"
# setup, helpers, tokenizer, numina §5 definitions
CELLS = (4, 6, 7, 9, 16, 23)  # 23: §5.3 long second pass


def _load_namespace() -> dict:
    nb = json.loads(NOTEBOOK.read_text())
    ns: dict = {"__name__": "__main__"}
    for idx in CELLS:
        code = "".join(nb["cells"][idx]["source"])
        exec(compile(code, f"sft_data_prep_cell_{idx}", "exec"), ns)
    return ns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first-pass-ready",
        type=Path,
        default=REPO / "data/sft_sources/numina_cot_clean_ready.jsonl",
    )
    parser.add_argument(
        "--out-ready",
        type=Path,
        default=REPO / "data/sft_sources/numina_cot_clean_ready_long.jsonl",
    )
    parser.add_argument(
        "--out-stats",
        type=Path,
        default=REPO / "data/sft_sources/numina_cot_clean_long_stats.json",
    )
    parser.add_argument("--candidate-heap", type=int, default=25_000)
    parser.add_argument("--max-ready", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.first_pass_ready.is_file():
        raise SystemExit(f"Missing first-pass ready: {args.first_pass_ready}")

    sys.path.insert(0, str(REPO))
    ns = _load_namespace()
    random.seed(args.seed)

    existing_ids = {r["source_id"] for r in ns["read_jsonl"](args.first_pass_ready)}
    print(f"First-pass source_ids to skip: {len(existing_ids):,}")

    ready, stats_payload = ns["run_numina_long_second_pass"](
        existing_source_ids=existing_ids,
        candidate_heap=args.candidate_heap,
        max_ready=args.max_ready,
    )
    ns["validate_ready_corpus"](
        ready, ns["NUMINA_LONG_SOURCE"], require_thinking_wrapper=True
    )

    ns["write_jsonl"](args.out_ready, ready)
    args.out_stats.write_text(json.dumps(stats_payload, indent=2) + "\n")

    print(json.dumps(stats_payload, indent=2))
    print(f"\nWrote {args.out_ready} ({len(ready)} rows)")
    print(f"Wrote {args.out_stats}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Smoke-run Numina Step 2 prep (limited scan). Full run: notebooks/sft_data_prep.ipynb §5.1."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "sft_data_prep.ipynb"
CELLS = (4, 6, 7, 9, 16)  # 9: datasets import


def _load_notebook_namespace() -> dict:
    nb = json.loads(NOTEBOOK.read_text())
    ns: dict = {"__name__": "__main__"}
    for idx in CELLS:
        code = "".join(nb["cells"][idx]["source"])
        exec(compile(code, f"sft_data_prep_cell_{idx}", "exec"), ns)
    return ns


def main() -> None:
    sys.path.insert(0, str(REPO))
    ns = _load_notebook_namespace()
    ns["NUMINA_MAX_SCAN"] = 25_000
    ns["NUMINA_MAX_READY"] = 400
    random.seed(ns["RANDOM_SEED"])

    ready, rejects, stats = ns["run_numina_prep"]()
    ns["validate_ready_corpus"](ready, ns["NUMINA_SOURCE"], require_thinking_wrapper=True)

    ready_path = ns["NUMINA_READY_PATH"]
    rejects_path = ns["NUMINA_REJECTS_PATH"]
    stats_path = ns["NUMINA_STATS_PATH"]

    ns["write_jsonl"](ready_path, ready)
    ns["write_jsonl"](rejects_path, rejects)

    trace_chars = [r["trace_chars"] for r in ready]
    template_tokens = [r["template_tokens"] for r in ready]
    payload = {
        **stats.to_dict(),
        "smoke": True,
        "thinking_template": ns["THINKING_TEMPLATE"],
        "ready_rows": len(ready),
        "mean_trace_chars": round(sum(trace_chars) / max(len(trace_chars), 1), 1),
        "mean_template_tokens": round(
            sum(template_tokens) / max(len(template_tokens), 1), 1
        ),
    }
    stats_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps(payload, indent=2))
    print(f"Wrote {ready_path} ({len(ready)} rows)")
    print(f"Wrote {rejects_path} ({len(rejects)} sampled rejects)")


if __name__ == "__main__":
    main()

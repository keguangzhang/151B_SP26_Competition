"""Load frozen eval watch sets (Q4 long, multi-blank ≥3) and score filtered results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "data" / "eval" / "watch_manifest.json"


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    path = manifest_path or DEFAULT_MANIFEST
    with open(path) as f:
        return json.load(f)


def load_watch_ids(name: str, manifest_path: Path | None = None) -> list[int]:
    manifest = load_manifest(manifest_path)
    return list(manifest["watch"][name]["ids"])


def filter_results_by_watch(
    results: list[dict[str, Any]],
    watch_name: str,
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    ids = set(load_watch_ids(watch_name, manifest_path))
    return [r for r in results if r.get("id") in ids]


def watch_accuracy(
    results: list[dict[str, Any]],
    watch_name: str,
    manifest_path: Path | None = None,
) -> tuple[float, int, int]:
    subset = filter_results_by_watch(results, watch_name, manifest_path)
    n = len(subset)
    if n == 0:
        return 0.0, 0, 0
    correct = sum(1 for r in subset if r.get("correct"))
    return correct / n * 100, correct, n

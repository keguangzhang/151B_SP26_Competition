"""Load frozen eval watch sets (geometry, sequences, Q4 long, multi-blank ≥3) and score filtered results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "data" / "eval" / "watch_manifest.json"
DEFAULT_GEOMETRY_DEV = REPO / "data" / "eval" / "geometry_dev.jsonl"
DEFAULT_SEQUENCES_DEV = REPO / "data" / "eval" / "sequences_dev.jsonl"


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    path = manifest_path or DEFAULT_MANIFEST
    with open(path) as f:
        return json.load(f)


def _ids_from_dev_jsonl(dev_path: Path) -> list[int]:
    with open(dev_path) as f:
        return [json.loads(line)["id"] for line in f]


def _geometry_ids_from_dev(geometry_dev_path: Path | None = None) -> list[int]:
    return _ids_from_dev_jsonl(geometry_dev_path or DEFAULT_GEOMETRY_DEV)


def _sequences_ids_from_dev(sequences_dev_path: Path | None = None) -> list[int]:
    return _ids_from_dev_jsonl(sequences_dev_path or DEFAULT_SEQUENCES_DEV)


def load_watch_ids(name: str, manifest_path: Path | None = None) -> list[int]:
    manifest = load_manifest(manifest_path)
    watch = manifest.get("watch", {})
    if name in watch:
        return list(watch[name]["ids"])
    eval_dir = (manifest_path or DEFAULT_MANIFEST).parent
    if name == "geometry":
        geo_path = eval_dir / "geometry_dev.jsonl"
        if geo_path.is_file():
            return _geometry_ids_from_dev(geo_path)
    if name == "sequences":
        seq_path = eval_dir / "sequences_dev.jsonl"
        if seq_path.is_file():
            return _sequences_ids_from_dev(seq_path)
    raise KeyError(
        f"Unknown watch set {name!r}; rebuild data/eval/watch_manifest.json "
        f"(have: {sorted(watch.keys())})"
    )


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

#!/usr/bin/env python3
"""
End-to-end private-set inference for CSE 151B final submission.

Mirrors notebooks/submission.ipynb: Qwen3-4B-Thinking, exact_v1 prompts, vLLM bf16 A100 profile.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import io
import json
import os
import site
import sys
import time
from pathlib import Path
# ── Frozen hyperparameters (final submission) ─────────────────────────────────
# Base model (competition-required). If you fine-tuned, push merged weights to the Hub
# and set this to your repo id, e.g. "your-username/qwen3-4b-thinking-exact-v1".
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

GPU_ID = "0"
MAX_TOKENS = 16384
PROMPT_BUDGET = 2048
MAX_MODEL_LEN = MAX_TOKENS + PROMPT_BUDGET

TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 0.0
REPETITION_PENALTY = 1.0

CHUNK_SIZE = 128
GPU_MEMORY_UTILIZATION = 0.88
KV_CACHE_DTYPE = "fp8"

from inference.prompts import PROMPT_VARIANT, build_prompt  # noqa: E402


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "data").is_dir():
        return here
    if (here.parent / "data").is_dir():
        return here.parent
    return here


def _prepend_nvidia_libs_to_ld_path() -> None:
    roots = list(site.getsitepackages())
    user_site = getattr(site, "getusersitepackages", lambda: None)()
    if user_site:
        roots.append(user_site)
    libdirs: list[str] = []
    seen: set[str] = set()
    for root in roots:
        for d in glob.glob(os.path.join(root, "nvidia", "*", "lib")):
            if os.path.isdir(d) and d not in seen:
                seen.add(d)
                libdirs.append(d)
    if libdirs:
        sep = os.pathsep
        os.environ["LD_LIBRARY_PATH"] = sep.join(
            libdirs + [os.environ.get("LD_LIBRARY_PATH", "")]
        ).strip(sep)


@contextlib.contextmanager
def _jupyter_stdout_for_vllm():
    try:
        sys.stdout.fileno()
    except (io.UnsupportedOperation, AttributeError, OSError):
        saved_out, saved_err = sys.stdout, sys.stderr
        dup1, dup2 = os.dup(1), os.dup(2)
        try:
            sys.stdout = os.fdopen(dup1, "w", buffering=1)
            sys.stderr = os.fdopen(dup2, "w", buffering=1)
            yield
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout, sys.stderr = saved_out, saved_err
    else:
        yield


def _submission_stem() -> str:
    token_k = MAX_TOKENS // 1024
    return f"submission_{PROMPT_VARIANT}_{token_k}k"


def _load_private_data(data_path: Path) -> list[dict]:
    with open(data_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _write_submission_csv(
    data: list[dict], completed: dict[int, str], csv_path: Path
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["id", "response"])
        for row in data:
            qid = row["id"]
            w.writerow([qid, completed[qid]])

    with open(csv_path, newline="", encoding="utf-8") as f:
        n = sum(1 for _ in csv.reader(f))
    expected = len(data) + 1
    if n != expected:
        raise RuntimeError(f"Expected header + {len(data)} rows, got {n} lines in {csv_path}")


def run_inference(
    output_csv: str | Path | None = None,
    data_path: str | Path | None = None,
    *,
    resume: bool = True,
) -> Path:
    """
    Load model, run inference on private.jsonl, write submission CSV.

    Parameters
    ----------
    output_csv
        Destination CSV (default: results/submission_exact_v1_16k.csv).
    data_path
        Input JSONL (default: data/private.jsonl under repo root).
    resume
        If True, append to and reuse an existing responses checkpoint.

    Returns
    -------
    Path to the written submission CSV.
    """
    root = repo_root()
    data_path = Path(data_path or root / "data/private.jsonl")
    stem = _submission_stem()
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if output_csv is None:
        output_csv = results_dir / f"{stem}.csv"
    output_csv = Path(output_csv)

    response_checkpoint = results_dir / f"{stem}.responses.jsonl"

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    _prepend_nvidia_libs_to_ld_path()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.perf_counter()
    data = _load_private_data(data_path)
    print(f"Loaded {len(data)} questions from {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    max_num_seqs = max(64, int(256 * 17500 / MAX_MODEL_LEN))
    llm_kwargs: dict = {
        "model": MODEL_ID,
        "dtype": "bfloat16",
        "enable_prefix_caching": True,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_model_len": MAX_MODEL_LEN,
        "trust_remote_code": True,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max(MAX_MODEL_LEN, 32768),
        "enable_chunked_prefill": True,
        "kv_cache_dtype": KV_CACHE_DTYPE,
    }
    print(f"Loading model {MODEL_ID!r} (max_model_len={MAX_MODEL_LEN})...")
    with _jupyter_stdout_for_vllm():
        llm = LLM(**llm_kwargs)

    load_s = time.perf_counter() - t0
    print(f"Model ready in {load_s:.1f}s (max_num_seqs={max_num_seqs})")

    default_sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        min_p=MIN_P,
        presence_penalty=PRESENCE_PENALTY,
        repetition_penalty=REPETITION_PENALTY,
    )

    completed: dict[int, str] = {}
    if resume and response_checkpoint.exists():
        with open(response_checkpoint, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                completed[r["id"]] = r["response"]
        print(f"Resumed checkpoint: {len(completed)} responses")

    if not resume and response_checkpoint.exists():
        response_checkpoint.unlink()

    remaining = [d for d in data if d["id"] not in completed]
    print(f"Generating {len(remaining)} remaining / {len(data)} total...")

    gen_t0 = time.perf_counter()
    for chunk_start in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[chunk_start : chunk_start + CHUNK_SIZE]

        prompts = []
        chunk_params = []
        for item in chunk:
            system, user = build_prompt(item["question"], item.get("options"))
            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt_text)
            chunk_params.append(default_sampling_params)

        with _jupyter_stdout_for_vllm():
            outputs = llm.generate(prompts, sampling_params=chunk_params)

        with open(response_checkpoint, "a", encoding="utf-8") as f:
            for item, out in zip(chunk, outputs):
                response = out.outputs[0].text.strip()
                completed[item["id"]] = response
                f.write(json.dumps({"id": item["id"], "response": response}) + "\n")

        done = len(completed)
        print(f"  {done}/{len(data)}  ({100.0 * done / len(data):.1f}%)")

    gen_s = time.perf_counter() - gen_t0
    if len(remaining) > 0:
        print(f"Generation wall time: {gen_s:.1f}s ({gen_s / len(remaining):.2f}s/item)")

    if len(completed) != len(data):
        raise RuntimeError("Missing ids — checkpoint vs data mismatch")

    _write_submission_csv(data, completed, output_csv)
    total_s = time.perf_counter() - t0
    print(f"Wrote {len(data)} rows to {output_csv.resolve()}")
    print(f"Total pipeline time: {total_s:.1f}s")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="CSE 151B private-set inference")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: results/submission_<variant>_<N>k.csv)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Input JSONL (default: data/private.jsonl)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore/delete checkpoint and regenerate all responses",
    )
    args = parser.parse_args()
    run_inference(
        output_csv=args.output,
        data_path=args.data,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

# vLLM inference configuration

How we load **Qwen/Qwen3-4B-Thinking-2507** with vLLM across notebooks. Colab install / env quirks: [`vllm-colab-l4.md`](vllm-colab-l4.md).

**Source of truth:** `LLM(...)` cells in each notebook (not this table alone).

---

## Profiles at a glance

| | **L4 / 24 GB** (starter, full public) | **A100 / 40 GB** (`notebooks/dev.ipynb`) |
|---|--------------------------------------|------------------------------------------|
| Notebooks | `starter_code_cse151b_comp.ipynb`, `notebooks/full_public.ipynb`, `submission.ipynb` | `notebooks/dev.ipynb` |
| Weights | **INT8** via `bitsandbytes` | **bf16** native (no quant) |
| `max_model_len` | **16384** | **32768** |
| `MAX_TOKENS` (generation) | 4096 (starter) / **8192** (pub-001) | **8192** or **16384** (§2) |
| `gpu_memory_utilization` | 0.88 | **0.92** |
| `max_num_seqs` | 64 | **128** |
| `max_num_batched_tokens` | 16384 | **32768** |
| `enable_chunked_prefill` | default (off in older runs) | **True** |
| `enable_prefix_caching` | **True** | **True** |
| `enforce_eager` | **off** (CUDA graphs + compile enabled) | **off** |

On A100, bf16 weights are ~7.6 GiB; the extra headroom funds a longer KV cache for 16k–32k contexts. On L4, INT8 is required to fit `max_model_len=16384` with batching.

---

## `notebooks/dev.ipynb` — A100 optimized load (§7)

Used for dev-007 (16k generation) on Colab **A100 40 GB**. One-time engine init ~50 s (includes `torch.compile` ~33 s).

```python
with _jupyter_stdout_for_vllm():
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",                # ~7.6 GiB weights; faster than INT8 dequant on A100
        enable_prefix_caching=True,      # shared system-prompt prefix across items
        gpu_memory_utilization=0.92,
        max_model_len=32768,             # headroom for 16k outputs + long prompts
        trust_remote_code=True,
        max_num_seqs=128,
        max_num_batched_tokens=32768,    # match max_model_len for chunked prefill throughput
        enable_chunked_prefill=True,     # long prompts without OOM on prefill
    )
```

### Why each flag

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `dtype="bfloat16"` | no `quantization` | 4B model fits comfortably on 40 GB; avoids bitsandbytes load/dequant overhead. §1 env note: skip `bitsandbytes` on A100. |
| `max_model_len=32768` | 2× pub-001 context cap | Supports `MAX_TOKENS=16384` plus long CoT traces and chat template overhead. Observed KV capacity ~188k tokens pooled (vLLM log). |
| `gpu_memory_utilization=0.92` | vs 0.88 on L4 | A100 has more VRAM; v0.20+ also reserves CUDA-graph pool (~0.38 GiB logged). |
| `enable_prefix_caching=True` | on | MCQ/free-form share fixed system strings; prefix cache cuts prefill cost across 225+ items. |
| `enable_chunked_prefill=True` | on | Splits long prefills across steps — important when `max_model_len` is 32k. |
| `max_num_batched_tokens=32768` | = `max_model_len` | Scheduler can batch prefill/decode up to full context (vLLM recommendation when using chunked prefill). |
| `max_num_seqs=128` | vs 64 on L4 | Higher concurrency on 40 GB; dev slice is small enough that KV rarely saturates. |
| *(no `enforce_eager`)* | CUDA graphs on | Starter removed `enforce_eager=True` after it blocked graphs/compile; dev inherits that. First load runs `torch.compile` (compile range 1–32768). |
| `_jupyter_stdout_for_vllm()` | context manager | vLLM workers need real stdout FDs; Jupyter wrappers lack `fileno()`. See [`vllm-colab-l4.md` §4](vllm-colab-l4.md#4-jupyter-stdout--stderr-fix). |

### Observed A100 startup (dev-007 run, vLLM 0.20.0)

| Metric | Value |
|--------|------:|
| Weight memory | 7.61 GiB |
| Available KV cache | 25.86 GiB (~188k tokens) |
| `torch.compile` time | ~33 s (one-time) |
| CUDA graph pool | ~0.38 GiB |
| Max concurrency @ 32k seq | ~5.75× |

### OOM / fallback

- **L4 or &lt;32 GB:** use the **L4 profile** below (INT8, `max_model_len=16384`, lower `max_num_seqs`).
- **OOM at 32k on A100:** lower `max_model_len` to **16384**, set `max_num_batched_tokens=16384`, reduce `max_num_seqs` to **64**.
- **OOM during generation:** lower `MAX_TOKENS` in §2 (8192) before shrinking `max_model_len` if prompts are short.

---

## L4 profile — starter / full public

```python
llm = LLM(
    model=MODEL_ID,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    enable_prefix_caching=True,
    gpu_memory_utilization=0.88,
    max_model_len=16384,
    trust_remote_code=True,
    max_num_seqs=64,
    max_num_batched_tokens=16384,
)
```

| Parameter | Rationale |
|-----------|-----------|
| INT8 quant | Fits weights + KV on **24 GB** L4 with 16k context. |
| `max_model_len=16384` | Matches pub-001 `max_tokens=8192` with room for prompt + thinking. |
| `gpu_memory_utilization=0.88` | Leaves margin for CUDA-graph profiling overhead (vLLM 0.20+ logs suggest ~0.89 effective). |
| No `enable_chunked_prefill` in starter | Optional on L4; add if long-prompt prefill OOMs. |

`notebooks/full_public.ipynb` wraps load in `_jupyter_stdout_for_vllm()` like dev; starter may omit the wrapper on non-Jupyter runs.

---

## Generation caps vs engine context

| Notebook | `max_model_len` | `MAX_TOKENS` | Notes |
|----------|----------------:|-------------:|-------|
| starter | 16384 | 4096 | Original cap |
| full_public / pub-001 | 16384 | **8192** | Shipped baseline |
| dev (dev-007) | **32768** | **16384** | §1.1 truncation fix |
| dev (dev-009) | **32768** | **32768** | 32k ablation — no lift vs 16k ([run note](../log/runs/dev-009-max-tokens-32k.md)) |

`max_tokens` in `SamplingParams` must be **≤ `max_model_len`** minus prompt length. Engine `max_model_len` is the hard per-request ceiling; raising `MAX_TOKENS` alone does nothing if the engine cap is lower. At `max_model_len=32768`, `MAX_TOKENS=32768` consumes nearly the full window (prompt + generation); validated ceiling for gains is **16384** ([D008](../log/decisions.md#d008--32k-max_tokens-rejected-stay-at-16k)).

---

## Shared notebook plumbing

Set **before** `from vllm import LLM` (see [`vllm-colab-l4.md`](vllm-colab-l4.md)):

- `CUDA_VISIBLE_DEVICES`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`
- `_prepend_nvidia_libs_to_ld_path()`
- `_jupyter_stdout_for_vllm()` around `LLM()` and `llm.generate()` in dev / full_public

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-24 | Document dev A100 bf16 profile (`max_model_len=32k`, chunked prefill, prefix cache). |
| 2026-05-24 | dev-009: `MAX_TOKENS=32768` ablation — no gain vs 16k; keep 16k cap. |
| earlier | L4 INT8 profile in starter; `enforce_eager` removed, prefix caching enabled. |

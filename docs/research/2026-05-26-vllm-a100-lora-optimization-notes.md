# vLLM on A100 (base + LoRA): optimization research notes

Date: 2026-05-26  
Scope: identify GPU-side and engine-side optimizations not yet applied in this repo for `Qwen/Qwen3-4B-Thinking-2507` on A100, including LoRA evaluation path.

---

## Executive findings

The current A100 profile is already strong (bf16 weights, prefix caching, chunked prefill, CUDA graphs). The highest-value missing experiments are:

1. `kv_cache_dtype="fp8"` (memory + concurrency lever)
2. recalibrating `gpu_memory_utilization` under newer CUDA graph memory profiling behavior
3. right-sizing `max_lora_rank` to the adapter's true rank
4. sweeping `max_num_batched_tokens` for decode-heavy math traces
5. optional cold-start optimization via `safetensors_load_strategy="prefetch"`

---

## Current repo state (confirmed)

### Base model path (`notebooks/dev.ipynb`)

- `dtype="bfloat16"`
- `enable_prefix_caching=True`
- `enable_chunked_prefill=True`
- `gpu_memory_utilization=0.92`
- `max_num_seqs=128`
- `max_num_batched_tokens=32768`
- `max_model_len=17500` (current active cell) or 32768 in documented profile history

### LoRA eval path (`notebooks/sft_eval.ipynb`)

- base config above, plus:
  - `enable_lora=True`
  - `max_loras=1`
  - `max_lora_rank=64`

### Log evidence from current notebooks

- Engine reports `kv_cache_dtype=auto` (no explicit KV cache quantization)
- vLLM warning notes that with CUDA graph memory profiling enabled, effective KV budget is lower than nominal `gpu_memory_utilization=0.92`
- `sft_eval.ipynb` LoRA load logs indicate higher graph pool memory than base-only run (expected overhead)

---

## High-priority optimizations not yet applied

## 1) FP8 KV cache

**Change to test**

- add `kv_cache_dtype="fp8"` in `LLM(...)`

**Why**

- KV cache dominates VRAM at long context/generation lengths.
- FP8 KV cache can reduce KV memory footprint, often translating into higher stable concurrency or more headroom for long responses.

**Risk**

- possible small quality drift on mathematically exact tasks; requires direct holdout/watch-set validation.

---

## 2) Re-tune `gpu_memory_utilization` for new memory profiler behavior

**Change to test**

- sweep `gpu_memory_utilization` from `0.92 -> 0.93 -> 0.935 -> 0.94` (and `0.945` only if stable)

**Why**

- notebook logs explicitly state that the same nominal value now yields lower effective KV memory due to CUDA graph memory estimation.
- recovering that budget can reduce preemption and increase throughput.

**Guardrail**

- stop increasing if preemption/OOM appears or if run stability drops.

---

## 3) Right-size LoRA rank budget

**Change to test**

- set `max_lora_rank` to the maximum actual rank among served adapters, not a generic high value.

**Why**

- vLLM LoRA guidance warns that over-sizing this value wastes memory and compute.
- current eval hardcodes `max_lora_rank=64`; if adapter is rank 16/32, this is avoidable overhead.

**Implementation note**

- read adapter rank from `adapter_config.json` in checkpoint directory, then set `max_lora_rank` programmatically.

---

## 4) Sweep `max_num_batched_tokens`

**Change to test**

- compare `8192`, `16384`, `32768` (keep all else fixed)

**Why**

- for vLLM V1 + chunked prefill, this value sets prefill/decode scheduling balance:
  - smaller: often better inter-token latency
  - larger: often better TTFT/prefill throughput
- math reasoning workload can be decode-heavy; current `32768` may not be globally optimal.

---

## 5) Weight-loading startup optimization (optional)

**Change to test**

- `safetensors_load_strategy="prefetch"`

**Why**

- current logs show auto-prefetch disabled due to filesystem detection (`OVERLAY`), so explicit prefetch may reduce cold-start variance.
- affects startup time/QoL more than steady-state accuracy.

---

## LoRA-specific guidance for this repo

- Keep `max_loras=1` for current `sft_eval` single-adapter workflow.
- Do not increase `max_loras` unless multi-adapter concurrency is required.
- Keep base tokenizer unless adapter explicitly requires alternate tokenizer behavior.
- If adding multiple LoRAs later, set:
  - `max_lora_rank = max(actual_ranks)`
  - `max_loras = expected concurrent distinct adapters`

---

## Recommended experiment matrix (minimal, high signal)

All runs should keep prompt template and decoding policy fixed, then compare only targeted engine knobs.

| Run ID | LoRA | `kv_cache_dtype` | `gpu_memory_utilization` | `max_num_batched_tokens` | `max_lora_rank` | Goal |
|---|---:|---|---:|---:|---:|---|
| a100-opt-001 | no | auto | 0.92 | 32768 | n/a | baseline repeat |
| a100-opt-002 | no | fp8 | 0.92 | 32768 | n/a | isolate FP8 KV effect |
| a100-opt-003 | no | fp8 | 0.935 | 32768 | n/a | recover effective KV budget |
| a100-opt-004 | no | fp8 | 0.935 | 16384 | n/a | decode/prefill balance |
| a100-opt-005 | yes | auto | 0.92 | 32768 | actual rank | LoRA rank right-size baseline |
| a100-opt-006 | yes | fp8 | 0.935 | 16384 | actual rank | full optimized LoRA pass |

### Success criteria

- No meaningful regression on holdout overall, MCQ, FF, and watch sets.
- Lower preemption incidence.
- Better throughput (tokens/s or wall-clock for fixed eval set).
- Stable engine init and no OOM.

---

## Proposed candidate config (A100)

```python
llm = LLM(
    model=MODEL_ID,
    dtype="bfloat16",
    trust_remote_code=True,
    max_model_len=17500,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    kv_cache_dtype="fp8",                 # test vs auto
    gpu_memory_utilization=0.935,         # sweep around this
    max_num_seqs=128,
    max_num_batched_tokens=16384,         # sweep 8192/16384/32768
    safetensors_load_strategy="prefetch", # optional startup QoL
)
```

LoRA variant:

```python
llm = LLM(
    ...,
    enable_lora=True,
    max_loras=1,
    max_lora_rank=ACTUAL_ADAPTER_RANK,
)
```

---

## Source links

- Repo docs:
  - `docs/infra/vllm-inference-config.md`
  - `docs/infra/vllm-colab-l4.md`
  - `notebooks/dev.ipynb`
  - `notebooks/sft_eval.ipynb`
- vLLM upstream docs:
  - [Optimization and tuning](https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md)
  - [LoRA feature guide](https://github.com/vllm-project/vllm/blob/main/docs/features/lora.md)
  - [Engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)

# vLLM on Google Colab — install & notebook plumbing

Colab-specific **install**, **env vars**, and **Jupyter workarounds** for vLLM. For `LLM(...)` tuning (L4 INT8 vs A100 bf16), see [`vllm-inference-config.md`](vllm-inference-config.md).

This document summarizes changes in `starter_code_cse151b_comp.ipynb` (and shared cells in `notebooks/dev.ipynb`) needed to run **vLLM** with **Qwen3-4B-Thinking** on **Colab**, instead of a local `uv` workflow. Primary target: **L4 24 GB**; dev runs on **A100** use a different load profile (documented there).

---

## 1. Install path: Option B (`%pip`) instead of `uv`

- **Colab** uses the built-in kernel; skip the `uv` / `.venv` cells.
- **Why not plain `pip install vllm`:** the default wheel is often built for **CUDA 13** and expects `libcudart.so.13` on `LD_LIBRARY_PATH`. Colab’s image does not provide that layout the same way, which leads to load failures.
- **What works:** align **PyTorch** and **vLLM** on the **CUDA 12.9** stack Colab’s drivers support:
  - Install PyTorch from the **`cu129`** index:  
    `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129`
  - Install the official **vLLM `+cu129`** wheel (example pinned in the notebook: **v0.20.0**) from the vLLM GitHub releases, with `--extra-index-url https://download.pytorch.org/whl/cu129`.
- **Architecture:** the notebook uses the **manylinux x86_64** wheel. On **aarch64** Colab runtimes, swap `x86_64` for `aarch64` in the wheel URL.
- Other deps installed in the same cell: `sympy`, `numpy`, `tqdm`, `antlr4-python3-runtime==4.11.1`, `transformers`, `bitsandbytes>=0.48.1`.
- After install: **Runtime → Restart runtime** once so `vllm` / `transformers` load cleanly.

---

## 2. Environment variables for vLLM + Jupyter

Set **before** importing `vllm`:

| Variable | Value | Purpose |
|----------|--------|---------|
| `CUDA_VISIBLE_DEVICES` | `"0"` | Single GPU (L4); use another index only if a second GPU exists. |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `"0"` | vLLM V1’s default **EngineCore subprocess** often breaks inside **Jupyter/Colab**. Disabling multiprocessing keeps execution **in-process**, which is stable in notebooks. |

---

## 3. `LD_LIBRARY_PATH`: NVIDIA wheels under `site-packages`

Colab/pip installs ship CUDA libraries under paths like  
`<site-packages>/nvidia/*/lib`. Those dirs are not always on the loader path.

The notebook defines **`_prepend_nvidia_libs_to_ld_path()`**, which:

- Collects `site.getsitepackages()`, optional user site, and
- For each `.../site-packages/nvidia/*/lib` directory, prepends it to **`LD_LIBRARY_PATH`**.

This avoids missing `.so` errors when vLLM/torch load CUDA components.

---

## 4. Jupyter `stdout` / `stderr` fix

In Jupyter, **`sys.stdout`** can be replaced with a wrapper that has **no `fileno()`**. vLLM worker code expects a real file descriptor.

The notebook wraps stdout/stderr with **`os.fdopen(1, ...)`** and **`os.fdopen(2, ...)`** when `sys.stdout.fileno()` fails (`UnsupportedOperation` / etc.), so workers see valid FDs.

---

## 5. `LLM(...)` constructor — see inference profiles

**Full parameter tables, A100 vs L4 rationale, and OOM fallbacks:** [`vllm-inference-config.md`](vllm-inference-config.md).

**L4 / starter** (`starter_code_cse151b_comp.ipynb`, `full_public.ipynb`): INT8 bitsandbytes, `max_model_len=16384`, `gpu_memory_utilization=0.88`, prefix caching on, **`enforce_eager` off** (CUDA graphs + compile enabled).

**A100 / dev** (`notebooks/dev.ipynb` §7): native **bf16**, `max_model_len=32768`, `enable_chunked_prefill=True`, higher batch limits — for 16k `MAX_TOKENS` runs without quantization overhead.

---

## 6. Google Drive workflow (data + judger + results)

Colab VMs do not include your git tree unless you clone or upload.

- **Data:** mount Drive and **`copy`** `public.jsonl` from e.g.  
  `/content/drive/MyDrive/CSE151B/data/public.jsonl` → `data/public.jsonl`.
- **`DRIVE_PROJECT_ROOT`:** parent of `data/` (e.g. `MyDrive/CSE151B`) used as:
  - **`sys.path`** root to **`import judger`** (copy `judger.py` next to your Drive project folder, or set `CSE151B_JUDGER_DIR`).
  - **Output directory** for JSONL: `DRIVE_PROJECT_ROOT/results/<filename>`.

---

## 7. Quick checklist (Colab L4)

1. **GPU runtime** (T4/L4/A100 — L4 is fine with this stack).
2. Run **Option B** `%pip` cell → **restart runtime**.
3. Run **imports/config** (env vars, `LD_LIBRARY_PATH`, stdout fix) **before** `from vllm import ...`.
4. Run **Drive mount + copy** if you need `public.jsonl` / `judger.py` from Drive.
5. Load model, then generate/score/save as in the notebook.

---

## References in the notebook

- Comments in the **Colab `%pip`** cell explain the **cu129** wheel choice vs default CUDA 13 vLLM wheels.
- Section **“2. Imports & Configuration”** cells document `GPU_ID`, `VLLM_ENABLE_V1_MULTIPROCESSING`, `_prepend_nvidia_libs_to_ld_path`, and the **stdout** workaround inline.

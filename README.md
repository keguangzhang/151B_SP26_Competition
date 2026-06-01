# CSE 151B Competition — Final submission

Math reasoning on **`data/private.jsonl`** with **`Qwen/Qwen3-4B-Thinking-2507`**, vLLM, and the **`exact_v1`** prompt variant (adaptive MCQ / single-blank / multi-blank layout). Full development notebooks live under [`notebooks/`](notebooks/); the graded reproduction path is **`run_inference()`** below.

## Hardware and runtime

| Item | Value |
|------|--------|
| **GPU** | NVIDIA **A100 40GB** (Google Colab) |
| **Weights** | bf16 (~7.6 GiB), FP8 KV cache |
| **Engine init** | ~90 s first run (includes vLLM `torch.compile` warmup); ~25 s on a warm cache |
| **Generation** | ~35–45 min for **943** private questions at **16k** `max_tokens`, batch size 128 |
| **Total** | ~**40–55 min** end-to-end on A100 |

No manual weight download is required: Hugging Face Hub caches under `~/.cache/huggingface/hub/` on first `run_inference()` call.

**Fine-tuning:** This submission uses the **base** checkpoint only (no LoRA at inference). If you add a fine-tune, push merged weights to the Hub and set `MODEL_ID` in [`run_inference.py`](run_inference.py).

## Environment setup

**Requirements:** Linux x86_64, CUDA 12.x GPU with ≥24 GB VRAM (A100 recommended for 16k context), Python 3.10+.

```bash
cd /path/to/151B_SP26_Competition
python -m venv .venv && source .venv/bin/activate

pip install sympy numpy tqdm "antlr4-python3-runtime==4.11.1"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
pip install "https://github.com/vllm-project/vllm/releases/download/v0.20.0/vllm-0.20.0+cu129-cp38-abi3-manylinux_2_31_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu129
pip install transformers
```

Optional: `export HF_TOKEN=...` for faster Hub downloads.

Place competition data at **`data/private.jsonl`** (included in this repo).

## Reproduce the submission CSV

### Python API

```python
from run_inference import run_inference

csv_path = run_inference()  # -> results/submission_exact_v1_16k.csv
print(csv_path)
```

### Command line

```bash
python run_inference.py
# Fresh run (no checkpoint resume):
python run_inference.py --no-resume
# Custom paths:
python run_inference.py --data data/private.jsonl --output results/submission.csv
```

**Output:** `results/submission_exact_v1_16k.csv` with header `id,response` (full model traces, RFC 4180 quoting). Checkpoint: `results/submission_exact_v1_16k.responses.jsonl`.

## Frozen hyperparameters

Defined at the top of [`run_inference.py`](run_inference.py) and in [`inference/prompts.py`](inference/prompts.py):

| Parameter | Value |
|-----------|--------|
| Model | `Qwen/Qwen3-4B-Thinking-2507` |
| Prompt variant | `exact_v1` (per-question MCQ vs multi-blank layout) |
| `max_tokens` | 16384 |
| `max_model_len` | 18432 (`max_tokens` + 2048 prompt budget) |
| `temperature` / `top_p` / `top_k` | 0.6 / 0.95 / 20 |
| `dtype` | bfloat16 |
| `kv_cache_dtype` | fp8 |
| `gpu_memory_utilization` | 0.88 |
| Batch chunk | 128 prompts per vLLM call |

## Repository layout

| Path | Role |
|------|------|
| [`run_inference.py`](run_inference.py) | **Single entry point** — load model, infer, write CSV |
| [`inference/prompts.py`](inference/prompts.py) | `exact_v1` system/user prompts |
| [`notebooks/submission.ipynb`](notebooks/submission.ipynb) | Colab notebook used to produce the leaderboard file |
| [`judger.py`](judger.py) | Local grading / answer extraction (public dev only) |
| [`data/private.jsonl`](data/private.jsonl) | Unlabeled test set |
| [`docs/README.md`](docs/README.md) | Experiment log and analysis index |

## Starter and development

- Course starter: [`starter_code_cse151b_comp.ipynb`](starter_code_cse151b_comp.ipynb)
- Fast iteration: [`notebooks/dev.ipynb`](notebooks/dev.ipynb)
- Agent / constraint summary: [`AGENTS.md`](AGENTS.md)

## AI Usage Disclosure

Generative AI tools were used during the development process to assist with programming support, debugging, software troubleshooting, and understanding machine learning concepts related to large language model inference and fine-tuning methods.

The final submissions consisted of:

1. A baseline inference pipeline using the designated base model.
2. A prompt-engineered variant of the same model.

The design, implementation, evaluation, prompt development, experiment selection, and final submission decisions were made by the team members. AI-generated suggestions were reviewed and validated by the team before use.

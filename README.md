# CSE 151B Competition — Starter Code

Open **`starter_code_cse151b_comp.ipynb`** at the repo root to get started (environment setup, inference with Qwen3-4B-Thinking INT8, scoring on the public set).

Experimentation and submission notebooks live under **`notebooks/`**:

| Notebook | Purpose |
|----------|---------|
| `notebooks/dev.ipynb` | Stratified dev slice of `public.jsonl`, faster iteration |
| `notebooks/submission.ipynb` | Full `private.jsonl` → `results/submission.csv` |

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Main entry point (repo root) |
| `notebooks/dev.ipynb` | Dev split / experiments |
| `notebooks/submission.ipynb` | Private-set inference → leaderboard CSV |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output JSONL / CSV files written at runtime |

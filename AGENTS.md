# Agent instructions — CSE 151B math reasoning competition

This repository targets the UCSD CSE 151B competition: maximize **unified accuracy** on mathematical reasoning using a **single mid-sized open-weight model** (~4B parameters), without delegating inference to larger models or external tooling.

## Hard constraints (must not violate)

- **No external calls at inference time:** no remote APIs, no retrieval against live services, no Python REPL/symbolic calculator **as tools invoked during generation** for the submission path. The official evaluation extracts answers from the model’s **raw text trace**; generation must be self-contained model output.
- **No tool-augmented test-time generation** in the sense described by the competition (e.g., wiring a code interpreter or calculator into the generation loop for the final submission). Training-time data prep and offline experiments may use normal development tools; the **leaderboard submission** is model-only generations per problem.
- **Submission artifact:** a CSV with every `id` from `data/private.jsonl`, column `response` = **full** model output (chain-of-thought / thinking + final answer). Answers are **parsed from** this trace during grading — do not submit “answer only” unless the pipeline still stores the full trace somewhere compliant with rules.

Clarify course-specific model locks with the course staff if they differ from the starter (the starter uses **Qwen3-4B-Thinking** in INT8 in the notebook).

## Evaluation summary

- **Metric:** unified accuracy = total correct / total questions across all benchmarks; **equal weight per question** regardless of subset or difficulty.
- **Splits:** `data/public.jsonl` includes labels for development; `data/private.jsonl` has **no** answers and drives leaderboard / finals. Public and private are **distribution-aligned** (domains, difficulty mix, formats); leaderboard uses ~30% of private until finals.

## Data format (JSONL)

Each line is one JSON object:

| Field | Meaning |
|--------|--------|
| `id` | Unique integer |
| `question` | LaTeX problem text; free-form uses `[ANS]` placeholders |
| `answer` | Free-form: list of strings (one per `[ANS]`). MCQ: single capital letter |
| `options` | MCQ only: list of LaTeX choices |

**Free-form:** every `[ANS]` sub-answer must be correct for the item to score.

**Multiple-choice:** graded by matching the selected letter to ground truth.

## Submission CSV

Required header:

```text
id,response
```

- `response` must be **properly CSV-escaped** (quoted field; double quotes doubled inside).
- Include **complete** traces — not stripped reasoning.
- Free-form with multiple placeholders: final extraction must yield **all** sub-answers correct (e.g. `\boxed{a,b,c}` style per competition/starter conventions).

## Repository map

| Path | Role |
|------|------|
| `starter_code_cse151b_comp.ipynb` | Official starter (repo root): setup, inference, public scoring |
| `notebooks/dev.ipynb` | Local development / stratified dev slice |
| `notebooks/sft_train.ipynb` | QLoRA SFT on `data/sft_corpus.jsonl` (smoke + full run; Drive checkpoints) |
| `notebooks/sft_eval.ipynb` | SFT checkpoint eval on frozen `data/eval/holdout.jsonl` (LoRA + extended metrics) |
| `notebooks/submission.ipynb` | Full `private.jsonl` inference → `results/submission.csv` |
| `judger.py` | Grading logic aligned with the competition extractors |
| `utils.py` | Shared normalization/utilities for `judger.py` |
| `data/public.jsonl` | Labeled development set |
| `data/eval/holdout.jsonl` | Frozen stratified eval holdout from public (225 rows, seed 42) |
| `data/eval/geometry_dev.jsonl` | Full-public geometry dev slice (~133 rows; sft-005 primary eval) |
| `data/eval/sequences_dev.jsonl` | Full-public sequences/recurrences dev slice (~74 rows; sft-006 primary eval) |
| `data/eval/watch_manifest.json` | Frozen watch sets: geometry, sequences, Q4-long (30 ids), multi-blank ≥3 (20 ids) |
| `scripts/build_eval_holdout.py` | Build holdout from `public.jsonl` |
| `scripts/build_eval_geometry_set.py` | Build `geometry_dev.jsonl` from `public.jsonl` |
| `scripts/build_eval_sequences_set.py` | Build `sequences_dev.jsonl` from `public.jsonl` |
| `scripts/build_eval_watch_sets.py` | Build watch JSONLs + manifest from holdout |
| `scripts/eval_watch.py` | Filter/score results by watch set name |
| `data/private.jsonl` | Unlabeled test IDs for submission |
| `results/` | Runtime outputs (e.g. JSONL predictions) |
| `docs/README.md` | Docs index, current best scores, layout |
| `docs/log/experiments.md` | Experiment registry (all runs) |
| `docs/log/decisions.md` | Decision log for report / rationale |
| `docs/roadmap.md` | Forward-looking improvement priorities |
| `docs/report-outline.md` | Final report skeleton with links |
| `docs/analysis/baseline-public-8k.md` | Full-public 8k baseline error analysis |
| `docs/sft/pipeline.md` | SFT run plan |
| `docs/infra/vllm-colab-l4.md` | Colab install, env vars, Jupyter vLLM quirks |
| `docs/infra/vllm-inference-config.md` | `LLM(...)` profiles: L4 INT8 (starter) vs A100 bf16 (dev) |

## Allowed improvement directions (high level)

- Inference: prompting (CoT, few-shot, self-consistency, progressive hints), decoding choices, token budgets, constrained decoding where compatible with the stack.
- Training: supervised fine-tuning (LoRA / QLoRA / full), RL-style alignment (GRPO, DPO, outcome rewards, etc.) on **public** data and other permitted public datasets — subject to course rules.

## Conventions for AI contributors

- Prefer changing **`notebooks/dev.ipynb`** for exploratory work; keep **`starter_code_cse151b_comp.ipynb`** aligned with what you intend to submit or share unless the user asks otherwise.
- After logic changes that affect extraction or formatting, re-run scoring on **`data/public.jsonl`** using the same code path as the notebook / `judger.py`.
- Do not commit secrets, API keys, or private leaderboard tokens.
- Match existing code style in `judger.py` / `utils.py`: minimal churn, no unrelated refactors.

## Upstream starter

Course materials reference: [151B_SP26_Competition](https://github.com/brooksniu/151B_SP26_Competition) — use for comparing against the canonical starter if this fork diverges.

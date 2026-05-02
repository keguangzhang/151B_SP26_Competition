# Claude — CSE 151B competition workspace

Read **`AGENTS.md`** first. It is the canonical contract for constraints, data layout, submission format, and repo structure.

## Quick facts

- **Goal:** Maximize **unified accuracy** on math reasoning with a **single ~4B** thinking-style model; **no** external APIs or tool-augmented **inference** for the graded submission path.
- **Train/dev:** `data/public.jsonl` (labels). **Submit predictions for:** `data/private.jsonl` (no labels).
- **Output:** CSV `id,response` with **full** model traces; graders extract `\boxed{…}` / letters from `response`.
- **Scoring reference:** `judger.py` + `utils.py`; notebooks wire inference + evaluation.

## Where to work

| Task | Start here |
|------|------------|
| Reproduce baseline | `starter_code_cse151b_comp.ipynb` |
| Experiments / iteration | `notebooks/dev.ipynb` |
| Private CSV submission | `notebooks/submission.ipynb` |
| Understanding grades | `judger.py` |
| Empirical priorities (truncation, MCQ, etc.) | `docs/improvement-directions.md` |

Notebooks under `notebooks/` set **`REPO_ROOT`** so `data/` and `results/` resolve whether the kernel cwd is the repo root or `notebooks/`.

## Editing discipline

- Keep diffs **focused** on the requested change; don’t refactor `judger.py` unless fixing correctness or the user asks.
- Notebook outputs: clear heavy outputs before commit if the user cares about repo size (follow `.gitignore`).
- LaTeX in questions uses `[ANS]` placeholders and `\boxed{}` in model outputs — preserve escaping when generating strings or CSV.

## Clarifications

If course policy **fixes** a different checkpoint or forbids certain training data, **course instructions override** generic ML advice in docs.

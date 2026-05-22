# Competition constraints (reference)

Canonical rules live in [`AGENTS.md`](../AGENTS.md) at the repo root. Summary for report writing:

- **Model:** `Qwen/Qwen3-4B-Thinking-2507` — fine-tune allowed, no swap to a different base model for submission unless course staff say otherwise.
- **Inference:** model-only generation; no external APIs, retrieval, or tool loops (calculator / REPL) in the graded path.
- **Submission:** CSV `id,response` with **full** traces; graders extract `\boxed{…}` / MCQ letters from `response`.
- **Metric:** unified accuracy = correct / total; equal weight per question (MCQ + free-form).
- **Data:** `data/public.jsonl` for dev (labeled); `data/private.jsonl` for leaderboard (unlabeled).

Grading implementation: `judger.py`, `utils.py`.

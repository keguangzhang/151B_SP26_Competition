# Final report outline

Fill each section with links to registry rows and analysis — avoid copying metric tables here. Update as experiments ship.

---

## 1. Introduction

- Problem: unified math reasoning accuracy on competition benchmarks.
- Constraints: [`reference/constraints.md`](reference/constraints.md), [`AGENTS.md`](../AGENTS.md).

## 2. Baseline system

- Model and stack: Qwen3-4B-Thinking, vLLM — L4 INT8 ([`infra/vllm-colab-l4.md`](infra/vllm-colab-l4.md)); A100 bf16 dev profile ([`infra/vllm-inference-config.md`](infra/vllm-inference-config.md)).
- Evaluation protocol: `judger.py`, dev slice definition — [`log/runs/dev-001-baseline.md`](log/runs/dev-001-baseline.md).
- **Shipped baseline:** [pub-001](log/experiments.md#pub-001) — [`analysis/baseline-public-8k.md`](analysis/baseline-public-8k.md).

## 3. Error analysis (baseline)

- MCQ: format vs truncation — analysis § MCQ format compliance.
- Free-form: multi-blank gap, trace length — analysis § weaknesses.
- Topic weaknesses: number theory, sequences, geometry — analysis topic table.
- Link: [`analysis/baseline-public-8k.md`](analysis/baseline-public-8k.md).

## 4. Inference experiments

- Summary table: [`log/experiments.md`](log/experiments.md) (dev + pub rows).
- Key decisions: [D001](log/decisions.md#d001) (8k tokens), [D002](log/decisions.md#d002) (§1.3), [D003](log/decisions.md#d003) (guided decoding).
- Per-run narrative: [`log/runs/`](log/runs/).
- Open ideas not yet run: [`roadmap.md`](roadmap.md).

## 5. Training (SFT)

- Decision to start Numina-only: [D004](log/decisions.md#d004).
- Procedure and hyperparameters: [`sft/pipeline.md`](sft/pipeline.md).
- Data contract: [`sft/data-spec.md`](sft/data-spec.md).
- Data QA: [`sft/data-issues.md`](sft/data-issues.md).
- Results: *(add `sft-001` row to experiments when complete)*.

## 6. Results and comparison

- Best model config: [`README.md`](README.md) — Current best.
- Leaderboard / private: *(priv-001 when submitted)*.

## 7. Discussion

- What worked / failed vs hypotheses in [`roadmap.md`](roadmap.md).
- Limitations: model size, INT8, dev slice noise, data license.

## 8. Conclusion and future work

- Unfinished roadmap items (RL, self-consistency, BF16, …).
- Backlog: [`roadmap.md`](roadmap.md) § open questions.

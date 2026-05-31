"""OpenMathReasoning row qualification for SFT corpus builders (sft-003/005/006/007)."""

from __future__ import annotations

import re
from typing import Any, Optional

from scripts.build_sft_corpus import normalize_question_for_overlap
from scripts.sft_prompt import (
    THINKING_CLOSE,
    THINKING_OPEN,
    is_figure_dependent,
    split_thinking_response,
    wrap_thinking_response,
)
from utils import last_boxed_only_string, remove_boxed

FREEFORM_FINAL_RE = re.compile(r"^\\boxed\{.+}$")
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")
DECONTAM_PREFIX_CHARS = 200
THINKING_TEMPLATE = "explicit_redacted_thinking"


def last_non_empty_line(text: str) -> str:
    for line in reversed(text.rstrip().splitlines()):
        if line.strip():
            return line.strip()
    return ""


def extract_boxed_answer(response: str) -> Optional[str]:
    boxed = last_boxed_only_string(response)
    if boxed is None:
        return None
    return remove_boxed(boxed)


def normalize_freeform_response(solution: str) -> tuple[Optional[str], Optional[str]]:
    text = solution.strip()
    if not text:
        return None, None
    answer = extract_boxed_answer(text)
    if answer is None:
        return None, None
    boxed = last_boxed_only_string(text)
    assert boxed is not None
    idx = text.rfind(boxed)
    prefix = text[:idx].rstrip()
    final_line = boxed.strip()
    response = f"{prefix}\n{final_line}" if prefix else final_line
    return response, answer.strip()


def validate_final_line(response: str, task_type: str) -> bool:
    line = last_non_empty_line(response)
    if task_type == "mcq":
        return bool(re.match(r"^\\boxed\{[A-J]\}$", line))
    return bool(FREEFORM_FINAL_RE.match(line))


def contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def has_thinking_wrapper(response: str) -> bool:
    return THINKING_OPEN in response and THINKING_CLOSE in response


def assemble_thinking_response(reasoning: str, final_line: str) -> str:
    parts = [THINKING_OPEN]
    if reasoning.strip():
        parts.append(reasoning.strip())
    parts.append(THINKING_CLOSE)
    parts.append("")
    parts.append(final_line.strip())
    return "\n".join(parts)


def format_openr1_response(raw_trace: str) -> tuple[Optional[str], Optional[str]]:
    trace = raw_trace.strip()
    if not trace:
        return None, None

    has_wrapper = THINKING_OPEN in trace and THINKING_CLOSE in trace
    if not has_wrapper:
        norm, answer = normalize_freeform_response(trace)
        if norm is None or answer is None:
            return None, None
        final_line = last_non_empty_line(norm)
        body = norm.rstrip()
        reasoning = body[: -len(final_line)].rstrip() if body.endswith(final_line) else body
        return assemble_thinking_response(reasoning, final_line), answer.strip()

    reasoning, after_close = split_thinking_response(trace)
    after_close = after_close.strip()
    boxed_after = extract_boxed_answer(after_close) if after_close else None
    boxed_inside = extract_boxed_answer(reasoning) if reasoning else None

    if boxed_after is not None:
        final_line = last_non_empty_line(after_close)
        if not FREEFORM_FINAL_RE.match(final_line):
            boxed = last_boxed_only_string(after_close)
            if boxed is None:
                return None, None
            final_line = boxed.strip()
        response = assemble_thinking_response(reasoning, final_line)
        answer = extract_boxed_answer(final_line)
        return (response, answer.strip()) if answer else (None, None)

    if boxed_inside is not None:
        boxed = last_boxed_only_string(reasoning)
        if boxed is None:
            return None, None
        idx = reasoning.rfind(boxed)
        reasoning_clean = reasoning[:idx].rstrip() if idx >= 0 else reasoning
        final_line = boxed.strip()
        response = assemble_thinking_response(reasoning_clean, final_line)
        return response, extract_boxed_answer(final_line)

    return None, None


def load_decontam_prefixes(*paths, prefix_chars: int = DECONTAM_PREFIX_CHARS) -> list[str]:
    from pathlib import Path

    prefixes: list[str] = []
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        with open(p) as f:
            for line in f:
                norm = normalize_question_for_overlap(
                    __import__("json").loads(line)["question"][:prefix_chars]
                )
                if len(norm) >= 20:
                    prefixes.append(norm)
    return prefixes


def decontam_hit(
    problem: str,
    prefixes: list[str],
    competition_keys: set[str],
) -> bool:
    p_full = normalize_question_for_overlap(problem)
    p_prefix = normalize_question_for_overlap(problem[:DECONTAM_PREFIX_CHARS])
    if p_full in competition_keys:
        return True
    for pref in prefixes:
        if pref in p_full or p_prefix in pref or pref in p_prefix:
            return True
    return False


def render_training_messages(
    question: str,
    options: Optional[list],
    response: str,
) -> list[dict[str, str]]:
    from scripts.sft_prompt import build_prompt_baseline

    system, user = build_prompt_baseline(question, options)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": response},
    ]


def count_template_tokens(tokenizer, messages: list[dict[str, str]]) -> int:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def qualify_openmath_ex(
    ex: dict[str, Any],
    *,
    idx: int,
    source: str,
    source_id_prefix: str,
    decontam_prefixes: list[str],
    competition_keys: set[str],
    min_response_chars: int,
    max_response_chars: int,
    min_pass_rate: float,
    max_pass_rate: float,
    tokenizer=None,
    max_template_tokens: int = 7900,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    reject_base = {"source": source, "source_id": f"{source_id_prefix}:{idx}"}

    problem = (ex.get("problem") or "").strip()
    if not problem:
        return None, {**reject_base, "reason": "empty_problem"}

    solution = (ex.get("generated_solution") or "").strip()
    if not solution:
        return None, {**reject_base, "reason": "empty_solution"}

    pass_rate = ex.get("pass_rate_72b_tir")
    try:
        pass_rate = float(pass_rate)
    except (TypeError, ValueError):
        pass_rate = None
    if pass_rate is None or not (min_pass_rate <= pass_rate <= max_pass_rate):
        return None, {**reject_base, "reason": "pass_rate_out_of_range", "pass_rate": pass_rate}

    raw_len = len(solution)
    if not (min_response_chars <= raw_len <= max_response_chars):
        return None, {**reject_base, "reason": "response_length_out_of_range", "raw_len": raw_len}

    if contains_cjk(problem) or contains_cjk(solution):
        return None, {**reject_base, "reason": "cjk_text", "question_preview": problem[:200]}
    if is_figure_dependent(problem) or is_figure_dependent(solution):
        return None, {**reject_base, "reason": "figure_dependent", "question_preview": problem[:200]}
    if decontam_hit(problem, decontam_prefixes, competition_keys):
        return None, {**reject_base, "reason": "decontam", "question_preview": problem[:200]}

    response, answer = format_openr1_response(solution)
    if response is None or answer is None or not str(answer).strip():
        return None, {**reject_base, "reason": "bad_format", "solution_preview": solution[:200]}
    if not has_thinking_wrapper(response):
        return None, {**reject_base, "reason": "missing_thinking_wrapper"}
    if len(response) < min_response_chars:
        return None, {**reject_base, "reason": "short_response_after_format", "trace_chars": len(response)}
    if not validate_final_line(response, "freeform"):
        return None, {**reject_base, "reason": "bad_final_line", "response_tail": response[-120:]}

    template_tokens = 0
    if tokenizer is not None:
        messages = render_training_messages(problem, None, response)
        template_tokens = count_template_tokens(tokenizer, messages)
        if template_tokens > max_template_tokens:
            return None, {
                **reject_base,
                "reason": "too_long_tokens",
                "template_tokens": template_tokens,
            }

    return {
        "source": source,
        "source_id": f"{source_id_prefix}:{idx}",
        "task_type": "freeform",
        "question": problem,
        "options": None,
        "response": response,
        "answer": answer.strip(),
        "thinking_template": THINKING_TEMPLATE,
        "trace_chars": len(response),
        "template_tokens": template_tokens,
        "pass_rate_72b_tir": float(pass_rate),
    }, None

"""Prompt and answer helpers for SFT corpus builders (matches full_public / pub-002)."""

from __future__ import annotations

import re
from typing import Optional

THINKING_OPEN = "<think>"
THINKING_CLOSE = "</think>"

_MATH_BASELINE = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

_MCQ_BASELINE = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

_MATH_MULTI_BLANK = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "For problems with multiple [ANS] placeholders, put each sub-answer in its own "
    "\\boxed{}, separated by commas, in the order the blanks appear "
    "(e.g. \\boxed{3}, \\boxed{7}). Do not use labels like 'Answer 1:' between boxes. "
    "For single-answer problems, use one \\boxed{}."
)

FIGURE_RE = re.compile(
    r"\\begin\{asy\}|\\begin\{tikzpicture\}|in the figure shown|in the diagram|"
    r"shown in the figure|as shown in the figure|see the figure",
    re.I,
)
GEOMETRY_RE = re.compile(
    r"\b(triangle|polygon|angle|circle|quadrilateral|parallelogram|"
    r"perpendicular|parallel|inradius|circumradius|circumcircle)\b",
    re.I,
)
CONTIGUOUS_BOXED_GAP_RE = re.compile(r"^[\s,\$\.\;\:\-\&\\]*$")


def n_ans_placeholders(question: str) -> int:
    return question.count("[ANS]")


def build_prompt_baseline(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return _MCQ_BASELINE, f"{question}\n\nOptions:\n{opts_text}"
    return _MATH_BASELINE, question


def build_prompt_multi_blank(question: str, options: Optional[list]) -> tuple[str, str]:
    """Inference-time multi-blank path (pub-002 / full_public.ipynb)."""
    if options:
        return build_prompt_baseline(question, options)
    n_blanks = n_ans_placeholders(question)
    if n_blanks <= 1:
        return build_prompt_baseline(question, options)
    example = ", ".join("\\boxed{...}" for _ in range(n_blanks))
    user = (
        f"{question}\n\n"
        f"The problem has {n_blanks} [ANS] blanks. "
        f"After your reasoning, give {n_blanks} comma-separated \\boxed{{}} values "
        f"in order: {example}"
    )
    return _MATH_MULTI_BLANK, user


def is_figure_dependent(text: str) -> bool:
    return bool(FIGURE_RE.search(text))


def is_geometry_flavored(text: str) -> bool:
    return bool(GEOMETRY_RE.search(text))


def split_thinking_response(response: str) -> tuple[str, str]:
    if THINKING_OPEN in response and THINKING_CLOSE in response:
        start = response.index(THINKING_OPEN) + len(THINKING_OPEN)
        end = response.index(THINKING_CLOSE)
        reasoning = response[start:end].strip()
        final = response[end + len(THINKING_CLOSE) :].strip()
        return reasoning, final
    lines = response.rstrip().splitlines()
    if not lines:
        return "", ""
    final = lines[-1].strip()
    reasoning = "\n".join(lines[:-1]).strip()
    return reasoning, final


def wrap_thinking_response(reasoning: str, final_line: str) -> str:
    parts = [THINKING_OPEN]
    if reasoning:
        parts.append(reasoning)
    parts.append(THINKING_CLOSE)
    parts.append("")
    parts.append(final_line)
    return "\n".join(parts)


def extract_all_boxed(text: str) -> list[str]:
    """Last contiguous \\boxed{} group (judger-compatible, no sympy)."""
    entries: list[tuple[int, int, str]] = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth = 1
        i = brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            content = text[brace_start : i - 1]
            if content:
                entries.append((idx, i, content.strip()))
        start = i

    if not entries:
        return []

    last_group = [entries[-1]]
    for j in range(len(entries) - 2, -1, -1):
        gap = text[entries[j][1] : entries[j + 1][0]]
        if CONTIGUOUS_BOXED_GAP_RE.match(gap):
            last_group.insert(0, entries[j])
        else:
            break
    return [e[2] for e in last_group]


def final_section(response: str) -> str:
    if THINKING_CLOSE in response:
        return response.split(THINKING_CLOSE)[-1]
    return response


def multi_boxed_final_line(answers: list[str]) -> str:
    return ", ".join(f"\\boxed{{{a}}}" for a in answers)


def ensure_ans_placeholders(question: str, n_blanks: int) -> str:
    """Add trailing 'Answer: [ANS]' lines until placeholder count reaches n_blanks."""
    q = question.strip()
    have = n_ans_placeholders(q)
    if have >= n_blanks:
        return q
    extra = n_blanks - have
    suffix = "\n".join(f"Answer: [ANS]" for _ in range(extra))
    return f"{q}\n\n{suffix}" if q else suffix

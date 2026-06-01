"""exact_v1 prompts — mirrors notebooks/submission.ipynb §5 and notebooks/dev.ipynb §6."""

from __future__ import annotations

from typing import Optional

_ROLE_SOLVE = (
    "You are a precise mathematical reasoner. Solve the problem rigorously, but keep "
    "your reasoning concise: do not re-derive steps you have already completed or go "
    "in circles."
)
_ROLE_MCQ = (
    "You are a precise mathematical reasoner. Read the problem and the answer choices, "
    "then identify the single best answer. Keep your reasoning concise; you may confirm "
    "by elimination."
)

_PRECISION_RULE = (
    "Answer precision is graded against a rounded reference with a very tight tolerance, "
    "so the FORM of your answer matters:\n"
    "- Prefer an EXACT closed form whenever one exists - fractions (5/8), radicals "
    "(3\\sqrt{2}), \\pi, e, logarithms (\\log_3 35). Box it exactly; the grader matches "
    "exact forms reliably and they carry no rounding error.\n"
    "- Use a decimal ONLY when no exact form exists (e.g. a statistical test statistic, "
    "p-value, regression coefficient, or physical measurement). Then use the CONVENTIONAL "
    "precision for the problem: standard statistical constants in their usual rounded form "
    "(z = 1.96, 1.645, 2.576; the usual t-table values), and otherwise 2-4 decimal places, "
    "or the precision of the data you were given.\n"
    "- NEVER pad an answer with extra significant figures. An over-precise decimal such as "
    "1.6448536270 will FAIL against a rounded gold value like 1.645. Do not round a true "
    "exact form to a decimal, and do not over-expand a conventional decimal."
)

_REPRESENTATION_RULE = (
    "Give the answer in the simplest standard form and do not rewrite it into a fancier "
    "equivalent (keep \\log_3 35 rather than \\frac{\\ln 35}{\\ln 3}; keep a clean factored "
    "or simplified expression). For True/False or Yes/No answers, use exactly the words the "
    "question uses - write True / False when it says 'True or False', and Yes / No when it "
    "asks a yes/no question."
)

_FORMAT_RULE = (
    "Box the value only - no units or surrounding words (\\boxed{5}, not \\boxed{5 cm}). "
    "Write a probability as a decimal (0.5, not 50%) and a percentage as a bare number "
    "(50, no % sign). Keep any coordinate, tuple, interval, or set together with its "
    "brackets, e.g. \\boxed{(-1, -3)} or \\boxed{[0, 14]}, so its internal commas are not "
    "mistaken for answer separators."
)

_LAYOUT_SINGLE = (
    "End with a single line 'Final Answer:' followed by exactly one \\boxed{} containing "
    "your answer. Put \\boxed{} nowhere else in the response."
)
_LAYOUT_MCQ = (
    "End with a single line 'Final Answer:' followed by exactly one \\boxed{} containing "
    "ONLY the letter of your chosen option (e.g. \\boxed{C}). If the slot says 'select all "
    "that apply', concatenate the letters alphabetically inside that one box with no "
    "separator (e.g. \\boxed{AB}). Put \\boxed{} nowhere else in the response."
)

SYSTEM_PROMPT_MATH = "\n\n".join(
    [_ROLE_SOLVE, _PRECISION_RULE, _REPRESENTATION_RULE, _FORMAT_RULE]
)
SYSTEM_PROMPT_MCQ = "\n\n".join([_ROLE_MCQ, _REPRESENTATION_RULE])

PROMPT_VARIANT = "exact_v1"


def n_ans_placeholders(question: str) -> int:
    return question.count("[ANS]")


def _math_layout_rule(n_blanks: int) -> str:
    if n_blanks > 1:
        return (
            f"This problem has {n_blanks} blanks ([ANS]). End with a single line "
            f"'Final Answer:' followed by exactly ONE \\boxed{{}} that contains all "
            f"{n_blanks} answers, comma-separated, in the order the [ANS] blanks appear, "
            f"e.g. \\boxed{{a, b, c}}. Keep any tuple/interval/set in its own brackets "
            f"inside the box. Use only this one box; put \\boxed{{}} nowhere else."
        )
    return _LAYOUT_SINGLE


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the exact_v1 variant."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts}\n\n{_LAYOUT_MCQ}"
    n_blanks = n_ans_placeholders(question)
    return SYSTEM_PROMPT_MATH, f"{question}\n\n{_math_layout_rule(n_blanks)}"


def prompt_mode(question: str, options: Optional[list]) -> str:
    if options:
        return "mcq/exact_v1"
    return "multi-blank/exact_v1" if n_ans_placeholders(question) > 1 else "single-blank/exact_v1"

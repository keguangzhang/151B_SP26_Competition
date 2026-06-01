"""Add grader-grounded box-hygiene rules to the `precision` variant in dev.ipynb.
Grounded by probing judger.is_equal (units, percent, tuple brackets, yes/no)."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "dev.ipynb"
nb = json.load(open(NB))

CLAUSE_DEF_END = ('    "8 significant figures and do not round."\n'
                  ")\n")

FORMAT_CLAUSE = r'''    "8 significant figures and do not round."
)

# Box-hygiene rules grounded in judger.is_equal probes:
#   "5 cm" != "5" (bare units fail); "-1,-3" != "(-1,-3)" (brackets required);
#   "50" != "0.5" (% is stripped, can't mean /100); Yes/No/True/False interchange.
_GRADER_FORMAT_CLAUSE = (
    "Box the value only — no units or words (write \\boxed{5}, not \\boxed{5 cm}). "
    "Do not use a percent sign: give a probability as a decimal (e.g. 0.5) and a "
    "percentage as a bare number (e.g. 50). Keep a coordinate, tuple, or interval "
    "together in ONE box with its brackets, e.g. \\boxed{(-1,-3)} — never split it "
    "across boxes. For yes/no answers box just \\boxed{Yes} or \\boxed{No}."
)
'''

CLAUSE_F = '    f"{_PRECISION_CLAUSE} "\n'
CLAUSE_F_NEW = '    f"{_PRECISION_CLAUSE} "\n    f"{_GRADER_FORMAT_CLAUSE} "\n'

patched = False
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "_MATH_PRECISION" not in src:
        continue
    if "_GRADER_FORMAT_CLAUSE" in src:
        print("v2 already applied — skipping")
        break
    assert CLAUSE_DEF_END in src and CLAUSE_F in src, "v2 anchors not found"
    src = src.replace(CLAUSE_DEF_END, FORMAT_CLAUSE, 1)
    src = src.replace(CLAUSE_F, CLAUSE_F_NEW, 1)
    compile(src, "<dev_prompt_cell>", "exec")
    cell["source"] = src.splitlines(keepends=True)
    patched = True
    break

if patched:
    json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
    print("patched dev.ipynb with grader format clause")
else:
    print("no changes")

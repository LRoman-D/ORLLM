from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


GENPRM_STEP_TITLES = [
    "Problem Description",
    "Sets and Parameters",
    "Decision Variables",
    "Objective Function",
    "Constraints",
    "Mathematical Model",
    "Nonlinear Relationships",
    "Final Model",
    "Python Code Using OR-Tools",
]


GENPRM_SYSTEM_PROMPT = """You are a generative process reward model for operations research.
Audit a complete OR modeling trajectory step by step. Judge modeling semantics, not only whether the code runs.
Return a strict 9-step verdict using the requested text format only."""


GENPRM_OUTPUT_FORMAT = """STEP_1: [CORRECT/INCORRECT]
EXPLANATION_1: [...]
STEP_2: [CORRECT/INCORRECT]
EXPLANATION_2: [...]
STEP_3: [CORRECT/INCORRECT]
EXPLANATION_3: [...]
STEP_4: [CORRECT/INCORRECT]
EXPLANATION_4: [...]
STEP_5: [CORRECT/INCORRECT]
EXPLANATION_5: [...]
STEP_6: [CORRECT/INCORRECT]
EXPLANATION_6: [...]
STEP_7: [CORRECT/INCORRECT]
EXPLANATION_7: [...]
STEP_8: [CORRECT/INCORRECT]
EXPLANATION_8: [...]
STEP_9: [CORRECT/INCORRECT]
EXPLANATION_9: [...]"""


@dataclass
class GenPRMReport:
    step_correct: list[bool]
    explanations: list[str] = field(default_factory=list)
    raw_output: str = ""
    model: str = "unknown"

    @property
    def correct_count(self) -> int:
        return sum(1 for item in self.step_correct if item)

    @property
    def total_steps(self) -> int:
        return len(GENPRM_STEP_TITLES)

    @property
    def all_correct(self) -> bool:
        return self.correct_count == self.total_steps and len(self.step_correct) == self.total_steps

    @property
    def score(self) -> float:
        if len(self.step_correct) != self.total_steps:
            return 0.0
        return round(self.correct_count / self.total_steps, 4)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "correct_count": self.correct_count,
                "total_steps": self.total_steps,
                "all_correct": self.all_correct,
                "score": self.score,
            }
        )
        return payload


def build_genprm_prompt(
    *,
    question: str,
    trajectory: str,
    verification: dict[str, Any],
    template_name: str | None = None,
    reference_context: str | None = None,
) -> str:
    verification_text = json.dumps(
        {
            "execution_ok": verification.get("execution_ok"),
            "status": verification.get("status"),
            "objective_value": verification.get("objective_value"),
            "objective_match": verification.get("objective_match"),
            "error_message": verification.get("error_message"),
        },
        ensure_ascii=False,
        indent=2,
    )
    context_block = reference_context or "No external reference excerpts were supplied."
    template_text = template_name or "unknown"
    criteria = "\n".join(f"{idx}. {title}" for idx, title in enumerate(GENPRM_STEP_TITLES, start=1))
    return f"""# Context
Template family: {template_text}

Problem:
<problem>
{question}
</problem>

Solver execution summary:
<solver_verification>
{verification_text}
</solver_verification>

Trusted OR reference excerpts:
<reference_context>
{context_block}
</reference_context>

# Task
Review the following 9-step trajectory. For each step, decide whether it is semantically correct in the context of the original problem and all preceding steps.

Full trajectory:
<trajectory>
{trajectory}
</trajectory>

# Cascading Error Rule
Flag the first step where a modeling or code-implementation error appears. All later dependent steps must be marked INCORRECT.

# Step Criteria
{criteria}

# Critical Modeling Checks
- The problem summary must preserve every entity, number, objective direction, and feasibility condition.
- Sets, parameters, and indices must match the problem; no hallucinated constants or omitted resources.
- Variable domains must be correct: continuous, integer, binary, routing index, or activation variable as required.
- Objective and constraints must implement the same semantics as the natural language problem.
- The final OR-Tools code must implement the final model exactly, including coefficients, bounds, linking constraints, and objective sense.
- Treat successful execution as necessary but not sufficient. Running code with a semantically wrong model is INCORRECT.

# Output Format
Use exactly this format and nothing else:
{GENPRM_OUTPUT_FORMAT}
"""


def parse_genprm_output(text: str, *, model: str = "unknown") -> GenPRMReport:
    stripped = text.strip()
    if not stripped:
        return GenPRMReport(step_correct=[], explanations=[], raw_output=text, model=model)
    parsed = _parse_json_report(stripped)
    if parsed is not None:
        parsed.model = model
        return parsed

    step_correct: list[bool] = []
    explanations: list[str] = []
    for idx in range(1, len(GENPRM_STEP_TITLES) + 1):
        verdict_match = re.search(rf"STEP_{idx}\s*:\s*(CORRECT|INCORRECT)", stripped, flags=re.IGNORECASE)
        if verdict_match is None:
            continue
        step_correct.append(verdict_match.group(1).upper() == "CORRECT")
        explanation_match = re.search(
            rf"EXPLANATION_{idx}\s*:\s*(.*?)(?=\n\s*STEP_{idx + 1}\s*:|\Z)",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        explanations.append(explanation_match.group(1).strip() if explanation_match else "")
    return GenPRMReport(step_correct=step_correct, explanations=explanations, raw_output=text, model=model)


def process_score_from_audit(response_text: str, verification: dict[str, Any], audit: dict[str, Any] | None = None) -> float:
    step_hits = sum(1 for title in GENPRM_STEP_TITLES if title in response_text)
    structure_score = step_hits / len(GENPRM_STEP_TITLES)
    exec_bonus = 0.2 if verification.get("execution_ok") else 0.0
    match_bonus = 0.25 if verification.get("objective_match") else 0.0
    genprm_score = 0.0
    if audit:
        try:
            genprm_score = float(audit.get("score", 0.0))
        except (TypeError, ValueError):
            genprm_score = 0.0
    return round(min(1.0, structure_score * 0.2 + exec_bonus + match_bonus + genprm_score * 0.35), 4)


def audit_passes_threshold(
    audit: dict[str, Any] | None,
    *,
    min_correct_steps: int = 8,
    require_all_correct: bool = False,
) -> bool:
    if not audit:
        return False
    if require_all_correct:
        return bool(audit.get("all_correct"))
    try:
        correct_count = int(audit.get("correct_count", 0))
    except (TypeError, ValueError):
        return False
    return correct_count >= min_correct_steps


def _parse_json_report(text: str) -> GenPRMReport | None:
    candidate = text
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    raw_steps = payload.get("steps") or payload.get("step_correct")
    if not isinstance(raw_steps, list):
        return None
    step_correct = []
    explanations = []
    for item in raw_steps[: len(GENPRM_STEP_TITLES)]:
        if isinstance(item, dict):
            step_correct.append(bool(item.get("is_correct", item.get("correct", False))))
            explanations.append(str(item.get("explanation", "")))
        else:
            step_correct.append(bool(item))
            explanations.append("")
    return GenPRMReport(step_correct=step_correct, explanations=explanations, raw_output=text)


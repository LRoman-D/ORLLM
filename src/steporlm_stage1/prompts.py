from __future__ import annotations

from steporlm_stage1.data_factory.zhipu_teacher import build_solver_hint


SYSTEM_PROMPT = """You are an operations research modeling assistant for a small code model.
Solve the user's optimization problem in exactly 8 <step>...</step> blocks with these titles in order:
1. Problem Description
2. Sets and Parameters
3. Decision Variables
4. Objective Function
5. Constraints
6. Mathematical Model Summary
7. Nonlinear Relationships
8. Final Model and Implementation Considerations

Each step must be a single short line in this style:
<step>Problem Description: ...</step>
Keep every step under 25 words. Do not use bullet lists, equations, or markdown tables inside the steps.
Use ASCII only in reasoning and code. Do not use Unicode math symbols, placeholders, or pseudocode.

After the 8 steps, output exactly one complete fenced ```python``` block and nothing after it.
The Python code must:
- be self-contained and executable top to bottom
- import every module it uses explicitly
- use OR-Tools faithfully for the given problem
- match the variable domains in the question
- define a solve-status variable before reading results
- print one final line that starts with __STEPORLM_RESULT__= followed by JSON with keys status and objective_value

If space is tight, shorten the reasoning further instead of truncating the code.
Never return partial code."""

USER_PROMPT_TEMPLATE = """Below is an optimization modeling question. Build a mathematical model and corresponding Python code using OR-Tools.

{question}
"""


def build_rollout_user_prompt(question: str, template_name: str) -> str:
    solver_hint = build_solver_hint(template_name)
    domain_hint = _build_rollout_domain_hint(template_name)
    return f"""Template family: {template_name}
Preferred solver guidance:
{solver_hint}

Variable-domain reminder:
{domain_hint}

Optimization question:
{question}

Additional requirements:
- Keep the 8 reasoning steps extremely concise and single-line.
- Use plain ASCII in reasoning and code.
- Do not use markdown tables.
- Return one complete Python block, not a partial snippet.
"""


def _build_rollout_domain_hint(template_name: str) -> str:
    hints = {
        "resource_allocation": "Use continuous nonnegative decision variables unless the question explicitly asks for integers.",
        "production_planning": "Use integer quantity variables plus binary activation variables when setup decisions are present.",
        "assignment": "Use binary assignment variables.",
        "knapsack": "Use binary item-selection variables.",
        "tsp": "Use the routing solver and report the route objective value.",
    }
    return hints.get(template_name, "Match the variable domain stated or implied by the question.")

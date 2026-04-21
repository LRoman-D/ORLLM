from __future__ import annotations

from steporlm_stage1.data_factory.zhipu_teacher import build_solver_hint


SYSTEM_PROMPT = """You are an operations research modeling assistant.
Solve the user's optimization problem by following these 8 steps:
1. Problem Description
2. Sets and Parameters
3. Decision Variables
4. Objective Function
5. Constraints
6. Mathematical Model Summary
7. Nonlinear Relationships
8. Final Model and Implementation Considerations

Keep the reasoning concise but accurate. Ensure the mathematical logic is sound.
Use ASCII only in reasoning and code.

After the 8 steps, output exactly one complete fenced ```python``` block.
The Python code must:
- be self-contained and executable.
- import every module it uses explicitly (e.g., from ortools.linear_solver import pywraplp).
- use solver.BoolVar() for binary variables in OR-Tools.
- define a 'result' dictionary with keys 'status' and 'objective_value'.
- mapping solver status to "OPTIMAL", "FEASIBLE", "INFEASIBLE", or "UNBOUNDED".
- print the result at the end using: print(f"__STEPORLM_RESULT__={{json.dumps(result)}}")

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
- Ensure the Python code is complete and correctly implements the model.
- Use plain ASCII.
- Do not use markdown tables.
- The objective value in result should be float.
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

from __future__ import annotations

from steporlm_stage1.teachers.zhipu import build_solver_hint
from steporlm_stage1.quality.genprm import GENPRM_STEP_TITLES


ORTOOLS_STYLE_GUIDE = """OR-Tools coding standard:
1. Use `from ortools.linear_solver import pywraplp` for LP/MILP models.
2. Use `solver = pywraplp.Solver.CreateSolver("GLOP")` only for continuous LP.
3. Use `solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")` for integer, binary, or mixed-integer models. Use SCIP only if explicitly requested and available.
4. Create variables with the correct domains: `NumVar(lb, ub, name)`, `IntVar(lb, ub, name)`, or `BoolVar(name)`.
5. Add constraints with `solver.Add(...)`; every natural-language capacity, balance, assignment, demand, linking, and logical condition must appear exactly once unless intentionally equivalent.
6. Set the objective with `solver.Maximize(...)` or `solver.Minimize(...)`; never flip signs without explaining and preserving the reported objective.
7. Solve with `status = solver.Solve()` exactly once after the full model is built.
8. Map statuses explicitly: OPTIMAL, FEASIBLE, INFEASIBLE, UNBOUNDED, NOT_SOLVED, ABNORMAL.
9. Define `result = {"status": status_name, "objective_value": objective_value}` and print exactly one final marker with `__STEPORLM_RESULT__=`.
10. For TSP/routing tasks, use `from ortools.constraint_solver import routing_enums_pb2, pywrapcp`, build a `RoutingIndexManager` and `RoutingModel`, register a transit callback, and read the answer through `solution.ObjectiveValue()`.
11. Never use deprecated `pywrapknapsack_solver`, unavailable commercial solvers, hard-coded optimal objective values, randomization, hidden input files, or network calls.
12. Keep generated code self-contained; import every module used, including `json`."""


STEP_TEMPLATE = "\n".join(f"{idx}. {title}" for idx, title in enumerate(GENPRM_STEP_TITLES, start=1))


SYSTEM_PROMPT = f"""You are an expert operations research modeling assistant.
Solve each optimization problem with faithful mathematical modeling and executable Python code using OR-Tools.

Your response must contain exactly 9 reasoning steps, each enclosed in `<step>...</step>`, using these titles in order:
{STEP_TEMPLATE}

After the 9 steps, output exactly one complete fenced ```python``` block.

Modeling requirements:
- Preserve every number, bound, unit, entity, objective direction, and feasibility condition from the problem.
- Do not silently simplify away constraints.
- Separate modeling semantics from implementation details.
- If the model is linear, say so. If nonlinear terms appear, identify and linearize them before coding.
- The Python code must implement the final model exactly.

{ORTOOLS_STYLE_GUIDE}

Do not output any extra text after the Python block."""


USER_PROMPT_TEMPLATE = """Below is an optimization modeling question. Build a correct mathematical model and corresponding executable Python code using OR-Tools.

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

OR-Tools coding standard:
{ORTOOLS_STYLE_GUIDE}

Optimization question:
{question}

Additional requirements:
- Follow the exact 9-step template.
- Ensure the code is complete, deterministic, self-contained, and executable.
- Use plain ASCII in code.
- Do not use markdown tables.
- Report `objective_value` as a float or int when available, otherwise `None`.
"""


def _build_rollout_domain_hint(template_name: str) -> str:
    hints = {
        "external_or": (
            "Use the variable domain stated or implied by the external benchmark question. "
            "The reference value may be the optimal objective or the requested optimal decision value; "
            "put the numeric benchmark answer in `objective_value`."
        ),
        "resource_allocation": "Use continuous nonnegative decision variables unless the question explicitly asks for integers.",
        "production_planning": "Use integer quantity variables plus binary activation variables when setup/opening decisions are present.",
        "assignment": "Use binary assignment variables with one-to-one coverage constraints.",
        "knapsack": "Use binary item-selection variables and one capacity constraint per stated capacity.",
        "tsp": "Use the OR-Tools routing solver and report the closed-tour objective value.",
    }
    return hints.get(template_name, "Match the variable domain stated or implied by the question.")

from __future__ import annotations

import json
from typing import Any

from steporlm_stage1.clients import ZhipuChatClient


QUESTION_REWRITE_SYSTEM_PROMPT = """You are generating high-quality operations research training questions.
Rewrite each seed question into realistic variants while preserving:
- every numeric value
- optimization sense
- feasibility semantics
- all entities and resource relationships

Vary only the narrative framing and wording. Return JSON only:
{
  "variants": ["...", "..."]
}
"""


TRAJECTORY_SYSTEM_PROMPT = """You are an expert OR teacher creating supervised trajectories for a small reasoning model.
Solve the user's optimization problem in exactly 9 reasoning steps enclosed by <step>...</step>.
Use these step titles in order:
1. Problem Description
2. Sets and Parameters
3. Decision Variables
4. Objective Function
5. Constraints
6. Mathematical Model
7. Nonlinear Relationships
8. Final Model
9. Python Code Using OR-Tools

After the 9 steps, output exactly one fenced ```python``` block.
The Python code must:
- be self-contained
- include `import json`
- use OR-Tools
- solve the problem faithfully
- call solver.Solve()
- print one final line that starts with __STEPORLM_RESULT__= followed by JSON with keys status and objective_value
- import every module it uses explicitly
- not use helper functions unless they are fully defined in the code
- prefer the linear-solver API `from ortools.linear_solver import pywraplp` for MILP/LP problems
- never use the deprecated import `from ortools.algorithms import pywrapknapsack_solver`
- avoid CP-SAT (`cp_model`) unless absolutely necessary
- for TSP problems, use the routing solver APIs instead of writing a manual MTZ/x_ij MILP
- do not call `solver.status_name()`
- when printing the result marker, use exactly:
  `print("__STEPORLM_RESULT__=" + json.dumps({"status": "OPTIMAL", "objective_value": 123.0}, ensure_ascii=False))`
- do not hard-code the optimal objective; compute it from the solver object
- preserve all numeric values, sets, and constraints from the question exactly (never alter coefficients or bounds)
- avoid randomization in code unless the question explicitly requires stochastic simulation
- ensure `objective_value` is numeric (float/int) when available; use `None` when no valid objective is available
- keep code deterministic, minimal, and directly executable in one run

Do not output any extra text after the Python block.
"""


def build_question_rewrite_prompt(
    template_name: str,
    canonical_question: str,
    instance: dict[str, Any],
    num_variants: int,
    rewrite_styles: list[str],
) -> str:
    style_text = ", ".join(rewrite_styles) if rewrite_styles else "operations memo"
    instance_json = json.dumps(instance, ensure_ascii=False, indent=2)
    return f"""Template: {template_name}
Need {num_variants} rewritten variants.
Preferred narrative styles: {style_text}

Seed question:
{canonical_question}

Structured instance data:
{instance_json}

Return variants that preserve all numbers and optimization semantics exactly.
JSON only.
"""


def build_trajectory_prompt(question: str, template_name: str) -> str:
    solver_hint = build_solver_hint(template_name)
    return f"""Template family: {template_name}
Preferred solver guidance:
{solver_hint}

Optimization question:
{question}

Produce the full 9-step trajectory and executable OR-Tools Python solver now.
"""


def build_solver_hint(template_name: str) -> str:
    hints = {
        "external_or": (
            "- Build a faithful OR-Tools model for the question as written\n"
            "- The benchmark answer can be an optimal objective value or the requested optimal decision value\n"
            "- Store the numeric answer requested by the question in the JSON field `objective_value`\n"
            "- If the question asks for multiple decision values, report the benchmark's numeric target value in `objective_value` and explain the full solution in the steps"
        ),
        "resource_allocation": (
            "- Use `from ortools.linear_solver import pywraplp`\n"
            "- Build a continuous LP with `solver = pywraplp.Solver.CreateSolver('GLOP')`\n"
            "- Use decision variables directly in objective and constraints\n"
            "- Define `status = solver.Solve()`"
        ),
        "production_planning": (
            "- Use `from ortools.linear_solver import pywraplp`\n"
            "- This is a MILP with integer quantity variables and binary activation variables\n"
            "- Build it with `solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')`\n"
            "- Use `IntVar` and `BoolVar`, not `cp_model`\n"
            "- Define `status = solver.Solve()`"
        ),
        "assignment": (
            "- Use `from ortools.linear_solver import pywraplp`\n"
            "- Model it as a binary assignment MILP\n"
            "- Build it with `solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')`\n"
            "- Do not use `cp_model`\n"
            "- Define `status = solver.Solve()`"
        ),
        "knapsack": (
            "- Prefer a binary MILP using `from ortools.linear_solver import pywraplp`\n"
            "- Build it with `solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')`\n"
            "- Do not use `pywrapknapsack_solver`\n"
            "- Use `BoolVar` for item selection and `status = solver.Solve()`"
        ),
        "tsp": (
            "- Use the routing solver via `from ortools.constraint_solver import routing_enums_pb2, pywrapcp`\n"
            "- Do not formulate a manual x_ij / MTZ MILP for these generated TSP tasks\n"
            "- Build `RoutingIndexManager` and `RoutingModel`, register a distance callback, and solve with routing search parameters\n"
            "- Ensure the final result marker reports the route objective"
        ),
    }
    return hints.get(
        template_name,
        "- Use a standard OR-Tools solver with explicit imports and a valid JSON result marker.",
    )


class ZhipuTeacherGenerator:
    def __init__(self, client: ZhipuChatClient) -> None:
        self.client = client

    @classmethod
    def from_env(
        cls,
        model_env_var: str = "ZHIPUAI_DATA_MODEL",
        timeout_seconds: int = 60,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
        fallback_model: str = "glm-4.5-air",
    ) -> "ZhipuTeacherGenerator | None":
        client = ZhipuChatClient.from_env(
            model_env_var=model_env_var,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            fallback_model=fallback_model,
        )
        return cls(client) if client is not None else None

    def rewrite_question_variants(
        self,
        template_name: str,
        canonical_question: str,
        instance: dict[str, Any],
        num_variants: int,
        rewrite_styles: list[str],
        temperature: float = 0.6,
        max_tokens: int = 1200,
    ) -> list[str]:
        payload = self.client.json_chat(
            [
                {"role": "system", "content": QUESTION_REWRITE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_question_rewrite_prompt(
                        template_name,
                        canonical_question,
                        instance,
                        num_variants,
                        rewrite_styles,
                    ),
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_variants = payload.get("variants", [])
        variants: list[str] = []
        for item in raw_variants:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned and cleaned not in variants:
                    variants.append(cleaned)
        return variants

    def generate_trajectories(
        self,
        question: str,
        template_name: str,
        temperatures: list[float],
        max_tokens: int = 2000,
    ) -> list[str]:
        trajectories = []
        for temperature in temperatures:
            content = self.client.chat(
                [
                    {"role": "system", "content": TRAJECTORY_SYSTEM_PROMPT},
                    {"role": "user", "content": build_trajectory_prompt(question, template_name)},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            cleaned = content.strip()
            if cleaned:
                trajectories.append(cleaned)
        return trajectories

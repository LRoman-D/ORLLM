from __future__ import annotations

import random

from steporlm_stage1.schemas import ReferenceSolution
from steporlm_stage1.solvers.ortools_api import create_linear_solver, linear_status_name
from steporlm_stage1.templates.base import ProblemTemplate


class KnapsackTemplate(ProblemTemplate):
    name = "knapsack"
    optimization_sense = "maximize"

    def sample_instance(self, rng: random.Random) -> dict:
        items = rng.sample(
            ["portable scanner", "thermal camera", "relay box", "backup battery", "edge server", "sensor pod", "field router"],
            rng.randint(5, 6),
        )
        values = {item: rng.randint(12, 45) for item in items}
        weights = {item: rng.randint(2, 10) for item in items}
        capacity = int(sum(weights.values()) * rng.uniform(0.4, 0.65))
        return {
            "items": items,
            "values": values,
            "weights": weights,
            "capacity": capacity,
            "scenario": rng.choice(
                [
                    "emergency maintenance kit",
                    "field deployment equipment pack",
                    "mobile inspection resource bag",
                    "temporary network recovery kit",
                    "remote site diagnostics loadout",
                ]
            ),
        }

    def render_question(self, instance: dict, rng: random.Random) -> str:
        lines = []
        for item in instance["items"]:
            lines.append(f"- {item}: value {instance['values'][item]}, weight {instance['weights'][item]}.")
        framing = rng.choice(
            [
                "Select a subset of items without exceeding the weight limit.",
                "The team can either pack an item once or leave it behind.",
                "The goal is to maximize total carried value under the capacity limit.",
            ]
        )
        return (
            f"For a {instance['scenario']}, the maximum total weight is {instance['capacity']}. {framing}\n"
            + "\n".join(lines)
            + "\nBuild a 0-1 knapsack model and provide executable OR-Tools Python code."
        )

    def solve_reference(self, instance: dict) -> ReferenceSolution:
        solver = create_linear_solver(has_integer=True)
        x = {item: solver.BoolVar(f"x_{idx}") for idx, item in enumerate(instance["items"])}
        solver.Add(sum(instance["weights"][item] * x[item] for item in instance["items"]) <= instance["capacity"])
        solver.Maximize(sum(instance["values"][item] * x[item] for item in instance["items"]))
        status = solver.Solve()
        status_name = linear_status_name(status)
        chosen_items = [item for item in instance["items"] if x[item].solution_value() > 0.5] if status_name == "OPTIMAL" else []
        return ReferenceSolution(
            status=status_name,
            objective_value=solver.Objective().Value() if status_name == "OPTIMAL" else None,
            metadata={"chosen_items": chosen_items},
        )

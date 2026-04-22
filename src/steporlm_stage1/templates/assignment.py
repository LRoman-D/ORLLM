from __future__ import annotations

import random

from ortools.linear_solver import pywraplp

from steporlm_stage1.schemas import ReferenceSolution
from steporlm_stage1.templates.base import ProblemTemplate


class AssignmentTemplate(ProblemTemplate):
    name = "assignment"
    optimization_sense = "minimize"

    def sample_instance(self, rng: random.Random) -> dict:
        workers = rng.sample(["Alex", "Bailey", "Casey", "Drew", "Emery", "Finley"], 4)
        tasks = rng.sample(["routing audit", "inventory sync", "line inspection", "order batching"], 4)
        costs = {worker: {task: rng.randint(8, 28) for task in tasks} for worker in workers}
        return {
            "workers": workers,
            "tasks": tasks,
            "costs": costs,
            "scenario": rng.choice(
                [
                    "transport scheduling team",
                    "operations optimization group",
                    "logistics coordination shift",
                    "field service dispatch desk",
                    "warehouse process-improvement team",
                ]
            ),
        }

    def render_question(self, instance: dict, rng: random.Random) -> str:
        lines = []
        for worker in instance["workers"]:
            costs = "; ".join(f"{task}: {instance['costs'][worker][task]}" for task in instance["tasks"])
            lines.append(f"- Costs for {worker}: {costs}.")
        framing = rng.choice(
            [
                "Each person must take exactly one task and each task needs exactly one person.",
                "The manager wants a one-to-one assignment with minimum total cost.",
                "No worker can be assigned to two tasks, and no task can be left uncovered.",
            ]
        )
        return (
            f"A {instance['scenario']} must assign {len(instance['tasks'])} tasks to {len(instance['workers'])} workers. {framing}\n"
            + "\n".join(lines)
            + "\nFormulate the assignment MILP and provide executable OR-Tools Python code."
        )

    def solve_reference(self, instance: dict) -> ReferenceSolution:
        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
        if solver is None:
            raise RuntimeError("Failed to create CBC solver.")
        x = {
            (worker, task): solver.BoolVar(f"x_{w_idx}_{t_idx}")
            for w_idx, worker in enumerate(instance["workers"])
            for t_idx, task in enumerate(instance["tasks"])
        }
        for worker in instance["workers"]:
            solver.Add(sum(x[(worker, task)] for task in instance["tasks"]) == 1)
        for task in instance["tasks"]:
            solver.Add(sum(x[(worker, task)] for worker in instance["workers"]) == 1)
        solver.Minimize(
            sum(instance["costs"][worker][task] * x[(worker, task)] for worker in instance["workers"] for task in instance["tasks"])
        )
        status = solver.Solve()
        status_name = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FAILED"
        assignment = {}
        if status_name == "OPTIMAL":
            for worker in instance["workers"]:
                for task in instance["tasks"]:
                    if x[(worker, task)].solution_value() > 0.5:
                        assignment[worker] = task
        return ReferenceSolution(
            status=status_name,
            objective_value=solver.Objective().Value() if status_name == "OPTIMAL" else None,
            metadata={"assignment": assignment},
        )

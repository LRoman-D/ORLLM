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
            "scenario": rng.choice(["运筹调度团队", "运营优化小组", "物流协调班组"]),
        }

    def render_question(self, instance: dict, rng: random.Random) -> str:
        lines = []
        for worker in instance["workers"]:
            costs = "，".join(f"{task}: {instance['costs'][worker][task]}" for task in instance["tasks"])
            lines.append(f"- {worker} 执行各任务的成本为：{costs}。")
        return (
            f"某{instance['scenario']}要把 {len(instance['tasks'])} 个任务分配给 {len(instance['workers'])} 名成员，"
            "每个任务恰好分配给 1 人，每人恰好承担 1 个任务，目标是最小化总分配成本。\n"
            + "\n".join(lines)
            + "\n请建立 assignment 问题的 MILP 模型并给出 OR-Tools Python 代码。"
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

from __future__ import annotations

import random

from ortools.linear_solver import pywraplp

from steporlm_stage1.schemas import ReferenceSolution
from steporlm_stage1.templates.base import ProblemTemplate


class TSPTemplate(ProblemTemplate):
    name = "tsp"
    optimization_sense = "minimize"

    def sample_instance(self, rng: random.Random) -> dict:
        cities = rng.sample(["Aster", "Brook", "Cedar", "Delta", "Elm", "Fjord", "Grove"], rng.randint(5, 6))
        distances = {}
        for i, city_i in enumerate(cities):
            distances[city_i] = {}
            for j, city_j in enumerate(cities):
                if i == j:
                    distances[city_i][city_j] = 0
                elif city_j in distances and city_i in distances[city_j]:
                    distances[city_i][city_j] = distances[city_j][city_i]
                else:
                    distances[city_i][city_j] = rng.randint(12, 60)
        return {
            "cities": cities,
            "distances": distances,
            "scenario": rng.choice(["巡检巴士路线", "城际配送路线", "服务工程师访问路线"]),
        }

    def render_question(self, instance: dict, rng: random.Random) -> str:
        lines = []
        for city in instance["cities"]:
            costs = "，".join(
                f"到 {other} 为 {instance['distances'][city][other]}" for other in instance["cities"] if other != city
            )
            lines.append(f"- 从 {city} 出发：{costs}。")
        return (
            f"某{instance['scenario']}需要从一个城市出发并最终回到起点，且每个城市恰好访问一次，目标是最小化总路程成本。\n"
            f"城市集合为：{', '.join(instance['cities'])}。\n"
            + "\n".join(lines)
            + "\n请建立一个包含 subtour elimination 的 TSP 模型，并给出 OR-Tools Python 代码。"
        )

    def solve_reference(self, instance: dict) -> ReferenceSolution:
        cities = instance["cities"]
        n = len(cities)
        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
        if solver is None:
            raise RuntimeError("Failed to create CBC solver.")
        x = {(i, j): solver.BoolVar(f"x_{i}_{j}") for i in range(n) for j in range(n) if i != j}
        u = {i: solver.NumVar(0.0, n - 1, f"u_{i}") for i in range(1, n)}
        for i in range(n):
            solver.Add(sum(x[(i, j)] for j in range(n) if i != j) == 1)
            solver.Add(sum(x[(j, i)] for j in range(n) if i != j) == 1)
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    solver.Add(u[i] - u[j] + n * x[(i, j)] <= n - 1)
        solver.Minimize(
            sum(instance["distances"][cities[i]][cities[j]] * x[(i, j)] for i in range(n) for j in range(n) if i != j)
        )
        status = solver.Solve()
        status_name = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FAILED"
        return ReferenceSolution(
            status=status_name,
            objective_value=solver.Objective().Value() if status_name == "OPTIMAL" else None,
            metadata={"city_count": n},
        )

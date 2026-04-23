from __future__ import annotations

import random

from steporlm_stage1.schemas import ReferenceSolution
from steporlm_stage1.solvers.ortools_api import create_linear_solver, linear_status_name
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
            "scenario": rng.choice(
                [
                    "inspection shuttle route",
                    "intercity delivery tour",
                    "service engineer visit route",
                    "regional calibration trip",
                    "same-day spare-parts circuit",
                ]
            ),
        }

    def render_question(self, instance: dict, rng: random.Random) -> str:
        lines = []
        for city in instance["cities"]:
            costs = "; ".join(f"to {other}: {instance['distances'][city][other]}" for other in instance["cities"] if other != city)
            lines.append(f"- From {city}: {costs}.")
        framing = rng.choice(
            [
                "Start at one city, visit every other city exactly once, and return to the start.",
                "Find the minimum-cost closed tour over all listed cities.",
                "The route must avoid subtours and cover every city exactly once.",
            ]
        )
        return (
            f"For a {instance['scenario']}, {framing}\n"
            f"City set: {', '.join(instance['cities'])}.\n"
            + "\n".join(lines)
            + "\nBuild a TSP model and provide executable OR-Tools Python code."
        )

    def solve_reference(self, instance: dict) -> ReferenceSolution:
        cities = instance["cities"]
        n = len(cities)
        solver = create_linear_solver(has_integer=True)
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
        status_name = linear_status_name(status)
        return ReferenceSolution(
            status=status_name,
            objective_value=solver.Objective().Value() if status_name == "OPTIMAL" else None,
            metadata={"city_count": n},
        )

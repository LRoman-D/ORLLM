from __future__ import annotations

import random

from ortools.linear_solver import pywraplp

from steporlm_stage1.schemas import ReferenceSolution
from steporlm_stage1.templates.base import ProblemTemplate


class ProductionPlanningTemplate(ProblemTemplate):
    name = "production_planning"
    optimization_sense = "maximize"

    def sample_instance(self, rng: random.Random) -> dict:
        product_pool = ["battery packs", "control units", "sensor kits", "power modules", "cooling racks", "charging hubs"]
        num_products = rng.randint(3, 4)
        products = rng.sample(product_pool, num_products)
        margin = {product: rng.randint(18, 36) for product in products}
        setup_cost = {product: rng.randint(45, 110) for product in products}
        max_units = {product: rng.randint(8, 20) for product in products}
        labor_hours = {product: rng.randint(2, 6) for product in products}
        total_hours = int(sum(labor_hours[p] * max_units[p] for p in products) * rng.uniform(0.45, 0.7))
        return {
            "products": products,
            "margin": margin,
            "setup_cost": setup_cost,
            "max_units": max_units,
            "labor_hours": labor_hours,
            "total_hours": total_hours,
            "scenario": rng.choice(
                [
                    "smart factory weekly planning",
                    "electronics assembly schedule",
                    "small-batch industrial equipment planning",
                    "after-sales parts production",
                    "prototype manufacturing slotting",
                ]
            ),
        }

    def render_question(self, instance: dict, rng: random.Random) -> str:
        lines = []
        for product in instance["products"]:
            lines.append(
                f"- {product}: unit margin {instance['margin'][product]}, setup cost {instance['setup_cost'][product]}, "
                f"labor {instance['labor_hours'][product]} hours per unit, maximum {instance['max_units'][product]} units."
            )
        framing = rng.choice(
            [
                "Choose which products to open and how many units to make.",
                "Prepare a MILP that links setup decisions to production quantities.",
                "The plan may skip a product if its setup cost is not justified.",
            ]
        )
        return (
            f"For {instance['scenario']}, total labor cannot exceed {instance['total_hours']} hours. {framing} "
            "The objective is to maximize net profit after setup costs.\n"
            + "\n".join(lines)
            + "\nBuild a MILP with integer production variables and binary activation variables, then provide OR-Tools Python code."
        )

    def solve_reference(self, instance: dict) -> ReferenceSolution:
        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
        if solver is None:
            raise RuntimeError("Failed to create CBC solver.")
        x = {product: solver.IntVar(0, instance["max_units"][product], f"x_{idx}") for idx, product in enumerate(instance["products"])}
        y = {product: solver.BoolVar(f"y_{idx}") for idx, product in enumerate(instance["products"])}
        for product in instance["products"]:
            solver.Add(x[product] <= instance["max_units"][product] * y[product])
        solver.Add(sum(instance["labor_hours"][p] * x[p] for p in instance["products"]) <= instance["total_hours"])
        solver.Maximize(sum(instance["margin"][p] * x[p] - instance["setup_cost"][p] * y[p] for p in instance["products"]))
        status = solver.Solve()
        status_name = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FAILED"
        return ReferenceSolution(
            status=status_name,
            objective_value=solver.Objective().Value() if status_name == "OPTIMAL" else None,
            metadata={
                "production": {product: x[product].solution_value() for product in instance["products"]},
                "open": {product: y[product].solution_value() for product in instance["products"]},
            },
        )

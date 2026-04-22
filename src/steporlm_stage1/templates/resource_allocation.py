from __future__ import annotations

import random

from ortools.linear_solver import pywraplp

from steporlm_stage1.schemas import ReferenceSolution
from steporlm_stage1.templates.base import ProblemTemplate


class ResourceAllocationTemplate(ProblemTemplate):
    name = "resource_allocation"
    optimization_sense = "maximize"

    def sample_instance(self, rng: random.Random) -> dict:
        product_pool = ["EcoFuel", "PrimeSteel", "SmartGlass", "AeroFoam", "FlexCircuit", "BioResin", "MedGel", "NanoCable"]
        resource_pool = ["machine hours", "raw material", "skilled labor"]
        num_products = rng.randint(3, 5)
        num_resources = rng.randint(2, 3)
        products = rng.sample(product_pool, num_products)
        resources = resource_pool[:num_resources]
        profits = {product: rng.randint(18, 48) for product in products}
        usage = {resource: {product: rng.randint(1, 6) for product in products} for resource in resources}
        capacities = {resource: int(sum(usage[resource].values()) * rng.uniform(4.5, 6.5)) for resource in resources}
        return {
            "products": products,
            "resources": resources,
            "profits": profits,
            "usage": usage,
            "capacities": capacities,
            "scenario": rng.choice(
                [
                    "advanced materials workshop",
                    "precision fabrication studio",
                    "green manufacturing line",
                    "regional contract manufacturer",
                    "pilot production cell",
                ]
            ),
        }

    def render_question(self, instance: dict, rng: random.Random) -> str:
        product_lines = []
        for product in instance["products"]:
            parts = [f"{resource}: {instance['usage'][resource][product]}" for resource in instance["resources"]]
            product_lines.append(
                f"- Producing 1 unit of {product} earns profit {instance['profits'][product]} and consumes "
                + "; ".join(parts)
                + "."
            )
        capacities = "; ".join(f"{resource} capacity is {instance['capacities'][resource]}" for resource in instance["resources"])
        framing = rng.choice(
            [
                "The planning team needs a production mix recommendation.",
                "A manager is preparing a capacity allocation memo.",
                "The operations analyst is building a weekly production plan.",
            ]
        )
        return (
            f"In a {instance['scenario']}, {framing} Decide the production quantity of each product to maximize total profit.\n"
            f"Candidate products: {', '.join(instance['products'])}.\n"
            f"Available resources: {capacities}.\n"
            + "\n".join(product_lines)
            + "\nBuild a linear programming model and provide executable OR-Tools Python code."
        )

    def solve_reference(self, instance: dict) -> ReferenceSolution:
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:
            raise RuntimeError("Failed to create GLOP solver.")
        variables = {product: solver.NumVar(0.0, solver.infinity(), f"x_{idx}") for idx, product in enumerate(instance["products"])}
        for resource in instance["resources"]:
            solver.Add(
                sum(instance["usage"][resource][product] * variables[product] for product in instance["products"])
                <= instance["capacities"][resource]
            )
        solver.Maximize(sum(instance["profits"][product] * variables[product] for product in instance["products"]))
        status = solver.Solve()
        status_name = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FAILED"
        return ReferenceSolution(
            status=status_name,
            objective_value=solver.Objective().Value() if status_name == "OPTIMAL" else None,
            metadata={"solution": {product: variables[product].solution_value() for product in instance["products"]}},
        )

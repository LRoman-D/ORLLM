from __future__ import annotations

import random

from ortools.linear_solver import pywraplp

from steporlm_stage1.schemas import ReferenceSolution
from steporlm_stage1.templates.base import ProblemTemplate


class ResourceAllocationTemplate(ProblemTemplate):
    name = "resource_allocation"
    optimization_sense = "maximize"

    def sample_instance(self, rng: random.Random) -> dict:
        product_pool = ["EcoFuel", "PrimeSteel", "SmartGlass", "AeroFoam", "FlexCircuit", "BioResin"]
        resource_pool = ["machine hours", "raw material", "skilled labor"]
        num_products = rng.randint(3, 5)
        num_resources = rng.randint(2, 3)
        products = rng.sample(product_pool, num_products)
        resources = resource_pool[:num_resources]
        profits = {product: rng.randint(18, 48) for product in products}
        usage = {}
        for resource in resources:
            usage[resource] = {product: rng.randint(1, 6) for product in products}
        capacities = {resource: int(sum(usage[resource].values()) * rng.uniform(4.5, 6.5)) for resource in resources}
        return {
            "products": products,
            "resources": resources,
            "profits": profits,
            "usage": usage,
            "capacities": capacities,
            "scenario": rng.choice(["advanced materials workshop", "precision fabrication studio", "green manufacturing line"]),
        }

    def render_question(self, instance: dict, rng: random.Random) -> str:
        product_lines = []
        for product in instance["products"]:
            parts = [f"{resource}: {instance['usage'][resource][product]}" for resource in instance["resources"]]
            product_lines.append(f"- 每生产 1 单位 {product} 可获得利润 {instance['profits'][product]}，资源消耗为 " + "，".join(parts) + "。")
        capacities = "；".join(f"{resource} 总量为 {instance['capacities'][resource]}" for resource in instance["resources"])
        return (
            f"某{instance['scenario']}需要决定各产品的生产数量，以最大化总利润。\n"
            f"当前可选产品为：{', '.join(instance['products'])}。\n"
            f"{capacities}。\n"
            + "\n".join(product_lines)
            + "\n请建立线性规划模型，并给出可执行的 OR-Tools Python 代码。"
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

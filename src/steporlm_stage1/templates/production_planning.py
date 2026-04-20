from __future__ import annotations

import random

from ortools.linear_solver import pywraplp

from steporlm_stage1.schemas import ReferenceSolution
from steporlm_stage1.templates.base import ProblemTemplate


class ProductionPlanningTemplate(ProblemTemplate):
    name = "production_planning"
    optimization_sense = "maximize"

    def sample_instance(self, rng: random.Random) -> dict:
        product_pool = ["battery packs", "control units", "sensor kits", "power modules", "cooling racks"]
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
            "scenario": rng.choice(["智能工厂周生产计划", "电子装配车间周排产", "工业设备小批量计划"]),
        }

    def render_question(self, instance: dict, rng: random.Random) -> str:
        lines = []
        for product in instance["products"]:
            lines.append(
                f"- {product}: 单位贡献利润 {instance['margin'][product]}，启用后固定准备成本 {instance['setup_cost'][product]}，"
                f"每单位需要 {instance['labor_hours'][product]} 小时工时，最多可生产 {instance['max_units'][product]} 单位。"
            )
        return (
            f"某{instance['scenario']}需要决定哪些产品投产以及各自的产量，以在总工时不超过 {instance['total_hours']} 的前提下最大化净利润。\n"
            + "\n".join(lines)
            + "\n要求建立一个包含整数变量与启用变量的 MILP 模型，并给出 OR-Tools Python 代码。"
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

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ortools.linear_solver import pywraplp


RESULT_MARKER = "__STEPORLM_RESULT__="


@dataclass(frozen=True)
class SolverChoice:
    family: str
    backend: str
    reason: str


def choose_linear_solver(*, has_integer: bool, preferred: str | None = None) -> SolverChoice:
    if preferred:
        return SolverChoice(family="linear_solver", backend=preferred, reason="explicit_preference")
    if has_integer:
        return SolverChoice(family="linear_solver", backend="CBC_MIXED_INTEGER_PROGRAMMING", reason="integer_or_binary_model")
    return SolverChoice(family="linear_solver", backend="GLOP", reason="continuous_linear_model")


def create_linear_solver(*, has_integer: bool = False, preferred: str | None = None) -> pywraplp.Solver:
    choice = choose_linear_solver(has_integer=has_integer, preferred=preferred)
    solver = pywraplp.Solver.CreateSolver(choice.backend)
    if solver is None and choice.backend == "SCIP":
        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        raise RuntimeError(f"Failed to create OR-Tools solver backend: {choice.backend}")
    return solver


def linear_status_name(status: int) -> str:
    mapping = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    return mapping.get(status, str(status))


def build_result(status: str, objective_value: float | int | None, **metadata: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status, "objective_value": objective_value}
    if metadata:
        result["metadata"] = metadata
    return result


def emit_result(result: dict[str, Any]) -> None:
    print(RESULT_MARKER + json.dumps(result, ensure_ascii=False))


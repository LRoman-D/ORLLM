# OR-Tools Coding Standard

This project trains Qwen3-8B to produce executable and semantically faithful OR-Tools models. Generated solutions should follow this standard.

## Required Response Shape

Use exactly 9 `<step>...</step>` sections:

1. Problem Description
2. Sets and Parameters
3. Decision Variables
4. Objective Function
5. Constraints
6. Mathematical Model
7. Nonlinear Relationships
8. Final Model
9. Python Code Using OR-Tools

Then emit one fenced Python block and no extra text.

## Modeling Rules

- Preserve every numeric value, bound, entity, index set, objective direction, and feasibility condition from the problem.
- Treat successful execution as necessary but not sufficient: code that solves the wrong model is wrong.
- Declare variable domains before writing the objective or constraints.
- State whether the model is LP, MILP, routing, or another OR-Tools family.
- If nonlinear terms appear, identify and linearize them before implementation.
- Do not omit linking constraints, assignment coverage constraints, resource capacities, demand constraints, or subtour/routing semantics.

## Solver API Rules

- Continuous LP: `from ortools.linear_solver import pywraplp`, then `pywraplp.Solver.CreateSolver("GLOP")`.
- Integer, binary, or mixed-integer LP: `pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")`.
- TSP/routing: `from ortools.constraint_solver import routing_enums_pb2, pywrapcp`, then `RoutingIndexManager`, `RoutingModel`, callback registration, and `solution.ObjectiveValue()`.
- Use `NumVar`, `IntVar`, and `BoolVar` with correct lower and upper bounds.
- Use `solver.Add(...)` for constraints and `solver.Maximize(...)` or `solver.Minimize(...)` for the objective.
- Call `status = solver.Solve()` once after the full model is built.
- Map statuses explicitly to `OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `UNBOUNDED`, `NOT_SOLVED`, or `ABNORMAL`.
- Print exactly one final marker:

```python
print("__STEPORLM_RESULT__=" + json.dumps(result, ensure_ascii=False))
```

## Forbidden Patterns

- Hard-coded objective values or hard-coded selected decisions.
- Deprecated `pywrapknapsack_solver`.
- Unavailable commercial solver APIs.
- Hidden input files, network calls, randomness, or dependence on current time.
- `solver.status_name()` or undocumented routing methods.
- Running code that computes an answer but does not implement the final model.


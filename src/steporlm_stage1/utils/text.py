from __future__ import annotations

import re

from steporlm_stage1.quality.genprm import GENPRM_STEP_TITLES


STEP_TITLES = GENPRM_STEP_TITLES


def extract_python_code(response_text: str) -> str:
    text = response_text.replace("\r\n", "\n").strip()
    matches = list(re.finditer(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE))
    if matches:
        return postprocess_python_code(matches[-1].group(1).strip())
    open_matches = list(re.finditer(r"```(?:python|py)?\s*", text, flags=re.IGNORECASE))
    if open_matches:
        return postprocess_python_code(text[open_matches[-1].end() :].strip())
    return postprocess_python_code(text)


def postprocess_python_code(code_text: str) -> str:
    text = code_text.replace("\r\n", "\n").strip()
    text = re.sub(r"^python\s*\n", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    start = _find_code_start(text)
    if start is not None:
        text = text[start:].strip()
    lines = []
    skipping_marker_block = False
    marker_balance = 0
    for line in text.splitlines():
        if skipping_marker_block:
            marker_balance += _delimiter_balance(line)
            if marker_balance <= 0:
                skipping_marker_block = False
                marker_balance = 0
            continue
        stripped = line.strip()
        if stripped in {"<step>", "</step>"}:
            continue
        if "__STEPORLM_RESULT__" in line:
            indent = re.match(r"^\s*", line).group(0)
            safe_print = indent + 'print("__STEPORLM_RESULT__=" + json.dumps(_steporlm_emit_result(locals(), globals()), ensure_ascii=False))'
            lines.append(safe_print)
            marker_balance = _delimiter_balance(line)
            if marker_balance > 0:
                skipping_marker_block = True
            continue
        lines.append(line)
    text = "\n".join(lines).strip()
    text = _upgrade_legacy_ortools_patterns(text)
    text = _repair_ortools_patterns(text)
    text = _inject_runtime_preamble(text)
    text = _ensure_result_footer(text)
    return text


def _delimiter_balance(line: str) -> int:
    return sum(line.count(ch) for ch in "([{") - sum(line.count(ch) for ch in ")]}")


def _find_code_start(text: str) -> int | None:
    patterns = [
        r"(?m)^import\s+\w+",
        r"(?m)^from\s+\w+\s+import\s+.+$",
        r"(?m)^def\s+\w+\s*\(",
        r"(?m)^model\s*=",
        r"(?m)^manager\s*=",
        r"(?m)^routing\s*=",
        r"(?m)^data\s*=\s*\{",
        r"(?m)^solver\s*=\s*",
    ]
    starts = []
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            starts.append(match.start())
    return min(starts) if starts else None


def _repair_ortools_patterns(text: str) -> str:
    repaired = text
    repaired = re.sub(r"solver\.Value\(([^)]+)\)", r"\1.solution_value()", repaired)
    repaired = re.sub(r"solver\.status_name\(\)", "_steporlm_status_name(status)", repaired)
    repaired = re.sub(r"solver\.StatusName\(([^)]+)\)", r"_steporlm_status_name(\1)", repaired)
    repaired = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\.solution\(\)", r"\1.solution_value()", repaired)
    repaired = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\.solution\b(?!_)", r"\1.solution_value()", repaired)
    repaired = re.sub(r"\bval\.value\(\)", "val.solution_value()", repaired)
    repaired = re.sub(r"\bvar\.value\(\)", "var.solution_value()", repaired)
    repaired = re.sub(r"\bx\.value\(\)", "x.solution_value()", repaired)
    repaired = repaired.replace("objective.Minimize()", "objective.SetMinimization()")
    repaired = repaired.replace("objective.Maximize()", "objective.SetMaximization()")
    repaired = repaired.replace("objective.SetMinObjective()", "objective.SetMinimization()")
    repaired = repaired.replace("objective.SetMaxObjective()", "objective.SetMaximization()")
    repaired = repaired.replace("math.round(", "round(")
    repaired = re.sub(r"\bsolver\.Dispose\(\)", "", repaired)
    if "solver.Solve()" not in repaired and ("solver.Maximize(" in repaired or "solver.Minimize(" in repaired):
        insert_patterns = [r"(?m)^status\s*=", r"(?m)^status_name\s*=", r"(?m)^result\s*="]
        insert_at = None
        for pattern in insert_patterns:
            match = re.search(pattern, repaired)
            if match:
                insert_at = match.start()
                break
        solve_stmt = 'status = solver.Solve()\n'
        if insert_at is not None:
            repaired = repaired[:insert_at] + solve_stmt + repaired[insert_at:]
        else:
            repaired = repaired + "\n" + solve_stmt
    return repaired.strip()


def _upgrade_legacy_ortools_patterns(text: str) -> str:
    upgraded = text
    uses_legacy_knapsack = "pywrapknapsack_solver" in upgraded or "knapsack_solver.KnapsackSolver" in upgraded
    upgraded = upgraded.replace(
        "from ortools.algorithms import pywrapknapsack_solver",
        "from ortools.algorithms.python import knapsack_solver",
    )
    upgraded = upgraded.replace("pywrapknapsack_solver.KnapsackSolver(", "knapsack_solver.KnapsackSolver(")
    upgraded = upgraded.replace(
        "pywrapknapsack_solver.KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER",
        "knapsack_solver.SolverType.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER",
    )
    if uses_legacy_knapsack:
        upgraded = re.sub(r"\.Init\(([^,]+),\s*([^,]+),\s*len\([^)]+\),\s*([^)]+)\)", r".init(\1, \2, \3)", upgraded)
        upgraded = upgraded.replace(".Init(", ".init(")
        upgraded = upgraded.replace("optimal_value = solver.OptimalValue()", "optimal_value = solver.solve()")
        upgraded = upgraded.replace("solver.Solve()", "solver.solve()")
        upgraded = upgraded.replace(".BestSolutionContains(", ".best_solution_contains(")
    return upgraded


def _inject_runtime_preamble(text: str) -> str:
    preamble_lines = [
        "import json",
        "def _steporlm_status_name(status_value):",
        "    try:",
        "        from ortools.linear_solver import pywraplp",
        "        if status_value == pywraplp.Solver.OPTIMAL:",
        '            return "OPTIMAL"',
        "        if status_value == pywraplp.Solver.FEASIBLE:",
        '            return "FEASIBLE"',
        "    except Exception:",
        "        pass",
        "    try:",
        "        from ortools.sat.python import cp_model",
        "        if status_value == cp_model.OPTIMAL:",
        '            return "OPTIMAL"',
        "        if status_value == cp_model.FEASIBLE:",
        '            return "FEASIBLE"',
        "    except Exception:",
        "        pass",
        "    return str(status_value)",
        "def _steporlm_find_solver(local_vars):",
        "    solver = local_vars.get('solver')",
        "    if solver is not None and hasattr(solver, 'Solve'):",
        "        return solver",
        "    for value in local_vars.values():",
        "        if hasattr(value, 'Solve') and hasattr(value, 'Objective'):",
        "            return value",
        "    return None",
        "def _steporlm_find_result_dict(local_vars):",
        "    result = local_vars.get('result')",
        "    if isinstance(result, dict):",
        "        return result",
        "    for value in local_vars.values():",
        "        if isinstance(value, dict) and ('status' in value or 'objective_value' in value):",
        "            return value",
        "    return None",
        "def _steporlm_emit_result(local_vars, global_vars):",
        "    result = _steporlm_find_result_dict(local_vars)",
        "    if isinstance(result, dict):",
        "        return {",
        "            'status': _steporlm_status_name(result.get('status')),",
        "            'objective_value': result.get('objective_value'),",
        "        }",
        "    solution = local_vars.get('solution')",
        "    if solution is not None and hasattr(solution, 'ObjectiveValue'):",
        "        try:",
        "            return {'status': 'OPTIMAL', 'objective_value': float(solution.ObjectiveValue())}",
        "        except Exception:",
        "            pass",
        "    solver = _steporlm_find_solver(local_vars)",
        "    status_value = local_vars.get('status', global_vars.get('status'))",
        "    if status_value is None and solver is not None and hasattr(solver, 'Solve'):",
        "        try:",
        "            status_value = solver.Solve()",
        "        except Exception:",
        "            status_value = None",
        "    objective_value = local_vars.get('objective_value')",
        "    if objective_value is None and solver is not None and hasattr(solver, 'Objective'):",
        "        try:",
        "            objective_value = solver.Objective().Value()",
        "        except Exception:",
        "            objective_value = None",
        "    return {'status': _steporlm_status_name(status_value), 'objective_value': objective_value}",
    ]
    preamble = "\n".join(preamble_lines).strip()
    repaired = text
    if re.search(r"\bpywraplp\b", repaired) and not re.search(r"(?m)^from\s+ortools\.linear_solver\s+import\s+pywraplp", repaired):
        repaired = "from ortools.linear_solver import pywraplp\n" + repaired
    if re.search(r"\bcp_model\b", repaired) and not re.search(r"(?m)^from\s+ortools\.sat\.python\s+import\s+cp_model", repaired):
        repaired = "from ortools.sat.python import cp_model\n" + repaired
    if re.search(r"\bpywrapcp\b", repaired) and not re.search(r"(?m)^from\s+ortools\.constraint_solver\s+import\s+pywrapcp", repaired):
        repaired = "from ortools.constraint_solver import pywrapcp\n" + repaired
    if re.search(r"\brouting_enums_pb2\b", repaired) and not re.search(
        r"(?m)^from\s+ortools\.constraint_solver\s+import\s+routing_enums_pb2", repaired
    ):
        repaired = "from ortools.constraint_solver import routing_enums_pb2\n" + repaired
    return preamble + "\n\n" + repaired


def _ensure_result_footer(text: str) -> str:
    if "__STEPORLM_RESULT__=" in text:
        return text.strip()
    footer = """
try:
    print("__STEPORLM_RESULT__=" + json.dumps(_steporlm_emit_result(globals(), globals()), ensure_ascii=False))
except Exception as exc:
    print("__STEPORLM_RESULT__=" + json.dumps({"status": "FAILED", "objective_value": None, "error": str(exc)}, ensure_ascii=False))
""".strip()
    return (text.rstrip() + "\n\n" + footer).strip()


def heuristic_process_score(response_text: str, verification: dict) -> float:
    step_hits = sum(1 for title in STEP_TITLES if title in response_text)
    structure_score = step_hits / len(STEP_TITLES)
    exec_bonus = 0.35 if verification.get("execution_ok") else 0.0
    match_bonus = 0.35 if verification.get("objective_match") else 0.0
    return round(min(1.0, structure_score * 0.3 + exec_bonus + match_bonus), 4)

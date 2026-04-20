from __future__ import annotations

from pathlib import Path

from steporlm_stage1.executors.python_executor import PythonCodeExecutor
from steporlm_stage1.schemas import ReferenceSolution
from steporlm_stage1.utils.io import read_jsonl, write_json
from steporlm_stage1.utils.text import extract_python_code


def evaluate_predictions(
    dataset_path: str | Path,
    predictions_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, float | int]:
    dataset_rows = read_jsonl(dataset_path)
    if predictions_path is None:
        predicted_rows = dataset_rows
    else:
        predicted_rows = read_jsonl(predictions_path)
    prediction_map = {row["problem_id"]: row for row in predicted_rows}
    executor = PythonCodeExecutor(timeout_seconds=20)

    total = 0
    execution_ok = 0
    feasible = 0
    matched = 0
    detailed = []

    for row in dataset_rows:
        prediction = prediction_map.get(row["problem_id"])
        if prediction is None:
            continue
        code = prediction.get("code") or extract_python_code(prediction["response"])
        reference = ReferenceSolution(
            status=row["reference_solution"]["status"],
            objective_value=row["reference_solution"]["objective_value"],
            metadata=row["reference_solution"].get("metadata", {}),
        )
        verification = executor.verify(code, reference)
        total += 1
        execution_ok += int(verification.execution_ok)
        feasible += int(verification.status == "OPTIMAL")
        matched += int(verification.success)
        detailed.append(
            {
                "problem_id": row["problem_id"],
                "template_name": row["template_name"],
                "verification": verification.to_dict(),
            }
        )

    metrics = {
        "num_examples": total,
        "execution_rate": round(execution_ok / total, 4) if total else 0.0,
        "feasible_rate": round(feasible / total, 4) if total else 0.0,
        "pass_at_1": round(matched / total, 4) if total else 0.0,
    }
    if output_path is not None:
        write_json(output_path, {"metrics": metrics, "details": detailed})
    return metrics

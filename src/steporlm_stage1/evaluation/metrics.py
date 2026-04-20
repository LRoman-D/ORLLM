from __future__ import annotations

from statistics import pvariance

from steporlm_stage1.preference.ranking import rank_key


def summarize_best_trajectories(rows: list[dict]) -> tuple[list[dict], dict[str, float], list[float]]:
    predictions = []
    execution_rates = []
    feasible_rates = []
    pass_rates = []
    objective_gaps = []
    teacher_scores = []

    for row in rows:
        ranked = sorted(row["trajectories"], key=rank_key, reverse=True)
        best = ranked[0]
        verification = best["verification"]
        reference_value = row["reference_solution"].get("objective_value")
        objective_value = verification.get("objective_value")
        objective_gap = None
        if reference_value is not None and objective_value is not None:
            objective_gap = abs(float(objective_value) - float(reference_value))
            objective_gaps.append(objective_gap)

        teacher = best.get("teacher_evaluation") or {}
        try:
            score = float(teacher["overall_score"])
            teacher_scores.append(score)
        except (KeyError, TypeError, ValueError):
            score = None

        predictions.append(
            {
                "problem_id": row["problem_id"],
                "template_name": row["template_name"],
                "question": row["question"],
                "response": best["response"],
                "code": best["code"],
                "verification": verification,
                "teacher_evaluation": best.get("teacher_evaluation"),
                "objective_gap": objective_gap,
                "source": best["source"],
            }
        )
        execution_rates.append(1.0 if verification.get("execution_ok") else 0.0)
        feasible_rates.append(1.0 if verification.get("status") == "OPTIMAL" else 0.0)
        pass_rates.append(1.0 if verification.get("success") else 0.0)

    metrics = {
        "num_examples": len(predictions),
        "execution_rate": round(sum(execution_rates) / len(execution_rates), 4) if execution_rates else 0.0,
        "feasible_rate": round(sum(feasible_rates) / len(feasible_rates), 4) if feasible_rates else 0.0,
        "pass_at_1": round(sum(pass_rates) / len(pass_rates), 4) if pass_rates else 0.0,
        "mean_abs_objective_gap": round(sum(objective_gaps) / len(objective_gaps), 4) if objective_gaps else 0.0,
        "teacher_mean_score": round(sum(teacher_scores) / len(teacher_scores), 4) if teacher_scores else 0.0,
        "teacher_var_score": round(pvariance(teacher_scores), 6) if len(teacher_scores) > 1 else 0.0,
    }
    return predictions, metrics, teacher_scores

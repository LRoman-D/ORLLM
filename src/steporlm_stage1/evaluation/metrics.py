from __future__ import annotations

from statistics import pvariance

from steporlm_stage1.preference.ranking import rank_key


def _trajectory_gap(reference_value, objective_value):
    if reference_value is None or objective_value is None:
        return None
    try:
        return abs(float(objective_value) - float(reference_value))
    except (TypeError, ValueError):
        return None


def _trajectory_summary(traj: dict, reference_value) -> dict:
    verification = traj.get("verification") or {}
    objective_gap = _trajectory_gap(reference_value, verification.get("objective_value"))
    teacher = traj.get("teacher_evaluation") or {}
    try:
        teacher_score = float(teacher.get("overall_score", 0.0))
    except (TypeError, ValueError):
        teacher_score = 0.0
    return {
        "trajectory_id": traj.get("trajectory_id"),
        "status": verification.get("status"),
        "execution_ok": bool(verification.get("execution_ok")),
        "success": bool(verification.get("success")),
        "objective_match": bool(verification.get("objective_match")),
        "objective_value": verification.get("objective_value"),
        "objective_gap": objective_gap,
        "process_score": float(traj.get("process_score", 0.0) or 0.0),
        "teacher_score": teacher_score,
        "response_chars": len(str(traj.get("response", ""))),
    }


def summarize_rollout_set(rows: list[dict]) -> tuple[list[dict], list[dict], dict[str, float], list[float]]:
    predictions = []
    detailed_rows = []
    execution_rates = []
    feasible_rates = []
    optimal_exec_rates = []
    pass_rates = []
    pass_at_k_rates = []
    optimal_exec_at_k_rates = []
    execution_at_k_rates = []
    objective_gaps = []
    best_objective_gaps = []
    teacher_scores = []
    process_scores = []
    process_passes = []
    successful_trajectory_counts = []
    optimal_trajectory_counts = []

    for row in rows:
        ranked = sorted(row["trajectories"], key=rank_key, reverse=True)
        if not ranked:
            continue

        reference_value = row["reference_solution"].get("objective_value")
        best = ranked[0]
        verification = best["verification"]
        objective_gap = _trajectory_gap(reference_value, verification.get("objective_value"))
        if objective_gap is not None:
            objective_gaps.append(objective_gap)

        teacher = best.get("teacher_evaluation") or {}
        try:
            score = float(teacher["overall_score"])
            teacher_scores.append(score)
        except (KeyError, TypeError, ValueError):
            score = None
        process = best.get("process_verification") or {}
        try:
            process_scores.append(float(process.get("score", 0.0)))
            process_passes.append(1.0 if int(process.get("correct_count", 0)) >= 8 else 0.0)
        except (TypeError, ValueError):
            pass

        trajectory_summaries = [_trajectory_summary(traj, reference_value) for traj in ranked]
        success_count = sum(1 for traj in trajectory_summaries if traj["success"])
        optimal_count = sum(1 for traj in trajectory_summaries if traj["execution_ok"] and str(traj["status"]).upper() == "OPTIMAL")
        successful_trajectory_counts.append(float(success_count))
        optimal_trajectory_counts.append(float(optimal_count))
        gap_candidates = [traj["objective_gap"] for traj in trajectory_summaries if traj["objective_gap"] is not None]
        best_gap = min(gap_candidates) if gap_candidates else None
        if best_gap is not None:
            best_objective_gaps.append(best_gap)

        predictions.append(
            {
                "problem_id": row["problem_id"],
                "template_name": row["template_name"],
                "question": row["question"],
                "response": best["response"],
                "code": best["code"],
                "verification": verification,
                "teacher_evaluation": best.get("teacher_evaluation"),
                "process_verification": best.get("process_verification"),
                "objective_gap": objective_gap,
                "source": best["source"],
            }
        )
        detailed_rows.append(
            {
                "problem_id": row["problem_id"],
                "template_name": row["template_name"],
                "source_name": row.get("source_name"),
                "reference_solution": row.get("reference_solution"),
                "best_trajectory_id": best.get("trajectory_id"),
                "best_verification": verification,
                "best_objective_gap": objective_gap,
                "best_rank_key": list(rank_key(best)),
                "num_trajectories": len(ranked),
                "num_success_trajectories": success_count,
                "num_optimal_trajectories": optimal_count,
                "pass_at_k": success_count > 0,
                "optimal_execution_at_k": optimal_count > 0,
                "min_objective_gap": best_gap,
                "trajectories": trajectory_summaries,
            }
        )
        execution_rates.append(1.0 if verification.get("execution_ok") else 0.0)
        feasible_rates.append(1.0 if verification.get("status") == "OPTIMAL" else 0.0)
        optimal_exec_rates.append(1.0 if verification.get("execution_ok") and verification.get("status") == "OPTIMAL" else 0.0)
        pass_rates.append(1.0 if verification.get("success") else 0.0)
        pass_at_k_rates.append(1.0 if success_count > 0 else 0.0)
        optimal_exec_at_k_rates.append(1.0 if optimal_count > 0 else 0.0)
        execution_at_k_rates.append(1.0 if any(traj["execution_ok"] for traj in trajectory_summaries) else 0.0)

    metrics = {
        "num_examples": len(predictions),
        "avg_num_trajectories": round(sum(row["num_trajectories"] for row in detailed_rows) / len(detailed_rows), 4) if detailed_rows else 0.0,
        "execution_rate": round(sum(execution_rates) / len(execution_rates), 4) if execution_rates else 0.0,
        "feasible_rate": round(sum(feasible_rates) / len(feasible_rates), 4) if feasible_rates else 0.0,
        "optimal_execution_at_1": round(sum(optimal_exec_rates) / len(optimal_exec_rates), 4) if optimal_exec_rates else 0.0,
        "pass_at_1": round(sum(pass_rates) / len(pass_rates), 4) if pass_rates else 0.0,
        "execution_at_k": round(sum(execution_at_k_rates) / len(execution_at_k_rates), 4) if execution_at_k_rates else 0.0,
        "optimal_execution_at_k": round(sum(optimal_exec_at_k_rates) / len(optimal_exec_at_k_rates), 4) if optimal_exec_at_k_rates else 0.0,
        "pass_at_k": round(sum(pass_at_k_rates) / len(pass_at_k_rates), 4) if pass_at_k_rates else 0.0,
        "mean_abs_objective_gap": round(sum(objective_gaps) / len(objective_gaps), 4) if objective_gaps else 0.0,
        "mean_best_abs_objective_gap": round(sum(best_objective_gaps) / len(best_objective_gaps), 4) if best_objective_gaps else 0.0,
        "teacher_mean_score": round(sum(teacher_scores) / len(teacher_scores), 4) if teacher_scores else 0.0,
        "teacher_var_score": round(pvariance(teacher_scores), 6) if len(teacher_scores) > 1 else 0.0,
        "genprm_mean_score": round(sum(process_scores) / len(process_scores), 4) if process_scores else 0.0,
        "genprm_pass_rate": round(sum(process_passes) / len(process_passes), 4) if process_passes else 0.0,
        "avg_successful_trajectories": round(sum(successful_trajectory_counts) / len(successful_trajectory_counts), 4) if successful_trajectory_counts else 0.0,
        "avg_optimal_trajectories": round(sum(optimal_trajectory_counts) / len(optimal_trajectory_counts), 4) if optimal_trajectory_counts else 0.0,
    }
    return predictions, detailed_rows, metrics, teacher_scores


def summarize_best_trajectories(rows: list[dict]) -> tuple[list[dict], dict[str, float], list[float]]:
    predictions, _details, metrics, teacher_scores = summarize_rollout_set(rows)
    return predictions, metrics, teacher_scores

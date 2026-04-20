from __future__ import annotations


def teacher_score(traj: dict) -> float:
    teacher = traj.get("teacher_evaluation") or {}
    try:
        return float(teacher.get("overall_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def rank_key(traj: dict) -> tuple[int, int, float, float]:
    verification = traj["verification"]
    success = 1 if verification.get("success") else 0
    objective_match = 1 if verification.get("objective_match") else 0
    process_score = float(traj.get("process_score", 0.0))
    return success, objective_match, teacher_score(traj), process_score


def build_weight(chosen: dict, rejected: dict) -> tuple[float, str]:
    c_ver = chosen["verification"]
    r_ver = rejected["verification"]
    if c_ver.get("success") and not r_ver.get("success"):
        return 1.0, "solver_success_beats_failure"
    if c_ver.get("objective_match") and not r_ver.get("objective_match"):
        return 0.85, "objective_match_beats_mismatch"
    teacher_gap = max(0.0, teacher_score(chosen) - teacher_score(rejected))
    if teacher_gap > 0:
        return round(min(0.8, 0.35 + teacher_gap * 0.45), 4), "teacher_score_gap"
    gap = max(0.05, float(chosen.get("process_score", 0.0)) - float(rejected.get("process_score", 0.0)))
    return round(min(0.8, 0.3 + gap), 4), "process_score_gap"

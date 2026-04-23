from __future__ import annotations


def teacher_score(traj: dict) -> float:
    teacher = traj.get("teacher_evaluation") or {}
    try:
        return float(teacher.get("overall_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def process_audit_score(traj: dict) -> float:
    audit = traj.get("process_verification") or {}
    try:
        return float(audit.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def process_audit_correct_count(traj: dict) -> int:
    audit = traj.get("process_verification") or {}
    try:
        return int(audit.get("correct_count", 0))
    except (TypeError, ValueError):
        return 0


def process_audit_all_correct(traj: dict) -> int:
    return 1 if (traj.get("process_verification") or {}).get("all_correct") else 0


def rank_key(traj: dict) -> tuple[int, int, int, int, float, float, float]:
    verification = traj["verification"]
    success = 1 if verification.get("success") else 0
    objective_match = 1 if verification.get("objective_match") else 0
    process_score = float(traj.get("process_score", 0.0))
    return (
        success,
        objective_match,
        process_audit_all_correct(traj),
        process_audit_correct_count(traj),
        process_audit_score(traj),
        teacher_score(traj),
        process_score,
    )


def build_weight(chosen: dict, rejected: dict) -> tuple[float, str]:
    c_ver = chosen["verification"]
    r_ver = rejected["verification"]
    if c_ver.get("success") and not r_ver.get("success"):
        return 1.0, "solver_success_beats_failure"
    if c_ver.get("objective_match") and not r_ver.get("objective_match"):
        return 0.85, "objective_match_beats_mismatch"
    if process_audit_all_correct(chosen) and not process_audit_all_correct(rejected):
        return 0.8, "genprm_all_correct_beats_process_error"
    audit_gap = process_audit_score(chosen) - process_audit_score(rejected)
    if audit_gap > 0:
        return round(min(0.8, 0.35 + audit_gap * 0.45), 4), "genprm_score_gap"
    teacher_gap = max(0.0, teacher_score(chosen) - teacher_score(rejected))
    if teacher_gap > 0:
        return round(min(0.8, 0.35 + teacher_gap * 0.45), 4), "teacher_score_gap"
    gap = max(0.05, float(chosen.get("process_score", 0.0)) - float(rejected.get("process_score", 0.0)))
    return round(min(0.8, 0.3 + gap), 4), "process_score_gap"

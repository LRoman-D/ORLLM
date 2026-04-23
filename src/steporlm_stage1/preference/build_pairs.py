from __future__ import annotations

from pathlib import Path

from steporlm_stage1.preference.ranking import build_weight, rank_key, teacher_score
from steporlm_stage1.quality.genprm import audit_passes_threshold
from steporlm_stage1.schemas import PreferencePair
from steporlm_stage1.utils.plots import save_preference_dashboard
from steporlm_stage1.utils.io import read_jsonl, write_json, write_jsonl
from steporlm_stage1.utils.run_dirs import create_timestamped_run_dir


def build_preference_pairs(
    rollout_path: str | Path,
    output_path: str | Path | None = None,
    report_dir: str | Path | None = None,
    run_root: str | Path = "runs",
    run_prefix: str = "preferences",
    timestamped: bool = True,
    require_chosen_success: bool = False,
    require_chosen_process_pass: bool = False,
    min_correct_steps: int = 8,
) -> dict[str, int | str]:
    rows = read_jsonl(rollout_path)
    run_dir = create_timestamped_run_dir(run_root, run_prefix) if timestamped else (Path(report_dir) if report_dir else Path(run_root))
    if output_path is None:
        output_path = run_dir / "preferences.jsonl"
    if report_dir is None:
        report_dir = run_dir
    pairs = []
    weights = []
    rationales = {}
    teacher_scores = []
    skipped_for_unsuccessful_chosen = 0
    skipped_for_process = 0
    for row in rows:
        ranked = sorted(row["trajectories"], key=rank_key, reverse=True)
        if len(ranked) < 2:
            continue
        chosen = ranked[0]
        rejected = ranked[-1]
        if require_chosen_success and not bool(chosen.get("verification", {}).get("success")):
            skipped_for_unsuccessful_chosen += 1
            continue
        if require_chosen_process_pass and not audit_passes_threshold(
            chosen.get("process_verification"),
            min_correct_steps=min_correct_steps,
            require_all_correct=False,
        ):
            skipped_for_process += 1
            continue
        if chosen["trajectory_id"] == rejected["trajectory_id"]:
            continue
        weight, rationale = build_weight(chosen, rejected)
        pairs.append(
            PreferencePair(
                problem_id=row["problem_id"],
                template_name=row["template_name"],
                question=row["question"],
                chosen=chosen,
                rejected=rejected,
                weight=weight,
                rationale=rationale,
            ).to_dict()
        )
        weights.append(weight)
        rationales[rationale] = rationales.get(rationale, 0) + 1
        teacher_scores.append(teacher_score(chosen))
    write_jsonl(output_path, pairs)
    report_path = Path(report_dir)
    if pairs:
        save_preference_dashboard(weights, rationales, teacher_scores, report_path / "preference_dashboard.png")
    summary = {
        "num_pairs": len(pairs),
        "output_path": str(output_path),
        "run_dir": str(report_path),
        "rationales": rationales,
        "require_chosen_success": require_chosen_success,
        "require_chosen_process_pass": require_chosen_process_pass,
        "min_correct_steps": min_correct_steps,
        "skipped_for_unsuccessful_chosen": skipped_for_unsuccessful_chosen,
        "skipped_for_process": skipped_for_process,
    }
    write_json(report_path / "summary.json", summary)
    return summary

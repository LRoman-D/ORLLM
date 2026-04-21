from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from steporlm_stage1.evaluators.teacher import ZhipuTeacherEvaluator
from steporlm_stage1.executors.python_executor import PythonCodeExecutor
from steporlm_stage1.preference.ranking import rank_key, teacher_score
from steporlm_stage1.prompts import SYSTEM_PROMPT, build_rollout_user_prompt
from steporlm_stage1.schemas import PreferenceTrajectory, ReferenceSolution
from steporlm_stage1.utils.io import ensure_dir, load_yaml_config, read_jsonl, write_json, write_jsonl
from steporlm_stage1.utils.modeling import load_causal_lm, load_tokenizer
from steporlm_stage1.utils.plots import save_rollout_dashboard
from steporlm_stage1.utils.run_dirs import create_timestamped_run_dir
from steporlm_stage1.utils.text import extract_python_code, heuristic_process_score


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _accumulate_rollout_stats(rows: list[dict]) -> tuple[int, dict[str, int], list[float]]:
    trajectory_count = 0
    execution_statuses: dict[str, int] = {}
    teacher_scores: list[float] = []
    for row in rows:
        for traj in row.get("trajectories", []):
            trajectory_count += 1
            verification = traj.get("verification") or {}
            status = str(verification.get("status", "unknown"))
            execution_statuses[status] = execution_statuses.get(status, 0) + 1
            if "teacher_evaluation" in traj:
                teacher_scores.append(teacher_score(traj))
    return trajectory_count, execution_statuses, teacher_scores


def _write_rollout_state(
    state_path: Path,
    *,
    output_path: Path,
    total_problems: int,
    processed_problems: int,
    trajectory_count: int,
    solver_status_counts: dict[str, int],
    last_problem_id: str | None,
    completed: bool,
) -> None:
    write_json(
        state_path,
        {
            "version": 1,
            "output_path": str(output_path),
            "total_problems": total_problems,
            "processed_problems": processed_problems,
            "remaining_problems": max(0, total_problems - processed_problems),
            "num_trajectories": trajectory_count,
            "solver_status_counts": solver_status_counts,
            "last_problem_id": last_problem_id,
            "completed": completed,
        },
    )


def run_rollout_generation(
    config: dict[str, Any],
    rows: list[dict] | None = None,
    initial_rows: list[dict] | None = None,
    skip_problem_ids: set[str] | None = None,
    incremental_output_path: str | Path | None = None,
    state_path: str | Path | None = None,
) -> tuple[list[dict], dict[str, Any], list[float]]:
    dataset_rows = rows if rows is not None else read_jsonl(config["dataset_path"])
    tokenizer = load_tokenizer(config["model_path"])
    model = load_causal_lm(
        config["model_path"],
        load_in_4bit=config.get("load_in_4bit", False),
        is_adapter=config.get("is_adapter", True),
        base_model_override=config.get("base_model_path"),
    )
    model.eval()
    executor = PythonCodeExecutor(timeout_seconds=int(config.get("timeout_seconds", 20)))
    teacher = ZhipuTeacherEvaluator.from_env() if config.get("teacher_evaluation", False) else None

    if torch.cuda.is_available():
        torch.manual_seed(int(config.get("seed", 2026)))

    output_rows = list(initial_rows or [])
    skipped_ids = skip_problem_ids or set()
    trajectory_count, execution_statuses, teacher_scores = _accumulate_rollout_stats(output_rows)
    processed_problems = len(output_rows)
    total_problems = len(dataset_rows) + len(output_rows)
    incremental_target = Path(incremental_output_path) if incremental_output_path is not None else None
    state_target = Path(state_path) if state_path is not None else None
    last_problem_id: str | None = None

    if state_target is not None and incremental_target is not None:
        _write_rollout_state(
            state_target,
            output_path=incremental_target,
            total_problems=total_problems,
            processed_problems=processed_problems,
            trajectory_count=trajectory_count,
            solver_status_counts=execution_statuses,
            last_problem_id=last_problem_id,
            completed=processed_problems >= total_problems,
        )

    try:
        for row in tqdm(dataset_rows, desc=config.get("progress_label", "Generating real rollouts")):
            problem_id = str(row.get("problem_id", ""))
            if problem_id and problem_id in skipped_ids:
                continue

            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_rollout_user_prompt(row["question"], row["template_name"])},
            ]
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {key: value.to(model.device) for key, value in inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    do_sample=config.get("do_sample", True),
                    temperature=float(config.get("temperature", 0.8)),
                    top_p=float(config.get("top_p", 0.9)),
                    max_new_tokens=int(config.get("max_new_tokens", 1500)),
                    num_return_sequences=int(config.get("num_return_sequences", 3)),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            reference = ReferenceSolution(
                status=row["reference_solution"]["status"],
                objective_value=row["reference_solution"]["objective_value"],
                metadata=row["reference_solution"].get("metadata", {}),
            )
            trajectories = []
            for idx in range(generated.shape[0]):
                output_ids = generated[idx][prompt_len:]
                response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                code = extract_python_code(response)
                verification = executor.verify(code, reference)
                trajectory = PreferenceTrajectory(
                    trajectory_id=f"{row['problem_id']}::sample{idx}",
                    response=response,
                    code=code,
                    verification=verification.to_dict(),
                    process_score=heuristic_process_score(response, verification.to_dict()),
                    source=str(config["model_path"]),
                ).to_dict()
                if teacher is not None:
                    try:
                        trajectory["teacher_evaluation"] = teacher.evaluate(row["question"], response, verification.to_dict())
                        teacher_scores.append(teacher_score(trajectory))
                    except Exception as exc:  # noqa: BLE001
                        trajectory["teacher_evaluation"] = {"overall_score": 0.0, "verdict": "bad", "issues": [str(exc)]}
                trajectories.append(trajectory)
                trajectory_count += 1
                status = verification.status
                execution_statuses[status] = execution_statuses.get(status, 0) + 1

            rollout_row = {
                "problem_id": row["problem_id"],
                "template_name": row["template_name"],
                "question": row["question"],
                "reference_solution": row["reference_solution"],
                "trajectories": sorted(trajectories, key=rank_key, reverse=True),
            }
            output_rows.append(rollout_row)
            processed_problems += 1
            last_problem_id = str(row["problem_id"])

            if incremental_target is not None:
                _append_jsonl(incremental_target, [rollout_row])
            if state_target is not None and incremental_target is not None:
                _write_rollout_state(
                    state_target,
                    output_path=incremental_target,
                    total_problems=total_problems,
                    processed_problems=processed_problems,
                    trajectory_count=trajectory_count,
                    solver_status_counts=execution_statuses,
                    last_problem_id=last_problem_id,
                    completed=processed_problems >= total_problems,
                )
    except KeyboardInterrupt:
        if state_target is not None and incremental_target is not None:
            _write_rollout_state(
                state_target,
                output_path=incremental_target,
                total_problems=total_problems,
                processed_problems=processed_problems,
                trajectory_count=trajectory_count,
                solver_status_counts=execution_statuses,
                last_problem_id=last_problem_id,
                completed=False,
            )
        raise

    summary = {
        "num_problems": len(output_rows),
        "num_trajectories": trajectory_count,
        "solver_status_counts": execution_statuses,
        "teacher_enabled": teacher is not None,
    }
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_rows, summary, teacher_scores


def generate_real_rollouts_from_config(config: dict[str, Any], run_dir: str | Path | None = None) -> dict[str, Any]:
    target_dir = ensure_dir(run_dir) if run_dir is not None else create_timestamped_run_dir(config.get("run_root", "runs"), config.get("run_prefix", "real_rollouts"))
    output_path = Path(config.get("output_path") or (target_dir / "real_rollouts.jsonl"))

    resume_from_checkpoint = bool(config.get("resume_from_checkpoint", True))
    state_file = str(config.get("state_file", "rollout_state.json"))
    state_path = Path(state_file)
    if not state_path.is_absolute():
        state_path = output_path.parent / state_file

    initial_rows: list[dict] = []
    skip_problem_ids: set[str] = set()
    if resume_from_checkpoint and output_path.exists():
        initial_rows = read_jsonl(output_path)
        skip_problem_ids = {str(row.get("problem_id", "")) for row in initial_rows if row.get("problem_id")}
    elif not resume_from_checkpoint and output_path.exists():
        output_path.unlink()

    if resume_from_checkpoint:
        all_rows = read_jsonl(config["dataset_path"])
        remaining_rows = [row for row in all_rows if str(row.get("problem_id", "")) not in skip_problem_ids]
        if remaining_rows:
            rows, summary, teacher_scores = run_rollout_generation(
                config,
                rows=remaining_rows,
                initial_rows=initial_rows,
                skip_problem_ids=None,
                incremental_output_path=output_path,
                state_path=state_path,
            )
        else:
            rows = initial_rows
            trajectory_count, status_counts, teacher_scores = _accumulate_rollout_stats(rows)
            summary = {
                "num_problems": len(rows),
                "num_trajectories": trajectory_count,
                "solver_status_counts": status_counts,
                "teacher_enabled": bool(config.get("teacher_evaluation", False)),
            }
            _write_rollout_state(
                state_path,
                output_path=output_path,
                total_problems=len(rows),
                processed_problems=len(rows),
                trajectory_count=trajectory_count,
                solver_status_counts=status_counts,
                last_problem_id=(rows[-1]["problem_id"] if rows else None),
                completed=True,
            )
    else:
        rows, summary, teacher_scores = run_rollout_generation(config)
        write_jsonl(output_path, rows)

    save_rollout_dashboard(summary["solver_status_counts"], teacher_scores, target_dir / "rollout_dashboard.png")
    summary["run_dir"] = str(target_dir)
    summary["output_path"] = str(output_path)
    summary["resume_from_checkpoint"] = resume_from_checkpoint
    summary["state_path"] = str(state_path)
    write_json(target_dir / "summary.json", summary)
    return summary


def generate_real_rollouts(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    return generate_real_rollouts_from_config(config)

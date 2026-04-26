from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from steporlm_stage1.teachers.factory import build_teacher_generator, teacher_model_name
from steporlm_stage1.evaluators.teacher import ZhipuTeacherEvaluator
from steporlm_stage1.executors.python_executor import PythonCodeExecutor
from steporlm_stage1.preference.ranking import rank_key, teacher_score
from steporlm_stage1.prompts import SYSTEM_PROMPT, build_rollout_user_prompt
from steporlm_stage1.quality.genprm import audit_passes_threshold, process_score_from_audit
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


def _row_has_success(row: dict[str, Any]) -> bool:
    for traj in row.get("trajectories", []):
        verification = traj.get("verification") or {}
        if bool(verification.get("success")):
            return True
        if bool(verification.get("execution_ok")) and str(verification.get("status", "")).upper() == "OPTIMAL":
            return True
    return False


def _count_rows_with_success(rows: list[dict]) -> int:
    return sum(1 for row in rows if _row_has_success(row))


def _zero_success_streak(rows: list[dict]) -> int:
    streak = 0
    for row in reversed(rows):
        if _row_has_success(row):
            break
        streak += 1
    return streak


def _build_zero_success_alert(row: dict[str, Any], consecutive_zero_success: int) -> dict[str, Any]:
    trajectories = row.get("trajectories", [])
    return {
        "problem_id": row.get("problem_id"),
        "source_name": row.get("source_name"),
        "consecutive_zero_success": consecutive_zero_success,
        "num_trajectories": len(trajectories),
        "trajectory_summaries": [
            {
                "trajectory_id": traj.get("trajectory_id"),
                "status": (traj.get("verification") or {}).get("status"),
                "success": bool((traj.get("verification") or {}).get("success")),
                "execution_ok": bool((traj.get("verification") or {}).get("execution_ok")),
                "objective_match": bool((traj.get("verification") or {}).get("objective_match")),
                "response_preview": str(traj.get("response", ""))[:600],
            }
            for traj in trajectories
        ],
    }


def _write_rollout_state(
    state_path: Path,
    *,
    output_path: Path,
    total_problems: int,
    processed_problems: int,
    trajectory_count: int,
    solver_status_counts: dict[str, int],
    problems_with_success: int,
    consecutive_zero_success: int,
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
            "problems_with_success": problems_with_success,
            "problem_success_rate": round(problems_with_success / processed_problems, 4) if processed_problems else 0.0,
            "consecutive_zero_success": consecutive_zero_success,
            "last_problem_id": last_problem_id,
            "completed": completed,
        },
    )


def _genprm_config(config: dict[str, Any]) -> dict[str, Any]:
    explicit = config.get("genprm_teacher")
    if isinstance(explicit, dict) and explicit:
        merged = dict(explicit)
    else:
        merged = {
            "teacher_backend": "qwen_rag",
            "teacher_model_path": config.get("teacher_model_path") or config.get("base_model_path") or "models/Qwen3-8B",
            "teacher_load_in_4bit": config.get("genprm_teacher_load_in_4bit", True),
            "rag_index_dir": config.get("rag_index_dir", "data/rag/or_books"),
            "rag_candidate_top_k": config.get("rag_candidate_top_k", 30),
            "rag_top_k": config.get("rag_top_k", 5),
            "rag_max_context_chars": config.get("rag_max_context_chars", 4500),
            "rag_use_semantic_rerank": config.get("rag_use_semantic_rerank", True),
            "rag_reranker_model": config.get("rag_reranker_model", "models/msmarco-minilm-reranker"),
            "rag_reranker_batch_size": config.get("rag_reranker_batch_size", 16),
            "rag_reranker_max_length": config.get("rag_reranker_max_length", 256),
            "rag_fail_on_reranker_error": config.get("rag_fail_on_reranker_error", False),
            "qwen_enable_thinking": config.get("qwen_enable_thinking", False),
        }
    merged.setdefault("teacher_backend", "qwen_rag")
    return merged


def _audit_rollout_rows(config: dict[str, Any], rows: list[dict], output_path: Path | None = None) -> dict[str, Any]:
    if not bool(config.get("genprm_evaluation", False)):
        return {"genprm_enabled": False, "genprm_evaluated": 0, "genprm_passed": 0}

    teacher = build_teacher_generator(_genprm_config(config))
    if teacher is None or not hasattr(teacher, "audit_trajectory"):
        return {
            "genprm_enabled": True,
            "genprm_evaluated": 0,
            "genprm_passed": 0,
            "genprm_error": "teacher_backend_does_not_support_audit",
        }

    evaluated = 0
    passed = 0
    min_correct_steps = int(config.get("genprm_min_correct_steps", 8))
    require_all_correct = bool(config.get("genprm_require_all_correct", False))
    max_tokens = int(config.get("genprm_max_tokens", 4200))

    for row in tqdm(rows, desc="Auditing rollouts with GenPRM"):
        for traj in row.get("trajectories", []):
            if traj.get("process_verification"):
                audit = traj["process_verification"]
            else:
                audit = teacher.audit_trajectory(
                    question=row["question"],
                    response=traj.get("response", ""),
                    verification=traj.get("verification", {}),
                    template_name=row.get("template_name", "external_or"),
                    max_tokens=max_tokens,
                )
                traj["process_verification"] = audit
            evaluated += 1
            if audit_passes_threshold(
                audit,
                min_correct_steps=min_correct_steps,
                require_all_correct=require_all_correct,
            ):
                passed += 1
            traj["process_score"] = process_score_from_audit(
                traj.get("response", ""),
                traj.get("verification", {}),
                audit,
            )
        row["trajectories"] = sorted(row.get("trajectories", []), key=rank_key, reverse=True)

    if output_path is not None:
        write_jsonl(output_path, rows)

    model_name = teacher_model_name(teacher)
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "genprm_enabled": True,
        "genprm_teacher_model": model_name,
        "genprm_evaluated": evaluated,
        "genprm_passed": passed,
        "genprm_pass_rate": round(passed / evaluated, 4) if evaluated else 0.0,
        "genprm_min_correct_steps": min_correct_steps,
        "genprm_require_all_correct": require_all_correct,
    }


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
    model.config.use_cache = bool(config.get("use_cache_for_generation", True))
    executor = PythonCodeExecutor(timeout_seconds=int(config.get("timeout_seconds", 20)))
    teacher = ZhipuTeacherEvaluator.from_env() if config.get("teacher_evaluation", False) else None

    if torch.cuda.is_available():
        torch.manual_seed(int(config.get("seed", 2026)))

    output_rows = list(initial_rows or [])
    skipped_ids = skip_problem_ids or set()
    trajectory_count, execution_statuses, teacher_scores = _accumulate_rollout_stats(output_rows)
    processed_problems = len(output_rows)
    problems_with_success = _count_rows_with_success(output_rows)
    consecutive_zero_success = _zero_success_streak(output_rows)
    total_problems = len(dataset_rows) + len(output_rows)
    incremental_target = Path(incremental_output_path) if incremental_output_path is not None else None
    state_target = Path(state_path) if state_path is not None else None
    zero_success_alert_target = (
        incremental_target.parent / str(config.get("zero_success_alert_file", "zero_success_alerts.jsonl"))
        if incremental_target is not None
        else None
    )
    monitor_success_rate = bool(config.get("monitor_success_rate", True))
    max_consecutive_zero_success = int(config.get("max_consecutive_zero_success", 3))
    stop_on_consecutive_zero_success = bool(config.get("stop_on_consecutive_zero_success", True))
    last_problem_id: str | None = None

    if state_target is not None and incremental_target is not None:
        _write_rollout_state(
            state_target,
            output_path=incremental_target,
            total_problems=total_problems,
            processed_problems=processed_problems,
            trajectory_count=trajectory_count,
            solver_status_counts=execution_statuses,
            problems_with_success=problems_with_success,
            consecutive_zero_success=consecutive_zero_success,
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
                {"role": "user", "content": build_rollout_user_prompt(row["question"], row.get("template_name", "external_or"))},
            ]
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {key: value.to(model.device) for key, value in inputs.items()}

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=config.get("do_sample", True),
                    temperature=float(config.get("temperature", 0.8)),
                    top_p=float(config.get("top_p", 0.9)),
                    max_new_tokens=int(config.get("max_new_tokens", 1500)),
                    num_return_sequences=int(config.get("num_return_sequences", 3)),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=bool(config.get("use_cache_for_generation", True)),
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
                "template_name": row.get("template_name", "external_or"),
                "source_name": row.get("source_name"),
                "question": row["question"],
                "reference_solution": row["reference_solution"],
                "trajectories": sorted(trajectories, key=rank_key, reverse=True),
            }
            output_rows.append(rollout_row)
            processed_problems += 1
            last_problem_id = str(row["problem_id"])
            if _row_has_success(rollout_row):
                problems_with_success += 1
                consecutive_zero_success = 0
            else:
                consecutive_zero_success += 1
                if monitor_success_rate and zero_success_alert_target is not None:
                    _append_jsonl(
                        zero_success_alert_target,
                        [_build_zero_success_alert(rollout_row, consecutive_zero_success)],
                    )
                if monitor_success_rate and stop_on_consecutive_zero_success and consecutive_zero_success >= max_consecutive_zero_success:
                    if state_target is not None and incremental_target is not None:
                        _write_rollout_state(
                            state_target,
                            output_path=incremental_target,
                            total_problems=total_problems,
                            processed_problems=processed_problems,
                            trajectory_count=trajectory_count,
                            solver_status_counts=execution_statuses,
                            problems_with_success=problems_with_success,
                            consecutive_zero_success=consecutive_zero_success,
                            last_problem_id=last_problem_id,
                            completed=False,
                        )
                    raise RuntimeError(
                        "Detected consecutive rollout problems without any successful trajectory. "
                        f"Reached streak {consecutive_zero_success} at problem {last_problem_id}. "
                        f"Inspect {zero_success_alert_target} and {incremental_target} before resuming."
                    )

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
                    problems_with_success=problems_with_success,
                    consecutive_zero_success=consecutive_zero_success,
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
                problems_with_success=problems_with_success,
                consecutive_zero_success=consecutive_zero_success,
                last_problem_id=last_problem_id,
                completed=False,
            )
        raise

    summary = {
        "num_problems": len(output_rows),
        "num_trajectories": trajectory_count,
        "solver_status_counts": execution_statuses,
        "problems_with_success": problems_with_success,
        "problem_success_rate": round(problems_with_success / len(output_rows), 4) if output_rows else 0.0,
        "consecutive_zero_success": consecutive_zero_success,
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
                problems_with_success=_count_rows_with_success(rows),
                consecutive_zero_success=_zero_success_streak(rows),
                last_problem_id=(rows[-1]["problem_id"] if rows else None),
                completed=True,
            )
    else:
        rows, summary, teacher_scores = run_rollout_generation(config)
        write_jsonl(output_path, rows)

    audit_summary = _audit_rollout_rows(config, rows, output_path)
    summary.update(audit_summary)
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


def audit_rollouts(
    rollout_path: str | Path,
    config_path: str | Path = "configs/stage1_real_rollout.yaml",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    config["genprm_evaluation"] = True
    source = Path(rollout_path)
    target = Path(output_path) if output_path is not None else source
    rows = read_jsonl(source)
    summary = _audit_rollout_rows(config, rows, target)
    write_json(target.parent / "genprm_audit_summary.json", summary)
    return summary

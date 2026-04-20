from __future__ import annotations

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


def run_rollout_generation(config: dict[str, Any], rows: list[dict] | None = None) -> tuple[list[dict], dict[str, Any], list[float]]:
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

    output_rows = []
    trajectory_count = 0
    execution_statuses: dict[str, int] = {}
    teacher_scores: list[float] = []

    for row in tqdm(dataset_rows, desc=config.get("progress_label", "Generating real rollouts")):
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
                max_new_tokens=int(config.get("max_new_tokens", 900)),
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

        output_rows.append(
            {
                "problem_id": row["problem_id"],
                "template_name": row["template_name"],
                "question": row["question"],
                "reference_solution": row["reference_solution"],
                "trajectories": sorted(trajectories, key=rank_key, reverse=True),
            }
        )

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
    rows, summary, teacher_scores = run_rollout_generation(config)
    write_jsonl(output_path, rows)
    save_rollout_dashboard(summary["solver_status_counts"], teacher_scores, target_dir / "rollout_dashboard.png")
    summary["run_dir"] = str(target_dir)
    summary["output_path"] = str(output_path)
    write_json(target_dir / "summary.json", summary)
    return summary


def generate_real_rollouts(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    return generate_real_rollouts_from_config(config)

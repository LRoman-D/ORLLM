from __future__ import annotations

import random
from pathlib import Path

from steporlm_stage1.evaluation.metrics import summarize_best_trajectories
from steporlm_stage1.rollout.generate_real_rollouts import run_rollout_generation
from steporlm_stage1.utils.io import load_yaml_config, read_jsonl, write_json, write_jsonl
from steporlm_stage1.utils.plots import save_comparison_dashboard
from steporlm_stage1.utils.run_dirs import create_timestamped_run_dir


def _load_comparison_questions(config: dict, output_path: Path) -> tuple[list[dict], dict]:
    rng = random.Random(int(config.get("seed", 2026)))
    target_questions = int(config.get("target_questions", 50))
    dataset_path = config.get("dataset_path") or config.get("eval_dataset_path")
    if not dataset_path:
        raise ValueError("benchmark.dataset_path must point to external eval questions.")

    questions = read_jsonl(dataset_path)
    if target_questions > 0 and len(questions) > target_questions:
        questions = rng.sample(questions, target_questions)
    for row in questions:
        row.setdefault("template_name", "external_or")

    write_jsonl(output_path, questions)
    source_counter = {}
    for row in questions:
        source = row.get("source_name", row.get("source", "unknown"))
        source_counter[source] = source_counter.get(source, 0) + 1
    summary = {
        "num_questions": len(questions),
        "dataset_path": str(dataset_path),
        "source_distribution": source_counter,
        "output_path": str(output_path),
    }
    return questions, summary


def compare_models(config_path: str | Path) -> dict:
    config = load_yaml_config(config_path)

    run_dir = create_timestamped_run_dir(config.get("run_root", "runs"), config.get("run_prefix", "model_compare"))
    benchmark_rows, benchmark_summary = _load_comparison_questions(config["benchmark"], run_dir / "comparison_questions.jsonl")

    comparison_rows = []
    model_summaries = []
    for model_cfg in config["models"]:
        rollout_config = dict(config["rollout"])
        rollout_config["model_path"] = model_cfg["model_path"]
        rollout_config["base_model_path"] = model_cfg.get("base_model_path")
        rollout_config["is_adapter"] = bool(model_cfg.get("is_adapter", True))
        rollout_config["teacher_evaluation"] = bool(config.get("teacher_evaluation", True))
        rollout_config["progress_label"] = f"Evaluating {model_cfg['name']}"

        rollouts, _, _ = run_rollout_generation(rollout_config, rows=benchmark_rows)
        predictions, metrics, _teacher_scores = summarize_best_trajectories(rollouts)
        metrics["model_name"] = model_cfg["name"]
        model_summaries.append(metrics)

        for row in predictions:
            comparison_rows.append({"model_name": model_cfg["name"], **row})

    save_comparison_dashboard(model_summaries, run_dir / "comparison_dashboard.png")
    summary = {
        "benchmark": benchmark_summary,
        "models": model_summaries,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "comparison_summary.json", summary)
    write_jsonl(run_dir / "comparison_details.jsonl", comparison_rows)
    return summary

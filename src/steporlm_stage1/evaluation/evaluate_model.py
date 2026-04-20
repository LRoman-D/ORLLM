from __future__ import annotations

from pathlib import Path

from steporlm_stage1.evaluation.metrics import summarize_best_trajectories
from steporlm_stage1.preference.build_pairs import build_preference_pairs
from steporlm_stage1.rollout.generate_real_rollouts import generate_real_rollouts_from_config
from steporlm_stage1.utils.io import load_yaml_config, read_jsonl, write_json, write_jsonl
from steporlm_stage1.utils.plots import save_eval_dashboard
from steporlm_stage1.utils.run_dirs import create_timestamped_run_dir


def evaluate_model(config_path: str | Path) -> dict:
    config = load_yaml_config(config_path)

    run_dir = create_timestamped_run_dir(config.get("run_root", "runs"), config.get("run_prefix", "test_eval"))
    rollout_cfg = dict(config)
    rollout_cfg["output_path"] = str(run_dir / "test_rollouts.jsonl")
    rollout_summary = generate_real_rollouts_from_config(rollout_cfg, run_dir=run_dir)

    rows = read_jsonl(run_dir / "test_rollouts.jsonl")
    predictions, metrics, teacher_scores = summarize_best_trajectories(rows)
    metrics["run_dir"] = str(run_dir)
    write_jsonl(run_dir / "test_predictions.jsonl", predictions)
    write_json(run_dir / "metrics.json", metrics)
    save_eval_dashboard(metrics, teacher_scores, run_dir / "evaluation_dashboard.png")

    build_preference_pairs(
        run_dir / "test_rollouts.jsonl",
        run_dir / "test_preferences_preview.jsonl",
        report_dir=run_dir / "preference_preview",
        timestamped=False,
    )

    return {"rollout_summary": rollout_summary, "metrics": metrics}

from __future__ import annotations

from pathlib import Path
from typing import Any

from steporlm_stage1.utils.io import ensure_dir, write_json
from steporlm_stage1.utils.paths import map_repo_relative_path
from steporlm_stage1.utils.plots import save_training_dashboard
from steporlm_stage1.utils.run_dirs import create_timestamped_run_dir


def prepare_training_artifact_dirs(
    output_dir: str | Path,
    run_root: str | Path | None = None,
    run_prefix: str = "train",
) -> dict[str, Path]:
    base_output = map_repo_relative_path(output_dir)
    root = map_repo_relative_path(run_root) if run_root is not None else base_output.parent / f"{base_output.name}_runs"
    run_dir = create_timestamped_run_dir(root, run_prefix)
    weights_dir = ensure_dir(run_dir / "weights")
    metrics_dir = ensure_dir(run_dir / "metrics")
    checkpoints_dir = ensure_dir(weights_dir / "checkpoints")
    final_dir = ensure_dir(weights_dir / "final_adapter")
    return {
        "run_dir": run_dir,
        "weights_dir": weights_dir,
        "metrics_dir": metrics_dir,
        "checkpoints_dir": checkpoints_dir,
        "final_dir": final_dir,
    }


def summarize_training_history(
    log_history: list[dict[str, Any]],
    train_metrics: dict[str, Any] | None = None,
    stage_name: str = "training",
) -> dict[str, Any]:
    train_metrics = dict(train_metrics or {})
    scalar_history = [row for row in log_history if isinstance(row, dict) and "loss" in row and "step" in row]

    summary: dict[str, Any] = {
        "stage_name": stage_name,
        "num_log_points": len(scalar_history),
    }
    if train_metrics:
        summary["train_metrics"] = train_metrics

    if not scalar_history:
        return summary

    losses = [float(row["loss"]) for row in scalar_history if row.get("loss") is not None]
    steps = [int(row["step"]) for row in scalar_history]
    summary.update(
        {
            "first_step": steps[0],
            "last_step": steps[-1],
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "best_loss": min(losses),
            "loss_delta": losses[-1] - losses[0],
            "loss_reduction": losses[0] - losses[-1],
            "relative_loss_reduction": (losses[0] - losses[-1]) / losses[0] if losses[0] else 0.0,
            "final_epoch": float(scalar_history[-1].get("epoch", train_metrics.get("epoch", 0.0) or 0.0)),
            "final_learning_rate": float(scalar_history[-1].get("learning_rate", 0.0) or 0.0),
        }
    )

    for key in ("mean_token_accuracy", "entropy", "grad_norm", "reward_margin", "preference_accuracy"):
        values = [float(row[key]) for row in scalar_history if row.get(key) is not None]
        if values:
            prefix = key
            summary[f"first_{prefix}"] = values[0]
            summary[f"last_{prefix}"] = values[-1]
            summary[f"best_{prefix}"] = max(values) if key in {"mean_token_accuracy", "reward_margin", "preference_accuracy"} else min(values)

    return summary


def write_training_artifacts(
    metrics_dir: str | Path,
    summary: dict[str, Any],
    log_history: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    dashboard_title: str = "Training Dashboard",
) -> dict[str, str]:
    target = ensure_dir(metrics_dir)
    write_json(target / "summary.json", summary)
    write_json(target / "log_history.json", {"log_history": log_history})
    if config is not None:
        write_json(target / "config_snapshot.json", config)
    save_training_dashboard(log_history, summary, target / "training_dashboard.png", title=dashboard_title)
    return {
        "summary_path": str(target / "summary.json"),
        "log_history_path": str(target / "log_history.json"),
        "dashboard_path": str(target / "training_dashboard.png"),
    }

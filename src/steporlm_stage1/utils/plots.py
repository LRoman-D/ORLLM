from __future__ import annotations

from pathlib import Path

from steporlm_stage1.utils.io import ensure_dir


def _load_plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    return plt


def save_rollout_dashboard(status_counts: dict[str, int], teacher_scores: list[float], path: str | Path) -> None:
    plt = _load_plt()
    if plt is None:
        return
    target = Path(path)
    ensure_dir(target.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    labels = list(status_counts.keys()) or ["none"]
    values = list(status_counts.values()) or [0]
    axes[0].bar(labels, values, color="#2F6BFF")
    axes[0].set_title("Solver Status")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)
    if teacher_scores:
        axes[1].hist(teacher_scores, bins=min(12, max(5, len(teacher_scores) // 3)), color="#FF8C42")
        axes[1].set_title("Teacher Scores")
        axes[1].set_xlabel("overall_score")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Teacher evaluation disabled", ha="center", va="center", fontsize=11)
    fig.suptitle("Rollout Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def save_preference_dashboard(weights: list[float], rationales: dict[str, int], teacher_scores: list[float], path: str | Path) -> None:
    plt = _load_plt()
    if plt is None:
        return
    target = Path(path)
    ensure_dir(target.parent)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=220)
    axes[0].hist(weights or [0.0], bins=min(12, max(5, len(weights) // 3 if weights else 5)), color="#2F6BFF")
    axes[0].set_title("Preference Weights")
    axes[0].set_xlabel("weight")
    labels = list(rationales.keys()) or ["none"]
    values = list(rationales.values()) or [0]
    axes[1].bar(labels, values, color="#5CB85C")
    axes[1].set_title("Rationale Mix")
    axes[1].tick_params(axis="x", rotation=20)
    if teacher_scores and any(score > 0 for score in teacher_scores):
        axes[2].hist(teacher_scores, bins=min(12, max(5, len(teacher_scores) // 3)), color="#FF8C42")
        axes[2].set_title("Chosen Teacher Scores")
        axes[2].set_xlabel("overall_score")
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "No teacher scores", ha="center", va="center", fontsize=11)
    fig.suptitle("Preference Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def save_eval_dashboard(metrics: dict[str, float], teacher_scores: list[float], path: str | Path) -> None:
    plt = _load_plt()
    if plt is None:
        return
    target = Path(path)
    ensure_dir(target.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    metric_labels = ["execution_rate", "feasible_rate", "pass_at_1"]
    metric_values = [float(metrics.get(label, 0.0)) for label in metric_labels]
    axes[0].bar(metric_labels, metric_values, color=["#2F6BFF", "#5CB85C", "#E45756"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Core Metrics")
    axes[0].tick_params(axis="x", rotation=20)
    gap_text = f"mean_abs_gap = {float(metrics.get('mean_abs_objective_gap', 0.0)):.4f}"
    teacher_text = f"teacher_mean = {float(metrics.get('teacher_mean_score', 0.0)):.4f}"
    axes[0].text(0.02, 0.98, gap_text + "\n" + teacher_text, transform=axes[0].transAxes, va="top", fontsize=10)
    if teacher_scores:
        axes[1].hist(teacher_scores, bins=min(12, max(5, len(teacher_scores) // 3)), color="#FF8C42")
        axes[1].set_title("Teacher Scores")
        axes[1].set_xlabel("overall_score")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No teacher scores", ha="center", va="center", fontsize=11)
    fig.suptitle("Evaluation Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def save_comparison_dashboard(model_metrics: list[dict[str, float | str]], path: str | Path) -> None:
    plt = _load_plt()
    if plt is None:
        return
    target = Path(path)
    ensure_dir(target.parent)
    names = [str(item["model_name"]) for item in model_metrics]
    execution = [float(item.get("execution_rate", 0.0)) for item in model_metrics]
    success = [float(item.get("pass_at_1", 0.0)) for item in model_metrics]
    gap = [float(item.get("mean_abs_objective_gap", 0.0)) for item in model_metrics]
    teacher_mean = [float(item.get("teacher_mean_score", 0.0)) for item in model_metrics]
    teacher_std = [float(item.get("teacher_var_score", 0.0)) ** 0.5 for item in model_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=220)
    x_positions = range(len(names))
    axes[0].bar(x_positions, execution, width=0.35, label="execution", color="#2F6BFF")
    axes[0].bar([x + 0.35 for x in x_positions], success, width=0.35, label="pass@1", color="#5CB85C")
    axes[0].set_xticks([x + 0.175 for x in x_positions], names)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Solver Metrics")
    axes[0].legend()

    axes[1].bar(names, gap, color="#E45756")
    axes[1].set_title("Mean Abs Objective Gap")
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(names, teacher_mean, yerr=teacher_std, color="#FF8C42", capsize=5)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Teacher Score Mean +/- Std")
    axes[2].tick_params(axis="x", rotation=15)

    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def save_training_dashboard(log_history: list[dict], summary: dict, path: str | Path, title: str = "Training Dashboard") -> None:
    plt = _load_plt()
    if plt is None:
        return
    rows = [row for row in log_history if isinstance(row, dict) and "loss" in row and "step" in row]
    if not rows:
        return

    target = Path(path)
    ensure_dir(target.parent)
    steps = [int(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    learning_rates = [float(row.get("learning_rate", 0.0) or 0.0) for row in rows]
    grad_norms = [float(row.get("grad_norm", 0.0) or 0.0) for row in rows]

    accuracy_values = [row.get("mean_token_accuracy") for row in rows]
    reward_margin_values = [row.get("reward_margin") for row in rows]
    preference_acc_values = [row.get("preference_accuracy") for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=220)

    axes[0, 0].plot(steps, losses, color="#2F6BFF", linewidth=2.2, marker="o")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(steps, learning_rates, color="#E45756", linewidth=2.0, marker="o")
    axes[0, 1].set_title("Learning Rate")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("LR")

    if any(value is not None for value in accuracy_values):
        y_values = [float(value or 0.0) for value in accuracy_values]
        axes[1, 0].plot(steps, y_values, color="#5CB85C", linewidth=2.2, marker="o")
        axes[1, 0].set_ylim(0.0, 1.05)
        axes[1, 0].set_title("Mean Token Accuracy")
        axes[1, 0].set_ylabel("Accuracy")
    elif any(value is not None for value in reward_margin_values):
        y_values = [float(value or 0.0) for value in reward_margin_values]
        axes[1, 0].plot(steps, y_values, color="#5CB85C", linewidth=2.2, marker="o")
        axes[1, 0].set_title("Reward Margin")
        axes[1, 0].set_ylabel("Margin")
    elif any(value is not None for value in preference_acc_values):
        y_values = [float(value or 0.0) for value in preference_acc_values]
        axes[1, 0].plot(steps, y_values, color="#5CB85C", linewidth=2.2, marker="o")
        axes[1, 0].set_ylim(0.0, 1.05)
        axes[1, 0].set_title("Preference Accuracy")
        axes[1, 0].set_ylabel("Accuracy")
    else:
        axes[1, 0].plot(steps, grad_norms, color="#5CB85C", linewidth=2.2, marker="o")
        axes[1, 0].set_title("Grad Norm")
        axes[1, 0].set_ylabel("Norm")
    axes[1, 0].set_xlabel("Step")

    axes[1, 1].axis("off")
    text_lines = [
        f"log points: {summary.get('num_log_points', 0)}",
        f"first loss: {float(summary.get('first_loss', 0.0)):.4f}" if summary.get("first_loss") is not None else "first loss: n/a",
        f"last loss: {float(summary.get('last_loss', 0.0)):.4f}" if summary.get("last_loss") is not None else "last loss: n/a",
        f"best loss: {float(summary.get('best_loss', 0.0)):.4f}" if summary.get("best_loss") is not None else "best loss: n/a",
        f"loss reduction: {float(summary.get('loss_reduction', 0.0)):.4f}",
        f"relative reduction: {float(summary.get('relative_loss_reduction', 0.0)):.2%}",
    ]
    if summary.get("last_mean_token_accuracy") is not None:
        text_lines.append(f"final token acc: {float(summary['last_mean_token_accuracy']):.4f}")
    if summary.get("best_mean_token_accuracy") is not None:
        text_lines.append(f"best token acc: {float(summary['best_mean_token_accuracy']):.4f}")
    if summary.get("last_reward_margin") is not None:
        text_lines.append(f"final reward margin: {float(summary['last_reward_margin']):.4f}")
    if summary.get("last_preference_accuracy") is not None:
        text_lines.append(f"final pref acc: {float(summary['last_preference_accuracy']):.4f}")
    if summary.get("final_epoch") is not None:
        text_lines.append(f"final epoch: {float(summary['final_epoch']):.3f}")
    axes[1, 1].text(0.02, 0.98, "\n".join(text_lines), va="top", ha="left", fontsize=11)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)

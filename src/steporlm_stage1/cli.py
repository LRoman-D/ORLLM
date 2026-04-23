from __future__ import annotations

import typer

from steporlm_stage1.utils.io import ensure_project_dirs

app = typer.Typer(help="Qwen3-8B ORLLM loop with RAG teacher, GenPRM audit, SFT, rollout, and DPO.")
ensure_project_dirs()


@app.command("generate-dataset")
def generate_dataset(config_path: str = "configs/stage1_data.yaml") -> None:
    from steporlm_stage1.data_factory.pipeline import Stage1DataFactory

    summary = Stage1DataFactory.from_yaml(config_path).generate()
    typer.echo(summary)


@app.command("build-rag-index")
def build_rag_index_cmd(config_path: str = "configs/rag_index.yaml") -> None:
    from steporlm_stage1.rag.index import build_rag_index
    from steporlm_stage1.utils.io import load_yaml_config

    config = load_yaml_config(config_path)
    summary = build_rag_index(**config)
    typer.echo(summary)


@app.command("evaluate-rag")
def evaluate_rag_cmd(config_path: str = "configs/rag_eval.yaml") -> None:
    from steporlm_stage1.rag.evaluation import evaluate_rag_retrieval

    summary = evaluate_rag_retrieval(config_path)
    typer.echo(summary)


@app.command("summarize-sft-quality")
def summarize_sft_quality_cmd(
    dataset_dir: str = "data/processed/qwen3_rag_teacher",
    output_path: str | None = None,
) -> None:
    from steporlm_stage1.rag.evaluation import summarize_sft_quality

    summary = summarize_sft_quality(dataset_dir, output_path)
    typer.echo(summary)


@app.command("prepare-sft")
def prepare_sft(
    input_dir: str = "data/processed/qwen3_rag_teacher",
    output_dir: str = "data/processed/qwen3_sft",
) -> None:
    from steporlm_stage1.training.sft.prepare_dataset import prepare_sft_dataset

    summary = prepare_sft_dataset(input_dir, output_dir)
    typer.echo(summary)


@app.command("generate-real-rollouts")
def generate_real_rollouts_cmd(config_path: str = "configs/stage1_real_rollout.yaml") -> None:
    from steporlm_stage1.rollout.generate_real_rollouts import generate_real_rollouts

    summary = generate_real_rollouts(config_path)
    typer.echo(summary)


@app.command("audit-rollouts")
def audit_rollouts_cmd(
    rollout_path: str,
    config_path: str = "configs/stage1_real_rollout.yaml",
    output_path: str | None = None,
) -> None:
    from steporlm_stage1.rollout.generate_real_rollouts import audit_rollouts

    summary = audit_rollouts(rollout_path, config_path, output_path)
    typer.echo(summary)


@app.command("build-preferences")
def build_preferences(
    rollout_path: str = "runs/qwen3_8b/rollouts/real_rollouts.jsonl",
    output_path: str | None = None,
    run_root: str = "runs/qwen3_8b",
    run_prefix: str = "preferences",
    require_chosen_success: bool = typer.Option(
        False,
        "--require-chosen-success/--allow-unsuccessful-chosen",
        help="Only keep preference pairs whose chosen trajectory already passed solver verification.",
    ),
    require_chosen_process_pass: bool = typer.Option(
        False,
        "--require-chosen-process-pass/--allow-process-warnings",
        help="Only keep preference pairs whose chosen trajectory passed the GenPRM process threshold.",
    ),
    min_correct_steps: int = typer.Option(8, "--min-correct-steps", help="Minimum GenPRM-correct steps for chosen trajectories."),
) -> None:
    from steporlm_stage1.preference.build_pairs import build_preference_pairs

    summary = build_preference_pairs(
        rollout_path,
        output_path,
        run_root=run_root,
        run_prefix=run_prefix,
        timestamped=True,
        require_chosen_success=require_chosen_success,
        require_chosen_process_pass=require_chosen_process_pass,
        min_correct_steps=min_correct_steps,
    )
    typer.echo(summary)


@app.command("evaluate")
def evaluate(
    dataset_path: str = "data/processed/qwen3_rag_teacher/test.jsonl",
    predictions_path: str | None = None,
    output_path: str = "reports/eval_stage1.json",
) -> None:
    from steporlm_stage1.evaluation.run_eval import evaluate_predictions

    metrics = evaluate_predictions(dataset_path, predictions_path, output_path)
    typer.echo(metrics)


@app.command("evaluate-model")
def evaluate_model_cmd(config_path: str = "configs/stage1_real_rollout.yaml") -> None:
    from steporlm_stage1.evaluation.evaluate_model import evaluate_model

    summary = evaluate_model(config_path)
    typer.echo(summary)


@app.command("compare-models")
def compare_models_cmd(config_path: str = "configs/stage1_compare.yaml") -> None:
    from steporlm_stage1.evaluation.compare_models import compare_models

    summary = compare_models(config_path)
    typer.echo(summary)


@app.command("rag-ablation")
def rag_ablation_cmd(config_path: str = "configs/rag_ablation_20_semantic_v2.yaml") -> None:
    from steporlm_stage1.evaluation.rag_ablation import run_rag_ablation

    summary = run_rag_ablation(config_path)
    typer.echo(summary)


@app.command("train-lora")
def train_lora_cmd(config_path: str = "configs/stage1_sft.yaml") -> None:
    from steporlm_stage1.training.sft.train_lora import train_lora

    summary = train_lora(config_path)
    typer.echo(summary)


@app.command("prepare-dpo")
def prepare_dpo_cmd(config_path: str = "configs/stage1_dpo_data.yaml") -> None:
    from steporlm_stage1.training.dpo.prepare_dataset import prepare_dpo_dataset

    summary = prepare_dpo_dataset(config_path)
    typer.echo(summary)


@app.command("train-dpo")
def train_dpo_cmd(config_path: str = "configs/stage1_dpo_train.yaml") -> None:
    from steporlm_stage1.training.dpo.weighted_dpo import train_weighted_dpo

    summary = train_weighted_dpo(config_path)
    typer.echo(summary)


if __name__ == "__main__":
    app()

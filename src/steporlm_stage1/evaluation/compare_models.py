from __future__ import annotations

import random
from pathlib import Path

import torch

from steporlm_stage1.data_factory.teacher_factory import build_teacher_generator, teacher_model_name
from steporlm_stage1.evaluation.metrics import summarize_best_trajectories
from steporlm_stage1.rollout.generate_real_rollouts import run_rollout_generation
from steporlm_stage1.templates.registry import TEMPLATE_REGISTRY
from steporlm_stage1.utils.io import load_yaml_config, write_json, write_jsonl
from steporlm_stage1.utils.plots import save_comparison_dashboard
from steporlm_stage1.utils.run_dirs import create_timestamped_run_dir


def _generate_comparison_questions(config: dict, output_path: Path) -> tuple[list[dict], dict]:
    rng = random.Random(int(config.get("seed", 2026)))
    target_questions = int(config.get("target_questions", 50))
    max_seed_problems = int(config.get("max_seed_problems", target_questions))
    include_canonical = bool(config.get("include_canonical_question", True))
    question_variants_per_seed = int(config.get("question_variants_per_seed", 1))
    rewrite_styles = list(config.get("question_rewrite_styles", []))
    template_weights = config["template_weights"]
    generator = build_teacher_generator(config)
    if generator is None:
        raise RuntimeError("Teacher generator is required for comparison question generation.")

    questions = []
    template_counter = {}
    for seed_idx in range(max_seed_problems):
        if len(questions) >= target_questions:
            break
        names = list(template_weights.keys())
        probs = [template_weights[name] for name in names]
        template_name = rng.choices(names, weights=probs, k=1)[0]
        template = TEMPLATE_REGISTRY[template_name]
        instance = template.sample_instance(rng)
        canonical_question = template.render_question(instance, rng)
        reference = template.solve_reference(instance)

        variants = [canonical_question] if include_canonical else []
        if question_variants_per_seed > len(variants):
            rewrites = generator.rewrite_question_variants(
                template_name=template_name,
                canonical_question=canonical_question,
                instance=instance,
                num_variants=question_variants_per_seed - len(variants),
                rewrite_styles=rewrite_styles,
                temperature=float(config.get("question_rewrite_temperature", 0.6)),
                max_tokens=int(config.get("question_rewrite_max_tokens", 1200)),
            )
            for item in rewrites:
                if item not in variants:
                    variants.append(item)
        if not variants:
            variants = [canonical_question]

        for variant_idx, question in enumerate(variants[:question_variants_per_seed]):
            questions.append(
                {
                    "problem_id": f"{template_name}-compare{seed_idx:04d}-q{variant_idx:02d}",
                    "template_name": template_name,
                    "question": question,
                    "instance": instance,
                    "reference_solution": reference.to_dict(),
                }
            )
            template_counter[template_name] = template_counter.get(template_name, 0) + 1
            if len(questions) >= target_questions:
                break

    write_jsonl(output_path, questions)
    summary = {
        "num_questions": len(questions),
        "teacher_model": teacher_model_name(generator),
        "template_distribution": template_counter,
        "output_path": str(output_path),
    }
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return questions, summary


def compare_models(config_path: str | Path) -> dict:
    config = load_yaml_config(config_path)

    run_dir = create_timestamped_run_dir(config.get("run_root", "runs"), config.get("run_prefix", "model_compare"))
    benchmark_rows, benchmark_summary = _generate_comparison_questions(config["benchmark"], run_dir / "comparison_questions.jsonl")

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

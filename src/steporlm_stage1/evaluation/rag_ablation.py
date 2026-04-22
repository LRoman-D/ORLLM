from __future__ import annotations

import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from steporlm_stage1.data_factory.qwen_rag_teacher import QwenRagTeacherGenerator
from steporlm_stage1.executors.python_executor import PythonCodeExecutor
from steporlm_stage1.schemas import ReferenceSolution
from steporlm_stage1.templates.registry import TEMPLATE_REGISTRY
from steporlm_stage1.utils.io import load_yaml_config, write_json, write_jsonl
from steporlm_stage1.utils.run_dirs import create_timestamped_run_dir
from steporlm_stage1.utils.text import extract_python_code


def _generate_shared_questions(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(int(config.get("seed", 2026)))
    target_questions = int(config.get("target_questions", 40))
    template_weights = dict(config.get("template_weights", {}))
    if not template_weights:
        raise ValueError("benchmark.template_weights must not be empty")

    template_counter: Counter[str] = Counter()
    questions: list[dict[str, Any]] = []
    names = list(template_weights.keys())
    probs = [template_weights[name] for name in names]
    for idx in range(target_questions):
        template_name = rng.choices(names, weights=probs, k=1)[0]
        template = TEMPLATE_REGISTRY[template_name]
        instance = template.sample_instance(rng)
        question = template.render_question(instance, rng)
        reference = template.solve_reference(instance)
        questions.append(
            {
                "problem_id": f"{template_name}-ablation{idx:04d}",
                "template_name": template_name,
                "question": question,
                "instance": instance,
                "reference_solution": reference.to_dict(),
            }
        )
        template_counter[template_name] += 1

    summary = {
        "num_questions": len(questions),
        "template_distribution": dict(template_counter),
    }
    return questions, summary


def _build_temperatures(rollout_config: dict[str, Any]) -> list[float]:
    explicit = rollout_config.get("trajectory_temperatures")
    if explicit:
        return [float(item) for item in explicit]

    count = int(rollout_config.get("num_return_sequences", 2))
    base_temp = float(rollout_config.get("temperature", 0.7))
    if count <= 1:
        return [base_temp]
    if count == 2:
        return [max(0.05, round(base_temp - 0.2, 2)), base_temp]

    low = max(0.05, base_temp - 0.25)
    high = min(1.2, base_temp + 0.15)
    step = (high - low) / (count - 1)
    return [round(low + idx * step, 2) for idx in range(count)]


def _summarize_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_questions = len(rows)
    total_trajectories = 0
    successful_trajectories = 0
    question_successes = 0
    execution_ok_count = 0
    status_counts: Counter[str] = Counter()
    objective_gap_sum = 0.0
    objective_gap_count = 0
    generation_seconds = 0.0

    for row in rows:
        generation_seconds += float(row.get("generation_seconds", 0.0))
        any_success = False
        reference_value = row.get("reference_solution", {}).get("objective_value")
        for traj in row.get("trajectories", []):
            total_trajectories += 1
            verification = traj.get("verification", {})
            if bool(verification.get("execution_ok")):
                execution_ok_count += 1
            status = str(verification.get("status", "unknown"))
            status_counts[status] += 1
            if bool(verification.get("success")):
                successful_trajectories += 1
                any_success = True

            objective_value = verification.get("objective_value")
            if reference_value is not None and objective_value is not None:
                try:
                    objective_gap_sum += abs(float(objective_value) - float(reference_value))
                    objective_gap_count += 1
                except (TypeError, ValueError):
                    pass
        if any_success:
            question_successes += 1

    return {
        "num_questions": total_questions,
        "num_trajectories": total_trajectories,
        "trajectory_success_rate": round(successful_trajectories / total_trajectories, 4) if total_trajectories else 0.0,
        "question_success_rate": round(question_successes / total_questions, 4) if total_questions else 0.0,
        "execution_rate": round(execution_ok_count / total_trajectories, 4) if total_trajectories else 0.0,
        "mean_abs_objective_gap": round(objective_gap_sum / objective_gap_count, 4) if objective_gap_count else 0.0,
        "status_counts": dict(status_counts),
        "total_generation_seconds": round(generation_seconds, 3),
        "avg_generation_seconds_per_question": round(generation_seconds / total_questions, 3) if total_questions else 0.0,
    }


def run_rag_ablation(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    benchmark_config = dict(config["benchmark"])
    rollout_config = dict(config["rollout"])

    run_dir = create_timestamped_run_dir(config.get("run_root", "runs"), config.get("run_prefix", "rag_ablation"))
    questions, question_summary = _generate_shared_questions(benchmark_config)
    write_jsonl(run_dir / "shared_questions.jsonl", questions)

    generator = QwenRagTeacherGenerator(
        model_path=benchmark_config["teacher_model_path"],
        rag_index_dir=benchmark_config.get("rag_index_dir", "data/rag/or_books"),
        load_in_4bit=bool(benchmark_config.get("teacher_load_in_4bit", True)),
        max_context_chars=int(benchmark_config.get("rag_max_context_chars", 5000)),
        top_k=int(benchmark_config.get("rag_top_k", 5)),
        candidate_top_k=int(benchmark_config.get("rag_candidate_top_k", 25)),
        use_semantic_rerank=bool(benchmark_config.get("rag_use_semantic_rerank", True)),
        reranker_model_name_or_path=str(benchmark_config.get("rag_reranker_model", "BAAI/bge-reranker-base")),
        reranker_batch_size=int(benchmark_config.get("rag_reranker_batch_size", 8)),
        reranker_max_length=int(benchmark_config.get("rag_reranker_max_length", 512)),
        use_query_rewrite=bool(benchmark_config.get("rag_use_query_rewrite", False)),
        fail_on_reranker_error=bool(benchmark_config.get("rag_fail_on_reranker_error", False)),
        use_bf16_if_available=bool(benchmark_config.get("use_bf16_if_available", True)),
    )
    executor = PythonCodeExecutor(timeout_seconds=int(rollout_config.get("timeout_seconds", 20)))
    temperatures = _build_temperatures(rollout_config)
    max_tokens = int(rollout_config.get("max_new_tokens", 700))

    modes = [
        ("with_rag", int(benchmark_config.get("rag_top_k", 5))),
        ("without_rag", int(rollout_config.get("no_rag_top_k", 0))),
    ]

    mode_summaries: dict[str, dict[str, Any]] = {}
    for mode_name, top_k in modes:
        generator.top_k = top_k
        mode_rows: list[dict[str, Any]] = []
        for row in tqdm(questions, desc=f"{mode_name} trajectories"):
            reference_dict = row["reference_solution"]
            reference = ReferenceSolution(
                status=str(reference_dict["status"]),
                objective_value=reference_dict.get("objective_value"),
                metadata=reference_dict.get("metadata", {}),
            )

            started = time.perf_counter()
            responses = generator.generate_trajectories(
                question=row["question"],
                template_name=row["template_name"],
                temperatures=temperatures,
                max_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - started
            if len(responses) < len(temperatures):
                responses.extend([""] * (len(temperatures) - len(responses)))

            trajectories: list[dict[str, Any]] = []
            for idx, response in enumerate(responses[: len(temperatures)]):
                code = extract_python_code(response)
                verification = executor.verify(code, reference).to_dict()
                trajectories.append(
                    {
                        "trajectory_id": f"{row['problem_id']}::{mode_name}::sample{idx}",
                        "temperature": temperatures[idx],
                        "response": response,
                        "code": code,
                        "verification": verification,
                        "retrieval_contexts": list(generator.last_retrieval_contexts),
                    }
                )

            mode_rows.append(
                {
                    "problem_id": row["problem_id"],
                    "template_name": row["template_name"],
                    "question": row["question"],
                    "instance": row["instance"],
                    "reference_solution": row["reference_solution"],
                    "mode": mode_name,
                    "rag_top_k": top_k,
                    "generation_seconds": elapsed,
                    "trajectories": trajectories,
                }
            )

        write_jsonl(run_dir / f"{mode_name}_results.jsonl", mode_rows)
        mode_summaries[mode_name] = _summarize_mode(mode_rows)
        mode_summaries[mode_name]["rag_top_k"] = top_k

    with_rag = mode_summaries["with_rag"]
    without_rag = mode_summaries["without_rag"]
    deltas = {
        "trajectory_success_rate_delta": round(with_rag["trajectory_success_rate"] - without_rag["trajectory_success_rate"], 4),
        "question_success_rate_delta": round(with_rag["question_success_rate"] - without_rag["question_success_rate"], 4),
        "execution_rate_delta": round(with_rag["execution_rate"] - without_rag["execution_rate"], 4),
        "mean_abs_objective_gap_delta": round(with_rag["mean_abs_objective_gap"] - without_rag["mean_abs_objective_gap"], 4),
        "avg_generation_seconds_per_question_delta": round(
            with_rag["avg_generation_seconds_per_question"] - without_rag["avg_generation_seconds_per_question"],
            3,
        ),
    }
    summary = {
        "run_dir": str(run_dir),
        "question_generation": question_summary,
        "rollout": {
            "num_return_sequences": len(temperatures),
            "trajectory_temperatures": temperatures,
            "max_new_tokens": max_tokens,
            "timeout_seconds": int(rollout_config.get("timeout_seconds", 20)),
        },
        "modes": mode_summaries,
        "deltas": deltas,
    }
    write_json(run_dir / "rag_ablation_summary.json", summary)

    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary

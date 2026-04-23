#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from steporlm_stage1.executors.python_executor import PythonCodeExecutor
from steporlm_stage1.prompts import SYSTEM_PROMPT
from steporlm_stage1.schemas import ReferenceSolution
from steporlm_stage1.utils.io import ensure_dir, read_jsonl, write_json
from steporlm_stage1.utils.modeling import load_causal_lm, load_tokenizer
from steporlm_stage1.utils.text import extract_python_code


BASE_MODEL = "models/Qwen3-8B"
SFT_ADAPTER = "artifacts/qwen3-8b/sft-lora"
DPO_ADAPTER = "artifacts/qwen3-8b/dpo-lora"


def _append_jsonl(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _score_trajectories(trajectories: list[dict]) -> tuple[bool, bool, float | None]:
    if not trajectories:
        return False, False, None
    first_success = bool(trajectories[0].get("verification", {}).get("success"))
    any_success = any(bool(item.get("verification", {}).get("success")) for item in trajectories)
    gaps = []
    for item in trajectories:
        gap = _safe_float(item.get("verification", {}).get("objective_gap"))
        if gap is not None:
            gaps.append(gap)
    return first_success, any_success, min(gaps) if gaps else None


def _summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["benchmark"], row["model_name"]), []).append(row)

    summary_rows = []
    for (benchmark, model_name), items in sorted(grouped.items()):
        first_successes = []
        any_successes = []
        execution_any = []
        best_gaps = []
        for item in items:
            first_success, any_success, best_gap = _score_trajectories(item.get("trajectories", []))
            first_successes.append(1.0 if first_success else 0.0)
            any_successes.append(1.0 if any_success else 0.0)
            execution_any.append(
                1.0
                if any(bool(traj.get("verification", {}).get("execution_ok")) for traj in item.get("trajectories", []))
                else 0.0
            )
            if best_gap is not None:
                best_gaps.append(best_gap)
        n = len(items)
        summary_rows.append(
            {
                "benchmark": benchmark,
                "model_name": model_name,
                "num_questions": n,
                "num_trajectories": sum(len(item.get("trajectories", [])) for item in items),
                "pass_at_1": round(sum(first_successes) / n, 4) if n else 0.0,
                "success_at_2": round(sum(any_successes) / n, 4) if n else 0.0,
                "execution_any_rate": round(sum(execution_any) / n, 4) if n else 0.0,
                "mean_best_abs_gap": round(sum(best_gaps) / len(best_gaps), 4) if best_gaps else None,
            }
        )
    return summary_rows


def _write_table(run_dir: Path, summary_rows: list[dict]) -> None:
    csv_path = run_dir / "comparison_table.csv"
    md_path = run_dir / "comparison_table.md"
    fields = [
        "benchmark",
        "model_name",
        "num_questions",
        "num_trajectories",
        "pass_at_1",
        "success_at_2",
        "execution_any_rate",
        "mean_best_abs_gap",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = ["| Benchmark | Model | Q | Traj | Pass@1 | Success@2 | Exec any | Mean best gap |"]
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        gap = row["mean_best_abs_gap"]
        gap_text = "" if gap is None else str(gap)
        lines.append(
            "| {benchmark} | {model_name} | {num_questions} | {num_trajectories} | "
            "{pass_at_1:.4f} | {success_at_2:.4f} | {execution_any_rate:.4f} | {gap} |".format(
                gap=gap_text,
                **row,
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_done(output_path: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    if not output_path.exists():
        return [], set()
    rows = read_jsonl(output_path)
    done = {(str(row["model_name"]), str(row["problem_id"])) for row in rows}
    return rows, done


def _prompt_for(question: str) -> list[dict]:
    user = f"""Below is an external operations research benchmark question.

Build a correct mathematical optimization model and solve it with executable Python code using OR-Tools.
The code must print exactly one JSON result marker:
print("__STEPORLM_RESULT__=" + json.dumps({{"status": "OPTIMAL", "objective_value": value}}, ensure_ascii=False))

Use CBC_MIXED_INTEGER_PROGRAMMING for integer/binary/mixed-integer models and GLOP only for continuous linear programs.
If the problem asks for a maximum or minimum objective value, set objective_value to that objective value.
Do not use unavailable commercial solvers.

Question:
{question}
"""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _extract_first_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if match is None:
        return None
    return _safe_float(match.group(0))


def _verify_response(
    executor: PythonCodeExecutor,
    response: str,
    reference_value: float,
    timeout_tolerance: float,
) -> tuple[str, dict]:
    code = extract_python_code(response)
    reference = ReferenceSolution(status="OPTIMAL", objective_value=reference_value, metadata={})
    tolerance = max(timeout_tolerance, abs(reference_value) * 1e-5)
    verification = executor.verify(code, reference, tolerance=tolerance).to_dict()
    objective = _safe_float(verification.get("objective_value"))
    if objective is not None:
        verification["objective_gap"] = abs(objective - reference_value)
    else:
        fallback = _extract_first_number(response)
        verification["fallback_numeric_answer"] = fallback
        verification["fallback_numeric_match"] = (
            fallback is not None and abs(fallback - reference_value) <= tolerance
        )
        verification["objective_gap"] = abs(fallback - reference_value) if fallback is not None else None
    return code, verification


def _evaluate_model(
    *,
    model_name: str,
    model_path: str,
    is_adapter: bool,
    base_model_path: str | None,
    examples: list[dict],
    output_path: Path,
    state_path: Path,
    done: set[tuple[str, str]],
    existing_rows: list[dict],
    args: argparse.Namespace,
) -> None:
    tokenizer = load_tokenizer(model_path)
    model = load_causal_lm(
        model_path,
        load_in_4bit=args.load_in_4bit,
        use_bf16_if_available=True,
        is_adapter=is_adapter,
        base_model_override=base_model_path,
    )
    model.eval()
    model.config.use_cache = True
    executor = PythonCodeExecutor(timeout_seconds=args.execution_timeout)

    if torch.cuda.is_available():
        torch.manual_seed(args.seed)

    for example in tqdm(examples, desc=f"Evaluating {model_name}"):
        key = (model_name, example["problem_id"])
        if key in done:
            continue

        prompt_text = tokenizer.apply_chat_template(_prompt_for(example["question"]), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

        start = time.time()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        trajectories = []
        for idx in range(generated.shape[0]):
            response = tokenizer.decode(generated[idx][prompt_len:], skip_special_tokens=True).strip()
            code, verification = _verify_response(executor, response, float(example["answer"]), args.absolute_tolerance)
            trajectories.append(
                {
                    "trajectory_id": f"{example['problem_id']}::{model_name}::sample{idx}",
                    "response": response,
                    "code": code,
                    "verification": verification,
                }
            )
        elapsed = round(time.time() - start, 3)
        row = {
            "model_name": model_name,
            "model_path": model_path,
            "problem_id": example["problem_id"],
            "benchmark": example["benchmark"],
            "question": example["question"],
            "reference_answer": example["answer"],
            "answer_key": example.get("answer_key"),
            "metadata": example.get("metadata", {}),
            "elapsed_seconds": elapsed,
            "trajectories": trajectories,
        }
        _append_jsonl(output_path, row)
        existing_rows.append(row)
        done.add(key)

        summary_rows = _summarize(existing_rows)
        _write_table(output_path.parent, summary_rows)
        write_json(
            state_path,
            {
                "completed": False,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "output_path": str(output_path),
                "processed_model_problem_pairs": len(done),
                "total_model_problem_pairs": getattr(args, "total_model_problem_pairs", len(examples)),
                "last_model": model_name,
                "last_problem_id": example["problem_id"],
                "summary": summary_rows,
            },
        )

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base and SFT Qwen models on external OR benchmarks.")
    parser.add_argument("--dataset", default="benchmarks/or_external/prepared/benchmark_samples.jsonl")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--num-return-sequences", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--execution-timeout", type=int, default=45)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    examples = read_jsonl(args.dataset)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir or f"benchmarks/or_external/runs/eval_{timestamp}")
    ensure_dir(run_dir)
    output_path = run_dir / "rollouts.jsonl"
    state_path = run_dir / "state.json"

    existing_rows, done = _load_done(output_path)
    models = [
        {"name": "qwen3_base", "path": BASE_MODEL, "is_adapter": False, "base": None},
        {"name": "qwen3_sft", "path": SFT_ADAPTER, "is_adapter": True, "base": BASE_MODEL},
        {"name": "qwen3_dpo", "path": DPO_ADAPTER, "is_adapter": True, "base": BASE_MODEL},
    ]
    models = [spec for spec in models if not spec["is_adapter"] or Path(spec["path"]).exists()]
    args.total_model_problem_pairs = len(examples) * len(models)

    write_json(
        run_dir / "run_config.json",
        {
            "dataset": args.dataset,
            "num_examples": len(examples),
            "models": models,
            "generation": {
                "num_return_sequences": args.num_return_sequences,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
        },
    )

    try:
        for spec in models:
            _evaluate_model(
                model_name=spec["name"],
                model_path=spec["path"],
                is_adapter=spec["is_adapter"],
                base_model_path=spec["base"],
                examples=examples,
                output_path=output_path,
                state_path=state_path,
                done=done,
                existing_rows=existing_rows,
                args=args,
            )
    finally:
        summary_rows = _summarize(existing_rows)
        _write_table(run_dir, summary_rows)
        completed = len(done) >= len(examples) * len(models)
        summary_payload = {
            "completed": completed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "num_examples": len(examples),
            "processed_model_problem_pairs": len(done),
            "total_model_problem_pairs": len(examples) * len(models),
            "summary": summary_rows,
            "output_path": str(output_path),
            "comparison_table_csv": str(run_dir / "comparison_table.csv"),
            "comparison_table_md": str(run_dir / "comparison_table.md"),
        }
        write_json(
            run_dir / "summary.json",
            summary_payload,
        )
        write_json(state_path, summary_payload)
    print((run_dir / "comparison_table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

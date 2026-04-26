#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return {}


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            count += 1
    return count


def _count_unique_questions(path: Path) -> int:
    if not path.exists():
        return 0
    questions: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = str(row.get("question", "")).strip()
            if question:
                questions.add(question)
    return len(questions)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _effective_tolerance(reference_value: Any, absolute_tolerance: float, relative_tolerance: float) -> float:
    tolerance = max(0.0, float(absolute_tolerance))
    try:
        reference_abs = abs(float(reference_value))
    except (TypeError, ValueError):
        return tolerance
    return max(tolerance, reference_abs * max(0.0, float(relative_tolerance)))


def _is_effective_success(row: dict[str, Any], absolute_tolerance: float, relative_tolerance: float) -> bool:
    verification = row.get("verification") or {}
    if verification.get("success"):
        return True
    if not verification.get("execution_ok"):
        return False
    if str(verification.get("status", "")).upper() != "OPTIMAL":
        return False
    objective_value = verification.get("objective_value")
    reference_value = row.get("answer")
    if objective_value is None or reference_value is None:
        return False
    try:
        objective = float(objective_value)
        reference = float(reference_value)
    except (TypeError, ValueError):
        return False
    return abs(objective - reference) <= _effective_tolerance(reference, absolute_tolerance, relative_tolerance)


def _trace_stats(trace_rows: list[dict[str, Any]], absolute_tolerance: float, relative_tolerance: float) -> dict[str, Any]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    source_attempts: Counter[str] = Counter()
    source_success_questions: Counter[str] = Counter()
    source_questions: Counter[str] = Counter()
    for row in trace_rows:
        problem_id = str(row.get("problem_id", ""))
        if problem_id:
            by_problem[problem_id].append(row)
        verification = row.get("verification") or {}
        status_counts[str(verification.get("status"))] += 1
        source_attempts[str(row.get("source_name"))] += 1

    for problem_id, rows in by_problem.items():
        source = str(rows[0].get("source_name"))
        source_questions[source] += 1
        if any(_is_effective_success(row, absolute_tolerance, relative_tolerance) for row in rows):
            source_success_questions[source] += 1

    source_success_rates = {
        source: round(source_success_questions[source] / count, 4)
        for source, count in source_questions.items()
        if count
    }
    return {
        "attempted_trajectories": len(trace_rows),
        "processed_trace_questions": len(by_problem),
        "successful_trace_questions": sum(
            1 for rows in by_problem.values() if any(_is_effective_success(row, absolute_tolerance, relative_tolerance) for row in rows)
        ),
        "status_counts": dict(status_counts),
        "source_attempts": dict(source_attempts),
        "source_success_rates": source_success_rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report current generation progress and success rate.")
    parser.add_argument("--config", default="configs/stage1_data.yaml", help="Path to dataset config yaml.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    output_dir = Path(str(cfg.get("external_data_dir", "data/external_or")))
    sft_dir = Path(str(cfg.get("sft_source_dir", output_dir / "sft_teacher")))
    summary_file = output_dir / "summary.json"
    sft_summary_file = sft_dir / "summary.json"
    sft_state_file = sft_dir / "generation_state.json"

    summary = _safe_read_json(summary_file)
    sft_summary = _safe_read_json(sft_summary_file)
    sft_state = _safe_read_json(sft_state_file)
    live_sft = {**sft_summary, **sft_state}
    trace_rows = _read_jsonl(sft_dir / "generation_trace.jsonl")
    verification_tolerance = float(cfg.get("verification_tolerance", 1e-4))
    verification_relative_tolerance = float(cfg.get("verification_relative_tolerance", 0.0))
    trace_stats = _trace_stats(trace_rows, verification_tolerance, verification_relative_tolerance)

    attempted = int(trace_stats["attempted_trajectories"]) or int(live_sft.get("attempted_trajectories", 0))
    accepted = _count_lines(sft_dir / "train.jsonl")
    verification_rate = round(accepted / attempted, 4) if attempted > 0 else 0.0

    split_counts = {}
    for split_name in ["sft", "rollout", "eval"]:
        split_counts[split_name] = _count_lines(output_dir / "splits" / f"{split_name}_questions.jsonl")

    report = {
        "config": str(cfg_path),
        "output_dir": str(output_dir),
        "sft_source_dir": str(sft_dir),
        "total_external_questions": int(summary.get("num_questions", 0)),
        "max_sft_samples": int(cfg.get("max_sft_samples", 0)),
        "accepted_sft_records": _count_lines(sft_dir / "train.jsonl"),
        "unique_sft_questions": _count_unique_questions(sft_dir / "train.jsonl"),
        "processed_questions": int(live_sft.get("processed_questions", 0)),
        "attempted_trajectories": attempted,
        "accepted_samples": accepted,
        "verification_rate": verification_rate,
        "verification_tolerance": verification_tolerance,
        "verification_relative_tolerance": verification_relative_tolerance,
        **trace_stats,
        "source_counts": summary.get("source_counts", {}),
        "source_statuses": summary.get("source_statuses", []),
        "split_counts": split_counts,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

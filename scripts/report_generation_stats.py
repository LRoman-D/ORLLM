#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Report current generation progress and success rate.")
    parser.add_argument("--config", default="configs/stage1_data.yaml", help="Path to dataset config yaml.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    output_dir = Path(str(cfg.get("output_dir", "data/processed/qwen3_rag_teacher")))
    pending_file = output_dir / str(cfg.get("pending_records_file", "accepted_pending.jsonl"))
    state_file = output_dir / str(cfg.get("state_file", "generation_state.json"))
    summary_file = output_dir / "summary.json"

    state = _safe_read_json(state_file)
    summary = _safe_read_json(summary_file)

    attempted = int(state.get("attempted_trajectories", summary.get("attempted_trajectories", 0)))
    accepted = int(state.get("accepted_samples", summary.get("accepted_samples", 0)))
    verification_rate = round(accepted / attempted, 4) if attempted > 0 else 0.0

    split_counts = {}
    for split_name in dict(cfg.get("splits", {"train": 1.0})).keys():
        split_counts[split_name] = _count_lines(output_dir / f"{split_name}.jsonl")

    report = {
        "config": str(cfg_path),
        "output_dir": str(output_dir),
        "target_verified_samples": int(cfg.get("target_verified_samples", 0)),
        "accepted_records_pending": _count_lines(pending_file),
        "unique_questions_pending": _count_unique_questions(pending_file),
        "attempted_trajectories": attempted,
        "accepted_samples": accepted,
        "verification_rate": verification_rate,
        "seed_problem_count": int(state.get("seed_problem_count", summary.get("seed_problem_count", 0))),
        "question_variant_count": int(state.get("question_variant_count", summary.get("question_variant_count", 0))),
        "api_request_errors": int(state.get("api_request_errors", summary.get("api_request_errors", 0))),
        "genprm_attempted": int(state.get("genprm_attempted", summary.get("genprm_attempted", 0))),
        "genprm_rejected": int(state.get("genprm_rejected", summary.get("genprm_rejected", 0))),
        "genprm_acceptance_rate": summary.get("genprm_acceptance_rate", 0.0),
        "completed": bool(state.get("completed", summary.get("completed", False))),
        "interrupted": bool(state.get("interrupted", summary.get("interrupted", False))),
        "split_counts": split_counts,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

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
    output_dir = Path(str(cfg.get("external_data_dir", "data/external_or")))
    sft_dir = Path(str(cfg.get("sft_source_dir", output_dir / "sft_teacher")))
    summary_file = output_dir / "summary.json"
    sft_summary_file = sft_dir / "summary.json"

    summary = _safe_read_json(summary_file)
    sft_summary = _safe_read_json(sft_summary_file)

    attempted = int(sft_summary.get("attempted_trajectories", 0))
    accepted = int(sft_summary.get("accepted_samples", 0))
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
        "attempted_trajectories": attempted,
        "accepted_samples": accepted,
        "verification_rate": verification_rate,
        "source_counts": summary.get("source_counts", {}),
        "source_statuses": summary.get("source_statuses", []),
        "split_counts": split_counts,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

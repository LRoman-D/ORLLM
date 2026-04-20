from __future__ import annotations

from pathlib import Path

from steporlm_stage1.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from steporlm_stage1.utils.io import ensure_dir, load_yaml_config, read_jsonl, write_json, write_jsonl
from steporlm_stage1.utils.run_dirs import create_timestamped_run_dir


def prepare_dpo_dataset(config_path: str | Path) -> dict[str, int | str]:
    config = load_yaml_config(config_path)

    rows = read_jsonl(config["preference_path"])
    if config.get("output_dir"):
        output_dir = ensure_dir(config["output_dir"])
    else:
        output_dir = create_timestamped_run_dir(config.get("run_root", "runs"), config.get("run_prefix", "dpo_data"))

    converted = []
    for row in rows:
        converted.append(
            {
                "problem_id": row["problem_id"],
                "template_name": row["template_name"],
                "prompt_messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=row["question"])},
                ],
                "chosen": row["chosen"]["response"],
                "rejected": row["rejected"]["response"],
                "weight": float(row.get("weight", 1.0)),
                "rationale": row.get("rationale", ""),
            }
        )

    split_idx = max(1, int(len(converted) * 0.9)) if converted else 0
    train_rows = converted[:split_idx]
    valid_rows = converted[split_idx:]
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "valid.jsonl", valid_rows)
    summary = {"train": len(train_rows), "valid": len(valid_rows), "output_dir": str(output_dir)}
    write_json(output_dir / "summary.json", summary)
    return summary

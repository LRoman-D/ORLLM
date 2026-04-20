from __future__ import annotations

from pathlib import Path

from steporlm_stage1.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from steporlm_stage1.utils.io import ensure_dir, read_jsonl, write_jsonl


def prepare_sft_dataset(input_dir: str | Path, output_dir: str | Path) -> dict[str, int]:
    source_dir = Path(input_dir)
    target_dir = ensure_dir(output_dir)
    counts = {}
    for split in ["train", "valid", "test"]:
        source_path = source_dir / f"{split}.jsonl"
        if not source_path.exists():
            continue
        rows = read_jsonl(source_path)
        converted = []
        for row in rows:
            converted.append(
                {
                    "problem_id": row["problem_id"],
                    "template_name": row["template_name"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=row["question"])},
                        {"role": "assistant", "content": row["response"]},
                    ],
                }
            )
        write_jsonl(target_dir / f"{split}.jsonl", converted)
        counts[split] = len(converted)
    return counts


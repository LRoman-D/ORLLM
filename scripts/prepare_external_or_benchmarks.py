#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
from pathlib import Path
from typing import Any

import requests


HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_SPLITS_URL = "https://datasets-server.huggingface.co/splits"

HF_BENCHMARKS = [
    {
        "name": "NL4Opt",
        "dataset": "CardinalOperations/NL4OPT",
        "config": "default",
        "split": "test",
        "limit": 30,
    },
    {
        "name": "MAMO Easy",
        "dataset": "CardinalOperations/MAMO",
        "config": "default",
        "split": "easy_lp",
        "limit": 30,
    },
    {
        "name": "MAMO Complex",
        "dataset": "CardinalOperations/MAMO",
        "config": "default",
        "split": "complex_lp",
        "limit": 30,
    },
    {
        "name": "IndustryOR",
        "dataset": "CardinalOperations/IndustryOR",
        "config": "default",
        "split": "test",
        "limit": 30,
    },
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_status(statuses: list[dict], benchmark: str, status: str, detail: str) -> None:
    statuses.append({"benchmark": benchmark, "status": status, "detail": detail})


def _numeric_answer(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or "no best" in text.lower() or "infeasible" in text.lower():
        return None
    try:
        result = float(text)
    except ValueError:
        match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
        if match is None:
            return None
        result = float(match.group(0))
    if not math.isfinite(result):
        return None
    if result <= -99990:
        return None
    return result


def _pick_result_value(results: dict[str, Any]) -> tuple[str, float] | None:
    priority = ("objective", "optimal", "max", "min", "profit", "cost", "revenue", "value")
    candidates: list[tuple[int, str, float]] = []
    for key, value in results.items():
        parsed = _numeric_answer(value)
        if parsed is None:
            continue
        key_lower = key.lower()
        rank = min((idx for idx, token in enumerate(priority) if token in key_lower), default=len(priority))
        candidates.append((rank, key, parsed))
    if not candidates:
        return None
    _, key, parsed = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    return key, parsed


def _hf_get_rows(dataset: str, config: str, split: str) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    page_size = 100
    while True:
        response = requests.get(
            HF_ROWS_URL,
            params={"dataset": dataset, "config": config, "split": split, "offset": offset, "length": page_size},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        page_rows = payload.get("rows", [])
        for item in page_rows:
            rows.append(item["row"])
        total = int(payload.get("num_rows_total") or len(rows))
        offset += len(page_rows)
        if not page_rows or offset >= total:
            break
    return rows


def _check_hf_access(dataset: str) -> tuple[bool, str]:
    try:
        response = requests.get(HF_SPLITS_URL, params={"dataset": dataset}, timeout=30)
        if response.status_code != 200:
            return False, response.text[:300]
        payload = response.json()
        if payload.get("failed"):
            return False, json.dumps(payload["failed"], ensure_ascii=False)
        if not payload.get("splits"):
            return False, "no splits returned"
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _sample_rows(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    indexes = sorted(rng.sample(range(len(rows)), limit))
    return [rows[idx] for idx in indexes]


def _prepare_hf(root: Path, statuses: list[dict], seed: int) -> list[dict]:
    prepared: list[dict] = []
    raw_dir = _ensure_dir(root / "raw" / "hf")
    for spec in HF_BENCHMARKS:
        ok, detail = _check_hf_access(spec["dataset"])
        if not ok:
            _append_status(statuses, spec["name"], "skipped", detail)
            continue
        try:
            rows = _hf_get_rows(spec["dataset"], spec["config"], spec["split"])
        except Exception as exc:  # noqa: BLE001
            _append_status(statuses, spec["name"], "skipped", f"download failed: {exc}")
            continue
        raw_path = raw_dir / f"{spec['name'].lower().replace(' ', '_')}.jsonl"
        _write_jsonl(raw_path, rows)

        numeric_rows = []
        for row_idx, row in enumerate(rows):
            answer = _numeric_answer(row.get("en_answer"))
            if answer is None:
                continue
            numeric_rows.append((row_idx, row, answer))
        chosen = _sample_rows(numeric_rows, spec["limit"], seed + len(prepared))
        for row_idx, row, answer in chosen:
            prepared.append(
                {
                    "problem_id": f"{spec['name'].lower().replace(' ', '_')}-{row_idx:04d}",
                    "benchmark": spec["name"],
                    "source": spec["dataset"],
                    "question": row["en_question"],
                    "answer": answer,
                    "answer_key": "en_answer",
                    "metadata": {
                        "hf_config": spec["config"],
                        "hf_split": spec["split"],
                        "difficulty": row.get("difficulty"),
                        "source_id": row.get("id"),
                        "row_idx": row_idx,
                    },
                }
            )
        _append_status(statuses, spec["name"], "included", f"{len(chosen)} numeric rows from {len(rows)} rows")
    return prepared


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _ensure_github_repo(root: Path, repo_url: str, target: Path, sparse_path: str | None = None) -> None:
    if target.exists():
        return
    _ensure_dir(target.parent)
    if sparse_path:
        _run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo_url, str(target)], root)
        _run(["git", "-C", str(target), "sparse-checkout", "set", sparse_path], root)
    else:
        _run(["git", "clone", "--depth", "1", "--filter=blob:none", repo_url, str(target)], root)


def _prepare_complex_or(root: Path, statuses: list[dict]) -> list[dict]:
    repo_dir = root / "raw" / "github" / "Chain-of-Experts"
    try:
        _ensure_github_repo(
            root,
            "https://github.com/xzymustbexzy/Chain-of-Experts.git",
            repo_dir,
            "dataset/ComplexOR",
        )
    except Exception as exc:  # noqa: BLE001
        _append_status(statuses, "ComplexOR", "skipped", f"clone failed: {exc}")
        return []

    base = repo_dir / "dataset" / "ComplexOR"
    rows: list[dict] = []
    for sample_path in sorted(base.glob("*/sample.json")):
        task_dir = sample_path.parent
        try:
            samples = json.loads(sample_path.read_text(encoding="utf-8"))
            description = (task_dir / "description.txt").read_text(encoding="utf-8").strip()
        except Exception:
            continue
        for idx, sample in enumerate(samples):
            output = sample.get("output")
            answer = None
            if isinstance(output, list) and output:
                answer = _numeric_answer(output[0])
            else:
                answer = _numeric_answer(output)
            if answer is None:
                continue
            input_json = json.dumps(sample.get("input", {}), ensure_ascii=False, indent=2)
            task_name = task_dir.name
            rows.append(
                {
                    "problem_id": f"complexor-{task_name}-{idx:03d}",
                    "benchmark": "ComplexOR",
                    "source": "xzymustbexzy/Chain-of-Experts",
                    "question": (
                        f"{description}\n\nConcrete input data:\n{input_json}\n\n"
                        "Build and solve the optimization model for this instance. Report the optimal objective value."
                    ),
                    "answer": answer,
                    "answer_key": "sample.output[0]",
                    "metadata": {"task": task_name, "sample_idx": idx},
                }
            )
    _append_status(statuses, "ComplexOR", "included" if rows else "skipped", f"{len(rows)} rows")
    return rows


def _prepare_resocratic(root: Path, statuses: list[dict], seed: int, limit: int) -> list[dict]:
    repo_dir = root / "raw" / "github" / "ReSocratic"
    try:
        _ensure_github_repo(root, "https://github.com/yangzhch6/ReSocratic.git", repo_dir)
    except Exception as exc:  # noqa: BLE001
        _append_status(statuses, "ReSocratic", "skipped", f"clone failed: {exc}")
        return []

    opt_path = repo_dir / "data" / "OptiBench.json"
    if not opt_path.exists():
        _append_status(statuses, "ReSocratic", "skipped", "OptiBench.json not found")
        return []
    data = json.loads(opt_path.read_text(encoding="utf-8"))
    numeric_rows = []
    for row in data:
        picked = _pick_result_value(row.get("results") or {})
        if picked is None:
            continue
        key, answer = picked
        numeric_rows.append((row, key, answer))
    chosen = _sample_rows(numeric_rows, limit, seed)
    prepared = []
    for row, key, answer in chosen:
        prepared.append(
            {
                "problem_id": f"resocratic-optibench-{int(row.get('index', len(prepared))):04d}",
                "benchmark": "ReSocratic",
                "source": "yangzhch6/ReSocratic/data/OptiBench.json",
                "question": row["question"],
                "answer": answer,
                "answer_key": key,
                "metadata": {"index": row.get("index"), "type": row.get("type")},
            }
        )
    _append_status(statuses, "ReSocratic", "included", f"{len(prepared)} numeric rows from {len(data)} rows")
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare external OR benchmark samples.")
    parser.add_argument("--root", default="benchmarks/or_external")
    parser.add_argument("--output", default="benchmarks/or_external/prepared/benchmark_samples.jsonl")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resocratic-limit", type=int, default=30)
    args = parser.parse_args()

    root = Path(args.root)
    _ensure_dir(root / "raw")
    _ensure_dir(root / "prepared")

    statuses: list[dict] = []
    rows: list[dict] = []
    rows.extend(_prepare_hf(root, statuses, args.seed))

    ok, detail = _check_hf_access("udell-lab/NLP4LP")
    if ok:
        _append_status(statuses, "NLP4LP", "skipped", "accessible but schema adapter not implemented")
    else:
        _append_status(statuses, "NLP4LP", "skipped", detail)

    rows.extend(_prepare_complex_or(root, statuses))
    rows.extend(_prepare_resocratic(root, statuses, args.seed + 1000, args.resocratic_limit))

    output_path = Path(args.output)
    _write_jsonl(output_path, rows)
    summary = {
        "num_samples": len(rows),
        "benchmarks": statuses,
        "counts": {name: sum(1 for row in rows if row["benchmark"] == name) for name in sorted({row["benchmark"] for row in rows})},
        "output_path": str(output_path),
    }
    _write_json(root / "prepared" / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

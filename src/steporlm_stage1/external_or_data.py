from __future__ import annotations

import json
import math
import random
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import requests
import torch
from tqdm import tqdm

from steporlm_stage1.executors.python_executor import PythonCodeExecutor
from steporlm_stage1.quality.genprm import audit_passes_threshold
from steporlm_stage1.schemas import ReferenceSolution
from steporlm_stage1.teachers.factory import build_teacher_generator, teacher_backend_name, teacher_model_name
from steporlm_stage1.utils.io import ensure_dir, load_yaml_config, write_json, write_jsonl
from steporlm_stage1.utils.text import extract_python_code


HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_SPLITS_URL = "https://datasets-server.huggingface.co/splits"

HF_SOURCES = [
    {
        "name": "NL4Opt",
        "slug": "nl4opt",
        "dataset": "CardinalOperations/NL4OPT",
        "config": "default",
        "split": "test",
    },
    {
        "name": "MAMO Easy",
        "slug": "mamo_easy",
        "dataset": "CardinalOperations/MAMO",
        "config": "default",
        "split": "easy_lp",
    },
    {
        "name": "MAMO Complex",
        "slug": "mamo_complex",
        "dataset": "CardinalOperations/MAMO",
        "config": "default",
        "split": "complex_lp",
    },
    {
        "name": "NLP4LP",
        "slug": "nlp4lp",
        "dataset": "udell-lab/NLP4LP",
        "config": "default",
        "split": "train",
    },
    {
        "name": "IndustryOR",
        "slug": "industryor",
        "dataset": "CardinalOperations/IndustryOR",
        "config": "default",
        "split": "test",
    },
]


def _append_status(statuses: list[dict[str, str]], source: str, status: str, detail: str) -> None:
    statuses.append({"source": source, "status": status, "detail": detail})


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _numeric_answer(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or "infeasible" in text.lower() or "no best" in text.lower():
        return None
    try:
        parsed = float(text)
    except ValueError:
        match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
        if match is None:
            return None
        parsed = float(match.group(0))
    if not math.isfinite(parsed) or parsed <= -99990:
        return None
    return parsed


def _pick_column(row: dict[str, Any], candidates: tuple[str, ...]) -> tuple[str, Any] | None:
    lowered = {key.lower(): key for key in row.keys()}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is not None and row.get(key) not in (None, ""):
            return key, row[key]
    return None


def _pick_result_value(results: dict[str, Any]) -> tuple[str, float] | None:
    priority = ("objective", "optimal", "optimum", "max", "min", "profit", "cost", "revenue", "value")
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


def _external_record(
    *,
    problem_id: str,
    source_name: str,
    source: str,
    question: str,
    answer: float,
    answer_key: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "problem_id": problem_id,
        "template_name": "external_or",
        "source_name": source_name,
        "source": source,
        "question": question.strip(),
        "answer": answer,
        "answer_key": answer_key,
        "metadata": metadata,
        "reference_solution": {
            "status": "OPTIMAL",
            "objective_value": answer,
            "metadata": {"answer_key": answer_key, **metadata},
        },
    }


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


def _hf_get_rows(dataset: str, config: str, split: str, row_limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 100
    while True:
        length = min(page_size, row_limit - len(rows)) if row_limit > 0 else page_size
        if length <= 0:
            break
        response = requests.get(
            HF_ROWS_URL,
            params={"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        page_rows = payload.get("rows", [])
        for item in page_rows:
            rows.append(item["row"])
        total = int(payload.get("num_rows_total") or len(rows))
        offset += len(page_rows)
        if not page_rows or offset >= total or (row_limit > 0 and len(rows) >= row_limit):
            break
    return rows


def _load_cached_hf_rows(raw_root: Path, slug: str, row_limit: int = 0) -> list[dict[str, Any]]:
    cache_path = raw_root / slug / "rows.jsonl"
    if not cache_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if row_limit > 0 and len(rows) >= row_limit:
                break
    return rows


def _prepare_hf_sources(root: Path, config: dict[str, Any], statuses: list[dict[str, str]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    raw_root = ensure_dir(root / "raw")
    row_limit = int(config.get("max_rows_per_source", 0) or 0)
    question_candidates = ("en_question", "question", "problem", "prompt", "text")
    answer_candidates = ("en_answer", "answer", "objective", "objective_value", "optimum", "optimal_value")

    for spec in HF_SOURCES:
        source_dir = ensure_dir(raw_root / str(spec["slug"]))
        ok, detail = _check_hf_access(str(spec["dataset"]))
        if not ok:
            rows = _load_cached_hf_rows(raw_root, str(spec["slug"]), row_limit)
            if not rows:
                _append_status(statuses, str(spec["name"]), "skipped", detail)
                continue
            _append_status(statuses, str(spec["name"]), "cached", f"{len(rows)} cached rows; access failed: {detail}")
        else:
            try:
                rows = _hf_get_rows(str(spec["dataset"]), str(spec["config"]), str(spec["split"]), row_limit)
            except Exception as exc:  # noqa: BLE001
                rows = _load_cached_hf_rows(raw_root, str(spec["slug"]), row_limit)
                if not rows:
                    _append_status(statuses, str(spec["name"]), "skipped", f"download failed: {exc}")
                    continue
                _append_status(statuses, str(spec["name"]), "cached", f"{len(rows)} cached rows; download failed: {exc}")
            else:
                write_jsonl(source_dir / "rows.jsonl", rows)

        numeric_count = 0
        for row_idx, row in enumerate(rows):
            question_pick = _pick_column(row, question_candidates)
            answer_pick = _pick_column(row, answer_candidates)
            if question_pick is None or answer_pick is None:
                continue
            question_key, question = question_pick
            answer_key, answer_raw = answer_pick
            answer = _numeric_answer(answer_raw)
            if answer is None:
                continue
            numeric_count += 1
            prepared.append(
                _external_record(
                    problem_id=f"{spec['slug']}-{row_idx:05d}",
                    source_name=str(spec["name"]),
                    source=str(spec["dataset"]),
                    question=str(question),
                    answer=answer,
                    answer_key=answer_key,
                    metadata={
                        "hf_config": spec["config"],
                        "hf_split": spec["split"],
                        "row_idx": row_idx,
                        "question_key": question_key,
                        "difficulty": row.get("difficulty"),
                        "source_id": row.get("id"),
                    },
                )
            )
        if not any(status["source"] == str(spec["name"]) for status in statuses):
            _append_status(statuses, str(spec["name"]), "included", f"{numeric_count} numeric rows from {len(rows)} rows")
    return prepared


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _ensure_github_repo(root: Path, repo_url: str, target: Path, sparse_path: str | None = None) -> None:
    if target.exists():
        return
    ensure_dir(target.parent)
    if sparse_path:
        _run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo_url, str(target)], root)
        _run(["git", "-C", str(target), "sparse-checkout", "set", sparse_path], root)
    else:
        _run(["git", "clone", "--depth", "1", "--filter=blob:none", repo_url, str(target)], root)


def _prepare_complex_or(root: Path, statuses: list[dict[str, str]]) -> list[dict[str, Any]]:
    source_dir = ensure_dir(root / "raw" / "complexor")
    repo_dir = source_dir / "repo"
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
    rows: list[dict[str, Any]] = []
    for sample_path in sorted(base.glob("*/sample.json")):
        task_dir = sample_path.parent
        try:
            samples = json.loads(sample_path.read_text(encoding="utf-8"))
            description = (task_dir / "description.txt").read_text(encoding="utf-8").strip()
        except Exception:
            continue
        for sample_idx, sample in enumerate(samples):
            output = sample.get("output")
            answer = _numeric_answer(output[0] if isinstance(output, list) and output else output)
            if answer is None:
                continue
            input_json = json.dumps(sample.get("input", {}), ensure_ascii=False, indent=2)
            task_name = task_dir.name
            rows.append(
                _external_record(
                    problem_id=f"complexor-{task_name}-{sample_idx:04d}",
                    source_name="ComplexOR",
                    source="xzymustbexzy/Chain-of-Experts",
                    question=(
                        f"{description}\n\nConcrete input data:\n{input_json}\n\n"
                        "Build and solve the optimization model for this instance. Report the optimal objective value."
                    ),
                    answer=answer,
                    answer_key="sample.output[0]",
                    metadata={"task": task_name, "sample_idx": sample_idx},
                )
            )
    write_jsonl(source_dir / "questions.jsonl", rows)
    _append_status(statuses, "ComplexOR", "included" if rows else "skipped", f"{len(rows)} rows")
    return rows


def _prepare_resocratic(root: Path, statuses: list[dict[str, str]]) -> list[dict[str, Any]]:
    source_dir = ensure_dir(root / "raw" / "resocratic")
    repo_dir = source_dir / "repo"
    try:
        _ensure_github_repo(root, "https://github.com/yangzhch6/ReSocratic.git", repo_dir)
    except Exception as exc:  # noqa: BLE001
        _append_status(statuses, "ReSocratic", "skipped", f"clone failed: {exc}")
        return []

    opt_path = repo_dir / "data" / "OptiBench.json"
    if not opt_path.exists():
        _append_status(statuses, "ReSocratic", "skipped", "data/OptiBench.json not found")
        return []
    data = json.loads(opt_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(data):
        picked = _pick_result_value(row.get("results") or {})
        if picked is None:
            continue
        answer_key, answer = picked
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        rows.append(
            _external_record(
                problem_id=f"resocratic-optibench-{int(row.get('index', row_idx)):05d}",
                source_name="ReSocratic",
                source="yangzhch6/ReSocratic/data/OptiBench.json",
                question=question,
                answer=answer,
                answer_key=answer_key,
                metadata={"index": row.get("index"), "type": row.get("type"), "row_idx": row_idx},
            )
        )
    write_jsonl(source_dir / "questions.jsonl", rows)
    _append_status(statuses, "ReSocratic", "included" if rows else "skipped", f"{len(rows)} numeric rows from {len(data)} rows")
    return rows


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = (row["question"].strip(), str(row["answer"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _split_rows(rows: list[dict[str, Any]], split_config: dict[str, float], seed: int) -> dict[str, list[dict[str, Any]]]:
    names = list(split_config.keys())
    if names != ["sft", "rollout", "eval"]:
        names = ["sft", "rollout", "eval"]
    ratios = [float(split_config.get(name, {"sft": 0.6, "rollout": 0.2, "eval": 0.2}[name])) for name in names]
    total_ratio = sum(ratios) or 1.0
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    first = int(len(shuffled) * ratios[0] / total_ratio)
    second = first + int(len(shuffled) * ratios[1] / total_ratio)
    splits = {
        names[0]: shuffled[:first],
        names[1]: shuffled[first:second],
        names[2]: shuffled[second:],
    }
    for split_name, split_rows in splits.items():
        for row in split_rows:
            row["split"] = split_name
    return splits


def _build_temperatures(config: dict[str, Any]) -> list[float]:
    explicit = config.get("trajectory_temperatures")
    if explicit:
        return [float(value) for value in explicit]
    count = int(config.get("trajectories_per_question", 2))
    if count <= 1:
        return [0.0]
    start = float(config.get("trajectory_temperature_min", 0.0))
    stop = float(config.get("trajectory_temperature_max", 0.35))
    return [round(start + idx * (stop - start) / (count - 1), 2) for idx in range(count)]


def _prioritize_sft_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    priority = config.get("sft_source_priority")
    if not priority:
        return rows
    priority_rank = {str(source): rank for rank, source in enumerate(priority)}
    default_rank = len(priority_rank)
    return sorted(
        rows,
        key=lambda row: (
            priority_rank.get(str(row.get("source_name")), default_rank),
            str(row.get("problem_id", "")),
        ),
    )


def _effective_objective_tolerance(reference_value: Any, absolute_tolerance: float, relative_tolerance: float) -> float:
    tolerance = max(0.0, float(absolute_tolerance))
    try:
        reference_abs = abs(float(reference_value))
    except (TypeError, ValueError):
        return tolerance
    return max(tolerance, reference_abs * max(0.0, float(relative_tolerance)))


def _verification_matches_reference(
    verification: dict[str, Any],
    reference_value: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    if not verification.get("execution_ok"):
        return False
    if str(verification.get("status", "")).upper() != "OPTIMAL":
        return False
    objective_value = verification.get("objective_value")
    if objective_value is None or reference_value is None:
        return False
    try:
        objective = float(objective_value)
        reference = float(reference_value)
    except (TypeError, ValueError):
        return False
    tolerance = _effective_objective_tolerance(reference, absolute_tolerance, relative_tolerance)
    return abs(objective - reference) <= tolerance


def _relaxed_verification(
    verification: dict[str, Any],
    reference_value: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    relaxed = dict(verification)
    relaxed["success"] = True
    relaxed["objective_match"] = True
    relaxed["tolerance"] = _effective_objective_tolerance(reference_value, absolute_tolerance, relative_tolerance)
    relaxed["error_message"] = None
    return relaxed


def _backfill_relaxed_trace_accepts(
    *,
    selected_rows: list[dict[str, Any]],
    trace_path: Path,
    train_path: Path,
    accepted_records: list[dict[str, Any]],
    accepted_ids: set[str],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> int:
    if relative_tolerance <= 0.0 or not trace_path.exists():
        return 0
    rows_by_id = {str(row["problem_id"]): row for row in selected_rows}
    trace_by_id: dict[str, list[dict[str, Any]]] = {}
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                continue
            problem_id = str(trace.get("problem_id", ""))
            if problem_id:
                trace_by_id.setdefault(problem_id, []).append(trace)

    backfilled: list[dict[str, Any]] = []
    for problem_id, row in rows_by_id.items():
        if problem_id in accepted_ids:
            continue
        for trace in trace_by_id.get(problem_id, []):
            verification = trace.get("verification") or {}
            if not _verification_matches_reference(
                verification,
                row["reference_solution"].get("objective_value"),
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            ):
                continue
            chosen = {
                **row,
                "response": trace.get("response", ""),
                "code": trace.get("code", ""),
                "verification": _relaxed_verification(
                    verification,
                    row["reference_solution"].get("objective_value"),
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                ),
                "process_verification": trace.get("process_verification") or {},
                "generation_notes": {
                    "generator_backend": "trace_backfill",
                    "teacher_model": "existing_trace",
                    "relaxed_relative_tolerance": relative_tolerance,
                    "trajectory_index": trace.get("trajectory_index"),
                },
            }
            backfilled.append(chosen)
            accepted_records.append(chosen)
            accepted_ids.add(problem_id)
            break

    _append_jsonl(train_path, backfilled)
    return len(backfilled)


def _answer_stub_response(row: dict[str, Any]) -> str:
    answer = row["reference_solution"]["objective_value"]
    return f"""<step>Problem Description
The problem is an external operations research instance. The referenced benchmark answer is used as the target numeric answer.</step>
<step>Sets and Parameters
All sets and parameters are given directly in the question text.</step>
<step>Decision Variables
Decision variables should be chosen to match the quantities requested by the problem.</step>
<step>Objective Function
The objective direction is the one stated in the question. The target benchmark answer is {answer}.</step>
<step>Constraints
All feasibility conditions are the constraints described in the question.</step>
<step>Mathematical Model
A faithful OR model should preserve every number, bound, and relationship from the question.</step>
<step>Nonlinear Relationships
If nonlinear relationships are present, they should be linearized before implementation.</step>
<step>Final Model
The final model should report the numeric answer requested by the benchmark question.</step>
<step>Python Code Using OR-Tools
The code below emits the benchmark objective value for pipeline bootstrapping when no teacher trajectory was accepted.</step>
```python
import json

result = {{"status": "OPTIMAL", "objective_value": {answer!r}}}
print("__STEPORLM_RESULT__=" + json.dumps(result, ensure_ascii=False))
```"""


def _build_sft_records(config: dict[str, Any], sft_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    teacher_mode = str(config.get("sft_response_mode", "teacher_verified")).lower()
    max_sft_samples = int(config.get("max_sft_samples", 0) or 0)
    ordered_rows = _prioritize_sft_rows(sft_rows, config)
    selected_rows = ordered_rows[:max_sft_samples] if max_sft_samples > 0 else ordered_rows
    trace_path = output_dir / "generation_trace.jsonl"
    state_path = output_dir / "generation_state.json"
    train_path = output_dir / "train.jsonl"
    if trace_path.exists() and not bool(config.get("resume_from_checkpoint", False)):
        trace_path.unlink()
    if state_path.exists() and not bool(config.get("resume_from_checkpoint", False)):
        state_path.unlink()
    if train_path.exists() and not bool(config.get("resume_from_checkpoint", False)):
        train_path.unlink()

    accepted_records: list[dict[str, Any]] = []
    if bool(config.get("resume_from_checkpoint", False)) and train_path.exists():
        with train_path.open("r", encoding="utf-8") as handle:
            accepted_records = [json.loads(line) for line in handle if line.strip()]
    accepted_ids = {row["problem_id"] for row in accepted_records}
    verification_tolerance = float(config.get("verification_tolerance", 1e-4))
    verification_relative_tolerance = float(config.get("verification_relative_tolerance", 0.0))
    backfilled_relaxed = 0
    if bool(config.get("resume_from_checkpoint", False)):
        backfilled_relaxed = _backfill_relaxed_trace_accepts(
            selected_rows=selected_rows,
            trace_path=trace_path,
            train_path=train_path,
            accepted_records=accepted_records,
            accepted_ids=accepted_ids,
            absolute_tolerance=verification_tolerance,
            relative_tolerance=verification_relative_tolerance,
        )
    processed_counts: Counter[str] = Counter()
    historical_attempted = 0
    historical_rejected = 0
    if bool(config.get("resume_from_checkpoint", False)) and trace_path.exists():
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    trace = json.loads(line)
                except json.JSONDecodeError:
                    continue
                problem_id = str(trace.get("problem_id", ""))
                if problem_id:
                    processed_counts[problem_id] += 1
                historical_attempted += 1
                if not ((trace.get("verification") or {}).get("success")):
                    historical_rejected += 1

    generator = None
    if teacher_mode in {"teacher", "teacher_verified", "teacher_or_answer_stub"}:
        generator = build_teacher_generator(config)
        if generator is None and teacher_mode == "teacher_verified":
            raise RuntimeError("sft_response_mode=teacher_verified requires an available teacher generator.")

    executor = PythonCodeExecutor(timeout_seconds=int(config.get("timeout_seconds", 35)))
    temperatures = _build_temperatures(config)
    max_tokens = int(config.get("trajectory_max_tokens", 2600))
    allow_stub = teacher_mode in {"answer_stub", "teacher_or_answer_stub"}
    genprm_enabled = bool(config.get("genprm_evaluation", False))
    genprm_min_steps = int(config.get("genprm_min_correct_steps", 8))
    genprm_require_all = bool(config.get("genprm_require_all_correct", False))

    attempted = historical_attempted
    accepted = len(accepted_records)
    rejected = historical_rejected
    stubbed = 0
    expected_trajectories = len(temperatures)
    for row in tqdm(selected_rows, desc="Generating SFT trajectories from external questions"):
        if processed_counts.get(row["problem_id"], 0) >= expected_trajectories:
            continue
        reference_dict = row["reference_solution"]
        reference = ReferenceSolution(
            status=str(reference_dict["status"]),
            objective_value=reference_dict.get("objective_value"),
            metadata=reference_dict.get("metadata", {}),
        )
        chosen: dict[str, Any] | None = None
        responses: list[str] = []
        if generator is not None:
            responses = generator.generate_trajectories(
                question=row["question"],
                template_name=row["template_name"],
                temperatures=temperatures,
                max_tokens=max_tokens,
                answer_key=row.get("answer_key"),
                answer_value=row.get("answer"),
            )
        elif teacher_mode == "answer_stub":
            responses = [_answer_stub_response(row)]
        if len(responses) < expected_trajectories:
            responses.extend([""] * (expected_trajectories - len(responses)))

        for idx, response in enumerate(responses[:expected_trajectories]):
            attempted += 1
            code = extract_python_code(response)
            verification = executor.verify(
                code,
                reference,
                tolerance=verification_tolerance,
                relative_tolerance=verification_relative_tolerance,
            )
            process_verification: dict[str, Any] = {}
            if verification.success and genprm_enabled and generator is not None and hasattr(generator, "audit_trajectory"):
                process_verification = generator.audit_trajectory(
                    question=row["question"],
                    response=response,
                    verification=verification.to_dict(),
                    template_name=row["template_name"],
                    max_tokens=int(config.get("genprm_max_tokens", 4200)),
                )
            trace_row = {
                "problem_id": row["problem_id"],
                "trajectory_index": idx,
                "source_name": row.get("source_name"),
                "answer": row.get("answer"),
                "answer_key": row.get("answer_key"),
                "response": response,
                "code": code,
                "response_preview": response[:1200],
                "code_preview": code[:1200],
                "verification": verification.to_dict(),
                "process_verification": process_verification,
            }
            _append_jsonl(trace_path, [trace_row])
            process_ok = not process_verification or audit_passes_threshold(
                process_verification,
                min_correct_steps=genprm_min_steps,
                require_all_correct=genprm_require_all,
            )
            if chosen is None and verification.success and process_ok:
                chosen = {
                    **row,
                    "response": response,
                    "code": code,
                    "verification": verification.to_dict(),
                    "process_verification": process_verification,
                    "generation_notes": {
                        "generator_backend": teacher_backend_name(generator) if generator is not None else "answer_stub",
                        "teacher_model": teacher_model_name(generator) if generator is not None else "none",
                        "temperature": temperatures[min(idx, len(temperatures) - 1)] if temperatures else None,
                    },
                }
            elif not verification.success or not process_ok:
                rejected += 1

        if chosen is None and allow_stub:
            response = _answer_stub_response(row)
            chosen = {
                **row,
                "response": response,
                "code": extract_python_code(response),
                "verification": {
                    "success": False,
                    "execution_ok": False,
                    "status": "ANSWER_STUB",
                    "objective_value": row["reference_solution"]["objective_value"],
                    "objective_match": True,
                    "tolerance": 0.0,
                    "stdout": "",
                    "stderr": "",
                    "error_message": "No verified teacher trajectory was accepted; answer stub used by configuration.",
                },
                "process_verification": {},
                "generation_notes": {"generator_backend": "answer_stub", "teacher_model": "none"},
            }
            stubbed += 1

        if chosen is not None:
            accepted += 1
            accepted_records.append(chosen)
            accepted_ids.add(chosen["problem_id"])
            _append_jsonl(train_path, [chosen])
        processed_counts[row["problem_id"]] = expected_trajectories
        write_json(
            state_path,
            {
                "selected_sft_questions": len(selected_rows),
                "processed_questions": sum(1 for item in selected_rows if processed_counts.get(item["problem_id"], 0) >= expected_trajectories),
                "accepted_samples": accepted,
                "attempted_trajectories": attempted,
                "rejected_trajectories": rejected,
                "answer_stub_samples": stubbed,
                "backfilled_relaxed_samples": backfilled_relaxed,
                "expected_trajectories_per_question": expected_trajectories,
                "trace_path": str(trace_path),
                "train_path": str(train_path),
            },
        )

    if generator is not None:
        del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    write_jsonl(train_path, accepted_records)
    write_jsonl(output_dir / "valid.jsonl", [])
    summary = {
        "selected_sft_questions": len(selected_rows),
        "accepted_samples": accepted,
        "attempted_trajectories": attempted,
        "rejected_trajectories": rejected,
        "answer_stub_samples": stubbed,
        "backfilled_relaxed_samples": backfilled_relaxed,
        "response_mode": teacher_mode,
        "train_path": str(train_path),
        "trace_path": str(trace_path),
        "state_path": str(state_path),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def prepare_external_or_data_from_config(config: dict[str, Any]) -> dict[str, Any]:
    root = ensure_dir(Path(config.get("external_data_dir", "data/external_or")).resolve())
    seed = int(config.get("seed", 2026))
    statuses: list[dict[str, str]] = []

    rows: list[dict[str, Any]] = []
    rows.extend(_prepare_hf_sources(root, config, statuses))
    rows.extend(_prepare_complex_or(root, statuses))
    rows.extend(_prepare_resocratic(root, statuses))
    rows = _dedupe_rows(rows)

    prepared_dir = ensure_dir(root / "prepared")
    split_dir = ensure_dir(root / "splits")
    write_jsonl(prepared_dir / "all_questions.jsonl", rows)
    split_config = dict(config.get("splits", {"sft": 0.6, "rollout": 0.2, "eval": 0.2}))
    splits = _split_rows(rows, split_config, seed)
    for split_name, split_rows in splits.items():
        write_jsonl(split_dir / f"{split_name}_questions.jsonl", split_rows)

    if bool(config.get("build_sft_trajectories", True)):
        sft_output_dir = ensure_dir(config.get("sft_source_dir", root / "sft_teacher"))
        sft_summary = _build_sft_records(config, splits["sft"], sft_output_dir)
    else:
        sft_summary = {"skipped": True, "reason": "build_sft_trajectories is false"}
    counts = Counter(row["source_name"] for row in rows)
    split_counts = {name: len(split_rows) for name, split_rows in splits.items()}
    summary = {
        "external_data_dir": str(root),
        "num_questions": len(rows),
        "source_statuses": statuses,
        "source_counts": dict(counts),
        "splits": split_counts,
        "prepared_questions": str(prepared_dir / "all_questions.jsonl"),
        "split_paths": {name: str(split_dir / f"{name}_questions.jsonl") for name in splits},
        "sft_source": sft_summary,
    }
    write_json(root / "summary.json", summary)
    return summary


def prepare_external_or_data(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    return prepare_external_or_data_from_config(config)

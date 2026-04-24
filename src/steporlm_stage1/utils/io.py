from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import yaml

from steporlm_stage1.utils.paths import PROJECT_ROOT, get_project_paths, map_repo_relative_path


CONFIG_PATH_KEYS = {
    "base_model_path",
    "config_path",
    "dataset_path",
    "eval_dataset_path",
    "external_data_dir",
    "input_dir",
    "index_dir",
    "model_name_or_path",
    "model_path",
    "output_dir",
    "output_path",
    "preference_path",
    "rag_index_dir",
    "report_dir",
    "root",
    "run_root",
    "sft_adapter_path",
    "sft_source_dir",
    "source_dir",
    "teacher_model_path",
}


def _looks_like_local_path(value: str) -> bool:
    if value.startswith(("http://", "https://")):
        return False
    if Path(value).is_absolute():
        return True
    if value.startswith(".") or "\\" in value or "/" in value:
        return True
    if value.endswith(
        (
            ".json",
            ".jsonl",
            ".yaml",
            ".yml",
            ".txt",
            ".png",
            ".jpg",
            ".jpeg",
            ".md",
            ".safetensors",
            ".bin",
            ".pt",
            ".pth",
            ".jinja",
        )
    ):
        return True
    return value.split("/", 1)[0] in {
        "artifacts",
        "configs",
        "data",
        "models",
        "outputs",
        "reports",
        "runs",
        "tools",
        "logs",
        ".cache",
        "cache",
    }


def resolve_local_path(path: str | Path, base_dir: str | Path | None = None) -> str:
    value = Path(path)
    if value.is_absolute():
        return str(value)

    mapped = map_repo_relative_path(value)
    if mapped.exists():
        return str(mapped)

    base = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    base_candidate = (base / value).resolve() if base.is_absolute() else (PROJECT_ROOT / base / value).resolve()
    if base_candidate.exists():
        return str(base_candidate)

    return str(mapped)


def _resolve_config_value(value, base_dir: Path):
    if isinstance(value, str):
        return resolve_local_path(value, base_dir) if _looks_like_local_path(value) else value
    if isinstance(value, list):
        return [_resolve_config_value(item, base_dir) for item in value]
    return value


def load_yaml_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return resolve_config_paths(config, config_path.parent)


def resolve_config_paths(config: dict, base_dir: str | Path) -> dict:
    resolved: dict = {}
    base = Path(base_dir)
    for key, value in config.items():
        if key in CONFIG_PATH_KEYS:
            resolved[key] = _resolve_config_value(value, base)
        elif isinstance(value, dict):
            resolved[key] = resolve_config_paths(value, base)
        elif isinstance(value, list):
            resolved[key] = [
                resolve_config_paths(item, base) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            resolved[key] = value
    return resolved


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_project_dirs() -> None:
    get_project_paths().ensure_base_dirs()


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

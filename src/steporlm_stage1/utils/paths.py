from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _env_path(name: str, default: str | Path) -> Path:
    value = os.getenv(name)
    if value:
        return Path(value).expanduser()
    return Path(default)


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_dir: Path
    model_dir: Path
    output_dir: Path
    log_dir: Path
    cache_dir: Path

    @property
    def run_root(self) -> Path:
        return self.output_dir / "runs"

    @property
    def checkpoint_root(self) -> Path:
        return self.output_dir / "checkpoints"

    def ensure_base_dirs(self) -> None:
        for path in [self.data_dir, self.model_dir, self.output_dir, self.log_dir, self.cache_dir, self.run_root, self.checkpoint_root]:
            path.mkdir(parents=True, exist_ok=True)


DEFAULT_PATHS = ProjectPaths(
    project_root=PROJECT_ROOT,
    data_dir=_env_path("PROJECT_DATA_DIR", PROJECT_ROOT / "data"),
    model_dir=_env_path("PROJECT_MODEL_DIR", PROJECT_ROOT / "models"),
    output_dir=_env_path("PROJECT_OUTPUT_DIR", PROJECT_ROOT / "outputs"),
    log_dir=_env_path("PROJECT_LOG_DIR", PROJECT_ROOT / "logs"),
    cache_dir=_env_path("PROJECT_CACHE_DIR", PROJECT_ROOT / ".cache"),
)


PATH_ALIAS_PREFIXES = {
    "data": lambda paths: paths.data_dir,
    "models": lambda paths: paths.model_dir,
    "outputs": lambda paths: paths.output_dir,
    "runs": lambda paths: paths.run_root,
    "logs": lambda paths: paths.log_dir,
    "cache": lambda paths: paths.cache_dir,
    ".cache": lambda paths: paths.cache_dir,
}


def get_project_paths() -> ProjectPaths:
    return DEFAULT_PATHS


def map_repo_relative_path(path: str | Path, paths: ProjectPaths | None = None) -> Path:
    current_paths = paths or get_project_paths()
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    parts = candidate.parts
    if parts:
        prefix = parts[0]
        if prefix in PATH_ALIAS_PREFIXES:
            base = PATH_ALIAS_PREFIXES[prefix](current_paths)
            remainder = Path(*parts[1:]) if len(parts) > 1 else Path()
            return (base / remainder).resolve()
    return (current_paths.project_root / candidate).resolve()


def project_env_exports(paths: ProjectPaths | None = None) -> dict[str, str]:
    current_paths = paths or get_project_paths()
    return {
        "PROJECT_ROOT": str(current_paths.project_root),
        "PROJECT_DATA_DIR": str(current_paths.data_dir),
        "PROJECT_MODEL_DIR": str(current_paths.model_dir),
        "PROJECT_OUTPUT_DIR": str(current_paths.output_dir),
        "PROJECT_LOG_DIR": str(current_paths.log_dir),
        "PROJECT_CACHE_DIR": str(current_paths.cache_dir),
    }

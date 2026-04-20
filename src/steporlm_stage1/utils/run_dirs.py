from __future__ import annotations

from datetime import datetime
from pathlib import Path

from steporlm_stage1.utils.io import ensure_dir
from steporlm_stage1.utils.paths import map_repo_relative_path


def create_timestamped_run_dir(root: str | Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_root = map_repo_relative_path(root)
    return ensure_dir(Path(resolved_root) / f"{prefix}_{timestamp}")

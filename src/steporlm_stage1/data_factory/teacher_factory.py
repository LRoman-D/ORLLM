from __future__ import annotations

from typing import Any

from steporlm_stage1.data_factory.qwen_rag_teacher import QwenRagTeacherGenerator
from steporlm_stage1.data_factory.zhipu_teacher import ZhipuTeacherGenerator


def build_teacher_generator(config: dict[str, Any]):
    backend = str(config.get("teacher_backend", "zhipu_api")).lower()
    if backend in {"zhipu", "zhipu_api", "api"}:
        return ZhipuTeacherGenerator.from_env(
            model_env_var=str(config.get("data_model_env_var", "ZHIPUAI_DATA_MODEL")),
            timeout_seconds=int(config.get("request_timeout_seconds", 60)),
            max_retries=int(config.get("request_max_retries", 2)),
            retry_backoff_seconds=float(config.get("request_retry_backoff_seconds", 1.5)),
            fallback_model=str(config.get("data_model_fallback", "glm-4.5-air")),
        )
    if backend in {"qwen_rag", "rag", "local_qwen_rag"}:
        return QwenRagTeacherGenerator.from_config(config)
    raise ValueError(f"Unsupported teacher_backend: {backend}")


def teacher_backend_name(generator) -> str:
    return str(getattr(generator, "backend_name", generator.__class__.__name__))


def teacher_model_name(generator) -> str:
    if hasattr(generator, "model_name"):
        return str(generator.model_name)
    client = getattr(generator, "client", None)
    config = getattr(client, "config", None)
    if config is not None and getattr(config, "model", None):
        return str(config.model)
    return "unknown"

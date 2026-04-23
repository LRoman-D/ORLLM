from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from steporlm_stage1.utils.io import ensure_dir, load_yaml_config
from steporlm_stage1.utils.paths import map_repo_relative_path
from steporlm_stage1.utils.training_reports import (
    prepare_training_artifact_dirs,
    summarize_training_history,
    write_training_artifacts,
)


def _load_quant_config(config: dict) -> BitsAndBytesConfig | None:
    if not config.get("load_in_4bit", False):
        return None
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "4-bit training was requested, but bitsandbytes is not installed. "
            "Use Linux/WSL for QLoRA on an 8GB GPU, or disable load_in_4bit only if you have enough memory."
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _path_from_config(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else map_repo_relative_path(path)


def _checkpoint_dirs_under(root: Path) -> list[Path]:
    if root.name.startswith("checkpoint-") and root.is_dir():
        return [root]
    if not root.exists():
        return []
    return [path for path in root.glob("checkpoint-*") if path.is_dir()]


def _find_latest_checkpoint(config: dict, current_checkpoints_dir: Path) -> Path | None:
    resume_value = config.get("resume_from_checkpoint", False)
    if not resume_value:
        return None

    if isinstance(resume_value, str) and resume_value.lower() not in {"auto", "true", "yes"}:
        checkpoint = _path_from_config(resume_value)
        return checkpoint if checkpoint.exists() else None

    roots: list[Path] = []
    if config.get("resume_checkpoint_dir"):
        roots.append(_path_from_config(config["resume_checkpoint_dir"]))

    run_root_value = config.get("run_root")
    if run_root_value:
        run_root = Path(run_root_value)
        roots.extend(path for path in run_root.glob("*/weights/checkpoints") if path.is_dir())

    output_dir = Path(config["output_dir"])
    default_run_root = output_dir.parent / f"{output_dir.name}_runs"
    roots.extend(path for path in default_run_root.glob("*/weights/checkpoints") if path.is_dir())
    roots.extend([current_checkpoints_dir, output_dir])

    checkpoints: dict[Path, Path] = {}
    for root in roots:
        for checkpoint in _checkpoint_dirs_under(root):
            checkpoints[checkpoint.resolve()] = checkpoint
    if not checkpoints:
        return None

    return max(
        checkpoints.values(),
        key=lambda path: (_checkpoint_step(path), path.stat().st_mtime),
    )


def _update_latest_adapter_link(link_path: str | Path, target_path: str | Path) -> Path:
    link = Path(link_path)
    target = Path(target_path)
    if link.resolve() == target.resolve():
        return link
    if link.exists() or link.is_symlink():
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    return link


def train_lora(config_path: str | Path) -> dict:
    config = load_yaml_config(config_path)

    artifact_dirs = prepare_training_artifact_dirs(
        output_dir=config["output_dir"],
        run_root=config.get("run_root"),
        run_prefix=config.get("run_prefix", "sft_train"),
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = _load_quant_config(config)
    torch_dtype = torch.float32
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if config.get("use_bf16_if_available", True) and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False

    dataset = load_dataset("json", data_files={"train": str(config["dataset_path"])})
    train_dataset = dataset["train"]

    def format_sample(row: dict) -> dict:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train_dataset = train_dataset.map(format_sample, remove_columns=train_dataset.column_names)

    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    training_kwargs = {
        "output_dir": str(artifact_dirs["checkpoints_dir"]),
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "num_train_epochs": config["num_train_epochs"],
        "logging_steps": config["logging_steps"],
        "save_steps": config["save_steps"],
        "bf16": torch.cuda.is_available()
        and config.get("use_bf16_if_available", True)
        and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": config.get("gradient_checkpointing", True),
        "gradient_checkpointing_kwargs": config.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
        "report_to": config.get("report_to", "none"),
        "dataset_text_field": "text",
        "max_length": config["max_seq_length"],
        "logging_first_step": config.get("logging_first_step", True),
    }
    for key in [
        "save_total_limit",
        "warmup_steps",
        "warmup_ratio",
        "lr_scheduler_type",
        "max_grad_norm",
        "optim",
        "packing",
        "dataset_num_proc",
    ]:
        if key in config:
            training_kwargs[key] = config[key]
    training_args = SFTConfig(**training_kwargs)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    resume_checkpoint = _find_latest_checkpoint(config, artifact_dirs["checkpoints_dir"])
    train_result = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    final_dir = artifact_dirs["final_dir"]
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    latest_dir = None
    if config.get("update_latest_symlink", False):
        latest_dir = _update_latest_adapter_link(config["output_dir"], final_dir)

    summary = summarize_training_history(
        trainer.state.log_history,
        train_metrics=train_result.metrics,
        stage_name="sft",
    )
    summary.update(
        {
            "run_dir": str(artifact_dirs["run_dir"]),
            "weights_dir": str(artifact_dirs["weights_dir"]),
            "final_model_dir": str(final_dir),
            "latest_model_dir": str(latest_dir) if latest_dir is not None else None,
            "metrics_dir": str(artifact_dirs["metrics_dir"]),
            "checkpoints_dir": str(artifact_dirs["checkpoints_dir"]),
            "resumed_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        }
    )
    report_paths = write_training_artifacts(
        artifact_dirs["metrics_dir"],
        summary,
        trainer.state.log_history,
        config=config,
        dashboard_title="SFT Training Dashboard",
    )
    summary.update(report_paths)
    return summary

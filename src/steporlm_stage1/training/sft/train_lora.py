from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from steporlm_stage1.utils.io import ensure_dir, load_yaml_config
from steporlm_stage1.utils.training_reports import (
    prepare_training_artifact_dirs,
    summarize_training_history,
    write_training_artifacts,
)


def _load_quant_config(config: dict) -> BitsAndBytesConfig | None:
    if not config.get("load_in_4bit", False):
        return None
    if importlib.util.find_spec("bitsandbytes") is None:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )


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
    training_args = SFTConfig(
        output_dir=str(artifact_dirs["checkpoints_dir"]),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        bf16=torch.cuda.is_available() and config.get("use_bf16_if_available", True) and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="none",
        dataset_text_field="text",
        max_length=config["max_seq_length"],
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    final_dir = artifact_dirs["final_dir"]
    latest_dir = ensure_dir(config["output_dir"])
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    trainer.save_model(str(latest_dir))
    tokenizer.save_pretrained(str(latest_dir))

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
            "latest_model_dir": str(latest_dir),
            "metrics_dir": str(artifact_dirs["metrics_dir"]),
            "checkpoints_dir": str(artifact_dirs["checkpoints_dir"]),
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

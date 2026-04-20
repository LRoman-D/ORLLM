from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer

from steporlm_stage1.utils.io import ensure_dir, load_yaml_config
from steporlm_stage1.utils.modeling import load_causal_lm, load_tokenizer
from steporlm_stage1.utils.training_reports import (
    prepare_training_artifact_dirs,
    summarize_training_history,
    write_training_artifacts,
)


def _tokenize_pair(tokenizer, prompt_messages, chosen: str, rejected: str, max_prompt_length: int, max_response_length: int):
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"][:max_prompt_length]
    chosen_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"][:max_response_length] + [tokenizer.eos_token_id]
    rejected_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"][:max_response_length] + [tokenizer.eos_token_id]
    return {
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
    }


@dataclass
class WeightedDPOCollator:
    tokenizer: Any

    @staticmethod
    def _pad_to_length(batch_tensor: torch.Tensor, target_length: int, padding_value: int) -> torch.Tensor:
        if batch_tensor.size(1) >= target_length:
            return batch_tensor
        return F.pad(batch_tensor, (0, target_length - batch_tensor.size(1)), value=padding_value)

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        chosen_input_ids = []
        chosen_attention_mask = []
        chosen_labels = []
        rejected_input_ids = []
        rejected_attention_mask = []
        rejected_labels = []
        weights = []

        for feature in features:
            prompt_ids = feature["prompt_ids"]
            chosen_ids = feature["chosen_ids"]
            rejected_ids = feature["rejected_ids"]

            chosen_seq = torch.tensor(prompt_ids + chosen_ids, dtype=torch.long)
            rejected_seq = torch.tensor(prompt_ids + rejected_ids, dtype=torch.long)
            chosen_label = torch.tensor([-100] * len(prompt_ids) + chosen_ids, dtype=torch.long)
            rejected_label = torch.tensor([-100] * len(prompt_ids) + rejected_ids, dtype=torch.long)

            chosen_input_ids.append(chosen_seq)
            chosen_attention_mask.append(torch.ones_like(chosen_seq))
            chosen_labels.append(chosen_label)
            rejected_input_ids.append(rejected_seq)
            rejected_attention_mask.append(torch.ones_like(rejected_seq))
            rejected_labels.append(rejected_label)
            weights.append(feature["weight"])

        target_length = max(
            max(seq.size(0) for seq in chosen_input_ids),
            max(seq.size(0) for seq in rejected_input_ids),
        )
        return {
            "chosen_input_ids": self._pad_to_length(
                pad_sequence(chosen_input_ids, batch_first=True, padding_value=pad_id),
                target_length,
                pad_id,
            ),
            "chosen_attention_mask": self._pad_to_length(
                pad_sequence(chosen_attention_mask, batch_first=True, padding_value=0),
                target_length,
                0,
            ),
            "chosen_labels": self._pad_to_length(
                pad_sequence(chosen_labels, batch_first=True, padding_value=-100),
                target_length,
                -100,
            ),
            "rejected_input_ids": self._pad_to_length(
                pad_sequence(rejected_input_ids, batch_first=True, padding_value=pad_id),
                target_length,
                pad_id,
            ),
            "rejected_attention_mask": self._pad_to_length(
                pad_sequence(rejected_attention_mask, batch_first=True, padding_value=0),
                target_length,
                0,
            ),
            "rejected_labels": self._pad_to_length(
                pad_sequence(rejected_labels, batch_first=True, padding_value=-100),
                target_length,
                -100,
            ),
            "weights": torch.tensor(weights, dtype=torch.float32),
        }


def _sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    loss_mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * loss_mask
    lengths = loss_mask.sum(dim=-1).clamp(min=1)
    return token_logps.sum(dim=-1) / lengths


class WeightedDPOTrainer(Trainer):
    def __init__(self, *args, beta: float, ref_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.ref_model = ref_model
        self._metric_buffer: list[dict[str, float]] = []
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weights").to(model.device)
        chosen_input_ids = inputs["chosen_input_ids"].to(model.device)
        chosen_attention_mask = inputs["chosen_attention_mask"].to(model.device)
        chosen_labels = inputs["chosen_labels"].to(model.device)
        rejected_input_ids = inputs["rejected_input_ids"].to(model.device)
        rejected_attention_mask = inputs["rejected_attention_mask"].to(model.device)
        rejected_labels = inputs["rejected_labels"].to(model.device)
        batch_size = chosen_input_ids.size(0)

        combined_inputs = {
            "input_ids": torch.cat([chosen_input_ids, rejected_input_ids], dim=0),
            "attention_mask": torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0),
        }
        combined_labels = torch.cat([chosen_labels, rejected_labels], dim=0)

        policy_outputs = model(**combined_inputs)
        policy_logps = _sequence_logps(policy_outputs.logits, combined_labels)
        policy_chosen_logps = policy_logps[:batch_size]
        policy_rejected_logps = policy_logps[batch_size:]

        with torch.no_grad():
            ref_device = next(self.ref_model.parameters()).device
            ref_inputs = {key: value.to(ref_device) for key, value in combined_inputs.items()}
            ref_labels = combined_labels.to(ref_device)
            ref_outputs = self.ref_model(**ref_inputs)
            ref_logps = _sequence_logps(ref_outputs.logits, ref_labels).to(model.device)
            ref_chosen_logps = ref_logps[:batch_size]
            ref_rejected_logps = ref_logps[batch_size:]

        logits = self.beta * (
            (policy_chosen_logps - policy_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        )
        reward_margin = (policy_chosen_logps - policy_rejected_logps).mean().detach()
        preference_accuracy = (logits > 0).float().mean().detach()
        self._metric_buffer.append(
            {
                "reward_margin": float(reward_margin.cpu()),
                "preference_accuracy": float(preference_accuracy.cpu()),
            }
        )
        loss = -(weights * F.logsigmoid(logits)).mean()
        outputs = {
            "loss": loss.detach(),
            "reward_margin": reward_margin,
        }
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if self._metric_buffer:
            reward_margin = sum(item["reward_margin"] for item in self._metric_buffer) / len(self._metric_buffer)
            preference_accuracy = sum(item["preference_accuracy"] for item in self._metric_buffer) / len(
                self._metric_buffer
            )
            logs = dict(logs)
            logs.setdefault("reward_margin", reward_margin)
            logs.setdefault("preference_accuracy", preference_accuracy)
            self._metric_buffer.clear()
        return super().log(logs, start_time=start_time)


def train_weighted_dpo(config_path: str | Path) -> dict:
    config = load_yaml_config(config_path)

    artifact_dirs = prepare_training_artifact_dirs(
        output_dir=config["output_dir"],
        run_root=config.get("run_root"),
        run_prefix=config.get("run_prefix", "dpo_train"),
    )

    tokenizer = load_tokenizer(config["sft_adapter_path"])
    train_rows = load_dataset("json", data_files={"train": str(config["dataset_path"])})["train"]

    def preprocess(row):
        sample = _tokenize_pair(
            tokenizer,
            row["prompt_messages"],
            row["chosen"],
            row["rejected"],
            config["max_prompt_length"],
            config["max_response_length"],
        )
        sample["weight"] = float(row["weight"])
        return sample

    train_dataset = train_rows.map(preprocess, remove_columns=train_rows.column_names)
    ref_model_on_cpu = bool(config.get("ref_model_on_cpu", False))

    policy_model = load_causal_lm(
        config["sft_adapter_path"],
        load_in_4bit=config.get("load_in_4bit", False),
        use_bf16_if_available=config.get("use_bf16_if_available", True),
        is_adapter=True,
        adapter_is_trainable=True,
        base_model_override=config["model_name_or_path"],
    )
    ref_model = load_causal_lm(
        config["sft_adapter_path"],
        load_in_4bit=False,
        use_bf16_if_available=False if ref_model_on_cpu else config.get("use_bf16_if_available", True),
        is_adapter=True,
        adapter_is_trainable=False,
        device_map_override="cpu" if ref_model_on_cpu else None,
        dtype_override=torch.float32 if ref_model_on_cpu else None,
        base_model_override=config["model_name_or_path"],
    )

    from transformers import TrainingArguments

    training_args = TrainingArguments(
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
        remove_unused_columns=False,
    )
    trainer = WeightedDPOTrainer(
        model=policy_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=WeightedDPOCollator(tokenizer),
        beta=config["beta"],
        ref_model=ref_model,
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
        stage_name="weighted_dpo",
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
        dashboard_title="Weighted DPO Training Dashboard",
    )
    summary.update(report_paths)
    return summary

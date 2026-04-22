from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from steporlm_stage1.utils.io import PROJECT_ROOT, resolve_local_path


def _resolve_model_reference(model_path: str | Path, relative_to: str | Path | None = None) -> str:
    path = Path(model_path)
    if path.is_absolute():
        return str(path)

    candidates = []
    if relative_to is not None:
        base = Path(relative_to)
        candidates.append(base / path)
    candidates.append(PROJECT_ROOT / path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return resolve_local_path(path, relative_to or PROJECT_ROOT)


def load_quant_config(enable_4bit: bool) -> BitsAndBytesConfig | None:
    if not enable_4bit:
        return None
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "4-bit loading was requested, but bitsandbytes is not installed. "
            "For a 7B model on an 8GB GPU, run under Linux/WSL with bitsandbytes or point the config to a GPTQ/AWQ quantized model."
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )


def select_torch_dtype(use_bf16_if_available: bool = True) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if use_bf16_if_available and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_tokenizer(model_path: str | Path):
    resolved_model_path = _resolve_model_reference(model_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_path: str | Path,
    load_in_4bit: bool = False,
    use_bf16_if_available: bool = True,
    is_adapter: bool = False,
    adapter_is_trainable: bool = False,
    device_map_override: str | dict | None = None,
    dtype_override: torch.dtype | None = None,
    base_model_override: str | Path | None = None,
):
    resolved_model_path = _resolve_model_reference(model_path)
    quant_config = load_quant_config(load_in_4bit)
    dtype = dtype_override or select_torch_dtype(use_bf16_if_available)
    common_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map_override if device_map_override is not None else ("auto" if torch.cuda.is_available() else None),
        "quantization_config": quant_config,
        "torch_dtype": dtype,
    }
    if is_adapter:
        adapter_path = resolved_model_path
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model_name = base_model_override or peft_config.base_model_name_or_path
        if not base_model_name:
            raise ValueError(
                f"Adapter at {adapter_path} does not record base_model_name_or_path. "
                "Pass base_model_override explicitly."
            )
        base_model_path = _resolve_model_reference(base_model_name, Path(adapter_path).parent)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **common_kwargs)
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=adapter_is_trainable)
    else:
        model = AutoModelForCausalLM.from_pretrained(resolved_model_path, **common_kwargs)
    model.config.use_cache = False
    return model

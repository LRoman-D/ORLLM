# ORLLM Qwen3-8B Stage Loop

This repo is a compact StepORLM-style loop for operations-research modeling:

1. Download OR questions and reference answers from external channels.
2. Filter them with solver execution plus GenPRM-style 9-step process audit.
3. Train the Qwen3-8B policy with SFT.
4. Generate multiple rollouts from the SFT policy.
5. Build preference pairs using solver success, objective match, and GenPRM semantic scores.
6. Train DPO.
7. Evaluate on the held-out external OR split.

The teacher remains frozen. RAG books, the built retrieval index, and retrieval logic are preserved under `referencebooks/`, `data/rag/`, and `src/steporlm_stage1/rag/`. The Qwen3-8B base model is intentionally not stored in Git; place or download it at `models/Qwen3-8B` before running training or evaluation.

## Layout

| Path | Purpose |
| --- | --- |
| `src/steporlm_stage1/` | External data preparation, RAG teacher, rollout, GenPRM audit, SFT, DPO, evaluation |
| `src/steporlm_stage1/quality/` | GenPRM prompt, parser, and process scoring |
| `src/steporlm_stage1/solvers/` | Internal OR-Tools status/result helpers |
| `docs/ortools_style_guide.md` | OR-Tools coding standard used by prompts |
| `docs/model_comparison_summary.md` | Interpretation of the saved three-model comparison run |
| `configs/` | Canonical Qwen3-8B configs for the full loop |
| `data/external_or/` | External raw questions, SFT/rollout/eval question splits, and SFT teacher trajectories |
| `data/processed/` | Converted SFT/DPO training datasets |
| `artifacts/` | Durable LoRA adapters and checkpoints |
| `runs/` | Rollouts, preferences, metrics, and timestamped experiment records |

## Setup

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
pip install -e .
```

Prepare the base/teacher model at `models/Qwen3-8B`. The released repository stores only the SFT/DPO LoRA adapters, not the original 8B base weights.

Build the RAG index once:

```bash
python -m steporlm_stage1.cli build-rag-index --config-path configs/rag_index.yaml
```

## Small Closed Loop

```bash
python -m steporlm_stage1.cli generate-dataset --config-path configs/stage1_data.yaml
python -m steporlm_stage1.cli prepare-sft --input-dir data/external_or/sft_teacher --output-dir data/processed/qwen3_sft
python -m steporlm_stage1.cli train-lora --config-path configs/stage1_sft.yaml
python -m steporlm_stage1.cli generate-real-rollouts --config-path configs/stage1_real_rollout.yaml
python -m steporlm_stage1.cli build-preferences \
  --rollout-path runs/qwen3_8b/rollouts/real_rollouts.jsonl \
  --output-path runs/qwen3_8b/preferences/preferences.jsonl \
  --require-chosen-success \
  --require-chosen-process-pass
python -m steporlm_stage1.cli prepare-dpo --config-path configs/stage1_dpo_data.yaml
python -m steporlm_stage1.cli train-dpo --config-path configs/stage1_dpo_train.yaml
python -m steporlm_stage1.cli compare-models --config-path configs/stage1_compare.yaml
```

Or run the same flow as one resumable server-side script:

```bash
bash scripts/run_qwen3_8b_pipeline.sh
```

Use `scripts/report_generation_stats.py` to inspect data-generation progress.

The saved three-model comparison is summarized in `docs/model_comparison_summary.md`.

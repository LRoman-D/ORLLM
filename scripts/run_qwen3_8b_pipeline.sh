#!/usr/bin/env bash
# Entry point for the Qwen3-8B SFT -> rollout -> DPO loop.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs runs/qwen3_8b/preferences artifacts/qwen3-8b

echo "[$(date '+%F %T')] Step 0/7: ensure RAG index exists"
if [[ ! -f data/rag/or_books/chunks.jsonl ]]; then
  python -m steporlm_stage1.cli build-rag-index --config-path configs/rag_index.yaml
fi

echo "[$(date '+%F %T')] Step 1/7: generate teacher data with RAG + GenPRM"
python -m steporlm_stage1.cli generate-dataset --config-path configs/stage1_data.yaml

echo "[$(date '+%F %T')] Step 2/7: convert accepted teacher data to chat SFT"
python -m steporlm_stage1.cli prepare-sft \
  --input-dir data/processed/qwen3_rag_teacher \
  --output-dir data/processed/qwen3_sft

echo "[$(date '+%F %T')] Step 3/7: train Qwen3-8B SFT LoRA"
python -m steporlm_stage1.cli train-lora --config-path configs/stage1_sft.yaml

echo "[$(date '+%F %T')] Step 4/7: generate SFT rollouts and audit with frozen RAG teacher"
python -m steporlm_stage1.cli generate-real-rollouts --config-path configs/stage1_real_rollout.yaml

echo "[$(date '+%F %T')] Step 5/7: build solver+GenPRM preference pairs"
python -m steporlm_stage1.cli build-preferences \
  --rollout-path runs/qwen3_8b/rollouts/real_rollouts.jsonl \
  --output-path runs/qwen3_8b/preferences/preferences.jsonl \
  --run-root runs/qwen3_8b \
  --run-prefix preferences \
  --require-chosen-success \
  --require-chosen-process-pass \
  --min-correct-steps 8

echo "[$(date '+%F %T')] Step 6/7: prepare DPO dataset"
python -m steporlm_stage1.cli prepare-dpo --config-path configs/stage1_dpo_data.yaml

echo "[$(date '+%F %T')] Step 7/7: train DPO adapter and compare"
python -m steporlm_stage1.cli train-dpo --config-path configs/stage1_dpo_train.yaml
python -m steporlm_stage1.cli compare-models --config-path configs/stage1_compare.yaml

echo "[$(date '+%F %T')] Qwen3-8B loop completed"

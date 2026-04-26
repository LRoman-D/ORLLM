#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m steporlm_stage1.cli build-preferences \
  --rollout-path runs/qwen3_8b/rollouts/real_rollouts.jsonl \
  --output-path runs/qwen3_8b/preferences/preferences_hard.jsonl \
  --run-root runs/qwen3_8b \
  --run-prefix preferences_hard \
  --allow-unsuccessful-chosen \
  --require-chosen-executable-optimal \
  --allow-process-warnings \
  --min-weight 0.9

python -m steporlm_stage1.cli prepare-dpo --config-path configs/stage1_dpo_data_hard.yaml
python -m steporlm_stage1.cli train-dpo --config-path configs/stage1_dpo_train_hard.yaml
python -m steporlm_stage1.cli compare-models --config-path configs/stage1_compare_hard_eval.yaml

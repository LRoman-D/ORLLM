#!/bin/bash
# scripts/run_example.sh - Run a training or evaluation example on remote Linux

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# [Optional] Map persistent storage to different mount points via env vars
export PROJECT_DATA_DIR="${PROJECT_DATA_DIR:-$PROJECT_ROOT/data}"
export PROJECT_MODEL_DIR="${PROJECT_MODEL_DIR:-$PROJECT_ROOT/models}"
export PROJECT_OUTPUT_DIR="${PROJECT_OUTPUT_DIR:-$PROJECT_ROOT/outputs}"
export PROJECT_LOG_DIR="${PROJECT_LOG_DIR:-$PROJECT_ROOT/logs}"
export PROJECT_CACHE_DIR="${PROJECT_CACHE_DIR:-$PROJECT_ROOT/.cache}"

# HF and Torch environment variables
export HF_HOME="$PROJECT_CACHE_DIR/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PYTHONUTF8="1"

# Load API keys from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "=== Running Model Evaluation Example ==="
echo "Data Dir:   $PROJECT_DATA_DIR"
echo "Output Dir: $PROJECT_OUTPUT_DIR"

# Example: Run evaluation on the test set
# Replace with your actual training command if needed
python3 -m steporlm_stage1.cli evaluate-model --config-path configs/stage1_test_eval.yaml

echo "---"
echo "Done. Results are in $PROJECT_OUTPUT_DIR/runs"

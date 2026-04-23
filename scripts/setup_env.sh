#!/usr/bin/env bash
# Initialize the local ORLLM environment.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [1/4] Creating persistent directories ==="
mkdir -p data/processed data/rag models artifacts runs logs .cache
touch data/.gitkeep models/.gitkeep

echo "=== [2/4] Environment setup ==="
if command -v conda >/dev/null 2>&1; then
  echo "Conda detected. Recommended commands:"
  echo "  conda env create -f environment.yml"
  echo "  conda activate orllm"
  echo "  pip install -r requirements.txt"
  echo "  pip install -e ."
else
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install -e .
fi

echo "=== [3/4] Local config ==="
if [[ ! -f .env ]]; then
  cat > .env <<'EOT'
TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOT
fi

echo "=== [4/4] Model reminder ==="
echo "Expected frozen teacher/base model path:"
echo "  $PROJECT_ROOT/models/Qwen3-8B"
echo "Download helper:"
echo "  bash scripts/download_qwen3_8b_tmux.sh"
echo "Build RAG index:"
echo "  python -m steporlm_stage1.cli build-rag-index --config-path configs/rag_index.yaml"
echo "Run the small loop:"
echo "  bash scripts/run_qwen3_8b_pipeline.sh"


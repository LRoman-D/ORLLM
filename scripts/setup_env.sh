#!/bin/bash
# scripts/setup_env.sh - Initialize Linux environment for StepORLM Stage-1

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [1/4] Creating Persistent Directories ==="
mkdir -p data models outputs logs .cache
mkdir -p data/processed/stage1_dataset data/processed/stage1_sft
mkdir -p outputs/qwen2.5-7b-stage1-lora

touch data/.gitkeep models/.gitkeep outputs/.gitkeep logs/.gitkeep

echo "=== [2/4] Environment Setup ==="
if command -v conda >/dev/null 2>&1; then
    echo "Conda detected. Recommended commands:"
    echo "  conda env create -f environment.yml"
    echo "  conda activate orllm"
    echo "  pip install -r requirements.txt"
    echo "  pip install -e ."
else
    echo "Conda not found. Setting up with venv and pip..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .
fi

echo "=== [3/4] Configuration ==="
if [ ! -f .env ]; then
    echo "Creating template .env file. Please update it with your API keys."
    cat <<EOT > .env
ZHIPUAI_API_KEY=your_api_key_here
ZHIPUAI_MODEL=glm-4.5
ZHIPUAI_DATA_MODEL=glm-4.5-air
EOT
fi

echo "=== [4/4] Model Path Reminder ==="
echo "This branch already includes SFT data and LoRA adapter artifacts."
echo "You still need to prepare the base model at:"
echo "  $PROJECT_ROOT/models/Qwen2.5-7B-Instruct"
echo "A symlink to the real model directory is also acceptable."

echo "---"
echo "Setup complete!"
echo "Next steps:"
echo "1. Activate your environment (conda activate orllm or source .venv/bin/activate)"
echo "2. Edit .env with your credentials"
echo "3. Ensure the base model path exists"
echo "4. Build the RAG index with: python -m steporlm_stage1.cli build-rag-index --config-path configs/rag_index.yaml"
echo "5. Generate RAG SFT data with: python -m steporlm_stage1.cli generate-dataset --config-path configs/stage1_data.yaml"
echo "6. Or continue training with: bash scripts/run_7b_dpo_pipeline.sh"

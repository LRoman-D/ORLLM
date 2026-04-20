#!/bin/bash
# scripts/setup_env.sh - Initialize Linux environment for StepORLM Stage-1

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [1/3] Creating Persistent Directories ==="
# Ensure directories for data, models, outputs, and logs exist
mkdir -p data models outputs logs .cache
touch data/.gitkeep models/.gitkeep outputs/.gitkeep logs/.gitkeep

echo "=== [2/3] Environment Setup ==="
if command -v conda >/dev/null 2>&1; then
    echo "Conda detected. To create the environment, run:"
    echo "  conda env create -f environment.yml"
    echo "  conda activate orllm"
else
    echo "Conda not found. Setting up with venv and pip..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .
fi

echo "=== [3/3] Configuration ==="
if [ ! -f .env ]; then
    echo "Creating template .env file. Please update it with your API keys."
    cat <<EOT > .env
ZHIPUAI_API_KEY=your_api_key_here
ZHIPUAI_MODEL=glm-4.5
ZHIPUAI_DATA_MODEL=glm-4.5-air
EOT
fi

echo "---"
echo "Setup complete!"
echo "Next steps:"
echo "1. Activate your environment (conda activate orllm)"
echo "2. Edit .env with your credentials"
echo "3. Run an example: bash scripts/run_example.sh"

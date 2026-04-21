#!/usr/bin/env bash
# Run full 7B SFT->DPO pipeline with resumable rollout and checkpointed training.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f .venv/bin/activate ]]; then
  echo "[error] .venv not found at $PROJECT_ROOT/.venv" >&2
  exit 1
fi

source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export PYTHONUNBUFFERED=1
mkdir -p logs runs/preferences_7b_720

echo "[$(date '+%F %T')] Step 0/5: prepare deterministic 500-sample rollout source"
python - <<'PY'
from pathlib import Path

src = Path("data/processed/stage1_dataset/train.jsonl")
dst = Path("data/processed/stage1_dataset/train_500.jsonl")
rows = []
with src.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 500:
            break
        rows.append(line)
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    f.writelines(rows)
print({"source": str(src), "target": str(dst), "rows": len(rows)})
PY

echo "[$(date '+%F %T')] Step 1/5: rollout on 500 training samples (resumable)"
python -m steporlm_stage1.cli generate-real-rollouts --config-path configs/stage1_real_rollout_7b_720.yaml

echo "[$(date '+%F %T')] Step 2/5: build preference pairs"
python -m steporlm_stage1.cli build-preferences \
  --rollout-path outputs/runs/real_rollouts_7b_720/real_rollouts.jsonl \
  --output-path outputs/runs/preferences_7b_720/preferences.jsonl \
  --run-root runs \
  --run-prefix preferences_7b_720 \
  --require-chosen-success

echo "[$(date '+%F %T')] Step 3/5: prepare DPO dataset (cap=300)"
python -m steporlm_stage1.cli prepare-dpo --config-path configs/stage1_dpo_data_7b.yaml

TRAIN_ROWS=$(wc -l < outputs/runs/dpo_data_7b_720/train.jsonl || echo 0)
echo "[$(date '+%F %T')] DPO train rows: $TRAIN_ROWS"
if [[ "$TRAIN_ROWS" -le 0 ]]; then
  echo "[error] No DPO training rows generated from success-constrained preferences. Stop here." >&2
  exit 2
fi

echo "[$(date '+%F %T')] Step 4/5: train DPO model (checkpointed)"
python -m steporlm_stage1.cli train-dpo --config-path configs/stage1_dpo_train_7b.yaml

echo "[$(date '+%F %T')] Step 5/5: compare base vs SFT vs DPO on 300 generated questions"
python -m steporlm_stage1.cli compare-models --config-path configs/stage1_compare_7b_300.yaml

echo "[$(date '+%F %T')] Pipeline completed successfully"

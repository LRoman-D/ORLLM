#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_PATH="${1:-benchmarks/or_external/prepared/benchmark_samples.jsonl}"
SESSION_NAME="${2:-or_benchmark_eval}"
LOG_DIR="${3:-logs}"

mkdir -p "$LOG_DIR" benchmarks/or_external/runs
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="benchmarks/or_external/runs/eval_${TIMESTAMP}"
LOG_FILE="$LOG_DIR/${SESSION_NAME}_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/${SESSION_NAME}.pid"

if ! [ -f "$DATASET_PATH" ]; then
  echo "Dataset not found: $DATASET_PATH" >&2
  exit 1
fi

CMD="cd '$ROOT_DIR' && PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./.venv/bin/python scripts/evaluate_external_or_benchmarks.py --dataset '$DATASET_PATH' --run-dir '$RUN_DIR'"

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "$SESSION_NAME" "$CMD 2>&1 | tee -a '$LOG_FILE'"
  echo "Started in tmux session: $SESSION_NAME"
  echo "Run dir: $RUN_DIR"
  echo "Log file: $LOG_FILE"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

if command -v setsid >/dev/null 2>&1; then
  setsid bash -lc "$CMD" >"$LOG_FILE" 2>&1 < /dev/null &
else
  nohup bash -lc "$CMD" >"$LOG_FILE" 2>&1 &
fi
PID=$!
echo "$PID" >"$PID_FILE"

echo "tmux not found; started with detached background runner."
echo "PID: $PID"
echo "PID file: $PID_FILE"
echo "Run dir: $RUN_DIR"
echo "Log file: $LOG_FILE"
echo "Tail with: tail -f $LOG_FILE"

sleep 5
if ! ps -p "$PID" >/dev/null 2>&1; then
  echo "Warning: process exited early. Check log: $LOG_FILE" >&2
  exit 1
fi

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_REPO="${1:-Qwen/Qwen3-8B}"
LOCAL_DIR="${2:-models/Qwen3-8B}"
SESSION_NAME="${3:-download_qwen3_8b}"
LOG_DIR="${4:-logs}"
MAX_WORKERS="${MAX_WORKERS:-4}"
REVISION="${REVISION:-main}"
HF_DOWNLOAD_BACKEND="${HF_DOWNLOAD_BACKEND:-curl}"

mkdir -p "$LOG_DIR" "$(dirname "$LOCAL_DIR")"
LOG_FILE="$LOG_DIR/${SESSION_NAME}_$(date -u +%Y%m%d_%H%M%S).log"
RUNNER_FILE="$LOG_FILE.runner.sh"

if [[ -x "$PROJECT_ROOT/.venv/bin/hf" ]]; then
  HF_BIN="$PROJECT_ROOT/.venv/bin/hf"
elif command -v hf >/dev/null 2>&1; then
  HF_BIN="$(command -v hf)"
else
  echo "hf CLI not found. Install dependencies first, for example: bash scripts/setup_env.sh" >&2
  exit 1
fi

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

cat >"$RUNNER_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$PROJECT_ROOT"

export HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET"
export HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER"

MODEL_REPO="$MODEL_REPO"
LOCAL_DIR="$LOCAL_DIR"
REVISION="$REVISION"
HF_BIN="$HF_BIN"
MAX_WORKERS="$MAX_WORKERS"
HF_DOWNLOAD_BACKEND="$HF_DOWNLOAD_BACKEND"

mkdir -p "\$LOCAL_DIR"

if [[ "\$MODEL_REPO" == "Qwen/Qwen3-8B" && "\$HF_DOWNLOAD_BACKEND" == "curl" ]]; then
  "\$HF_BIN" download "\$MODEL_REPO" \\
    --revision "\$REVISION" \\
    --local-dir "\$LOCAL_DIR" \\
    --exclude 'model-*.safetensors' \\
    --max-workers "\$MAX_WORKERS"

  base_url="https://huggingface.co/\$MODEL_REPO/resolve/\$REVISION"

  download_file() {
    local file="\$1"
    local expected_size="\$2"
    local path="\$LOCAL_DIR/\$file"

    if [[ -f "\$path" ]]; then
      local current_size
      current_size="\$(stat -c '%s' "\$path")"
      if [[ "\$current_size" == "\$expected_size" ]]; then
        echo "[skip] \$file already complete (\$current_size bytes)"
        return 0
      fi
      echo "[resume] \$file from \$current_size / \$expected_size bytes"
    else
      echo "[download] \$file (\$expected_size bytes)"
    fi

    until curl -L --fail --retry 50 --retry-all-errors --retry-delay 15 \\
      --connect-timeout 60 --speed-limit 1024 --speed-time 300 \\
      -C - -o "\$path" "\$base_url/\$file"; do
      echo "[retry] \$file after curl failure"
      sleep 20
    done

    local final_size
    final_size="\$(stat -c '%s' "\$path")"
    if [[ "\$final_size" != "\$expected_size" ]]; then
      echo "[error] \$file size mismatch: got \$final_size, expected \$expected_size" >&2
      return 1
    fi
    echo "[done] \$file"
  }

  failed=0
  active=0
  for item in \\
    "model-00001-of-00005.safetensors 3996250744" \\
    "model-00002-of-00005.safetensors 3993160032" \\
    "model-00003-of-00005.safetensors 3959604768" \\
    "model-00004-of-00005.safetensors 3187841392" \\
    "model-00005-of-00005.safetensors 1244659840"; do
    download_file \${item} &
    active=\$((active + 1))
    if (( active >= MAX_WORKERS )); then
      if ! wait -n; then
        failed=1
      fi
      active=\$((active - 1))
    fi
  done

  while (( active > 0 )); do
    if ! wait -n; then
      failed=1
    fi
    active=\$((active - 1))
  done

  exit "\$failed"
else
  "\$HF_BIN" download "\$MODEL_REPO" \\
    --revision "\$REVISION" \\
    --local-dir "\$LOCAL_DIR" \\
    --max-workers "\$MAX_WORKERS"
fi
EOF

chmod +x "$RUNNER_FILE"

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "$SESSION_NAME" "bash '$RUNNER_FILE' 2>&1 | tee -a '$LOG_FILE'"
  echo "Started download in tmux session: $SESSION_NAME"
  echo "Log file: $LOG_FILE"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

(
  cd "$PROJECT_ROOT"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash "$RUNNER_FILE" >"$LOG_FILE" 2>&1 < /dev/null &
  else
    nohup bash "$RUNNER_FILE" >"$LOG_FILE" 2>&1 < /dev/null &
  fi
  echo $!
) >"$LOG_FILE.pid"
PID="$(cat "$LOG_FILE.pid")"
echo "Started download with nohup. PID: $PID"
echo "PID file: $LOG_FILE.pid"
echo "Log file: $LOG_FILE"

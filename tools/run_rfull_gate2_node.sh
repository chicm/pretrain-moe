#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "usage: $0 CONFIG RUN_DIR PROJECT_DIR MEGATRON_DIR MASTER_PORT DATA_CACHE_PATH [launch-node options...]" >&2
  exit 2
fi

CONFIG=$(realpath "$1")
RUN_DIR=$2
PROJECT_DIR=$(realpath "$3")
MEGATRON_DIR=$(realpath "$4")
MASTER_PORT=$5
DATA_CACHE_PATH=$6
shift 6
PYTHON=${RFULL_PYTHON:-/opt/venv/bin/python}

if [[ "${TORCH_DISTRIBUTED_DEBUG:-}" =~ ^[Dd][Ee][Tt][Aa][Ii][Ll]$ ]]; then
  echo "REFUSE_TORCH_DISTRIBUTED_DEBUG_DETAIL: incompatible with the qualified distributed-optimizer path" >&2
  exit 8
fi
extra_mcore_args=${EXTRA_MCORE_ARGS:-}
if [[ -n "${extra_mcore_args//[[:space:]]/}" ]]; then
  echo "REFUSE_EXTRA_MCORE_ARGS: pass options as explicit launch-node arguments" >&2
  exit 9
fi

if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to overwrite immutable run directory: $RUN_DIR" >&2
  exit 3
fi
mkdir -p "$RUN_DIR"
mkdir -p "$DATA_CACHE_PATH"
LOG="$RUN_DIR/train.console.log"
TELEMETRY="$RUN_DIR/gpu.telemetry.csv"
TELEMETRY_STATUS="$RUN_DIR/gpu.telemetry.status.log"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing Python: $PYTHON" >&2
  exit 4
fi
export PYTHONPATH="$MEGATRON_DIR:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export USE_ROCM_AITER_ROPE_BACKEND=0
if ! "$PYTHON" -c 'from rfull_moe.pinned_mcore import verify_pinned_mcore_sources; verify_pinned_mcore_sources()'; then
  echo "pinned Megatron source guard failed" >&2
  exit 5
fi
GPU_COUNT=$($PYTHON -c 'import torch; print(torch.cuda.device_count())')
if [[ "$GPU_COUNT" != "8" ]]; then
  echo "Gate 2 requires exactly 8 visible GPUs, observed $GPU_COUNT" >&2
  exit 7
fi

export PYTHONUNBUFFERED=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=0
export NVTE_UNFUSED_ATTN=1
export NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-2}
export RFULL_RUNTIME_EVIDENCE=1
export RFULL_TRACEBACK_INTERVAL_SECONDS=${RFULL_TRACEBACK_INTERVAL_SECONDS:-180}

telemetry_loop() {
  local first=1
  echo "GPU_TELEMETRY_START,$(date -u +%Y-%m-%dT%H:%M:%SZ),$(hostname)" >>"$TELEMETRY_STATUS"
  while true; do
    local sample
    if ! sample=$(rocm-smi --showuse --showmemuse --csv 2>>"$TELEMETRY_STATUS"); then
      echo "GPU_TELEMETRY_SAMPLE_ERROR,$(date -u +%Y-%m-%dT%H:%M:%SZ),$(hostname)" >>"$TELEMETRY_STATUS"
      sleep 1
      continue
    fi
    if [[ $first -eq 1 ]]; then
      printf '%s\n' "$sample" >"$TELEMETRY"
      first=0
    else
      printf '%s\n' "$sample" | sed '1d' >>"$TELEMETRY"
    fi
    sleep 1
  done
}
: >"$TELEMETRY_STATUS"
telemetry_loop &
TELEMETRY_PID=$!

cleanup() {
  local rc=$?
  if kill -0 "$TELEMETRY_PID" 2>/dev/null; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
  echo "GPU_TELEMETRY_COMPLETE,$(date -u +%Y-%m-%dT%H:%M:%SZ),$(hostname),rc=$rc" >>"$TELEMETRY_STATUS"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "RFULL_NODE_START=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) run_dir=$RUN_DIR" | tee "$LOG"
printf 'RFULL_CONFIG_SHA256=' | tee -a "$LOG"
sha256sum "$CONFIG" | tee -a "$LOG"

set +e
"$PYTHON" "$PROJECT_DIR/tools/rfull_gate2.py" launch-node \
  --config "$CONFIG" \
  --project-dir "$PROJECT_DIR" \
  --megatron-dir "$MEGATRON_DIR" \
  --master-addr 127.0.0.1 \
  --master-port "$MASTER_PORT" \
  --data-cache-path "$DATA_CACHE_PATH" \
  --python "$PYTHON" \
  "$@" 2>&1 | tee -a "$LOG"
PIPE_RC=${PIPESTATUS[0]}
set -e

echo "RFULL_NODE_COMPLETE=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) rc=$PIPE_RC" | tee -a "$LOG"
if [[ $PIPE_RC -ne 0 ]]; then
  exit "$PIPE_RC"
fi

trap - EXIT INT TERM
cleanup
(
  cd "$RUN_DIR"
  sha256sum train.console.log gpu.telemetry.csv gpu.telemetry.status.log > evidence.sha256
)

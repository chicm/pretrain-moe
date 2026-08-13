#!/usr/bin/env bash
# Run preflight, one distributed collective probe, then stock Megatron training
# on one node. This script must be started concurrently on every node.
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 PROFILE_JSON PROJECT_DIR MEGATRON_DIR RUN_DIR NODE_RANK MASTER_ADDR MASTER_PORT DATA_CACHE_PATH" >&2
  exit 2
fi
PROFILE_JSON=$1
PROJECT_DIR=$2
MEGATRON_DIR=$3
RUN_DIR=$4
NODE_RANK=$5
MASTER_ADDR=$6
MASTER_PORT=$7
DATA_CACHE_PATH=$8
PYTHON_BIN=${PYTHON_BIN:-python3}
if [[ "$DATA_CACHE_PATH" != /* ]]; then
  echo "DATA_CACHE_PATH must be absolute: $DATA_CACHE_PATH" >&2
  exit 2
fi
mkdir -p "$DATA_CACHE_PATH"

readarray -t VALUES < <("$PYTHON_BIN" - "$PROFILE_JSON" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = json.load(handle)
print(cfg["upstream"]["commit"])
print(cfg["cluster"]["nnodes"])
print(cfg["cluster"]["gpus_per_node"])
for key in ("te_fused_attention", "te_flash_attention", "te_unfused_attention"):
    print("1" if cfg["runtime"][key] else "0")
PY
)
EXPECTED_COMMIT=${VALUES[0]}
NNODES=${VALUES[1]}
GPUS_PER_NODE=${VALUES[2]}
TE_FUSED_ATTN=${VALUES[3]}
TE_FLASH_ATTN=${VALUES[4]}
TE_UNFUSED_ATTN=${VALUES[5]}

mkdir -p "$RUN_DIR" "$RUN_DIR/data-cache"
export PYTHONPATH="$MEGATRON_DIR:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}
export NVTE_FUSED_ATTN=$TE_FUSED_ATTN
export NVTE_FLASH_ATTN=$TE_FLASH_ATTN
export NVTE_UNFUSED_ATTN=$TE_UNFUSED_ATTN
# Make Transformer Engine report backend filtering so the acceptance parser can
# verify that the requested backend controls actually took effect.
export NVTE_DEBUG=${NVTE_DEBUG:-1}
export NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-2}
# ROCm AIter's fused RoPE backend advertises lower precision in this image.
# Keep the conservative backend for semantic qualification and verify via logs
# that the warning disappears.
export USE_ROCM_AITER_ROPE_BACKEND=${USE_ROCM_AITER_ROPE_BACKEND:-0}
export MEGATRON_SMOKE_TRACEBACK_INTERVAL_SECONDS=${MEGATRON_SMOKE_TRACEBACK_INTERVAL_SECONDS:-180}

GPU_TELEMETRY="$RUN_DIR/gpu-node-$NODE_RANK.csv"
{
  printf 'sampled_utc,'
  rocm-smi --showuse --showmemuse --csv 2>/dev/null | sed -n '1p'
  while true; do
    sampled_utc=$(date -u +%FT%TZ)
    rocm-smi --showuse --showmemuse --csv 2>/dev/null \
      | tail -n +2 \
      | sed "s/^/$sampled_utc,/"
    sleep "${GPU_TELEMETRY_INTERVAL_SECONDS:-1}"
  done
} >"$GPU_TELEMETRY" &
GPU_TELEMETRY_PID=$!
cleanup_gpu_telemetry() {
  rc=$?
  kill "$GPU_TELEMETRY_PID" 2>/dev/null || true
  wait "$GPU_TELEMETRY_PID" 2>/dev/null || true
  printf '{"marker":"GPU_TELEMETRY_COMPLETE","hostname":"%s","node_rank":%s,"path":"%s","rc":%s}\n' \
    "$(hostname)" "$NODE_RANK" "$GPU_TELEMETRY" "$rc"
  return "$rc"
}
trap cleanup_gpu_telemetry EXIT

printf '{"marker":"NODE_RUN_START","hostname":"%s","node_rank":%s,"utc":"%s"}\n' \
  "$(hostname)" "$NODE_RANK" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"marker":"GPU_TELEMETRY_START","hostname":"%s","node_rank":%s,"path":"%s","pid":%s}\n' \
  "$(hostname)" "$NODE_RANK" "$GPU_TELEMETRY" "$GPU_TELEMETRY_PID"
"$PYTHON_BIN" "$PROJECT_DIR/tools/prepare_megatron_node.py" \
  --megatron-dir "$MEGATRON_DIR" \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-gpus "$GPUS_PER_NODE" \
  --compile-dataset-helpers

"$PYTHON_BIN" -m torch.distributed.run \
  --nnodes "$NNODES" \
  --nproc-per-node "$GPUS_PER_NODE" \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  "$PROJECT_DIR/tools/distributed_probe.py"

TRAIN_PORT=$((MASTER_PORT + 1))
"$PYTHON_BIN" "$PROJECT_DIR/tools/megatron_smoke.py" launch-node \
  --config "$PROFILE_JSON" \
  --megatron-dir "$MEGATRON_DIR" \
  --run-dir "$RUN_DIR" \
  --data-cache-path "$DATA_CACHE_PATH" \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$TRAIN_PORT" \
  --python "$PYTHON_BIN"

printf '{"marker":"NODE_RUN_COMPLETE","hostname":"%s","node_rank":%s,"utc":"%s"}\n' \
  "$(hostname)" "$NODE_RANK" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

#!/usr/bin/env bash
# Generic SSH fan-out for environments without a scheduler-provided torchrun
# rendezvous. Run this controller as the same unprivileged user on node 0.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 PROFILE_JSON PROJECT_DIR MEGATRON_DIR RUN_DIR" >&2
  echo "required env: HOSTS=host0,host1,...; optional: MASTER_ADDR, MASTER_PORT, PYTHON_BIN" >&2
  exit 2
fi
PROFILE_JSON=$1
PROJECT_DIR=$2
MEGATRON_DIR=$3
RUN_DIR=$4
: "${HOSTS:?HOSTS must be a comma-separated node list}"
PYTHON_BIN=${PYTHON_BIN:-python3}
MASTER_PORT=${MASTER_PORT:-29500}
IFS=',' read -r -a HOST_ARRAY <<< "$HOSTS"
MASTER_ADDR=${MASTER_ADDR:-${HOST_ARRAY[0]}}

safe_path='^[A-Za-z0-9_./:+-]+$'
safe_host='^[A-Za-z0-9_.-]+$'
for value in "$PROFILE_JSON" "$PROJECT_DIR" "$MEGATRON_DIR" "$RUN_DIR" "$PYTHON_BIN"; do
  if [[ ! "$value" =~ $safe_path ]]; then
    echo "unsafe path/token (spaces are unsupported): $value" >&2
    exit 2
  fi
done
for host in "${HOST_ARRAY[@]}"; do
  if [[ ! "$host" =~ $safe_host ]]; then
    echo "unsafe host: $host" >&2
    exit 2
  fi
done

EXPECTED_NNODES=$("$PYTHON_BIN" - "$PROFILE_JSON" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["cluster"]["nnodes"])
PY
)
if [[ ${#HOST_ARRAY[@]} -ne $EXPECTED_NNODES ]]; then
  echo "profile requires $EXPECTED_NNODES nodes, HOSTS has ${#HOST_ARRAY[@]}" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
if [[ -n "${MEGATRON_DATA_CACHE_PATH:-}" ]]; then
  # Replicated node-local cache: every node holds a byte-identical copy at the
  # SAME absolute path. Required because MCore memory-maps these .npy files and
  # mmap over blobfuse has already produced a SIGSEGV inside numpy open_memmap.
  # Used verbatim -- do NOT derive a per-run subdirectory, or the prebuilt
  # entries would be invisible and each rank would rebuild them over FUSE.
  if [[ ! "$MEGATRON_DATA_CACHE_PATH" =~ $safe_path ]]; then
    echo "unsafe MEGATRON_DATA_CACHE_PATH: $MEGATRON_DATA_CACHE_PATH" >&2
    exit 2
  fi
  DATA_CACHE_PATH="$MEGATRON_DATA_CACHE_PATH"
  CACHE_MODE=replicated
elif (( EXPECTED_NNODES > 1 )); then
  : "${MEGATRON_SHARED_CACHE_ROOT:?set MEGATRON_SHARED_CACHE_ROOT to a cross-node shared directory}"
  CACHE_KEY=$(printf '%s' "$RUN_DIR" | sha256sum | awk '{print $1}')
  DATA_CACHE_PATH="${MEGATRON_SHARED_CACHE_ROOT%/}/${CACHE_KEY}"
  CACHE_MODE=shared
else
  DATA_CACHE_PATH="$RUN_DIR/data-cache"
  CACHE_MODE=local
fi
mkdir -p "$DATA_CACHE_PATH"
printf '{"marker":"DATA_CACHE_PATH","nnodes":%d,"path":"%s","mode":"%s"}\n' \
  "$EXPECTED_NNODES" "$DATA_CACHE_PATH" "$CACHE_MODE"
cp "$PROFILE_JSON" "$RUN_DIR/profile.json"
printf '{"marker":"CONTROLLER_START","hosts":"%s","master_addr":"%s","master_port":%s,"utc":"%s"}\n' \
  "$HOSTS" "$MASTER_ADDR" "$MASTER_PORT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

pids=()
logs=()
for index in "${!HOST_ARRAY[@]}"; do
  host=${HOST_ARRAY[$index]}
  log="$RUN_DIR/node-${index}-${host}.log"
  logs+=("$log")
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$host" \
    env PYTHON_BIN="$PYTHON_BIN" \
    bash "$PROJECT_DIR/tools/run_megatron_smoke_node.sh" \
      "$PROFILE_JSON" "$PROJECT_DIR" "$MEGATRON_DIR" "$RUN_DIR" \
      "$index" "$MASTER_ADDR" "$MASTER_PORT" "$DATA_CACHE_PATH" \
    >"$log" 2>&1 &
  pids+=("$!")
  printf '{"marker":"NODE_SSH_STARTED","host":"%s","node_rank":%s,"pid":%s,"log":"%s"}\n' \
    "$host" "$index" "${pids[-1]}" "$log"
done

terminate_children() {
  for pid in "${pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    rc=0
  else
    rc=$?
    status=$rc
  fi
  printf '{"marker":"NODE_SSH_EXIT","host":"%s","node_rank":%s,"rc":%s,"log":"%s"}\n' \
    "${HOST_ARRAY[$index]}" "$index" "$rc" "${logs[$index]}"
  telemetry="$RUN_DIR/gpu-node-$index.csv"
  telemetry_tmp="$RUN_DIR/.gpu-node-$index.csv.tmp"
  if ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
      "${HOST_ARRAY[$index]}" cat "$telemetry" >"$telemetry_tmp"; then
    mv "$telemetry_tmp" "$telemetry"
    printf '{"marker":"GPU_TELEMETRY_COLLECTED","host":"%s","node_rank":%s,"path":"%s"}\n' \
      "${HOST_ARRAY[$index]}" "$index" "$telemetry"
  else
    telemetry_rc=$?
    rm -f "$telemetry_tmp"
    status=$telemetry_rc
    printf '{"marker":"GPU_TELEMETRY_COLLECTION_FAILED","host":"%s","node_rank":%s,"rc":%s,"path":"%s"}\n' \
      "${HOST_ARRAY[$index]}" "$index" "$telemetry_rc" "$telemetry" >&2
  fi
done
printf '{"marker":"CONTROLLER_COMPLETE","rc":%s,"utc":"%s"}\n' \
  "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$status"

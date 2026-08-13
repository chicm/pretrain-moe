#!/usr/bin/env bash
# Gate 3 multi-node controller for the R-Full MoE training path.
#
# Runs on node rank 0 as the same unprivileged user that owns the deployment and
# fans out over SSH to the remaining nodes. Every node executes the identical
# node runner (tools/run_rfull_gate2_node.sh); only RFULL_NODE_RANK differs.
#
# Contract:
#   * code lives on node-local storage under an identical path on every node
#   * dataset cache and checkpoints live on a genuinely shared filesystem
#   * per-node run dirs are created by the node runner, never by this controller
#   * a non-zero rc from any node fails the whole run
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "usage: $0 CONFIG RUN_ROOT PROJECT_DIR MEGATRON_DIR MASTER_PORT DATA_CACHE_PATH [launch-node options...]" >&2
  echo "required env: RFULL_HOSTS=host0,host1,...  (host0 must be this node)" >&2
  echo "optional env: RFULL_MASTER_ADDR (default: first host), RFULL_PYTHON" >&2
  exit 2
fi

CONFIG=$(realpath "$1")
RUN_ROOT=$2
PROJECT_DIR=$(realpath "$3")
MEGATRON_DIR=$(realpath "$4")
MASTER_PORT=$5
DATA_CACHE_PATH=$6
shift 6

: "${RFULL_HOSTS:?RFULL_HOSTS must be a comma-separated node list}"
PYTHON=${RFULL_PYTHON:-/opt/venv/bin/python}
IFS=',' read -r -a HOST_ARRAY <<< "$RFULL_HOSTS"
MASTER_ADDR=${RFULL_MASTER_ADDR:-${HOST_ARRAY[0]}}

safe_path='^[A-Za-z0-9_./:+-]+$'
safe_host='^[A-Za-z0-9_.-]+$'
for value in "$CONFIG" "$RUN_ROOT" "$PROJECT_DIR" "$MEGATRON_DIR" "$DATA_CACHE_PATH" "$PYTHON"; do
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
if [[ "$MASTER_ADDR" == "127.0.0.1" || "$MASTER_ADDR" == "localhost" ]]; then
  echo "REFUSE_LOOPBACK_MASTER: multi-node rendezvous needs a routable address" >&2
  exit 11
fi

EXPECTED_NNODES=$("$PYTHON" - "$CONFIG" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["cluster"]["nnodes"])
PY
)
if [[ ${#HOST_ARRAY[@]} -ne $EXPECTED_NNODES ]]; then
  echo "config requires $EXPECTED_NNODES nodes, RFULL_HOSTS has ${#HOST_ARRAY[@]}" >&2
  exit 2
fi
if (( EXPECTED_NNODES < 2 )); then
  echo "Gate 3 controller requires nnodes >= 2; use the node runner directly" >&2
  exit 2
fi

# The dataset cache must be visible to every rank. Prove it is shared before any
# GPU work starts: node 0 writes a probe, every other node must read it back.
mkdir -p "$DATA_CACHE_PATH"
PROBE="$DATA_CACHE_PATH/.shared-probe-$$"
date -u +%Y-%m-%dT%H:%M:%SZ > "$PROBE"
PROBE_SHA=$(sha256sum "$PROBE" | awk '{print $1}')

CONTROLLER_LOG="$RUN_ROOT/controller.log"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$CONTROLLER_LOG") 2>&1

printf '{"marker":"CONTROLLER_START","utc":"%s","host":"%s","pid":%s,"hosts":"%s","master_addr":"%s","master_port":%s,"nnodes":%s}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" "$$" "$RFULL_HOSTS" "$MASTER_ADDR" "$MASTER_PORT" "$EXPECTED_NNODES"
printf 'RFULL_CONFIG_SHA256=%s\n' "$(sha256sum "$CONFIG" | awk '{print $1}')"

for index in "${!HOST_ARRAY[@]}"; do
  host=${HOST_ARRAY[$index]}
  if (( index == 0 )); then
    observed=$(sha256sum "$PROBE" | awk '{print $1}')
  else
    observed=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$host" \
      "sha256sum '$PROBE' 2>/dev/null | awk '{print \$1}'" || true)
  fi
  if [[ "$observed" != "$PROBE_SHA" ]]; then
    printf '{"marker":"SHARED_CACHE_FAIL","host":"%s","node_rank":%s,"expected":"%s","observed":"%s"}\n' \
      "$host" "$index" "$PROBE_SHA" "$observed"
    rm -f "$PROBE"
    exit 12
  fi
  printf '{"marker":"SHARED_CACHE_OK","host":"%s","node_rank":%s,"sha256":"%s"}\n' \
    "$host" "$index" "$PROBE_SHA"
done
rm -f "$PROBE"

# The deployment must be byte-identical on every node.
DEPLOY_SHA_LOCAL=$(cd "$PROJECT_DIR" && find tools rfull_moe configs -type f -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
for index in "${!HOST_ARRAY[@]}"; do
  host=${HOST_ARRAY[$index]}
  if (( index == 0 )); then
    observed=$DEPLOY_SHA_LOCAL
  else
    observed=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$host" \
      "cd '$PROJECT_DIR' && find tools rfull_moe configs -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print \$1}'" || true)
  fi
  if [[ "$observed" != "$DEPLOY_SHA_LOCAL" ]]; then
    printf '{"marker":"DEPLOYMENT_MISMATCH","host":"%s","node_rank":%s,"expected":"%s","observed":"%s"}\n' \
      "$host" "$index" "$DEPLOY_SHA_LOCAL" "$observed"
    exit 13
  fi
  printf '{"marker":"DEPLOYMENT_OK","host":"%s","node_rank":%s,"sha256":"%s"}\n' \
    "$host" "$index" "$DEPLOY_SHA_LOCAL"
done

pids=()
logs=()
node_dirs=()
for index in "${!HOST_ARRAY[@]}"; do
  host=${HOST_ARRAY[$index]}
  node_dir="$RUN_ROOT/node-$index"
  ssh_log="$RUN_ROOT/node-$index-$host.ssh.log"
  node_dirs+=("$node_dir")
  logs+=("$ssh_log")
  if (( index == 0 )); then
    env RFULL_MASTER_ADDR="$MASTER_ADDR" RFULL_NODE_RANK="$index" RFULL_PYTHON="$PYTHON" \
      bash "$PROJECT_DIR/tools/run_rfull_gate2_node.sh" \
        "$CONFIG" "$node_dir" "$PROJECT_DIR" "$MEGATRON_DIR" "$MASTER_PORT" "$DATA_CACHE_PATH" \
        "$@" >"$ssh_log" 2>&1 &
  else
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$host" \
      env RFULL_MASTER_ADDR="$MASTER_ADDR" RFULL_NODE_RANK="$index" RFULL_PYTHON="$PYTHON" \
      bash "$PROJECT_DIR/tools/run_rfull_gate2_node.sh" \
        "$CONFIG" "$node_dir" "$PROJECT_DIR" "$MEGATRON_DIR" "$MASTER_PORT" "$DATA_CACHE_PATH" \
        "$@" >"$ssh_log" 2>&1 &
  fi
  pids+=("$!")
  printf '{"marker":"NODE_LAUNCHED","host":"%s","node_rank":%s,"pid":%s,"run_dir":"%s","log":"%s"}\n' \
    "$host" "$index" "${pids[-1]}" "$node_dir" "$ssh_log"
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
  printf '{"marker":"NODE_EXIT","host":"%s","node_rank":%s,"rc":%s,"run_dir":"%s"}\n' \
    "${HOST_ARRAY[$index]}" "$index" "$rc" "${node_dirs[$index]}"
done

# Pull remote node evidence back to node 0 so acceptance can read one tree.
for index in "${!HOST_ARRAY[@]}"; do
  (( index == 0 )) && continue
  host=${HOST_ARRAY[$index]}
  dest="$RUN_ROOT/collected/node-$index"
  mkdir -p "$dest"
  if scp -q -o BatchMode=yes -o StrictHostKeyChecking=no -r "$host:${node_dirs[$index]}/." "$dest/"; then
    printf '{"marker":"NODE_EVIDENCE_COLLECTED","host":"%s","node_rank":%s,"dest":"%s","files":%s}\n' \
      "$host" "$index" "$dest" "$(find "$dest" -type f | wc -l)"
  else
    printf '{"marker":"NODE_EVIDENCE_COLLECT_FAILED","host":"%s","node_rank":%s,"dest":"%s"}\n' \
      "$host" "$index" "$dest"
    status=14
  fi
done

printf '{"marker":"CONTROLLER_COMPLETE","utc":"%s","pid":%s,"rc":%s}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$status"
exit "$status"

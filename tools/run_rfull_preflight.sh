#!/usr/bin/env bash
# R-Full multi-node collective preflight.
#
# Proves rendezvous and every collective the MoE trainer depends on, at the
# exact topology of the run that follows, BEFORE spending GPU-hours on a
# trainer that would fail the same way but far more slowly.
#
# Scales to any node count: nodes are read from a hostfile, so the same tool
# covers the 2, 3, 5 and 15 node steps.
#
# Bug fixed relative to the throwaway v1: launching background jobs inside a
# command substitution puts them in a subshell, so the outer shell cannot wait
# on them and reports a false FAIL (rc=127) even when every rank passed.  Here
# PIDs are collected in the current shell and waited on directly.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_rfull_preflight.sh --hostfile F --out DIR --master-addr A --master-port P
                              [--gpus-per-node N] [--python PATH] [--timeout SEC]
EOF
  exit 2
}

GPUS_PER_NODE=8
PYTHON=/opt/venv/bin/python
TIMEOUT=900
HOSTFILE= ; OUT= ; MASTER_ADDR= ; MASTER_PORT=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hostfile) HOSTFILE=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --master-addr) MASTER_ADDR=$2; shift 2;;
    --master-port) MASTER_PORT=$2; shift 2;;
    --gpus-per-node) GPUS_PER_NODE=$2; shift 2;;
    --python) PYTHON=$2; shift 2;;
    --timeout) TIMEOUT=$2; shift 2;;
    *) echo "unknown argument: $1" >&2; usage;;
  esac
done
[[ -n $HOSTFILE && -n $OUT && -n $MASTER_ADDR && -n $MASTER_PORT ]] || usage
[[ -f $HOSTFILE ]] || { echo "no such hostfile: $HOSTFILE" >&2; exit 2; }
# Immutable evidence: never write into an existing preflight directory.
[[ -e $OUT ]] && { echo "REFUSE_OVERWRITE $OUT" >&2; exit 4; }

mapfile -t HOSTS < <(grep -vE '^\s*(#|$)' "$HOSTFILE")
NNODES=${#HOSTS[@]}
WORLD=$(( NNODES * GPUS_PER_NODE ))
mkdir -p "$OUT"

echo "PREFLIGHT_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "PREFLIGHT_NNODES=$NNODES"
echo "PREFLIGHT_WORLD=$WORLD"
echo "PREFLIGHT_OUT=$OUT"

cp "$(dirname "$0")/rfull_preflight_probe.py" "$OUT/probe.py"
echo "PROBE_SHA256=$(sha256sum "$OUT/probe.py" | cut -d' ' -f1)"

pids=()
for i in "${!HOSTS[@]}"; do
  host=${HOSTS[$i]}
  cmd="cd $OUT && NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
       $PYTHON -m torch.distributed.run \
         --nnodes=$NNODES --nproc-per-node=$GPUS_PER_NODE --node-rank=$i \
         --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
         $OUT/probe.py > $OUT/node-$i.log 2>&1"
  if [[ $i -eq 0 ]]; then
    bash -c "$cmd" & pids+=($!)
  else
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=30 "$host" "$cmd" & pids+=($!)
  fi
done

# Wait in THIS shell -- the v1 bug was waiting outside the subshell that owned
# these PIDs, which always failed regardless of what the ranks actually did.
rc_total=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "NODE_${i}_RC=0"
  else
    rc=$?
    echo "NODE_${i}_RC=$rc"
    rc_total=$(( rc_total + 1 ))
  fi
done
echo "NODES_NONZERO_RC=$rc_total"
echo "PREFLIGHT_RUN_DONE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0

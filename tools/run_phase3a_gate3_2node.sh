#!/usr/bin/env bash
# Phase 3a: 2-node EP8 MoE (gate3). First cross-node MoE all-to-all.
#
# IMPORTANT: MoE configs must go through tools/rfull_gate2.py (via
# run_rfull_gate2_node.sh), NOT tools/launch_megatron_multinode.sh. The latter
# drives tools/megatron_smoke.py, which is the DENSE path and validates a dense
# schema -- feeding it an R-Full config dies with KeyError: 'ffn_hidden_size'
# because MoE configs carry dense_ffn_hidden_size/expert_ffn_hidden_size instead.
#
# run_rfull_gate2_node.sh is already multi-node aware: it takes RFULL_MASTER_ADDR
# and RFULL_NODE_RANK and refuses a loopback master for non-zero ranks. So this
# script just fans it out over ssh, one invocation per node, and collects rcs.
#
# MASTER_ADDR is pinned to node-0 so that changing the host set can never silently
# relocate the rendezvous master.
set -uo pipefail

LOCAL=${LOCAL:-/scratch/rfull}
PROJECT_DIR=${PROJECT_DIR:-$LOCAL/src}
MEGATRON_DIR=${MEGATRON_DIR:-$LOCAL/megatron-lm}
PYTHON_BIN=${RFULL_PYTHON:-/opt/venv/bin/python}
PROFILE=${PROFILE:-$PROJECT_DIR/configs/gate3/rfull_ep8_full_geometry_2node.json}
STAMP=$(date -u +%Y%m%d-%H%M%S)
RUN_TAG=${RUN_TAG:-p3a-gate3-$STAMP}
RUN_ROOT=${RUN_ROOT:-$LOCAL/runs}
RUN_DIR="$RUN_ROOT/$RUN_TAG"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15"

NNODES=$("$PYTHON_BIN" -c "import json,sys;print(json.load(open(sys.argv[1]))['cluster']['nnodes'])" "$PROFILE")
MASTER_ADDR=${MASTER_ADDR:-node-0}
MASTER_PORT=${MASTER_PORT:-$(( 29900 + (RANDOM % 90) ))}

# mock data: rank 0 builds the indices and every other rank opens them by path,
# so this must be a cross-node visible location.
CACHE_ROOT=${CACHE_ROOT:-/scratch/workspaceblobstore/chec/pretrain-moe/smoke-cache/$RUN_TAG}

echo "profile=$PROFILE"
echo "nnodes=$NNODES master=$MASTER_ADDR:$MASTER_PORT"
echo "run_dir=$RUN_DIR (per node)"
mkdir -p "$RUN_ROOT"

pids=()
for (( r=0; r<NNODES; r++ )); do
  h="node-$r"
  ssh $SSH_OPTS "$h" \
    "RFULL_PYTHON='$PYTHON_BIN' RFULL_MASTER_ADDR='$MASTER_ADDR' RFULL_NODE_RANK=$r \
     bash '$PROJECT_DIR/tools/run_rfull_gate2_node.sh' \
       '$PROFILE' '$RUN_DIR' '$PROJECT_DIR' '$MEGATRON_DIR' \
       '$MASTER_PORT' '$CACHE_ROOT'" \
    > "$RUN_ROOT/$RUN_TAG.$h.log" 2>&1 &
  pids+=($!)
  echo "  launched $h (node_rank=$r)"
done

rc_total=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "  node-$i rc=0"
  else
    rc=$?
    echo "  node-$i rc=$rc"
    rc_total=1
  fi
done

echo
for (( r=0; r<NNODES; r++ )); do
  echo "--- node-$r tail ---"
  tail -5 "$RUN_ROOT/$RUN_TAG.node-$r.log" 2>/dev/null | sed 's/^/    /'
done

echo
if (( rc_total == 0 )); then echo "PHASE3A PASS"; else echo "PHASE3A FAIL"; fi
exit $rc_total

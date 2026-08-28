#!/usr/bin/env bash
# Phase 3a: 2-node EP8 MoE (gate3). Validates MoE across a real network hop.
#
# This is the first time EP=8 all-to-all traffic crosses nodes, so it isolates
# multi-node MoE transport from the single-node MoE correctness already shown in
# Phase 2. Production geometry (48 layers / 96 experts / top-6), short seq, mock
# data -- the point is transport, not tokens.
#
# MASTER_ADDR is pinned to node-0: the launcher would otherwise default it to the
# first host in the list, which silently couples "which hosts" to "who is the
# rendezvous master" and manufactures fake bad-node failures.
set -uo pipefail

LOCAL=${LOCAL:-/scratch/rfull}
PROJECT_DIR=${PROJECT_DIR:-$LOCAL/src}
MEGATRON_DIR=${MEGATRON_DIR:-$LOCAL/megatron-lm}
PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
PROFILE=${PROFILE:-$PROJECT_DIR/configs/gate3/rfull_ep8_full_geometry_2node.json}
RUN_TAG=${RUN_TAG:-p3a-gate3-$(date -u +%Y%m%d-%H%M%S)}
RUN_ROOT=${RUN_ROOT:-$LOCAL/runs}
RUN_DIR="$RUN_ROOT/$RUN_TAG"

NNODES=$("$PYTHON_BIN" -c "import json,sys;print(json.load(open(sys.argv[1]))['cluster']['nnodes'])" "$PROFILE")
HOSTS=$(seq -f 'node-%g' 0 $((NNODES - 1)) | paste -sd,)

export HOSTS
export MASTER_ADDR=node-0
export MASTER_PORT=$(( 29900 + (RANDOM % 90) ))
export PYTHON_BIN
# mock data -> indices must be visible from every node (rank 0 builds, others open
# by path). Small, read once at startup, so the shared mount is safe here.
export MEGATRON_SHARED_CACHE_ROOT=${MEGATRON_SHARED_CACHE_ROOT:-/scratch/workspaceblobstore/chec/pretrain-moe/smoke-cache}

mkdir -p "$RUN_DIR"
echo "profile=$PROFILE"
echo "nnodes=$NNODES hosts=$HOSTS master=$MASTER_ADDR:$MASTER_PORT"
echo "run_dir=$RUN_DIR"

bash "$PROJECT_DIR/tools/launch_megatron_multinode.sh" \
  "$PROFILE" "$PROJECT_DIR" "$MEGATRON_DIR" "$RUN_DIR"
rc=$?
echo "PHASE3A rc=$rc"
exit $rc

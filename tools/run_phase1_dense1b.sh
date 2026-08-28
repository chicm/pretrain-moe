#!/usr/bin/env bash
# Phase 1: dense-1B multi-node smoke on Megatron (framework validation).
#
# Purpose: prove the multi-node Megatron training path is sound BEFORE any MoE is
# involved. Dense only, mock data, 8 iterations.
#
# Discipline encoded here (all learned the hard way):
#  * MASTER_ADDR is pinned to node-0 explicitly. The launcher defaults it to
#    HOST_ARRAY[0], so changing the host list would silently move the rendezvous
#    master and produce a fake "bad node" failure (all ranks socket-timeout,
#    ranks=0, no fingerprints). Master is NOT a free variable.
#  * MASTER_PORT is rotated per run to avoid TIME_WAIT / stale rendezvous.
#  * Everything runs from node-local ext4 (/scratch/rfull), never blobfuse:
#    mmap'd index files and dlopen'd helpers .so fault uncatchably over FUSE.
#  * Run dir is timestamped so runs never overwrite each other.
set -euo pipefail

LOCAL=${LOCAL:-/scratch/rfull}
PROJECT_DIR=${PROJECT_DIR:-$LOCAL/src}
MEGATRON_DIR=${MEGATRON_DIR:-$LOCAL/megatron-lm}
PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
PROFILE=${PROFILE:-$PROJECT_DIR/configs/smoke/dense_1b.json}
RUN_TAG=${RUN_TAG:-p1-dense1b-$(date -u +%Y%m%d-%H%M%S)}
RUN_ROOT=${RUN_ROOT:-$LOCAL/runs}
RUN_DIR="$RUN_ROOT/$RUN_TAG"

# nnodes comes from the profile; take that many hosts starting at node-0.
NNODES=$("$PYTHON_BIN" -c "import json,sys;print(json.load(open(sys.argv[1]))['cluster']['nnodes'])" "$PROFILE")
HOSTS=$(seq -f 'node-%g' 0 $((NNODES - 1)) | paste -sd,)

export HOSTS
export MASTER_ADDR=node-0                        # pinned; never inferred
export MASTER_PORT=$(( 29600 + (RANDOM % 300) )) # rotate to dodge TIME_WAIT
export PYTHON_BIN
# Dataset index cache.
#
# Megatron builds the {document,sample,shuffle}_index.npy files on rank 0 ONLY;
# every other rank waits on a barrier and then opens those files BY PATH. If the
# path is node-local, the non-builder nodes die with FileNotFoundError while
# node-0 sits in the barrier -- node-0 looks like the hang but is only the victim.
#
# Phase 1 uses mock data, whose indices are small (~1 MB) and are read once at
# startup rather than mmap'd hot, so the shared blobfuse dir is safe and correct
# here. For real-data phases the cache is instead pre-built once with
# tools/prebuild_rfull_data_cache.py and replicated to node-local ext4 via
# MEGATRON_DATA_CACHE_PATH, because mmap over FUSE faults with an uncatchable
# SIGSEGV.
unset MEGATRON_DATA_CACHE_PATH
export MEGATRON_SHARED_CACHE_ROOT=${MEGATRON_SHARED_CACHE_ROOT:-/scratch/workspaceblobstore/chec/pretrain-moe/smoke-cache}

mkdir -p "$RUN_DIR"
echo "profile=$PROFILE"
echo "nnodes=$NNODES hosts=$HOSTS master=$MASTER_ADDR:$MASTER_PORT"
echo "run_dir=$RUN_DIR"

bash "$PROJECT_DIR/tools/launch_megatron_multinode.sh" \
  "$PROFILE" "$PROJECT_DIR" "$MEGATRON_DIR" "$RUN_DIR"

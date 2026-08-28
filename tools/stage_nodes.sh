#!/usr/bin/env bash
# Stage the R-Full deployment onto node-local disk on every node.
#
# WHY node-local: the blobfuse2 mount (/scratch/workspaceblobstore/...) cannot be
# used for anything that is mmap'd or dlopen'd. numpy.load(mmap_mode='r') and the
# compiled dataset helpers (.so) both fault over FUSE, and the kernel delivers
# SIGBUS/SIGSEGV directly to the faulting thread -- no errno, no Python exception,
# no retry. That is the root cause of the historical "MoE a2a wedge".
# /scratch is local ext4 on each node; that is where code + helpers must live.
set -euo pipefail

BLOB_SRC="${BLOB_SRC:-/scratch/workspaceblobstore/chec/pretrain-moe}"
LOCAL="${LOCAL:-/scratch/rfull}"
HOSTFILE="${HOSTFILE:-$HOME/hostfile}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15"

if [[ ! -f "$HOSTFILE" ]]; then
  echo "FAIL: hostfile not found: $HOSTFILE" >&2
  exit 1
fi

mapfile -t HOSTS < <(awk '{print $1}' "$HOSTFILE" | sed '/^$/d')
echo "stage: ${#HOSTS[@]} hosts -> $LOCAL"

stage_one() {
  local h="$1"
  ssh $SSH_OPTS "$h" bash -s <<EOF 2>&1 | sed "s/^/[\$h] /"
set -euo pipefail
mkdir -p "$LOCAL"
rm -rf "$LOCAL/src" "$LOCAL/megatron-lm"
cp -r "$BLOB_SRC/src-megatron" "$LOCAL/src"
cp -r "$BLOB_SRC/megatron-lm"  "$LOCAL/megatron-lm"
test -d "$LOCAL/src/rfull_moe"
test -d "$LOCAL/megatron-lm/megatron/core"
echo "\$(hostname) staged src=\$(ls "$LOCAL/src" | wc -l) mlm=\$(ls "$LOCAL/megatron-lm" | wc -l)"
EOF
}

pids=()
for h in "${HOSTS[@]}"; do
  stage_one "$h" &
  pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done

if [[ $rc -ne 0 ]]; then
  echo "FAIL: at least one host failed to stage" >&2
  exit 1
fi
echo "STAGE OK"

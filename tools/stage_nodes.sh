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
  ssh $SSH_OPTS "$h" bash -s <<EOF 2>&1 | sed "s/^/[$h] /"
set -euo pipefail
mkdir -p "$LOCAL"

# src is small and changes every push -> always replace.
rm -rf "$LOCAL/src"
cp -r "$BLOB_SRC/src-megatron" "$LOCAL/src"

# megatron-lm is pinned to a single upstream commit and is expensive to copy.
# Re-copying it also destroys the compiled dataset helpers (.so), which then have
# to be rebuilt on all 15 nodes. Only copy it when it is missing or incomplete.
if [[ -f "$LOCAL/megatron-lm/megatron/core/__init__.py" ]]; then
  mlm_state=kept
else
  rm -rf "$LOCAL/megatron-lm"
  cp -r "$BLOB_SRC/megatron-lm" "$LOCAL/megatron-lm"
  mlm_state=copied
fi

test -d "$LOCAL/src/rfull_moe"
test -d "$LOCAL/megatron-lm/megatron/core"
# NOTE: do not use \`ls *.so | wc -l\` here. Under \`set -euo pipefail\`, a glob that
# matches nothing makes ls exit 2, pipefail propagates that through the pipe, and
# set -e kills this script one line before the success echo -- staging looks like
# it failed on a fresh cluster where no helpers have been built yet. find returns
# 0 for "no matches", so it is safe.
nso=\$(find "$LOCAL/megatron-lm/megatron/core/datasets" -maxdepth 1 -name '*.so' 2>/dev/null | wc -l)
echo "\$(hostname) staged src=\$(ls "$LOCAL/src" | wc -l) mlm=\$(ls "$LOCAL/megatron-lm" | wc -l) mlm_state=\$mlm_state so=\$nso"
EOF
}

pids=()
for h in "${HOSTS[@]}"; do
  stage_one "$h" &
  pids+=($!)
done

rc=0
failed=()
for i in "${!pids[@]}"; do
  # Report WHICH host failed. An aggregate "at least one host failed" hides the
  # 14-good/1-bad asymmetry that is the usual shape of cluster problems.
  wait "${pids[$i]}" || { rc=1; failed+=("${HOSTS[$i]}"); }
done

if [[ $rc -ne 0 ]]; then
  echo "FAIL: staging failed on: ${failed[*]}" >&2
  exit 1
fi
echo "STAGE OK"

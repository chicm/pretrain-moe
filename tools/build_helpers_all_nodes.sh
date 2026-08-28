#!/usr/bin/env bash
# Build the Megatron dataset helpers (.so) on every node, on node-local disk.
#
# Two hard-won constraints are encoded here:
#  1. The helpers .so must live on LOCAL disk (ext4), never on blobfuse. dlopen +
#     mmap over FUSE faults with SIGSEGV/SIGBUS that Python cannot catch.
#  2. Node environments can be ASYMMETRIC. We previously lost a 15-node launch
#     because node-0's datasets dir was root-owned with no .so while nodes 1-14
#     were fine. So: build on EVERY node and print ONE LINE PER NODE. Never trust
#     an aggregate count -- it hides "14 good + 1 broken".
set -euo pipefail

LOCAL="${LOCAL:-/scratch/rfull}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
HOSTFILE="${HOSTFILE:-$HOME/hostfile}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15"

mapfile -t HOSTS < <(awk '{print $1}' "$HOSTFILE" | sed '/^$/d')
echo "helpers: ${#HOSTS[@]} hosts"

build_one() {
  local h="$1"
  ssh $SSH_OPTS "$h" bash -s <<EOF 2>&1 | tail -1
set -uo pipefail
D="$LOCAL/megatron-lm/megatron/core/datasets"
cd "\$D" || { echo "\$(hostname) FAIL no-datasets-dir"; exit 1; }
if [[ ! -w "\$D" ]]; then echo "\$(hostname) FAIL not-writable owner=\$(stat -c %U "\$D")"; exit 1; fi
if ls helpers*.so >/dev/null 2>&1; then
  echo "\$(hostname) OK already n=\$(ls helpers*.so | wc -l)"
  exit 0
fi
make >/dev/null 2>&1 || true
if ls helpers*.so >/dev/null 2>&1; then
  echo "\$(hostname) OK built n=\$(ls helpers*.so | wc -l)"
else
  echo "\$(hostname) FAIL no-so-after-make"
  exit 1
fi
EOF
}

tmp="$(mktemp -d)"
for h in "${HOSTS[@]}"; do
  build_one "$h" > "$tmp/$h.out" 2>&1 &
done
wait

rc=0
for h in "${HOSTS[@]}"; do
  line="$(cat "$tmp/$h.out" 2>/dev/null || echo "$h FAIL no-output")"
  echo "$line"
  case "$line" in *" OK "*) ;; *) rc=1 ;; esac
done
rm -rf "$tmp"

if [[ $rc -ne 0 ]]; then
  echo "FAIL: helpers missing on at least one node (see per-node lines above)" >&2
  exit 1
fi
echo "HELPERS OK on ${#HOSTS[@]} nodes"

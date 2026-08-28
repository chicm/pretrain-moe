#!/usr/bin/env bash
# Kill any leftover trainers on every node and PROVE the GPUs are free.
#
# Two traps encoded here, both previously paid for:
#  1. pgrep -f <pattern> MATCHES ITS OWN shell, because the pattern appears in
#     the command line of the shell running it. A naive check therefore reports
#     "1 trainer" forever. We exclude our own pid and our parent's.
#  2. Killing the controller leaves ORPHAN trainers holding GPU memory. Process
#     count alone is not proof; we also check VRAM. Cleanup is only complete when
#     every node reports trainers=0 AND vram_busy=0.
#
# Note the hostfile is "node-N slots=8", so hosts must be taken from field 1.
set -uo pipefail

HOSTFILE="${HOSTFILE:-$HOME/hostfile}"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15"
PATTERN='megatron_rocm_entrypoint|pretrain_gpt|torchrun|run_phase[0-9]'

mapfile -t HOSTS < <(awk '{print $1}' "$HOSTFILE" | sed '/^$/d')

for h in "${HOSTS[@]}"; do
  ssh $SSH_OPTS "$h" "pkill -9 -f 'megatron_rocm_entrypoint' ; pkill -9 -f 'pretrain_gpt' ; pkill -9 -f 'torchrun' ; true" >/dev/null 2>&1 &
done
wait
sleep 5

rc=0
for h in "${HOSTS[@]}"; do
  line=$(ssh $SSH_OPTS "$h" bash -s <<EOF 2>/dev/null | tail -1
self=\$\$
# -a prints the command line so we can drop our own shell; never trust a bare count.
n=\$(pgrep -af '$PATTERN' 2>/dev/null | awk -v s="\$self" '\$1 != s' | grep -vc 'bash -s' || true)
if command -v rocm-smi >/dev/null 2>&1; then
  busy=\$(rocm-smi --showmemuse 2>/dev/null | awk '/GPU memory use/ {if (\$NF+0 > 1) c++} END {print c+0}')
else
  busy=NA
fi
echo "\$(hostname) trainers=\$n vram_busy=\$busy"
EOF
)
  echo "${line:-$h NO_OUTPUT}"
  case "$line" in
    *"trainers=0 vram_busy=0"*) ;;
    *) rc=1 ;;
  esac
done

if (( rc != 0 )); then
  echo "WARN: some nodes still report trainers or busy VRAM (see above)" >&2
  exit 1
fi
echo "CLEAN: all ${#HOSTS[@]} nodes idle"

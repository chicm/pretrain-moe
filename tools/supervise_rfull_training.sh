#!/usr/bin/env bash
# PRODUCTION SUPERVISOR: crash-resilient long-run driver.
#
# WHY THIS EXISTS (independent of root cause):
#   A low-rate random per-rank SIGSEGV is present. Evidence so far:
#     8 nodes /  64 ranks: PASS, PASS, PASS      (~450 s each)
#    12 nodes /  96 ranks: 1 FAIL then 5 PASS    (~590 s each)
#    15 nodes / 120 ranks: repeated FAIL
#   Failure probability rises with rank-count x duration, exactly as a random
#   per-rank fault would.  Production is 256,856 iterations across 120 ranks
#   -- orders of magnitude more exposure than the ~70 rank-hours that already
#   produced crashes.  Therefore ANY residual crash rate WILL hit production,
#   and the run must survive it by resuming from the last checkpoint.
#
#   The current launcher uses static rendezvous with no restarts: one dead
#   rank kills the whole job permanently.  That is the real blocker for a
#   multi-week run, and it is fixable without knowing the root cause.
#
# SAFETY RULES:
#   * assert a clean cluster before every attempt (leaked trainers from a
#     previous attempt silently corrupt the next one)
#   * purge remote processes after every attempt -- local `timeout` does NOT
#     kill remote torchrun/trainers
#   * STOP if an attempt makes no forward progress (iteration did not advance)
#     -- otherwise a deterministic failure becomes an infinite crash loop
#   * never delete checkpoints here; retention is a separate, explicit tool
#
# Usage: _supervise.sh <config.json> <run_root> <ckpt_dir> <max_attempts> [hosts]
set -uo pipefail
SH=/scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/chec/pretrain-moe
D=/scratch/rfull-prod/deploy-v38
MEG=/scratch/mcore_probe
CACHE=/scratch/rfull-cache/production-seq4096

CFG=${1:?config}
RUNROOT=${2:?run_root}
CKPT=${3:?ckpt_dir}
MAXATT=${4:-10}
HOSTS=${5:-node-0,node-1,node-2,node-3,node-4,node-5,node-6,node-7,node-8,node-9,node-10,node-11,node-12,node-13,node-14}
# Hard wall-clock bound per attempt, and grace period granted to peer ranks to
# unwind after a SIGSEGV is first observed. Override via environment.
ATTEMPT_TIMEOUT=${ATTEMPT_TIMEOUT:-7200}
CRASH_GRACE=${CRASH_GRACE:-180}
LOG=$RUNROOT/supervisor.log

export RFULL_CACHE_MODE=replicated
export RFULL_HOSTS=$HOSTS
mkdir -p "$RUNROOT" "$CKPT"

log () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

latest_iter () {   # highest fully-saved iteration, per tracker file
  local t=$CKPT/latest_checkpointed_iteration.txt
  [ -f "$t" ] && tr -cd '0-9' < "$t" || echo 0
}

assert_clean () {
  local bad=0 n
  for i in $(seq 0 14); do
    n=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 node-$i \
        'pgrep -fc "[p]retrain_rfull_moe" 2>/dev/null | head -1' 2>/dev/null | tr -cd '0-9')
    n=${n:-0}
    [ "$((10#$n))" -ne 0 ] && { log "  DIRTY node-$i trainers=$n"; bad=1; }
  done
  return $bad
}

purge () {
  for i in $(seq 0 14); do
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 node-$i '
      ME=$$; PA=$PPID
      for pat in "[p]retrain_rfull_moe" "[t]orch.distributed.run" "[t]orchrun"; do
        for p in $(pgrep -f "$pat" 2>/dev/null); do
          [ "$p" = "$ME" ] || [ "$p" = "$PA" ] || kill -TERM $p 2>/dev/null
        done
      done
      sleep 5
      for pat in "[p]retrain_rfull_moe" "[t]orch.distributed.run" "[t]orchrun"; do
        for p in $(pgrep -f "$pat" 2>/dev/null); do
          [ "$p" = "$ME" ] || [ "$p" = "$PA" ] || kill -KILL $p 2>/dev/null
        done
      done' >/dev/null 2>&1 &
  done
  wait; sleep 5
}

classify () {   # run_dir -> failure class (drives retry decision)
  local R=$1
  grep -rqa 'The client socket has timed out' "$R" 2>/dev/null && { echo RENDEZVOUS; return; }
  grep -rqa 'exitcode: -11'                   "$R" 2>/dev/null && { echo SIGSEGV;    return; }
  grep -rqa 'out of memory'                   "$R" 2>/dev/null && { echo OOM;        return; }
  grep -rqa 'Watchdog caught'                 "$R" 2>/dev/null && { echo COLLECTIVE; return; }
  echo UNKNOWN
}

log "SUPERVISOR start cfg=$(basename $CFG) ckpt=$CKPT max_attempts=$MAXATT"
log "resume point: iteration $(latest_iter)"
prev_iter=-1
for att in $(seq 1 $MAXATT); do
  purge
  if ! assert_clean; then log "attempt $att ABORT: cluster not clean"; break; fi

  before=$(latest_iter)
  RUN=$RUNROOT/attempt-$att
  rm -rf "$RUN" 2>/dev/null
  PORT=$((28000 + RANDOM % 1500))
  log "attempt $att START from_iter=$before port=$PORT"
  s=$(date -u +%s)

  # The controller does NOT return when a rank dies: surviving ranks sit in a
  # 600 s collective watchdog and the per-node SSH wrappers never exit, so the
  # controller blocked for ~59 min in the first burn-in.  Two guards:
  #   (a) hard wall-clock bound via `timeout`
  #   (b) fail-fast: once a SIGSEGV appears, give peers GRACE seconds to
  #       unwind, then tear the attempt down instead of waiting out (a).
  timeout --kill-after=120 "$ATTEMPT_TIMEOUT" \
    bash $D/tools/run_rfull_gate3_controller.sh \
      "$CFG" "$RUN" "$D" "$MEG" "$PORT" "$CACHE" \
      --save-dir "$CKPT" --load-dir "$CKPT" --save-interval 25 \
      >> "$RUN.controller.log" 2>&1 &
  cpid=$!

  crash_at=0
  while kill -0 $cpid 2>/dev/null; do
    sleep 30
    if [ "$crash_at" = "0" ] && grep -rqa 'exitcode: -11' "$RUN" 2>/dev/null; then
      crash_at=$(date -u +%s)
      log "attempt $att: SIGSEGV detected, allowing ${CRASH_GRACE}s for teardown"
    fi
    if [ "$crash_at" != "0" ] && [ $(( $(date -u +%s) - crash_at )) -ge "$CRASH_GRACE" ]; then
      log "attempt $att: tearing down after crash (controller did not exit)"
      kill -TERM $cpid 2>/dev/null; sleep 10; kill -KILL $cpid 2>/dev/null
      break
    fi
  done
  wait $cpid 2>/dev/null
  rc=$?
  el=$(( $(date -u +%s) - s ))
  after=$(latest_iter)
  fp=$(grep -rac RFULL_BATCH_FINGERPRINT "$RUN" 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
  wd=$(grep -rac 'Watchdog caught'       "$RUN" 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')

  if [ "$rc" = "0" ]; then
    log "attempt $att SUCCESS rc=0 elapsed=${el}s iter=$after fp=$fp"
    log "SUPERVISOR done: training completed"
    purge; exit 0
  fi

  cls=$(classify "$RUN")
  log "attempt $att FAILED rc=$rc class=$cls elapsed=${el}s iter=$before->$after fp=$fp wd=$wd"

  # forward-progress guard: never loop forever on a deterministic failure
  if [ "$after" -le "$before" ] && [ "$before" = "$prev_iter" ]; then
    log "SUPERVISOR abort: no forward progress across two attempts (stuck at $after)"
    purge; exit 2
  fi
  prev_iter=$before
done
purge
log "SUPERVISOR exhausted $MAXATT attempts; last iteration=$(latest_iter)"
exit 3

#!/usr/bin/env bash
# Phase 2: single-node EP8 MoE validation.
#
# Runs, in order:
#   1. CPU-only unit tests (config/semantics/parameter-ledger) -- fail fast before
#      burning GPU time.
#   2. rfull_ep8_mini            (4 layers, hidden 512, 96 experts, top-6)
#   3. rfull_ep8_full_geometry   (production geometry, short run)
#
# Single node => EP=8 lives inside one box, so this isolates MoE correctness from
# any multi-node transport concern.
#
# Discipline:
#  * Everything runs from node-local ext4 (/scratch/rfull); blobfuse cannot back
#    mmap'd indices or dlopen'd helpers.
#  * MASTER_PORT is rotated per stage to avoid stale rendezvous state.
#  * Each stage gets its own immutable run dir (the node script refuses to
#    overwrite one, which is the behaviour we want).
set -uo pipefail

LOCAL=${LOCAL:-/scratch/rfull}
PROJECT_DIR=${PROJECT_DIR:-$LOCAL/src}
MEGATRON_DIR=${MEGATRON_DIR:-$LOCAL/megatron-lm}
PYTHON_BIN=${RFULL_PYTHON:-/opt/venv/bin/python}
STAMP=$(date -u +%Y%m%d-%H%M%S)
RUN_ROOT=${RUN_ROOT:-$LOCAL/runs}

export RFULL_PYTHON="$PYTHON_BIN"

echo "=== Phase 2 : single-node EP8 MoE ==="
echo "stamp=$STAMP project=$PROJECT_DIR megatron=$MEGATRON_DIR"
echo

cd "$PROJECT_DIR"
export PYTHONPATH="$MEGATRON_DIR:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "--- stage 0: CPU unit tests ---"
# tests/ is a package (tests/__init__.py). Note that when a test module fails to
# import, unittest prints "Ran 1 test ... FAILED (errors=1)", which looks like a
# real assertion failure but is only an ImportError -- so we echo the tail on
# failure rather than trusting the summary line.
cpu_rc=0
for t in test_rfull_semantics test_rfull_parameter_ledger test_rfull_gate2; do
  echo "  [$t]"
  out=$("$PYTHON_BIN" -m unittest "tests.$t" -v 2>&1)
  if printf '%s\n' "$out" | grep -qE '^OK'; then
    echo "  [$t] PASS  $(printf '%s\n' "$out" | grep -E '^Ran ' | tail -1)"
  else
    echo "  [$t] FAIL"
    printf '%s\n' "$out" | tail -20 | sed 's/^/      /'
    cpu_rc=1
  fi
done
if (( cpu_rc != 0 )); then
  echo "PHASE2 ABORT: CPU unit tests failed; not spending GPU time" >&2
  exit 1
fi
echo

run_stage() {
  local name="$1" cfg="$2" port="$3"
  local run_dir="$RUN_ROOT/p2-$name-$STAMP"
  mkdir -p "$RUN_ROOT"
  echo "--- stage: $name ---"
  echo "  config=$cfg"
  echo "  run_dir=$run_dir"
  # NOTE: run_rfull_gate2_node.sh refuses to reuse an existing RUN_DIR (good), so
  # the driver log must live beside it, not inside it.
  bash "$PROJECT_DIR/tools/run_rfull_gate2_node.sh" \
    "$cfg" "$run_dir" "$PROJECT_DIR" "$MEGATRON_DIR" \
    "$port" "$LOCAL/data-cache/p2-$name-$STAMP" > "$RUN_ROOT/p2-$name-$STAMP.driver.log" 2>&1
  local rc=$?
  echo "  rc=$rc"
  tail -6 "$RUN_ROOT/p2-$name-$STAMP.driver.log" 2>/dev/null | sed 's/^/    /'
  return $rc
}

rc_total=0
run_stage mini "$PROJECT_DIR/configs/gate2/rfull_ep8_mini.json" $((29700 + RANDOM % 100)) || rc_total=1
echo
if (( rc_total == 0 )); then
  run_stage fullgeo "$PROJECT_DIR/configs/gate2/rfull_ep8_full_geometry.json" $((29800 + RANDOM % 100)) || rc_total=1
else
  echo "skipping full_geometry because mini failed"
fi

echo
if (( rc_total == 0 )); then
  echo "PHASE2 PASS"
else
  echo "PHASE2 FAIL"
fi
exit $rc_total

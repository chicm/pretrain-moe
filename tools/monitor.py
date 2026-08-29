"""Monitor a Megatron run and decide PASS / SLOW / WEDGED / DEAD -- with evidence.

Diagnostic rules encoded here (each one cost me real compute to learn):

  1. A timeout is not a hang. "No new log lines" is not evidence either.
     A run is WEDGED only if BOTH hold over a long window:
       (a) no progress on an observable counter (iteration number), and
       (b) ranks are not aligned / GPUs still busy.
     Otherwise it is merely SLOW (first step can take 60-200 s on ROCm because
     of autotune, and autotune can re-trigger later).

  2. In a collective timeout the ranks that SHOUT are the victims.
     The diagnostic signal is in the SILENT ranks. So: count watchdog reports
     and compare against world size; if they differ, immediately list the nodes
     that produced NO watchdog output -- the answer is there.

  3. Classify the failure shape before counting it as a data point:
       all nodes socket-timeout + ranks=0  -> rendezvous/config bug, DISCARD
       exitcode -11 on a few ranks         -> real crash, COUNT
       rc=124 with iterations advancing    -> slow, not a crash
       rc=124 + frozen counter + busy GPUs -> genuine wedge
     A failure whose shape differs from the one under study is not a sample of
     that failure.

  4. Never grep -iE 'error|inf' blindly: "NCCL INFO" matches "inf". Exclude
     NCCL INFO/DEBUG lines before counting anything.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ITER_RE = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)")
LOSS_RE = re.compile(r"lm loss:\s*([0-9.eE+-]+)")
TPS_RE = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)")
ELAPSED_RE = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")
SEGV_RE = re.compile(r"exitcode\s*:?\s*(-?\d+)")
VRAM_PROBE = (
    "rocm-smi --showmeminfo vram 2>/dev/null "
    "| awk '/Used Memory/ {if ($NF+0 > 1000000000) n++} END {print n+0}'"
)

WATCHDOG_RE = re.compile(r"Watchdog caught collective operation timeout|NCCL.*timeout")


def ssh(host: str, cmd: str, timeout: int = 60) -> str:
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
             "-o", "LogLevel=ERROR", host, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return "__SSH_TIMEOUT__"


def read_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text())


def scan(run_dir: Path, meta: dict) -> dict:
    """Collect per-node progress. One row per node -- aggregates hide outliers."""
    st: dict = {"nodes": {}, "max_iter": 0, "trainers": 0, "busy_gpus": 0,
               "agents": 0, "iter_source": None}
    for rank, h in enumerate(meta["hosts"]):
        # NOTE: Megatron prints the per-iteration line with `print_rank_last`,
        # i.e. from the LAST global rank -- which lives on the LAST node, not
        # node-0. Scanning only node0.log shows zero iterations for a perfectly
        # healthy run and looks exactly like a hang. Scan every node.
        log = run_dir / f"node{rank}.log"
        out = ssh(h, (
            f"tail -c 400000 {log} 2>/dev/null | grep -a -E 'iteration +[0-9]+ */' | tail -3;"
            f"echo '---';"
            # Count the torchrun agent separately from the workers. A dead
            # agent shows up here as 41 -> 40, which is invisible in a single
            # total but is the difference between "healthy" and "this node is
            # already gone". The agent's argv contains the training script
            # path, so it must be excluded by its 'distributed.run' marker.
            f"pgrep -c -f '[t]orch[.]distributed[.]run' || true;"
            f"echo '---';"
            f"pgrep -f '[p]retrain_(entry|gpt)[.]py' | while read q; do "
            f"tr '\\0' ' ' < /proc/$q/cmdline 2>/dev/null "
            f"| grep -q distributed.run || echo $q; done | wc -l;"
            f"echo '---';"
            "VRAMPROBE;"
            f"echo '---';"
            f"grep -ac 'Watchdog caught collective' {log} 2>/dev/null || echo 0;"
            f"echo '---';"
            f"grep -a -oE 'exitcode: -?[0-9]+' {log} 2>/dev/null | sort | uniq -c | tr '\n' ' ';"
        ).replace("VRAMPROBE", VRAM_PROBE), timeout=90)
        parts = out.split("---")
        # parts: 0=iter lines, 1=agents, 2=workers, 3=vram, 4=watchdog, 5=exitcodes
        it, loss, tflops, ms = 0, None, None, None
        if parts and parts[0].strip():
            for line in parts[0].strip().splitlines():
                m = ITER_RE.search(line)
                if m:
                    it = max(it, int(m.group(1)))
                    lm = LOSS_RE.search(line)
                    if lm:
                        loss = float(lm.group(1))
                    tm = TPS_RE.search(line)
                    if tm:
                        tflops = float(tm.group(1))
                    em = ELAPSED_RE.search(line)
                    if em:
                        ms = float(em.group(1))

        def _int(i):
            try:
                return int(parts[i].strip().splitlines()[-1])
            except Exception:
                return 0

        agents = _int(1)
        trainers = _int(2)
        busy = _int(3)
        wd = _int(4)
        exits = parts[6].strip() if len(parts) > 6 else ""

        st["nodes"][h] = {
            "iter": it, "loss": loss, "tflops": tflops, "ms": ms,
            "agents": agents, "trainers": trainers, "busy_gpus": busy,
            "watchdogs": wd, "exitcodes": exits,
        }
        st["agents"] = st.get("agents", 0) + agents
        if it > st["max_iter"]:
            st["iter_source"] = h      # which node actually logs progress
        st["max_iter"] = max(st["max_iter"], it)
        st["trainers"] += trainers
        st["busy_gpus"] += busy
    return st


def classify(prev: dict, cur: dict, meta: dict, stalled_s: float) -> tuple[str, list[str]]:
    world = meta["world"]
    notes: list[str] = []
    nodes = cur["nodes"]

    # --- rule 1b: a dead torchrun agent kills its node silently ----------
    # The agent supervises the 8 local workers; if it dies they are orphaned,
    # and if it is the rendezvous agent (node rank 0) the TCPStore dies with
    # it and every other node reports `Broken pipe`. This is not visible in a
    # combined process count, so check it explicitly.
    dead_agents = [h for h, n in nodes.items() if n.get("agents", 0) == 0
                   and n.get("trainers", 0) > 0]
    if dead_agents:
        notes.append(
            f"AGENT DEAD on {', '.join(dead_agents)} -- workers orphaned. "
            f"Never send signals to a pattern matching the training script "
            f"path; the agent's argv contains it.")

    # --- rule 2: silent ranks hold the answer ---------------------------
    wd_total = sum(n["watchdogs"] for n in nodes.values())
    if wd_total:
        loud = [h for h, n in nodes.items() if n["watchdogs"] > 0]
        silent = [h for h, n in nodes.items() if n["watchdogs"] == 0]
        notes.append(f"watchdogs={wd_total} on {len(loud)} nodes")
        if silent:
            notes.append(f"SILENT NODES (look here first): {' '.join(silent)}")

    segv = {h: n["exitcodes"] for h, n in nodes.items() if "-11" in n["exitcodes"]}
    if segv:
        notes.append(f"SIGSEGV(-11) on: {' '.join(segv)} -- these are the culprits")

    # --- rule 3: classify the shape --------------------------------------
    if cur["trainers"] == 0:
        if cur["max_iter"] == 0:
            notes.append("no trainers, zero iterations -> rendezvous/config failure, "
                         "DISCARD as a crash sample")
            return "DEAD_NOSTART", notes
        return "EXITED", notes

    if prev is None:
        return "RUNNING", notes

    advanced = cur["max_iter"] > prev["max_iter"]
    if advanced:
        return "RUNNING", notes

    # counter frozen -- is it slow or wedged?
    if stalled_s < 900:
        notes.append(f"counter frozen {stalled_s:.0f}s but < 900s -> SLOW "
                     f"(ROCm first-step autotune can take 60-200s)")
        return "SLOW", notes

    iters = [n["iter"] for n in nodes.values() if n["trainers"] > 0]
    aligned = len(set(iters)) <= 1
    notes.append(f"counter frozen {stalled_s:.0f}s; per-node iters aligned={aligned}; "
                 f"busy_gpus={cur['busy_gpus']}")
    if cur["busy_gpus"] > 0 and not advanced and stalled_s > 1800:
        return "WEDGED", notes
    return "SLOW", notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--max-wait", type=int, default=3600)
    ap.add_argument("--until-iter", type=int, default=0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    meta = read_meta(run_dir)
    print(f"monitoring {run_dir.name}  world={meta['world']} nodes={len(meta['hosts'])}")

    prev = None
    t0 = time.time()
    last_progress = t0

    while True:
        cur = scan(run_dir, meta)
        now = time.time()
        if prev and cur["max_iter"] > prev["max_iter"]:
            last_progress = now
        state, notes = classify(prev, cur, meta, now - last_progress)

        rows = cur["nodes"]
        sample = next((n for n in rows.values() if n["iter"] == cur["max_iter"]), {})
        print(f"[{now-t0:6.0f}s] {state:12s} iter={cur['max_iter']}/"
              f"{meta.get('train_iters','?')} "
              f"loss={sample.get('loss')} tflops={sample.get('tflops')} "
              f"ms={sample.get('ms')} trainers={cur['trainers']}/{meta['world']} "
              f"busy_gpus={cur['busy_gpus']}")
        for note in notes:
            print(f"           ! {note}")

        if state in ("DEAD_NOSTART", "EXITED", "WEDGED"):
            for h, n in rows.items():
                print(f"           {h:10s} iter={n['iter']} tr={n['trainers']} "
                      f"wd={n['watchdogs']} exits={n['exitcodes']}")
            return 1 if state != "EXITED" else 0

        if args.until_iter and cur["max_iter"] >= args.until_iter:
            print(f"reached iteration {cur['max_iter']} -> PASS")
            return 0
        if args.once or now - t0 > args.max_wait:
            return 0

        prev = cur
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

"""Stop every trainer on the cluster and verify the GPUs are actually free.

Two traps this avoids:

  1. Self-matching. A stop script whose own command line contains the pattern
     it greps for will match itself and commit suicide (or report phantom
     survivors). Every pattern here is bracketed -- '[p]retrain_(entry|gpt)[.]py' matches
     the trainer but never this script's own argv -- and the current PID and
     its parents are excluded explicitly.

  2. Orphaned trainers. Killing only the torchrun agent leaves the per-GPU
     worker processes alive, still holding HBM, so the next launch OOMs for
     no visible reason. We kill agents AND workers, then poll until every node
     reports trainers=0 AND vram_busy=0. Nothing is "stopped" until that holds.
"""

from __future__ import annotations

import argparse
import subprocess
import time

PATTERNS = ["[p]retrain_(entry|gpt)[.]py", "[t]orch[.]distributed[.]run"]


def ssh(host: str, cmd: str, timeout: int = 60) -> str:
    try:
        r = subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
             "-o", "LogLevel=ERROR", host, cmd],
            capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return ((r.stdout or "") + (r.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        return "__SSH_TIMEOUT__"


VRAM_PROBE = (
    "rocm-smi --showmeminfo vram 2>/dev/null "
    "| awk '/Used Memory/ {if ($NF+0 > 1000000000) n++} END {print n+0}'"
)


def status(host: str) -> str:
    cmd = (
        'echo "trainers=$(pgrep -c -f \'[p]retrain_(entry|gpt)[.]py\' || true)'
        ' agents=$(pgrep -c -f \'[t]orch[.]distributed[.]run\' || true)'
        ' vram_busy=$(VRAMPROBE)"'
    ).replace('VRAMPROBE', VRAM_PROBE)
    return ssh(host, cmd)


def kill_node(host: str, sig: str) -> str:
    pats = " ".join(f"-f '{p}'" for p in PATTERNS)
    # Exclude our own shell ($$) and its parent so the kill cannot hit itself.
    cmd = " ; ".join(
        f"for p in $(pgrep -f '{p}'); do "
        f"[ \"$p\" != \"$$\" ] && [ \"$p\" != \"$PPID\" ] && kill -{sig} $p 2>/dev/null; "
        f"done"
        for p in PATTERNS
    )
    return ssh(host, cmd + " ; echo killed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnodes", type=int, default=15)
    ap.add_argument("--wait", type=int, default=90)
    args = ap.parse_args()
    hosts = [f"node-{i}" for i in range(args.nnodes)]

    print("before:")
    for h in hosts:
        print(f"  {h:10s} {status(h)}")

    for sig in ("TERM", "KILL"):
        print(f"sending SIG{sig} ...")
        for h in hosts:
            kill_node(h, sig)
        time.sleep(10 if sig == "TERM" else 5)

    # Poll until genuinely clean -- orphans holding HBM are the usual failure.
    deadline = time.time() + args.wait
    while True:
        rows = {h: status(h) for h in hosts}
        dirty = [h for h, s in rows.items()
                 if "trainers=0" not in s or "vram_busy=0" not in s]
        if not dirty or time.time() > deadline:
            print("after:")
            for h, s in rows.items():
                mark = "" if h not in dirty else "   <-- STILL BUSY"
                print(f"  {h:10s} {s}{mark}")
            if dirty:
                print(f"WARNING: {len(dirty)} node(s) not clean: {' '.join(dirty)}")
                return 1
            print("all nodes clean: trainers=0 vram_busy=0")
            return 0
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())

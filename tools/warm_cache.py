"""Pre-build the GPTDataset index cache on node-0, then fan it out.

Background
----------
Megatron builds the document/sample/shuffle indices on global rank 0 only and
expects every other rank to read them off a shared filesystem. Our cache is
deliberately node-local, because the load path uses
`numpy.load(..., mmap_mode='r')` and an mmap page fault on blobfuse raises
SIGBUS/SIGSEGV in the faulting thread -- an uncatchable crash that presents as
a cluster-wide deadlock.

So we build the cache once, in a single process, then copy it to every node.

Correctness
-----------
The cache key is `md5(json(class, dataset_path, num_samples, index_split,
random_seed, sequence_length, split, split_matrix, tokenizer))`. Every one of
those inputs is a deterministic function of the run's arguments, and
`num_samples` in particular depends on train_iters/global_batch_size/
eval_interval/eval_iters. This script therefore runs the REAL argv (same spec,
same data blend), just with world_size=1 and MOE_CACHE_ONLY=1, so the hashes it
produces are exactly the ones the distributed job will look for.

A mismatch is not silent: ranks would report `FileNotFoundError`, which is what
motivated this tool in the first place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe_rebuild.config import (  # noqa: E402
    GPUS_PER_NODE,
    LOCAL_CACHE,
    MEGATRON_DIR,
    PROJECT_DIR,
    PYTHON_BIN,
    build_argv,
)
from moe_rebuild.specs import REGISTRY  # noqa: E402
from tools.launch import env_block, hosts, ssh  # noqa: E402


def build_cache(spec, argv: list[str], port: int, timeout: int) -> int:
    """Run one single-GPU process that builds the cache and exits."""
    env = dict(os.environ)
    env.update(env_block(Path("/tmp"), 1))
    env.update({
        "MOE_CACHE_ONLY": "1",
        "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
        "CUDA_VISIBLE_DEVICES": "0",
    })
    # world_size=1 => no model/expert parallelism during the cache build.
    slim = list(argv)
    for flag in ("--expert-model-parallel-size", "--tensor-model-parallel-size",
                 "--pipeline-model-parallel-size"):
        if flag in slim:
            i = slim.index(flag)
            slim[i + 1] = "1"
    # global batch must equal micro batch when there is a single rank, but the
    # dataset cache key depends on the ORIGINAL global batch size, so we keep
    # --global-batch-size untouched and only shrink the world.
    cmd = [PYTHON_BIN, f"{PROJECT_DIR}/tools/pretrain_entry.py"] + slim
    print(f"  building cache with world_size=1 (timeout {timeout}s) ...", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, env=env, cwd=MEGATRON_DIR,
                       capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    tail = (r.stdout or "")[-2500:]
    print(f"  rc={r.returncode} in {dt:.0f}s")
    for line in tail.splitlines():
        if any(k in line for k in ("cache", "indices", "samples", "epochs",
                                   "Error", "error", "Traceback", "rocm_shim")):
            print("   |", line[:160])
    return r.returncode


def _inventory_cmd(src: str) -> str:
    """Shell command printing a digest of the cache file inventory.

    Deliberately contains no backslash escapes. Two earlier versions of this
    check failed for formatting reasons alone rather than any real difference:
    `du -sb` also counts the directory inode, and awk saturated the 2.4 GB
    byte sum at INT32_MAX (2147483647). Sorting "<size> <name>" lines and
    hashing them compares the inventory exactly, with no integer formatting
    involved. `find -printf` newline escapes are avoided because a literal
    backslash-n does not reliably survive Python -> ssh -> bash quoting.
    """
    return (f"cd {src} && find . -type f -exec stat -c '%s %n' {{}} + "
            f"| LC_ALL=C sort | md5sum")


def fan_out(nnodes: int) -> int:
    """Copy the cache to every node, then verify the inventories match."""
    src = str(LOCAL_CACHE)
    files = sorted(f for f in Path(src).glob("*") if f.is_file())
    total = sum(f.stat().st_size for f in files)
    print(f"  cache: {len(files)} files, {total/2**20:.1f} MiB")

    local = subprocess.run(["bash", "-c", _inventory_cmd(src)],
                           capture_output=True, text=True, timeout=300)
    local_digest = local.stdout.split()[0] if local.stdout.strip() else "?"
    print(f"  inventory md5: {local_digest[:12]}")

    bad = 0
    for h in hosts(nnodes)[1:]:
        ssh(h, f"mkdir -p {src}", timeout=30)
        r = subprocess.run(
            ["rsync", "-a", "--delete", "-e",
             "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o LogLevel=ERROR",
             f"{src}/", f"{h}:{src}/"],
            capture_output=True, text=True, timeout=900,
        )
        chk = ssh(h, _inventory_cmd(src), timeout=180)
        digest = (chk.stdout or "").split()[0] if (chk.stdout or "").strip() else "?"
        ok = digest == local_digest and local_digest != "?"
        if not ok:
            bad += 1
        print(f"  {h:10s} rsync_rc={r.returncode} md5={digest[:12]} "
              f"{'OK' if ok else '<-- MISMATCH'}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", choices=sorted(REGISTRY))
    ap.add_argument("--nnodes", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--data", default="/scratch/rfull/data_blend.txt")
    ap.add_argument("--timeout", type=int, default=3000)
    ap.add_argument("--port", type=int, default=29123)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    spec = REGISTRY[args.spec]()
    if args.nnodes:
        spec.topology.nnodes = args.nnodes
    if args.iters:
        spec.schedule.train_iters = args.iters
        spec.schedule.lr_decay_iters = args.iters
        spec.schedule.lr_warmup_iters = min(
            spec.schedule.lr_warmup_iters, max(1, args.iters // 10))
    blend = Path(args.data)
    if blend.exists():
        spec.data_blend = blend.read_text().split()

    argv = build_argv(spec)
    nnodes = spec.topology.nnodes
    print(f"warm cache for {spec.run_id}: nodes={nnodes} "
          f"gbs={spec.schedule.global_batch_size} iters={spec.schedule.train_iters}")

    if not args.skip_build:
        rc = build_cache(spec, argv, args.port, args.timeout)
        if rc != 0:
            print("  cache build FAILED")
            return rc

    print("fanning out:")
    bad = fan_out(nnodes)
    print("cache warm on all nodes" if not bad else f"{bad} node(s) mismatched")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

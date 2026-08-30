"""Launch a Megatron run on chec-mi300-7.

Hard-won rules encoded here:

  * The rendezvous master is ALWAYS node-0, never "the first entry of the host
    list". Binding those two variables together once produced an 8-node arm
    where every rank dialled node-7 and the whole job timed out with ranks=0 --
    which looks like a bad node but is a launcher bug.
  * Every launch gets a fresh port, so a previous run's TIME_WAIT or a stale
    rendezvous can't be joined by accident.
  * The controller writes one log per rank-0 plus a per-node log, and records
    the resolved argv as JSON. Acceptance checks read the JSON, not shell
    strings.
  * Trainers are started detached (setsid + nohup) so an SSH drop never kills
    the run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe_rebuild.config import (  # noqa: E402
    GPUS_PER_NODE,
    LOCAL_CACHE,
    MEGATRON_DIR,
    PADDED_VOCAB,
    PROJECT_DIR,
    PYTHON_BIN,
    build_argv,
)
from moe_rebuild.specs import REGISTRY  # noqa: E402

RUN_ROOT = Path("/scratch/rfull/runs")
MASTER = "node-0"

# Counts GPUs holding >1 GB of VRAM. The previous probe grepped
# "GPU memory use (%): [1-9]", but this rocm-smi prints
# "VRAM Total Used Memory (B): 36563726336" -- so it always returned 0 and
# reported an idle cluster while 8 GPUs ran at 100% with 36 GB each.
VRAM_PROBE = (
    "rocm-smi --showmeminfo vram 2>/dev/null "
    "| awk '/Used Memory/ {if ($NF+0 > 1000000000) n++} END {print n+0}'"
)          # INVARIANT: never derived from the host list


def hosts(nnodes: int) -> list[str]:
    """node-0 first, always. Only the tail varies between arms."""
    return [f"node-{i}" for i in range(nnodes)]


def env_block(run_dir: Path, nnodes: int) -> dict[str, str]:
    e = {
        # --- RCCL / transport -------------------------------------------
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_HCA": "mlx5_ib",
        "NCCL_IB_TIMEOUT": "22",
        "NCCL_DEBUG": "WARN",          # INFO floods logs and contains "inf"
        "NCCL_DEBUG_SUBSYS": "INIT",
        # --- ROCm --------------------------------------------------------
        "HSA_ENABLE_SCRATCH_ASYNC_RECLAIM": "0",
        "USE_ROCM_AITER_ROPE_BACKEND": "0",   # aiter RoPE is lower precision
        # --- torch -------------------------------------------------------
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "OMP_NUM_THREADS": "8",
        # NOTE: expandable_segments is NOT supported by this ROCm build
        # ("HIPAllocatorConfig option expandable_segments is unrecognized").
        # Do not re-add it, and do not credit it with fixing anything --
        # an env var the runtime rejects cannot be the cause of any effect.
        "TOKENIZERS_PARALLELISM": "false",
        # --- crash forensics ---------------------------------------------
        # A SIGSEGV is not a Python exception; only faulthandler's "Current
        # thread" frame identifies where a rank actually died.
        "PYTHONFAULTHANDLER": "1",
        # Dump every thread's Python stack on SIGUSR1 *without* killing the
        # process. Plain PYTHONFAULTHANDLER only covers fatal signals, so
        # sending SIGUSR1 to a stalled rank terminates it (exitcode -10) and
        # takes the whole job down with it -- I did exactly that once while
        # trying to find out where a rank was stuck.
        "MOE_FAULTHANDLER_SIGUSR1": "1",
        # The first iteration takes ~16 min: ROCm autotunes every distinct
        # kernel shape on first use (~16.9 s per MoE layer, 48 layers, 8
        # gradient-accumulation steps). Upstream leaves
        # EXPERT_MODEL_PARALLEL_GROUP on PyTorch's 10-minute default, so EP
        # peers abort mid-warmup and every other rank dies as a bystander.
        # See rocm_shim._install_ep_group_timeout_fix.
        "MOE_EP_TIMEOUT_MINUTES": "120",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{MEGATRON_DIR}:/scratch/rfull/moe",
    }
    return e


def build_launch_script(
    node_rank: int,
    nnodes: int,
    port: int,
    run_dir: Path,
    argv: list[str],
) -> str:
    """Full bash script for one node. Written to the node, then run detached."""
    env = env_block(run_dir, nnodes)
    exports = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items())
    # Always emit POSIX paths: the script runs on Linux, but the launcher
    # may be driven from Windows, where Path renders backslashes and would
    # silently corrupt both the redirect and mkdir.
    rd = str(run_dir).replace("\\", "/")
    log = f"{rd}/node{node_rank}.log"
    # One long line, no backslash continuations. A mangled continuation
    # once turned "--tensor-model-parallel-size 1" into "... 'n'";
    # argparse rejected it, rank 0 exited 2 and every other rank died
    # of SIGTERM as a bystander.
    args = " ".join(shlex.quote(a) for a in argv)
    return f"""#!/bin/bash
{exports}
cd {MEGATRON_DIR}
mkdir -p {rd}
exec > {log} 2>&1 < /dev/null
echo "=== node_rank={node_rank} nnodes={nnodes} master={MASTER}:{port} host=$(hostname) ==="
echo "=== started $(date -u) ==="
{PYTHON_BIN} -m torch.distributed.run --nnodes={nnodes} --nproc-per-node={GPUS_PER_NODE} --node-rank={node_rank} --master-addr={MASTER} --master-port={port} --max-restarts=0 {PROJECT_DIR}/tools/pretrain_entry.py {args}
echo "=== exited rc=$? $(date -u) ==="
"""


def ssh(host: str, cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh", "-n", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "LogLevel=ERROR", host, cmd,
        ],
        capture_output=True, text=True, timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


def ssh_put(host: str, path: str, content: str, timeout: int = 60) -> int:
    """Write a file on a remote node via stdin (no scp round-trip)."""
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
         "-o", "LogLevel=ERROR", host, f"cat > {path} && chmod +x {path}"],
        input=content.encode().replace(b"\r\n", b"\n"),
        capture_output=True, timeout=timeout,
    )
    return r.returncode


def ssh_detach(host: str, script: str, timeout: int = 45) -> str:
    """Start a script fully detached.

    ssh will not close the channel while any descendant still holds its stdout,
    so the remote script re-opens all three fds itself (`exec > log 2>&1
    < /dev/null`) and we additionally detach with setsid and swallow the
    wrapper's own output. Without this the launcher blocks on node-0 and the
    remaining nodes never start -- which then looks like a dead cluster.
    """
    cmd = f"setsid {script} >/dev/null 2>&1 & echo started=$!"
    try:
        r = subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
             "-o", "LogLevel=ERROR", host, cmd],
            capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return (r.stdout or r.stderr).strip()
    except subprocess.TimeoutExpired:
        # Non-fatal: the process may well be running. Verified in the poll below.
        return "SSH_TIMEOUT(verify below)"



def preflight(hs: list[str], run_dir: Path) -> bool:
    """Assert the cluster is symmetric and idle before burning a launch.

    Node-0 has historically been the asymmetric one (root-owned dirs, missing
    compiled helpers), so this prints ONE LINE PER NODE -- an aggregate count
    would hide "14 good + 1 broken".
    """
    print("preflight:")
    ok = True
    probe = (
        f"so=$(ls {MEGATRON_DIR}/megatron/core/datasets/helpers_cpp*.so 2>/dev/null | wc -l);"
        f"own=$(stat -c %U {MEGATRON_DIR}/megatron/core/datasets 2>/dev/null);"
        # Bracket the first char so the pattern cannot match this very ssh
        # command line -- a self-matching pgrep once made a stop-script kill
        # itself and here it reported phantom trainers.
        f"tr=$(pgrep -c -f '[p]retrain_(entry|gpt)[.]py' || true);"
        "vr=$(" + VRAM_PROBE + ");"
        f"idx=$(ls /scratch/rfull/data 2>/dev/null | wc -l);"
        f"av=$(df -BG --output=avail /scratch | tail -1 | tr -d ' G');"
        f'echo "so=$so own=$own trainers=$tr vram_busy=$vr corpora=$idx scratchGB=$av"'
    )
    for h in hs:
        r = ssh(h, probe, timeout=60)
        line = (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else "NO_OUTPUT"
        flag = ""
        if "so=0" in line or "trainers=0" not in line or "vram_busy=0" not in line:
            flag = "  <-- CHECK"
            ok = False
        print(f"  {h:10s} {line}{flag}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", choices=sorted(REGISTRY))
    ap.add_argument("--nnodes", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--data", default="/scratch/rfull/data_blend.txt",
                    help="file with a Megatron --data-path blend (weight path ...)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    spec = REGISTRY[args.spec]()
    if args.nnodes:
        spec.topology.nnodes = args.nnodes
    if args.iters:
        spec.schedule.train_iters = args.iters
        spec.schedule.lr_decay_iters = args.iters
        spec.schedule.lr_warmup_iters = min(
            spec.schedule.lr_warmup_iters, max(1, args.iters // 10)
        )
    if args.run_id:
        spec.run_id = args.run_id

    blend_file = Path(args.data)
    if blend_file.exists():
        spec.data_blend = blend_file.read_text().split()

    nnodes = spec.topology.nnodes
    hs = hosts(nnodes)
    port = args.port or random.randint(29600, 29990)
    stamp = time.strftime("%m%d_%H%M%S")
    run_dir = RUN_ROOT / f"{spec.run_id}_{stamp}"

    # TensorBoard event files live inside the run directory, so a run's
    # scalars travel with its logs and argv.json. Megatron writes them from
    # the last rank only; that rank's node holds the files.
    #
    # This must happen BEFORE build_argv: the run dir is only known here, and
    # setting the field afterwards silently produced an argv with no
    # --tensorboard-dir at all, so the flag looked configured but nothing was
    # ever written.
    if spec.tensorboard_dir is None:
        spec.tensorboard_dir = f"{str(run_dir).replace(chr(92), '/')}/tensorboard"

    argv = build_argv(spec)

    print(f"run       : {spec.run_id}")
    print(f"spec      : {args.spec}")
    print(f"nodes     : {nnodes} ({hs[0]} .. {hs[-1]})  world={nnodes*GPUS_PER_NODE}")
    print(f"master    : {MASTER}:{port}   (fixed, NOT hosts[0]-derived)")
    print(f"topology  : TP{spec.topology.tensor_parallel} PP{spec.topology.pipeline_parallel} "
          f"EP{spec.topology.expert_parallel} DP{spec.topology.data_parallel}")
    print(f"model     : {spec.model.name} L{spec.model.num_layers} h{spec.model.hidden_size} "
          f"experts={spec.model.num_experts} seq={spec.model.seq_length}")
    print(f"batch     : gbs={spec.schedule.global_batch_size} mbs={spec.schedule.micro_batch_size} "
          f"iters={spec.schedule.train_iters}")
    print(f"run_dir   : {run_dir}")

    if args.dry_run:
        print("\nargv:\n  " + " ".join(argv))
        return 0

    if not args.skip_preflight and not preflight(hs, run_dir):
        print("PREFLIGHT FAILED -- fix asymmetry/stale trainers before launching")
        return 2

    os.makedirs(run_dir, exist_ok=True)
    (run_dir / "argv.json").write_text(json.dumps(argv, indent=1))
    (run_dir / "spec.json").write_text(spec.to_json())
    (run_dir / "meta.json").write_text(json.dumps({
        "run_id": spec.run_id, "spec": args.spec, "nnodes": nnodes,
        "world": nnodes * GPUS_PER_NODE, "master": MASTER, "port": port,
        "hosts": hs, "padded_vocab": PADDED_VOCAB, "started": stamp,
        "data_cache": LOCAL_CACHE,
    }, indent=1))

    print("launching:")
    for rank, h in enumerate(hs):
        script_path = f"/tmp/_launch_{spec.run_id}_{rank}.sh"
        body = build_launch_script(rank, nnodes, port, run_dir, argv)
        rc = ssh_put(h, script_path, body)
        out = ssh_detach(h, script_path)
        print(f"  {h:10s} rank={rank} put_rc={rc} {out[:50]}")

    # Verify every node really has a torchrun agent -- do not trust the ssh rc.
    print("\nverifying trainers came up:")
    time.sleep(20)
    live = 0
    for rank, h in enumerate(hs):
        r = ssh(h, 'echo "tr=$(pgrep -c -f \'[p]retrain_(entry|gpt)[.]py\' || true) '
                   'rn=$(pgrep -c -f \'[t]orch.distributed.run\' || true)"', timeout=45)
        line = (r.stdout or "").strip().replace("\n", " ")
        good = "rn=0" not in line
        live += good
        print(f"  {h:10s} {line}{'' if good else '   <-- NO AGENT'}")
    print(f"  agents up on {live}/{nnodes} nodes")

    print(f"\nmonitor: python tools/monitor.py {run_dir}")
    print(f"logs   : {run_dir}/node*.log")
    return 0 if live == nnodes else 3


if __name__ == "__main__":
    raise SystemExit(main())

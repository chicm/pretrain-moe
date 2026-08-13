#!/usr/bin/env python3
"""Launch exact eight-process R-Full EP parity on CUDA/NCCL or CPU/Gloo."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile


WORLD_SIZE = 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    megatron = pathlib.Path(args.megatron_root).resolve()
    if not (megatron / "megatron" / "core").is_dir():
        raise SystemExit(f"invalid Megatron root: {megatron}")
    if args.timeout_seconds < 30:
        raise SystemExit("timeout must be at least 30 seconds")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.output_dir:
        output = pathlib.Path(args.output_dir).resolve()
        if output.exists():
            raise SystemExit(f"refusing to overwrite output directory: {output}")
        output.mkdir(parents=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="rfull-ep8-parity-")
        output = pathlib.Path(temporary.name)
    rendezvous = (output / "file-rendezvous").resolve().as_uri()

    processes: list[subprocess.Popen[str]] = []
    handles = []
    try:
        for rank in range(WORLD_SIZE):
            environment = os.environ.copy()
            environment.update(
                RANK=str(rank),
                LOCAL_RANK=str(rank),
                WORLD_SIZE=str(WORLD_SIZE),
                RFULL_INIT_METHOD=rendezvous,
                PYTHONPATH=os.pathsep.join(
                    filter(None, [str(megatron), str(root), environment.get("PYTHONPATH")])
                ),
            )
            handle = (output / f"rank-{rank}.log").open("w", encoding="utf-8")
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    [sys.executable, str(root / "tools" / "rfull_ep_parity.py")],
                    cwd=root,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        return_codes = [
            process.wait(timeout=args.timeout_seconds) for process in processes
        ]
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        for handle in handles:
            handle.close()

    rank_evidence = []
    failed = any(return_codes)
    for rank, return_code in enumerate(return_codes):
        log_path = output / f"rank-{rank}.log"
        text = log_path.read_text(encoding="utf-8", errors="replace")
        candidates = []
        for line in text.splitlines():
            if "RFULL_EP_PARITY_PASS" not in line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        valid = [
            item
            for item in candidates
            if item.get("marker") == "RFULL_EP_PARITY_PASS"
            and item.get("rank") == rank
            and item.get("world_size") == WORLD_SIZE
            and item.get("ep_world_size") == WORLD_SIZE
            and item.get("finite") is True
        ]
        if return_code != 0 or len(valid) != 1:
            failed = True
            print(f"===== rank {rank} rc={return_code} INVALID =====")
            print("\n".join(text.splitlines()[-200:]))
            continue
        rank_evidence.append(valid[0])
        print(json.dumps(valid[0], sort_keys=True))

    summary = {
        "marker": "RFULL_EP8_PARITY_LAUNCH_PASS" if not failed else "RFULL_EP8_PARITY_LAUNCH_FAIL",
        "world_size": WORLD_SIZE,
        "ranks": sorted(item["rank"] for item in rank_evidence),
        "return_codes": return_codes,
    }
    (output / "launch.acceptance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    if temporary is not None:
        temporary.cleanup()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

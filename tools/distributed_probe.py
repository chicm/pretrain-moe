#!/usr/bin/env python3
"""One-collective distributed probe used before every multi-node smoke run."""

from __future__ import annotations

import json
import os
import socket
import time

import torch
import torch.distributed as dist


def main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    local_device = torch.device("cuda", local_rank)
    started = time.monotonic()
    dist.init_process_group(
        backend="nccl", init_method="env://", device_id=local_device
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(
        json.dumps(
            {
                "marker": "DISTRIBUTED_PROBE_DEVICE_BIND",
                "rank": rank,
                "local_rank": local_rank,
                "device_id": str(local_device),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    value = torch.tensor(float(rank + 1), device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    expected = world_size * (world_size + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"all_reduce mismatch: expected {expected}, got {value.item()}")

    identity = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "device": torch.cuda.get_device_name(local_rank),
    }
    gathered: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(gathered, identity)
    payload = {
        "marker": "DISTRIBUTED_PROBE_RANK_OK",
        **identity,
        "world_size": world_size,
        "all_reduce_sum": value.item(),
        "seconds": time.monotonic() - started,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    if rank == 0:
        print(
            json.dumps(
                {
                    "marker": "DISTRIBUTED_PROBE_WORLD_OK",
                    "world_size": world_size,
                    "ranks": gathered,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

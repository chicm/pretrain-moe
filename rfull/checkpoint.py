"""Transactional checkpointing for R-Full.

A 254,313-update run will be interrupted -- preemption, a node reboot, a spike
that outlives even the widened timeout. What must never happen is resuming from
a checkpoint that is half-written, or one whose data cursor disagrees with its
weights, because that silently retrains or skips tokens and the loss curve will
not show it.

Two rules make that safe:

1. **Commit is a rename.** Every rank writes into a per-update staging
   directory. Only after all ranks finish, and only from rank 0, is
   ``LATEST_COMMITTED`` replaced via ``os.replace`` -- atomic on POSIX. A reader
   therefore sees either the old committed update or the new one, never a
   partial directory. Crash at any earlier point and the staging directory is
   simply garbage to be swept.

2. **The cursor travels with the weights.** ``successful_updates`` is stored in
   the same payload as the parameters. The data scheduler derives its position
   from that number alone, so weights and cursor cannot drift apart.

Expert parameters are sharded across the expert-parallel group: rank r of an EP
group owns a distinct slice of every MoE layer. Each rank therefore saves its
own shard, and restore requires the identical topology. The topology is recorded
and checked, because loading an EP8 checkpoint into an EP4 job would silently
mis-assign experts rather than fail.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

LATEST = "LATEST_COMMITTED"


def _shard_name(rank: int) -> str:
    return f"shard_{rank:05d}.pt"


def _topology() -> Dict[str, int]:
    from megatron.core import parallel_state as ps

    return {
        "world_size": dist.get_world_size(),
        "expert_model_parallel_size": ps.get_expert_model_parallel_world_size(),
        "data_parallel_size": ps.get_data_parallel_world_size(),
    }


def save_checkpoint(root: str, successful_updates: int, model, optimizer,
                    extra: Optional[Dict[str, Any]] = None) -> str:
    """Write a checkpoint and commit it atomically. Returns the directory."""
    rank = dist.get_rank()
    stage = os.path.join(root, f"update_{successful_updates:08d}")
    if rank == 0:
        os.makedirs(stage, exist_ok=True)
    dist.barrier()

    payload = {
        "successful_updates": successful_updates,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rank": rank,
        "topology": _topology(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, os.path.join(stage, _shard_name(rank)))

    # Every rank must land before the pointer moves, or a restart could read a
    # directory that is missing shards.
    dist.barrier()
    if rank == 0:
        meta = {
            "successful_updates": successful_updates,
            "topology": _topology(),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "shards": dist.get_world_size(),
        }
        with open(os.path.join(stage, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        tmp = os.path.join(root, LATEST + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(os.path.basename(stage))
        os.replace(tmp, os.path.join(root, LATEST))
    dist.barrier()
    return stage


def latest_checkpoint(root: str) -> Optional[str]:
    """Directory named by LATEST_COMMITTED, or None if nothing is committed."""
    ptr = os.path.join(root, LATEST)
    if not os.path.isfile(ptr):
        return None
    with open(ptr, encoding="utf-8") as f:
        name = f.read().strip()
    path = os.path.join(root, name)
    return path if os.path.isdir(path) else None


def load_checkpoint(root: str, model, optimizer) -> int:
    """Restore from the committed checkpoint. Returns ``successful_updates``."""
    path = latest_checkpoint(root)
    if path is None:
        return 0
    rank = dist.get_rank()
    shard = os.path.join(path, _shard_name(rank))
    if not os.path.isfile(shard):
        raise RuntimeError(
            f"checkpoint {path} has no shard for rank {rank}. Expert weights "
            "are sharded per rank, so the job must resume with the same world "
            "size it was saved with.")
    payload = torch.load(shard, map_location="cpu", weights_only=False)

    saved_topo = payload.get("topology", {})
    now_topo = _topology()
    if saved_topo != now_topo:
        raise RuntimeError(
            f"topology mismatch: checkpoint {saved_topo} vs current {now_topo}. "
            "Each rank owns a distinct slice of every MoE layer, so resuming "
            "under a different layout would silently mis-assign experts.")

    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state(payload["cuda_rng"])
    return int(payload["successful_updates"])


def sweep_stale(root: str, keep: int = 3) -> None:
    """Delete all but the newest ``keep`` committed checkpoints (rank 0 only).

    Never removes the committed one, whatever ``keep`` says.
    """
    if dist.get_rank() != 0:
        return
    committed = latest_checkpoint(root)
    dirs = sorted(d for d in os.listdir(root) if d.startswith("update_"))
    for d in dirs[:-keep] if keep else []:
        full = os.path.join(root, d)
        if full == committed:
            continue
        shutil.rmtree(full, ignore_errors=True)

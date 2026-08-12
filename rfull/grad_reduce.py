"""Bucketed gradient all-reduce for the R-Full MoE model.

Why this module exists
----------------------
Reducing gradients one tensor at a time is catastrophic here. The model has
1242 parameter tensors per rank; on two nodes that measured 270.96 s of
communication per step, i.e. 99.5% of the step. Coalescing the same gradients
into 128 MB flat buckets brought it to 1.28 s -- a 212x improvement.

Correctness detail that is easy to get wrong
--------------------------------------------
Routed experts are sharded across the expert-parallel (EP) group: each rank owns
a distinct subset of experts, so their gradients must be averaged only over the
*expert-data-parallel* group (the ranks holding the same expert shard).
Every other parameter is replicated across the ordinary data-parallel group and
must be averaged there. Averaging expert gradients over the wrong group silently
corrupts training, so parameters are partitioned explicitly by the
``is_expert_parallel`` marker Megatron sets on them.

Reduction is done in float32 regardless of parameter dtype: summing bf16
gradients across many ranks loses precision.
"""
from __future__ import annotations

from typing import Iterable, List

import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps

__all__ = ["allreduce_gradients", "split_expert_params"]

_DEFAULT_BUCKET_BYTES = 128 << 20


def _is_expert_param(p: torch.nn.Parameter) -> bool:
    """True if the parameter belongs to an EP-sharded routed expert.

    Megatron marks these with ``allreduce`` False or an explicit
    ``is_expert_parallel`` attribute depending on version, so check both.
    """
    if getattr(p, "is_expert_parallel", False):
        return True
    # Some versions express it as "do not all-reduce over the normal DP group".
    return getattr(p, "allreduce", True) is False


def split_expert_params(model: torch.nn.Module):
    """Partition parameters into (expert_sharded, replicated)."""
    expert: List[torch.nn.Parameter] = []
    dense: List[torch.nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (expert if _is_expert_param(p) else dense).append(p)
    return expert, dense


def _flush(bucket: List[torch.Tensor], group, world: int) -> None:
    if not bucket:
        return
    if len(bucket) == 1:
        g = bucket[0]
        buf = g.float()
        dist.all_reduce(buf, group=group)
        buf /= world
        g.copy_(buf.to(g.dtype))
        return
    flat = torch.cat([g.reshape(-1).float() for g in bucket])
    dist.all_reduce(flat, group=group)
    flat /= world
    off = 0
    for g in bucket:
        n = g.numel()
        g.copy_(flat[off:off + n].view_as(g).to(g.dtype))
        off += n


def _reduce_group(params: Iterable[torch.nn.Parameter], group,
                  bucket_bytes: int) -> None:
    if group is None:
        return
    world = dist.get_world_size(group)
    if world <= 1:
        return
    bucket: List[torch.Tensor] = []
    nbytes = 0
    for p in params:
        g = p.grad
        if g is None:
            continue
        # float32 accumulation: bucket cost is 4 bytes per element
        b = g.numel() * 4
        if bucket and nbytes + b > bucket_bytes:
            _flush(bucket, group, world)
            bucket, nbytes = [], 0
        bucket.append(g)
        nbytes += b
    _flush(bucket, group, world)


def allreduce_gradients(model: torch.nn.Module,
                        bucket_bytes: int = _DEFAULT_BUCKET_BYTES) -> None:
    """Average gradients over the correct group for each parameter class.

    Must be called after backward and BEFORE gradient clipping, so that every
    rank clips an identical gradient and stays in lockstep.
    """
    expert, dense = split_expert_params(model)
    _reduce_group(dense, ps.get_data_parallel_group(), bucket_bytes)
    _reduce_group(expert, ps.get_expert_data_parallel_group(), bucket_bytes)

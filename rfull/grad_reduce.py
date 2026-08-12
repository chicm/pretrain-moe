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
    """All-reduce one bucket of gradients and scatter the result back.

    Flattening is done with ``contiguous().view(-1)`` and restored with
    ``reshape``. The obvious ``reshape(-1)`` / ``view_as`` pairing happens to
    produce correct values on CPU, but ``reshape(-1)`` silently returns a copy
    for a non-contiguous tensor while ``view_as`` demands a compatible layout,
    so the pairing is only accidentally safe. Being explicit costs nothing.

    NOTE: this was tightened while investigating a two-node
    "Memory access fault by GPU ... Reason: Unknown", but a CPU regression test
    showed the old form still returned correct values, so it is NOT established
    as the cause of that fault. The fault remains under investigation.
    """
    if not bucket:
        return
    if len(bucket) == 1:
        g = bucket[0]
        buf = g.detach().float().contiguous()
        dist.all_reduce(buf, group=group)
        buf /= world
        g.copy_(buf.reshape(g.shape).to(g.dtype))
        return
    flat = torch.cat([g.detach().float().contiguous().view(-1)
                      for g in bucket])
    dist.all_reduce(flat, group=group)
    flat /= world
    off = 0
    for g in bucket:
        n = g.numel()
        g.copy_(flat[off:off + n].reshape(g.shape).to(g.dtype))
        off += n


def _reduce_group(params: Iterable[torch.nn.Parameter], group,
                  bucket_bytes: int) -> None:
    if group is None:
        return
    world = dist.get_world_size(group)
    if world <= 1:
        return

    # A bucketed reducer is a collective SCHEDULE: every rank in the group must
    # issue the same number of all_reduce calls, in the same order, with the
    # same sizes. Bucket boundaries are derived from each rank's own gradient
    # list, so if the ranks disagree about which parameters have gradients they
    # build different numbers of buckets and deadlock -- one rank waits on a
    # call the other never makes, and it surfaces 600 s later as
    #   DistBackendError: ... is setting up NCCL communicator ... wait timeout
    # Build the plan first, then verify all ranks agree before communicating.
    grads = [p.grad for p in params if p.grad is not None]
    plan: List[List[torch.Tensor]] = []
    bucket: List[torch.Tensor] = []
    nbytes = 0
    for g in grads:
        b = g.numel() * 4              # float32 accumulation
        if bucket and nbytes + b > bucket_bytes:
            plan.append(bucket)
            bucket, nbytes = [], 0
        bucket.append(g)
        nbytes += b
    if bucket:
        plan.append(bucket)

    n = torch.tensor([len(plan), len(grads)], dtype=torch.int64,
                     device=grads[0].device if grads else "cpu")
    alln = [torch.zeros_like(n) for _ in range(world)]
    dist.all_gather(alln, n, group=group)
    if any(not torch.equal(x, alln[0]) for x in alln):
        detail = ", ".join(f"rank{i}:buckets={int(x[0])},grads={int(x[1])}"
                           for i, x in enumerate(alln))
        raise RuntimeError(
            "expert-DP gradient reduction plan differs across ranks, which "
            f"would deadlock: {detail}")

    for bucket in plan:
        _flush(bucket, group, world)


def allreduce_gradients(model: torch.nn.Module,
                        bucket_bytes: int = _DEFAULT_BUCKET_BYTES) -> None:
    """Average gradients over the correct group for each parameter class.

    Must be called after backward and BEFORE gradient clipping, so that every
    rank clips an identical gradient and stays in lockstep.
    """
    # Imported lazily so the pure-tensor helpers stay testable without Megatron.
    from megatron.core import parallel_state as ps

    expert, dense = split_expert_params(model)
    _reduce_group(dense, ps.get_data_parallel_group(), bucket_bytes)
    _reduce_group(expert, ps.get_expert_data_parallel_group(), bucket_bytes)

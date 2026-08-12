"""EP all-to-all serialisation workaround for MCore MoE on ROCm.

Problem
-------
With a deep MoE stack (46 MoE layers, EP8) on this ROCm/RCCL build, training
deadlocks inside an EP all-to-all, typically during the second step. At the
hang all ranks are aligned on the same collective and the split tables are
provably correct:

  * 276 traced all_to_all_single calls: sum(in) == sum(out) on every call and
    zero pairwise mismatches (rank_i.in_splits[j] == rank_j.out_splits[i])
  * raw RCCL replaying the exact stuck split table: 200/200 iterations, 10.1 s
  * 1 layer x 300 steps passes; 8 layers x 6 steps passes; 46 layers hangs

So neither the routing data, the split computation, nor RCCL itself is at
fault. The trigger is many in-flight all-to-alls overlapping with compute
during backward.

Fix
---
Serialise each EP all-to-all by synchronising immediately after issuing it.

Evidence (46 MoE layers, 1024 tokens/rank, 12 steps):
  serialise=OFF -> deadlock, killed at 900 s timeout          (control)
  serialise=ON  -> 12/12 steps, steady 0.64-0.66 s/step       (repeat1)
  serialise=ON  -> 12/12 steps, steady 1.02-1.63 s/step       (repeat2, repeat3)

Cost: gives up communication/compute overlap. Measured cost is acceptable, but
re-measure on the full 48-layer / 4096-token configuration.

This is a workaround for a specific runtime combination, not a permanent
design choice. Re-run the serialise=OFF control after any upgrade of Megatron
Core, PyTorch, ROCm or RCCL to check whether it is still required.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

__all__ = ["enable_a2a_serialisation", "is_enabled", "call_count"]

_installed = False
_original = None
_count = 0


def is_enabled() -> bool:
    """True if the serialisation patch is currently installed."""
    return _installed


def call_count() -> int:
    """Number of all-to-all calls issued since the patch was installed."""
    return _count


def enable_a2a_serialisation(verbose: bool = True) -> bool:
    """Install the serialising wrapper around ``dist.all_to_all_single``.

    Idempotent: calling it twice is a no-op. Returns True if the patch is
    installed after the call.
    """
    global _installed, _original, _count

    if _installed:
        return True

    _original = dist.all_to_all_single

    def _serialised_all_to_all_single(
        output,
        input,
        output_split_sizes=None,
        input_split_sizes=None,
        group=None,
        async_op=False,
    ):
        global _count
        _count += 1
        result = _original(
            output, input, output_split_sizes, input_split_sizes, group,
            async_op,
        )
        # The synchronisation -- not the call itself -- is what avoids the
        # deadlock. Keep it immediately after the collective is issued.
        torch.cuda.synchronize()
        return result

    dist.all_to_all_single = _serialised_all_to_all_single
    _installed = True

    if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
        print("[rfull] EP all-to-all serialisation ENABLED "
              "(workaround for deep-MoE deadlock on this ROCm/RCCL build)",
              flush=True)
    return True

"""World-consensus finite checks with stop-and-replay (blocker NUM-001).

Frozen contract (design doc sections 8 / 16.4):

* every update is validated for finiteness on EVERY rank
* the decision is a WORLD consensus: if any rank sees a non-finite or
  implausibly large gradient norm, all ranks must agree to reject
* a rejected update is NOT silently skipped: the run terminates without
  committing, and training is replayed from the last committed checkpoint
* no overflow-driven silent loss scaling fallback (BF16 needs none)

The consensus is a single all-reduce on a 2-element FP32 tensor, so the
per-update overhead is one tiny collective.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.distributed as dist

__all__ = [
    "FiniteConsensusError",
    "ConsensusVerdict",
    "world_finite_consensus",
    "check_update_or_die",
]


class FiniteConsensusError(RuntimeError):
    """Raised when the world rejects an update. Must terminate the run."""


@dataclass
class ConsensusVerdict:
    accepted: bool
    local_finite: bool
    world_all_finite: bool
    grad_norm: float
    max_grad_norm_seen: float
    loss: float
    offending_ranks: int
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def world_finite_consensus(
    loss: torch.Tensor,
    grad_norm: Optional[torch.Tensor | float],
    device: torch.device,
    max_grad_norm_allowed: float = 1.0e4,
    group=None,
) -> ConsensusVerdict:
    """Reach a world consensus on whether this update may be committed.

    Args:
        loss: scalar loss tensor for this rank.
        grad_norm: total gradient norm for this rank (post clip computation).
        device: device to run the collective on.
        max_grad_norm_allowed: reject finite-but-absurd norms too. The frozen
            design rejects "large finite" values, not just NaN/Inf.
        group: process group; ``None`` == WORLD.

    Returns:
        ConsensusVerdict. ``accepted`` is guaranteed identical on all ranks.
    """
    loss_v = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
    if grad_norm is None:
        gn_v = 0.0
    elif torch.is_tensor(grad_norm):
        gn_v = float(grad_norm.detach().item())
    else:
        gn_v = float(grad_norm)

    import math

    local_finite = math.isfinite(loss_v) and math.isfinite(gn_v)
    local_ok = local_finite and abs(gn_v) <= max_grad_norm_allowed

    # [n_bad_ranks, max_grad_norm]
    probe = torch.tensor(
        [0.0 if local_ok else 1.0, gn_v if math.isfinite(gn_v) else float("inf")],
        dtype=torch.float32, device=device,
    )
    if dist.is_initialized() and _world_size() > 1:
        bad = probe[:1].clone()
        dist.all_reduce(bad, op=dist.ReduceOp.SUM, group=group)
        mx = probe[1:].clone()
        # inf survives MAX; NaN would not, hence the isfinite guard above
        dist.all_reduce(mx, op=dist.ReduceOp.MAX, group=group)
        offending = int(bad.item())
        max_gn = float(mx.item())
    else:
        offending = int(probe[0].item())
        max_gn = gn_v

    accepted = offending == 0
    if accepted:
        reason = "ok"
    elif not local_finite:
        reason = "non_finite_local"
    else:
        reason = f"world_reject:{offending}_rank(s)"

    return ConsensusVerdict(
        accepted=accepted,
        local_finite=local_finite,
        world_all_finite=accepted,
        grad_norm=gn_v,
        max_grad_norm_seen=max_gn,
        loss=loss_v,
        offending_ranks=offending,
        reason=reason,
    )


def check_update_or_die(
    loss: torch.Tensor,
    grad_norm: Optional[torch.Tensor | float],
    device: torch.device,
    step: int,
    max_grad_norm_allowed: float = 1.0e4,
    group=None,
) -> ConsensusVerdict:
    """Consensus check that raises instead of skipping.

    The frozen design forbids silently dropping a bad batch: a rejected update
    terminates the run so it can be replayed deterministically from the last
    committed checkpoint.
    """
    verdict = world_finite_consensus(
        loss, grad_norm, device,
        max_grad_norm_allowed=max_grad_norm_allowed, group=group)
    if not verdict.accepted:
        raise FiniteConsensusError(
            f"update {step} rejected by world consensus: {verdict.reason}; "
            f"loss={verdict.loss} grad_norm={verdict.grad_norm} "
            f"max_grad_norm_seen={verdict.max_grad_norm_seen}. "
            f"Run must terminate WITHOUT commit and replay from the last "
            f"committed checkpoint."
        )
    return verdict

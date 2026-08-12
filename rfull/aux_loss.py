"""EP-global auxiliary load-balancing loss and router z-loss (blocker AUX-001).

Frozen contract (design doc section 6.3):

    p_{t,i} = softmax over ALL N logits          (global 96-way, FP32)
    n_i     = # times expert i appears in Top-k  (detached)
    T       = tokens participating in the statistic, aggregated over the EP group
    f_i     = n_i / (k * T)
    P_i     = (1/T) * sum_t p_{t,i}
    L_aux   = N * sum_i f_i * P_i

Why the stock Megatron-Core function is not sufficient
------------------------------------------------------
``megatron.core.transformer.moe.moe_utils.switch_load_balancing_loss_func``
computes the statistic over the tokens that are local to the calling rank and
then relies on ordinary data-parallel gradient averaging.  Averaging per-rank
scalar aux losses is NOT the same function as computing f_i and P_i over the
union of the EP group's tokens: mean_r(N * sum_i f_i^(r) P_i^(r)) has a
different value *and* a different gradient than N * sum_i f_i^(G) P_i^(G),
because f_i and P_i are multiplied before reduction.  The frozen design
requires the latter, so this module aggregates the raw statistics.

Gradient scaling
----------------
Every rank of an EP group adds the same group-level scalar to its objective.
Megatron then averages gradients over the full dense-DP group (120 ranks).
With G = world/ep EP groups:

    DP-average = (1/world) * sum_r  d L_aux^{G(r)} / d W
               = (1/world) * sum_G  d L_aux^{G} / d W     (8 ranks share a group,
                                                           each supplying the
                                                           gradient of its own
                                                           tokens)
               = (1/ep) * mean_G  d L_aux^{G} / d W

To make the effective objective equal to ``mean_G L_aux^{G}`` -- which reduces
to the plain per-rank definition when ep=1 -- the term entering backward is
multiplied by ``ep_size``.  ``scale_for_dp_average=True`` (default) applies it.
The value reported for telemetry is always the UNSCALED group loss.

z-loss
------
    L_z = (1/T) * sum_t (logsumexp_i z_{t,i})^2

is separable per token, so a rank-local mean followed by DP averaging already
equals the global mean whenever ranks carry equal token counts (guaranteed by
the frozen fixed-shape batch contract).  It therefore needs no cross-rank
aggregation and no EP scaling.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

__all__ = [
    "ep_global_aux_loss",
    "router_z_loss",
    "expert_load_metrics",
]


class _DifferentiableAllReduceSum(torch.autograd.Function):
    """all_reduce(SUM) that propagates gradient unchanged to every rank.

    For y = sum_r x_r, dL/dx_r = dL/dy for every rank r, so the backward pass
    is the identity on the incoming gradient.  ``torch.distributed.all_reduce``
    is not differentiable, hence this wrapper.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group) -> torch.Tensor:  # type: ignore[override]
        ctx.group = group
        if group is None or not dist.is_initialized():
            return tensor
        if dist.get_world_size(group=group) == 1:
            return tensor
        out = tensor.contiguous().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return grad_output, None


def _all_reduce_sum_no_grad(tensor: torch.Tensor, group) -> torch.Tensor:
    if group is None or not dist.is_initialized():
        return tensor
    if dist.get_world_size(group=group) == 1:
        return tensor
    out = tensor.contiguous().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    return out


def _ep_world_size(group) -> int:
    if group is None or not dist.is_initialized():
        return 1
    return dist.get_world_size(group=group)


def ep_global_aux_loss(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
    topk: int,
    ep_group=None,
    scale_for_dp_average: bool = True,
    return_stats: bool = True,
):
    """Compute the EP-global auxiliary load-balancing loss.

    Args:
        logits: ``[T_local, num_experts]`` router logits. Promoted to FP32.
        topk_indices: ``[T_local, topk]`` selected expert ids (long).
        num_experts: N.
        topk: k.
        ep_group: expert-parallel process group whose tokens form the
            statistic. ``None`` means "this rank alone" (the EP1 reference).
        scale_for_dp_average: multiply the returned loss by ``ep_size`` so that
            downstream dense-DP gradient averaging yields ``mean_G L_aux^G``.
        return_stats: also return detached telemetry.

    Returns:
        ``(loss, stats)`` where ``loss`` is a scalar tensor carrying router
        gradient and ``stats`` is a dict of detached tensors/floats (or None).
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D [T, N], got {tuple(logits.shape)}")
    if logits.shape[-1] != num_experts:
        raise ValueError(
            f"logits last dim {logits.shape[-1]} != num_experts {num_experts}")
    if topk_indices.shape[0] != logits.shape[0]:
        raise ValueError("logits and topk_indices disagree on token count")
    if topk_indices.shape[-1] != topk:
        raise ValueError(
            f"topk_indices last dim {topk_indices.shape[-1]} != topk {topk}")

    fp32_logits = logits.float()
    n_tokens_local = fp32_logits.shape[0]

    # ---- differentiable part: sum_t p_{t,i} over the EP group ----------------
    probs = torch.softmax(fp32_logits, dim=-1)          # [T_local, N]
    prob_sum_local = probs.sum(dim=0)                   # [N], keeps grad
    prob_sum_group = _DifferentiableAllReduceSum.apply(prob_sum_local, ep_group)

    # ---- detached part: counts and token total ------------------------------
    with torch.no_grad():
        counts_local = torch.zeros(
            num_experts, dtype=torch.float32, device=logits.device)
        counts_local.scatter_add_(
            0,
            topk_indices.reshape(-1).long(),
            torch.ones(topk_indices.numel(), dtype=torch.float32,
                       device=logits.device),
        )
        counts_group = _all_reduce_sum_no_grad(counts_local, ep_group)

        tokens_local = torch.tensor(
            [float(n_tokens_local)], dtype=torch.float32, device=logits.device)
        tokens_group = _all_reduce_sum_no_grad(tokens_local, ep_group)
        total_tokens = tokens_group.item()

    if total_tokens <= 0:
        raise RuntimeError("EP group observed zero tokens for the aux statistic")

    # f_i = n_i / (k*T)   ;   P_i = S_i / T
    f_i = counts_group / (float(topk) * total_tokens)          # detached
    p_i = prob_sum_group / total_tokens                        # differentiable

    loss = float(num_experts) * torch.sum(f_i * p_i)

    ep_size = _ep_world_size(ep_group)
    reported = loss.detach()
    if scale_for_dp_average and ep_size > 1:
        loss = loss * float(ep_size)

    if not return_stats:
        return loss, None

    with torch.no_grad():
        mean_count = counts_group.mean()
        std_count = counts_group.std(unbiased=False)
        cv = (std_count / mean_count) if mean_count > 0 else torch.zeros_like(mean_count)
        entropy = -(p_i.detach() * torch.log(p_i.detach().clamp_min(1e-12))).sum()
        stats = {
            "aux_loss": reported,                       # unscaled, group-level
            "ep_size": ep_size,
            "tokens_in_statistic": total_tokens,
            "expert_counts": counts_group,              # [N]
            "expert_prob_sum": prob_sum_group.detach(),  # [N]
            "load_cv": cv,
            "load_min_over_mean": (counts_group.min() / mean_count)
            if mean_count > 0 else torch.zeros_like(mean_count),
            "load_max_over_mean": (counts_group.max() / mean_count)
            if mean_count > 0 else torch.zeros_like(mean_count),
            "zero_load_experts": int((counts_group == 0).sum().item()),
            "router_prob_entropy": entropy,
        }
    return loss, stats


def router_z_loss(logits: torch.Tensor, return_stats: bool = True):
    """L_z = mean_t (logsumexp_i z_{t,i})^2, computed in FP32.

    Rank-local by construction; see module docstring.
    """
    fp32_logits = logits.float()
    lse = torch.logsumexp(fp32_logits, dim=-1)
    loss = torch.mean(lse ** 2)
    if not return_stats:
        return loss, None
    with torch.no_grad():
        stats = {
            "z_loss": loss.detach(),
            "max_abs_logit": fp32_logits.abs().max(),
            "mean_logsumexp": lse.mean(),
        }
    return loss, stats


def expert_load_metrics(
    topk_indices: torch.Tensor,
    num_experts: int,
    ep_group=None,
) -> dict:
    """Detached per-layer expert assignment telemetry (design doc section 6.7)."""
    with torch.no_grad():
        counts = torch.zeros(num_experts, dtype=torch.float32,
                             device=topk_indices.device)
        counts.scatter_add_(
            0,
            topk_indices.reshape(-1).long(),
            torch.ones(topk_indices.numel(), dtype=torch.float32,
                       device=topk_indices.device),
        )
        counts = _all_reduce_sum_no_grad(counts, ep_group)
        mean = counts.mean()
        return {
            "counts": counts,
            "mean": mean,
            "std": counts.std(unbiased=False),
            "cv": (counts.std(unbiased=False) / mean) if mean > 0 else mean * 0,
            "min_over_mean": (counts.min() / mean) if mean > 0 else mean * 0,
            "max_over_mean": (counts.max() / mean) if mean > 0 else mean * 0,
            "zero_load_experts": int((counts == 0).sum().item()),
        }

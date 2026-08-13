"""Pure tensor contracts frozen by ``rfull_v0_1.source.json``.

These helpers intentionally contain no Megatron imports.  The distributed MCore
adapter builds on them, while unit tests can compare the equations directly.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


DEFAULT_LIMIT = 10.0


def limited_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    limit: float = DEFAULT_LIMIT,
) -> torch.Tensor:
    """Apply the frozen limited-SwiGLU equation.

    ``SiLU(min(gate, limit)) * clamp(up, -limit, limit)``

    The gate branch intentionally has no lower clamp.  This detail differs from
    applying a symmetric clamp to both halves and is covered by parity tests.
    """

    if gate.shape != up.shape:
        raise ValueError(f"gate/up shapes differ: {gate.shape} != {up.shape}")
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    limited_gate = torch.clamp_max(gate, limit)
    limited_up = torch.clamp(up, min=-limit, max=limit)
    return F.silu(limited_gate) * limited_up


def limited_swiglu_from_fused(
    fused_gate_up: torch.Tensor,
    *,
    limit: float = DEFAULT_LIMIT,
) -> torch.Tensor:
    """Split a fused ``[gate, up]`` projection and apply limited-SwiGLU."""

    if fused_gate_up.shape[-1] % 2:
        raise ValueError(
            "limited-SwiGLU requires an even fused projection width, got "
            f"{fused_gate_up.shape[-1]}"
        )
    gate, up = torch.chunk(fused_gate_up, 2, dim=-1)
    return limited_swiglu(gate, up, limit=limit)


def selected_topk_softmax(
    logits: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return Top-K indices and a softmax over only the selected FP32 logits.

    This is the main routing path.  The full-expert softmax used by the auxiliary
    loss is deliberately *not* returned as the routing weight.
    """

    if logits.ndim < 2:
        raise ValueError(f"router logits must have rank >= 2, got {logits.shape}")
    num_experts = logits.shape[-1]
    if not 0 < topk <= num_experts:
        raise ValueError(f"topk must be in [1, {num_experts}], got {topk}")
    logits_fp32 = logits.float()
    selected_logits, selected_indices = torch.topk(logits_fp32, k=topk, dim=-1)
    selected_probabilities = torch.softmax(selected_logits, dim=-1)
    return selected_probabilities, selected_indices


def load_balancing_loss_from_statistics(
    expert_counts: torch.Tensor,
    probability_sums: torch.Tensor,
    *,
    token_count: int | torch.Tensor,
    topk: int,
    coefficient: float = 1.0,
) -> torch.Tensor:
    """Compute the frozen global load-balancing loss from sufficient statistics.

    With ``N`` experts and ``T`` tokens, the definition is

    ``coefficient * N * sum_i ((n_i / (topk*T)) * (sum_t p_ti / T))``.

    ``probability_sums`` remains differentiable; ``expert_counts`` is treated as
    a non-differentiable routing statistic.  Callers performing expert-parallel
    training must aggregate both vectors over the EP group before invoking this
    function.
    """

    if expert_counts.ndim != 1 or probability_sums.ndim != 1:
        raise ValueError("expert_counts and probability_sums must be rank-1")
    if expert_counts.shape != probability_sums.shape:
        raise ValueError(
            "statistics shapes differ: "
            f"{expert_counts.shape} != {probability_sums.shape}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    if isinstance(token_count, torch.Tensor):
        if token_count.numel() != 1:
            raise ValueError("token_count tensor must be scalar")
        tokens = token_count.to(device=probability_sums.device, dtype=torch.float32)
        if int(token_count.detach().cpu().item()) <= 0:
            raise ValueError(f"token_count must be positive, got {token_count}")
    else:
        if token_count <= 0:
            raise ValueError(f"token_count must be positive, got {token_count}")
        tokens = torch.tensor(
            float(token_count), device=probability_sums.device, dtype=torch.float32
        )

    counts = expert_counts.to(device=probability_sums.device, dtype=torch.float32)
    sums = probability_sums.float()
    num_experts = counts.numel()
    fractions = counts / (float(topk) * tokens)
    mean_probabilities = sums / tokens
    return float(coefficient) * float(num_experts) * torch.sum(
        fractions * mean_probabilities
    )


def z_loss_from_statistics(
    squared_log_partition_sum: torch.Tensor,
    *,
    token_count: int | torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    """Compute mean squared log-partition z-loss from an aggregated sum."""

    if squared_log_partition_sum.numel() != 1:
        raise ValueError("squared_log_partition_sum must be scalar")
    if isinstance(token_count, torch.Tensor):
        if token_count.numel() != 1:
            raise ValueError("token_count tensor must be scalar")
        if int(token_count.detach().cpu().item()) <= 0:
            raise ValueError(f"token_count must be positive, got {token_count}")
        tokens = token_count.to(
            device=squared_log_partition_sum.device, dtype=torch.float32
        )
    else:
        if token_count <= 0:
            raise ValueError(f"token_count must be positive, got {token_count}")
        tokens = torch.tensor(
            float(token_count),
            device=squared_log_partition_sum.device,
            dtype=torch.float32,
        )
    return float(coefficient) * squared_log_partition_sum.float() / tokens

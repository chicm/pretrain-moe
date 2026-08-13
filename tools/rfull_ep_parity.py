"""Distributed EP=8 scalar and gradient parity test for the R-Full router.

Run with exactly eight processes.  On GPU nodes use torchrun with one process per
GPU; the same program can run under Gloo on CPU for a transport-independent
check.  It compares the custom EP-global aux/z-loss attachment against one
single-process global reference and explicitly models the dense-DP gradient
average performed by Megatron DDP.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    clear_aux_losses_tracker,
    reduce_aux_losses_tracker_across_ranks,
)
from megatron.core.transformer.transformer_config import TransformerConfig

from rfull_moe.mcore import RFullTopKRouter
from rfull_moe.pinned_mcore import verify_pinned_mcore_sources
from rfull_moe.semantics import (
    load_balancing_loss_from_statistics,
    z_loss_from_statistics,
)


WORLD_SIZE = 8
TOKENS_PER_RANK = 4
HIDDEN_SIZE = 2048
NUM_EXPERTS = 96
TOPK = 6
AUX_COEFFICIENT = 1.0e-3
Z_COEFFICIENT = 1.0e-4


def _config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=48,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=32,
        num_query_groups=4,
        kv_channels=128,
        ffn_hidden_size=5504,
        moe_ffn_hidden_size=896,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_shared_expert_intermediate_size=896,
        moe_grouped_gemm=True,
        moe_use_legacy_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_router_dtype="fp32",
        moe_router_pre_softmax=False,
        moe_router_score_function="softmax",
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=AUX_COEFFICIENT,
        moe_z_loss_coeff=Z_COEFFICIENT,
        moe_expert_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False,
        moe_shared_expert_overlap=False,
        normalization="RMSNorm",
        layernorm_epsilon=1.0e-6,
        qk_layernorm=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=WORLD_SIZE,
        expert_tensor_parallel_size=1,
        moe_layer_freq=[0, 0] + [1] * 46,
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        moe_router_enable_expert_bias=False,
        moe_enable_deepep=False,
        moe_permute_fusion=False,
        calculate_per_token_loss=False,
    )


def _fixed_weight(device: torch.device) -> torch.Tensor:
    index = torch.arange(NUM_EXPERTS * HIDDEN_SIZE, device=device, dtype=torch.float32)
    values = 0.009 * torch.sin(index * 0.017) + 0.004 * torch.cos(index * 0.011)
    return values.reshape(NUM_EXPERTS, HIDDEN_SIZE).to(torch.bfloat16)


def _fixed_hidden(rank: int, device: torch.device) -> torch.Tensor:
    begin = rank * TOKENS_PER_RANK * HIDDEN_SIZE
    index = torch.arange(
        begin,
        begin + TOKENS_PER_RANK * HIDDEN_SIZE,
        device=device,
        dtype=torch.float32,
    )
    values = 0.7 * torch.sin(index * 0.013) + 0.2 * torch.cos(index * 0.007)
    return values.reshape(TOKENS_PER_RANK, HIDDEN_SIZE).to(torch.bfloat16)


def _global_reference(
    weight_value: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.cat(
        [_fixed_hidden(rank, device) for rank in range(WORLD_SIZE)], dim=0
    ).detach()
    hidden.requires_grad_(True)
    weight = weight_value.detach().clone().requires_grad_(True)
    logits = F.linear(hidden.float(), weight.float())
    full_probabilities = torch.softmax(logits, dim=-1)
    top_indices = torch.topk(logits, k=TOPK, dim=-1).indices
    counts = torch.bincount(
        top_indices.reshape(-1), minlength=NUM_EXPERTS
    ).to(torch.float32)
    aux = load_balancing_loss_from_statistics(
        counts,
        full_probabilities.sum(dim=0),
        token_count=hidden.shape[0],
        topk=TOPK,
        coefficient=AUX_COEFFICIENT,
    )
    z_loss = z_loss_from_statistics(
        torch.logsumexp(logits, dim=-1).square().sum(),
        token_count=hidden.shape[0],
        coefficient=Z_COEFFICIENT,
    )
    (aux + z_loss).backward()
    assert hidden.grad is not None and weight.grad is not None
    return aux.detach(), z_loss.detach(), hidden.grad.detach(), weight.grad.detach(), logits.detach()


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    max_abs = float((actual_fp32 - expected_fp32).abs().max().cpu())
    torch.testing.assert_close(
        actual_fp32,
        expected_fp32,
        rtol=4.0e-2,
        atol=2.0e-6,
        msg=f"{name}: actual={actual_fp32} expected={expected_fp32}",
    )
    return max_abs


def main() -> None:
    verify_pinned_mcore_sources()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"R-Full parity requires world_size=8, got {world_size}")

    use_cuda = torch.cuda.is_available()
    explicit_init_method = os.environ.get("RFULL_INIT_METHOD")
    common_init = {
        "timeout": timedelta(minutes=10),
    }
    if explicit_init_method:
        common_init.update(
            init_method=explicit_init_method,
            rank=rank,
            world_size=world_size,
        )
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=device,
            **common_init,
        )
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo", **common_init)

    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=WORLD_SIZE,
            expert_tensor_parallel_size=1,
            order="tp-cp-ep-dp-pp",
            create_gloo_process_groups=use_cuda,
        )
        ep_group = parallel_state.get_expert_model_parallel_group()
        if dist.get_world_size(ep_group) != WORLD_SIZE:
            raise RuntimeError("expert-parallel group is not EP=8")

        clear_aux_losses_tracker()
        MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1.0, device=device))
        router = RFullTopKRouter(_config()).to(device)
        router.layer_number = 3
        weight_value = _fixed_weight(device)
        with torch.no_grad():
            router.weight.copy_(weight_value)

        hidden = _fixed_hidden(rank, device).detach().requires_grad_(True)
        # Calling gating() is equivalent on GPU, but pinned MCore's gating helper
        # unconditionally asks CUDA for a current device.  Spell out its audited
        # FP32 linear operation so this exact parity test also runs under Gloo.
        logits = F.linear(hidden.float(), router.weight.float())
        scores, routing_map = router.routing(logits)
        if not torch.all(routing_map.sum(dim=-1) == TOPK):
            raise AssertionError("router did not select exactly Top-6")
        # The main path contributes zero; both auxiliary losses arrive solely via
        # MoEAuxLossAutoScaler, exactly as they do alongside a real main loss.
        (scores.float().sum() * 0.0).backward()
        if hidden.grad is None or router.weight.grad is None:
            raise AssertionError("router parity backward produced a missing gradient")

        reference_aux, reference_z, reference_hidden_grad, reference_weight_grad, _ = (
            _global_reference(weight_value, device)
        )

        # Megatron DDP averages replicated router/common-model parameter gradients.
        averaged_weight_grad = router.weight.grad.detach().clone()
        dist.all_reduce(averaged_weight_grad, op=dist.ReduceOp.SUM)
        averaged_weight_grad.div_(WORLD_SIZE)

        hidden_slice = reference_hidden_grad[
            rank * TOKENS_PER_RANK : (rank + 1) * TOKENS_PER_RANK
        ]
        # The adapter multiplies local activation gradients by EP size so the
        # subsequent DDP parameter-gradient average yields the global reference.
        hidden_max_abs = _assert_close(
            "hidden gradient / EP",
            hidden.grad.detach() / WORLD_SIZE,
            hidden_slice,
        )
        weight_max_abs = _assert_close(
            "DDP-averaged router weight gradient",
            averaged_weight_grad,
            reference_weight_grad,
        )

        tracker = parallel_state.get_moe_layer_wise_logging_tracker()
        for metric_name in ("load_balancing_loss", "z_loss"):
            if tracker[metric_name]["avg_group"] is not ep_group:
                raise AssertionError(f"{metric_name} is not configured for EP averaging")
        # Gloo has no ReduceOp.AVG.  The pre-reduction values are already the
        # replicated global scalar, so CPU can validate them directly; ROCm/NCCL
        # additionally qualifies MCore's real tracker reduction path.
        if use_cuda:
            reduce_aux_losses_tracker_across_ranks()
        # MCore's tracker convention stores both metrics before their training
        # coefficients; scale them back when comparing to the attached losses.
        tracked_aux = tracker["load_balancing_loss"]["values"][2] * AUX_COEFFICIENT
        tracked_z = tracker["z_loss"]["values"][2] * Z_COEFFICIENT
        scalar_aux_abs = _assert_close("tracked scaled aux scalar", tracked_aux, reference_aux)
        scalar_z_abs = _assert_close("tracked scaled z scalar", tracked_z, reference_z)

        marker = {
            "marker": "RFULL_EP_PARITY_PASS",
            "rank": rank,
            "world_size": world_size,
            "device": str(device),
            "ep_world_size": dist.get_world_size(ep_group),
            "tracker_reduction": "nccl_avg" if use_cuda else "gloo_pre_reduce",
            "hidden_grad_max_abs": hidden_max_abs,
            "router_weight_grad_max_abs": weight_max_abs,
            "aux_scalar_abs": scalar_aux_abs,
            "z_scalar_abs": scalar_z_abs,
            "finite": bool(
                torch.isfinite(hidden.grad).all()
                and torch.isfinite(averaged_weight_grad).all()
                and torch.isfinite(tracked_aux)
                and torch.isfinite(tracked_z)
            ),
        }
        print(json.dumps(marker, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

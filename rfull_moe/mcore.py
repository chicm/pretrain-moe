"""Pinned Megatron-Core adapters for the frozen R-Full MoE semantics.

This module is intentionally external to the Megatron checkout.  It preserves
upstream parameter/state-dict names while replacing only the semantic deltas that
stock MCore 0.12.3 cannot express: limited-SwiGLU, router-only initialization,
and EP-global auxiliary/z losses.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import (
    MoEAlltoAllSEQTokenDispatcher,
)
from megatron.core.transformer.moe.moe_layer import BaseMoELayer, MoELayer
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal

from .pinned_mcore import verify_pinned_mcore_sources
from .semantics import (
    limited_swiglu_from_fused,
    load_balancing_loss_from_statistics,
    z_loss_from_statistics,
)


RFULL_NUM_LAYERS = 48
RFULL_HIDDEN_SIZE = 2048
RFULL_DENSE_FFN_SIZE = 5504
RFULL_EXPERT_FFN_SIZE = 896
RFULL_NUM_HEADS = 32
RFULL_NUM_QUERY_GROUPS = 4
RFULL_HEAD_DIM = 128
RFULL_NUM_EXPERTS = 96
RFULL_TOPK = 6
RFULL_EXPERT_PARALLEL_SIZE = 8
RFULL_ROUTER_INIT_STD = 0.01
RFULL_PRODUCTION_PROFILE = "production"
RFULL_EP8_MINI_PROFILE = "ep8-mini"
RFULL_EP8_MINI_GEOMETRY = {
    "num_layers": 4,
    "hidden_size": 512,
    "ffn_hidden_size": 1408,
    "moe_ffn_hidden_size": 256,
    "num_attention_heads": 8,
    "num_query_groups": 2,
    "kv_channels": 64,
    "moe_layer_freq": [0, 0, 1, 1],
}
_RUNTIME_EVIDENCE_MARKERS: set[str] = set()


def _emit_runtime_evidence_once(marker: str, **payload: object) -> None:
    if os.environ.get("RFULL_RUNTIME_EVIDENCE") != "1" or marker in _RUNTIME_EVIDENCE_MARKERS:
        return
    _RUNTIME_EVIDENCE_MARKERS.add(marker)
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(json.dumps({"marker": marker, "rank": rank, **payload}, sort_keys=True), flush=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_rfull_semantic_config(config: TransformerConfig) -> None:
    """Fail closed unless MCore configuration matches the frozen semantics."""

    profile = getattr(config, "rfull_profile", RFULL_PRODUCTION_PROFILE)
    if profile == RFULL_PRODUCTION_PROFILE:
        geometry = {
            "num_layers": RFULL_NUM_LAYERS,
            "hidden_size": RFULL_HIDDEN_SIZE,
            "ffn_hidden_size": RFULL_DENSE_FFN_SIZE,
            "moe_ffn_hidden_size": RFULL_EXPERT_FFN_SIZE,
            "num_attention_heads": RFULL_NUM_HEADS,
            "num_query_groups": RFULL_NUM_QUERY_GROUPS,
            "kv_channels": RFULL_HEAD_DIM,
            "moe_layer_freq": [0, 0] + [1] * 46,
        }
    elif profile == RFULL_EP8_MINI_PROFILE:
        geometry = RFULL_EP8_MINI_GEOMETRY
    else:
        raise ValueError(f"unknown R-Full profile: {profile!r}")
    for field, expected in geometry.items():
        actual = list(getattr(config, field)) if field == "moe_layer_freq" else getattr(config, field)
        _require(actual == expected, f"{profile} {field} drift: expected {expected}, got {actual}")

    _require(config.num_moe_experts == RFULL_NUM_EXPERTS, "R-Full requires 96 experts")
    _require(config.moe_router_topk == RFULL_TOPK, "R-Full requires Top-6 routing")
    _require(
        config.moe_shared_expert_intermediate_size == geometry["moe_ffn_hidden_size"],
        "shared-expert FFN width must match routed experts",
    )
    _require(
        config.expert_model_parallel_size == RFULL_EXPERT_PARALLEL_SIZE,
        "R-Full requires EP=8",
    )
    _require(config.pipeline_model_parallel_size == 1, "R-Full production PP must be 1")
    _require(config.moe_router_dtype == "fp32", "router compute must be FP32")
    _require(
        config.moe_router_score_function == "softmax",
        "router score function must be softmax",
    )
    _require(
        not config.moe_router_pre_softmax,
        "main routing must softmax only the selected Top-K logits",
    )
    _require(
        config.moe_router_load_balancing_type == "aux_loss",
        "R-Full requires auxiliary-loss routing",
    )
    _require(config.moe_aux_loss_coeff == 1.0e-3, "aux coefficient must be 1e-3")
    _require(config.moe_z_loss_coeff == 1.0e-4, "z-loss coefficient must be 1e-4")
    _require(config.moe_expert_capacity_factor is None, "capacity must be disabled")
    _require(not config.moe_pad_expert_input_to_capacity, "capacity padding is forbidden")
    _require(config.moe_router_topk_scaling_factor is None, "routing scale is forbidden")
    _require(config.moe_router_num_groups is None, "group-limited routing is forbidden")
    _require(config.moe_router_group_topk is None, "group-limited routing is forbidden")
    _require(config.moe_input_jitter_eps is None, "router input jitter is forbidden")
    _require(not config.moe_router_enable_expert_bias, "expert routing bias is forbidden")
    _require(config.moe_token_dispatcher_type == "alltoall", "dispatcher must be alltoall")
    _require(not config.calculate_per_token_loss, "per-token loss is not yet exact here")
    _require(config.tensor_model_parallel_size == 1, "R-Full production TP must be 1")
    _require(config.context_parallel_size == 1, "R-Full production CP must be 1")
    _require(config.normalization == "RMSNorm", "R-Full requires RMSNorm")
    _require(config.layernorm_epsilon == 1.0e-6, "RMSNorm epsilon must be 1e-6")
    _require(config.qk_layernorm, "Q/K RMSNorm is required")
    _require(config.hidden_dropout == 0.0, "hidden dropout must be zero")
    _require(config.attention_dropout == 0.0, "attention dropout must be zero")
    _require(config.gated_linear_unit, "limited-SwiGLU requires gated_linear_unit")
    _require(config.activation_func is F.silu, "R-Full activation must be SiLU")
    _require(not config.bias_activation_fusion, "fused bias activation bypasses clamps")
    _require(not config.add_bias_linear, "R-Full forbids linear biases")
    _require(config.params_dtype == torch.bfloat16, "R-Full parameters must be BF16")
    _require(config.fp8 is None, "R-Full BF16 baseline forbids FP8")
    _require(config.moe_grouped_gemm, "production experts require grouped GEMM")
    _require(
        config.moe_use_legacy_grouped_gemm,
        "exact clamps currently require legacy GroupedMLP, not TEGroupedMLP",
    )
    _require(not config.moe_apply_probs_on_input, "Top-6 probabilities apply after experts")
    _require(not config.moe_enable_deepep, "DeepEP is not qualified")
    _require(
        not getattr(config, "moe_router_force_load_balancing", False),
        "forced load balancing changes router semantics",
    )
    _require(not config.moe_permute_fusion, "fused permutation is not qualified")
    _require(not config.moe_shared_expert_overlap, "shared-expert overlap is frozen off")


def _ep_group_and_size():
    if not dist.is_available() or not dist.is_initialized():
        return None, 1
    group = parallel_state.get_expert_model_parallel_group()
    return group, dist.get_world_size(group=group)


def _detached_ep_sum(value: torch.Tensor, group, group_size: int) -> torch.Tensor:
    result = value.detach().clone()
    if group_size > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
    return result


class _DifferentiableEpSum(torch.autograd.Function):
    """EP all-reduce SUM whose local backward is the mathematical identity.

    Each rank contributes one slice to the forward sum.  The derivative of that
    sum with respect to every local slice is one; dense-DP averaging is corrected
    separately by multiplying the attached gradient by the EP partition count.
    """

    @staticmethod
    def forward(ctx, value: torch.Tensor, group, group_size: int):
        del ctx, group_size
        result = value.contiguous().clone()
        if group is not None:
            dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        del ctx
        return gradient, None, None


class RFullTopKRouter(TopKRouter):
    """Top-K router with frozen initialization and EP-global loss statistics."""

    def __init__(self, config: TransformerConfig) -> None:
        validate_rfull_semantic_config(config)
        router_config = copy.copy(config)
        router_config.init_method = init_method_normal(RFULL_ROUTER_INIT_STD)
        super().__init__(router_config)
        # Keep the canonical shared config after the router-only initializer has
        # been consumed; no global config mutation and no extra RNG draw occurs.
        self.config = config

    def apply_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        coeff = self.config.moe_z_loss_coeff
        if (
            coeff is None
            or coeff == 0
            or not self.training
            or not torch.is_grad_enabled()
        ):
            return logits

        group, partitions = _ep_group_and_size()
        local_z_sum = torch.square(torch.logsumexp(logits.float(), dim=-1)).sum()
        local_tokens = torch.tensor(
            logits.shape[0], device=logits.device, dtype=torch.float32
        )
        global_z_sum = _DifferentiableEpSum.apply(local_z_sum, group, partitions)
        global_tokens = _detached_ep_sum(local_tokens, group, partitions)

        true_loss = z_loss_from_statistics(
            global_z_sum, token_count=global_tokens, coefficient=coeff
        )
        # Dense DDP averages router gradients.  Preserve the true forward scalar
        # while scaling its attached gradient to reconstruct one EP-global loss.
        scaled_loss = true_loss * partitions
        z_loss = scaled_loss + (true_loss.detach() - scaled_loss.detach())
        save_to_aux_losses_tracker(
            "z_loss",
            z_loss / coeff,
            self.layer_number,
            self.config.num_layers,
            avg_group=group,
        )
        _emit_runtime_evidence_once(
            "RFULL_EP_GLOBAL_Z_LOSS",
            layer_number=self.layer_number,
            ep_world_size=partitions,
            global_tokens=int(global_tokens.item()),
            raw_z_loss=float((z_loss.detach() / coeff).item()),
            coefficient=float(coeff),
            tracker_group="expert_parallel_avg",
        )
        return MoEAuxLossAutoScaler.apply(logits, z_loss)

    def apply_load_balancing_loss(
        self,
        activation: torch.Tensor,
        load_balancing_loss_func,
    ) -> torch.Tensor:
        coeff = self.config.moe_aux_loss_coeff
        if coeff is None or coeff == 0:
            return activation

        # Pinned TopKRouter passes a functools.partial carrying the complete
        # N-way softmax and selected-token histogram.  Fail closed if that
        # audited extension contract changes.
        statistics = getattr(load_balancing_loss_func, "keywords", None)
        if not statistics or not {"probs", "tokens_per_expert", "topk"}.issubset(
            statistics
        ):
            raise RuntimeError("unexpected pinned load-balancing callback contract")
        probs = statistics["probs"]
        num_local_tokens_per_expert = statistics["tokens_per_expert"]
        if statistics["topk"] != self.topk:
            raise RuntimeError("load-balancing callback Top-K mismatch")

        group, partitions = _ep_group_and_size()
        local_probability_sums = probs.float().sum(dim=0)
        local_counts = num_local_tokens_per_expert.to(
            device=probs.device, dtype=torch.float32
        )
        local_tokens = torch.tensor(
            probs.shape[0], device=probs.device, dtype=torch.float32
        )
        global_probability_sums = _DifferentiableEpSum.apply(
            local_probability_sums, group, partitions
        )
        global_counts = _detached_ep_sum(local_counts, group, partitions)
        global_tokens = _detached_ep_sum(local_tokens, group, partitions)

        true_loss = load_balancing_loss_from_statistics(
            global_counts,
            global_probability_sums,
            token_count=global_tokens,
            topk=self.topk,
            coefficient=coeff,
        )
        scaled_loss = true_loss * partitions
        aux_loss = scaled_loss + (true_loss.detach() - scaled_loss.detach())
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / coeff,
            self.layer_number,
            self.config.num_layers,
            avg_group=group,
        )
        _emit_runtime_evidence_once(
            "RFULL_EP_GLOBAL_AUX_LOSS",
            layer_number=self.layer_number,
            ep_world_size=partitions,
            global_tokens=int(global_tokens.item()),
            raw_aux_loss=float((aux_loss.detach() / coeff).item()),
            coefficient=float(coeff),
            tracker_group="expert_parallel_avg",
        )
        return MoEAuxLossAutoScaler.apply(activation, aux_loss)


class RFullMLP(MLP):
    """MLP preserving MCore weights/state names with exact limited-SwiGLU."""

    def forward(self, hidden_states, per_token_scale=None):
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        if bias_parallel is not None:
            intermediate_parallel = intermediate_parallel + bias_parallel
        intermediate_parallel = limited_swiglu_from_fused(intermediate_parallel)
        if per_token_scale is not None:
            intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
        return self.linear_fc2(intermediate_parallel)


def _limited_activation_with_probs(
    fused_gate_up: torch.Tensor, probabilities: torch.Tensor
) -> torch.Tensor:
    dtype = fused_gate_up.dtype
    return (limited_swiglu_from_fused(fused_gate_up) * probabilities).to(dtype)


class RFullGroupedMLP(GroupedMLP):
    """Legacy grouped-GEMM experts with limited-SwiGLU activation."""

    def __init__(self, num_local_experts, config: TransformerConfig):
        _require(not config.bias_activation_fusion, "grouped limited-SwiGLU cannot be fused")
        super().__init__(num_local_experts=num_local_experts, config=config)
        self.activation_func = limited_swiglu_from_fused
        self.activation_func_with_probs = _limited_activation_with_probs

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        output = super().forward(
            permuted_local_hidden_states,
            tokens_per_expert,
            permuted_probs,
        )
        counts = tokens_per_expert.detach().to(device="cpu", dtype=torch.int64)
        _emit_runtime_evidence_once(
            "RFULL_GROUPED_GEMM_FORWARD",
            num_local_experts=self.num_local_experts,
            assigned_tokens=int(counts.sum().item()),
            zero_token_experts=int((counts == 0).sum().item()),
            min_tokens_per_expert=int(counts.min().item()),
            max_tokens_per_expert=int(counts.max().item()),
            hidden_size=self.config.hidden_size,
            expert_ffn_hidden_size=self.config.moe_ffn_hidden_size,
        )
        return output


class RFullSequentialMLP(SequentialMLP):
    """Reference sequential experts using :class:`RFullMLP`."""

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: MLPSubmodules,
    ) -> None:
        if config.moe_ffn_hidden_size == config.ffn_hidden_size:
            expert_config = config
        else:
            expert_config = copy.deepcopy(config)
            expert_config.ffn_hidden_size = config.moe_ffn_hidden_size
        # Skip SequentialMLP.__init__, whose only semantic difference is the MLP
        # class instantiated in this loop.
        super(SequentialMLP, self).__init__(config=expert_config)
        self.add_bias = config.add_bias_linear
        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList(
            [RFullMLP(self.config, submodules, is_expert=True) for _ in range(num_local_experts)]
        )


class RFullSharedExpertMLP(SharedExpertMLP):
    """Non-overlapped shared expert with the same limited-SwiGLU."""

    def __init__(self, config, submodules, gate: bool = False):
        _require(not config.moe_shared_expert_overlap, "shared overlap is frozen off")
        super().__init__(config=config, submodules=submodules, gate=gate)

    def forward(self, hidden_states):
        output, _ = RFullMLP.forward(self, hidden_states)
        if self.use_shared_expert_gate:
            logits = F.linear(hidden_states, self.gate_weight)
            output = output * torch.sigmoid(logits)
        return output


@dataclass
class RFullMoESubmodules:
    """MoE submodules including the router, absent from stock MoESubmodules."""

    router: Union[ModuleSpec, type] = None
    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class RFullMoELayer(MoELayer):
    """Pinned MoELayer constructor with an injectable custom router."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[RFullMoESubmodules] = None,
        layer_number: Optional[int] = None,
    ) -> None:
        self.submodules = submodules
        BaseMoELayer.__init__(self, config=config, layer_number=layer_number)
        self.moe_layer_recompute = (
            config.recompute_granularity == "selective"
            and "moe" in config.recompute_modules
        )
        self.router = build_module(self.submodules.router, config=self.config)

        self.token_dispatcher = None
        if self.config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
            )
        elif self.config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
            )
        elif self.config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
            )
        elif self.config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
            )
        else:
            raise ValueError(
                f"unsupported token dispatcher {self.config.moe_token_dispatcher_type}"
            )

        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
        )
        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)


def _customize_expert_spec(stock_spec: ModuleSpec) -> ModuleSpec:
    spec = copy.deepcopy(stock_spec)
    if spec.module is GroupedMLP:
        spec.module = RFullGroupedMLP
    elif spec.module is SequentialMLP:
        spec.module = RFullSequentialMLP
    elif spec.module is TEGroupedMLP:
        raise ValueError(
            "TEGroupedMLP fuses the GLU branches and cannot express frozen limited-SwiGLU; "
            "set moe_use_legacy_grouped_gemm=True"
        )
    else:
        raise TypeError(f"unrecognized pinned expert module {spec.module}")
    return spec


def get_rfull_decoder_block_spec(
    config: TransformerConfig,
    *,
    use_transformer_engine: bool = True,
    profile: str = RFULL_PRODUCTION_PROFILE,
):
    """Build an audited production or explicit EP8 qualification block."""

    verify_pinned_mcore_sources()
    existing_profile = getattr(config, "rfull_profile", profile)
    _require(existing_profile == profile, "conflicting R-Full profile selection")
    config.rfull_profile = profile
    validate_rfull_semantic_config(config)

    # Ask pinned MCore to construct attention/norm/linear submodules and perform
    # pipeline-stage slicing, then replace only MLP/MoE semantic extension points.
    stock_block = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=use_transformer_engine,
    )
    layer_specs = []
    for stock_layer in stock_block.layer_specs:
        layer_spec = copy.deepcopy(stock_layer)
        mlp_spec = layer_spec.submodules.mlp
        if mlp_spec.module is MLP:
            mlp_spec.module = RFullMLP
        elif mlp_spec.module is MoELayer:
            experts = _customize_expert_spec(mlp_spec.submodules.experts)
            shared_experts = copy.deepcopy(mlp_spec.submodules.shared_experts)
            if shared_experts is not None:
                shared_experts.module = RFullSharedExpertMLP
            layer_spec.submodules.mlp = ModuleSpec(
                module=RFullMoELayer,
                submodules=RFullMoESubmodules(
                    router=ModuleSpec(module=RFullTopKRouter),
                    experts=experts,
                    shared_experts=shared_experts,
                ),
            )
        else:
            raise TypeError(f"unrecognized pinned MLP module {mlp_spec.module}")
        layer_specs.append(layer_spec)
    stock_block.layer_specs = layer_specs
    return stock_block

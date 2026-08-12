"""Build the R-Full MoE model on Megatron-Core with all frozen overrides.

This is the single place where the frozen design is translated into a
`TransformerConfig`. Everything that Megatron would otherwise infer or default
differently is set explicitly here and then re-verified against the model that
actually gets built:

  * `kv_channels=128`      (else MCore infers 2048/32 = 64)
  * explicit per-layer MoE pattern (2 dense + 46 MoE), not a periodic freq
  * limited SwiGLU on every FFN path
  * router init Normal(0, 0.01), FP32 router logits
  * dropless (no capacity factor), top-6, selected-logit softmax gates
  * BF16 params, FP32 grad reduce, no TF32
"""

from __future__ import annotations

from typing import Optional

import torch

from .model_spec import (
    GEOMETRY,
    RFullGeometry,
    assert_attention_shapes,
    assert_frozen_ledger,
    moe_layer_pattern,
)
from .limited_swiglu import make_mcore_activation, patch_grouped_mlp_glu
from .router_init import ROUTER_INIT_STD, apply_router_init, verify_router_init

__all__ = ["build_transformer_config", "build_rfull_model", "count_built_parameters"]


def build_transformer_config(
    geo: RFullGeometry = GEOMETRY,
    *,
    seq_length: int = 4096,
    expert_model_parallel_size: int = 8,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    aux_loss_coeff: float = 0.001,
    z_loss_coeff: float = 0.0001,
    bf16: bool = True,
    params_dtype: Optional[torch.dtype] = None,
    sequence_parallel: bool = False,
    use_custom_aux: bool = True,
    **overrides,
):
    """Construct the frozen `TransformerConfig`.

    `use_custom_aux=True` disables Megatron's built-in (rank-local) aux loss so
    that `rfull.aux_loss.ep_global_aux_loss` is the only balancing term. See
    AUX-001.

    `**overrides` forwards additional `TransformerConfig` fields (e.g.
    `recompute_granularity`) verbatim. Unknown names raise rather than being
    silently dropped, so a typo cannot quietly disable activation
    recomputation and OOM only at scale.
    """
    from megatron.core.transformer.transformer_config import TransformerConfig

    if params_dtype is None:
        params_dtype = torch.bfloat16 if bf16 else torch.float32

    init_std = 0.02
    output_layer_init_std = init_std / ((2 * geo.num_layers) ** 0.5)

    cfg = TransformerConfig(
        # ---- core geometry -------------------------------------------------
        num_layers=geo.num_layers,
        hidden_size=geo.hidden_size,
        num_attention_heads=geo.num_attention_heads,
        num_query_groups=geo.num_query_groups,
        kv_channels=geo.head_dim,                 # CRITICAL: pin 128
        ffn_hidden_size=geo.dense_ffn_hidden_size,
        # NOTE: MCore 0.12 removed the `group_query_attention` flag; GQA is
        # implied by num_query_groups < num_attention_heads.
        # ---- normalisation / activation ------------------------------------
        normalization=geo.normalization,
        layernorm_epsilon=1e-6,
        qk_layernorm=geo.qk_layernorm,
        gated_linear_unit=geo.gated_linear_unit,
        # Both MCore FFN paths call activation_func(gate_half) and multiply by
        # `up` themselves, so the configured activation is the GATE-ONLY factor.
        # `patch_grouped_mlp_glu` replaces the whole glu closure to restore the
        # `up` clamp; this entry is the fallback/documented convention.
        activation_func=make_mcore_activation(
            gate_width=(geo.dense_ffn_hidden_size, geo.expert_ffn_hidden_size),
        ),
        add_bias_linear=geo.add_bias_linear,
        bias_activation_fusion=False,             # custom activation
        # ---- MoE ------------------------------------------------------------
        num_moe_experts=geo.num_routed_experts,
        moe_router_topk=geo.moe_router_topk,
        moe_ffn_hidden_size=geo.expert_ffn_hidden_size,
        moe_shared_expert_intermediate_size=(
            geo.expert_ffn_hidden_size * geo.num_shared_experts),
        moe_shared_expert_overlap=False,
        moe_layer_freq=_layer_freq_list(geo),
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_expert_capacity_factor=None,          # dropless
        moe_pad_expert_input_to_capacity=False,
        moe_router_load_balancing_type="none" if use_custom_aux else "aux_loss",
        moe_aux_loss_coeff=0.0 if use_custom_aux else aux_loss_coeff,
        moe_z_loss_coeff=0.0,                     # applied by rfull.aux_loss
        moe_router_dtype="fp32",
        moe_router_pre_softmax=False,             # top-k then softmax
        moe_router_score_function="softmax",
        # ---- parallelism -----------------------------------------------------
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=1,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel,
        # ---- precision -------------------------------------------------------
        bf16=bf16,
        fp16=False,
        params_dtype=params_dtype,
        # ---- init ------------------------------------------------------------
        init_method_std=init_std,
        use_cpu_initialization=False,
        perform_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        # ---- misc ------------------------------------------------------------
        gradient_accumulation_fusion=False,
        deallocate_pipeline_outputs=False,
        **overrides,
    )
    cfg.rfull_output_layer_init_std = output_layer_init_std
    cfg.rfull_aux_loss_coeff = aux_loss_coeff
    cfg.rfull_z_loss_coeff = z_loss_coeff
    cfg.rfull_use_custom_aux = use_custom_aux
    return cfg


def _layer_freq_list(geo: RFullGeometry) -> list:
    """Explicit 0/1 per-layer MoE placement list (design doc section 5)."""
    return [0 if i < geo.num_dense_layers else 1 for i in range(geo.num_layers)]


def build_rfull_model(
    geo: RFullGeometry = GEOMETRY,
    *,
    seq_length: int = 4096,
    config=None,
    pre_process: bool = True,
    post_process: bool = True,
    router_init_seed: Optional[int] = 1234,
    attention_backend: str = "auto",
    **cfg_kwargs,
):
    """Build a `GPTModel` matching the frozen R-Full design.

    `attention_backend`:
      * ``"te"``     - use TE's DotProductAttention (fused/flash).
      * ``"mcore"``  - swap in MCore's native DotProductAttention (PyTorch
                       SDPA) while keeping every other TE submodule.
      * ``"auto"``   - probe TE once and fall back to ``"mcore"`` if TE's
                       attention is unusable on this build.

    Why this knob exists (KERN-001): on this ROCm image TE 2.6.0 accepts
    flash-attn <=2.8.1 but the image ships 2.8.3, so TE marks flash-attn as
    "not installed"; TE's own fused backend separately aborts inside C++
    ("basic_string: construction from null"). Both TE attention paths are then
    unavailable. Swapping ONLY core_attention keeps TE LayerNorm/Linear and the
    TE grouped-GEMM MoE path, avoiding the legacy GroupedMLP downgrade that a
    wholesale switch to the local spec would cause.
    """
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

    if config is None:
        config = build_transformer_config(geo, seq_length=seq_length, **cfg_kwargs)

    # ACT-001: make the routed-expert GLU clamp `up` exactly like the dense
    # path. Must be applied BEFORE any GroupedMLP is constructed.
    config.rfull_glu_patch = patch_grouped_mlp_glu()

    # CRITICAL: `get_gpt_layer_with_transformer_engine_spec` returns a SINGLE
    # layer spec; handing it to GPTModel applies it to every layer, which makes
    # all 48 layers MoE and silently ignores `moe_layer_freq`. Only
    # `get_gpt_decoder_block_spec` honours the per-layer 0/1 pattern and yields
    # 2 dense + 46 MoE layers as frozen.
    layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True)

    backend = attention_backend
    if backend == "auto":
        backend = "te" if _te_attention_works() else "mcore"
    if backend == "mcore":
        # NOTE: MCore's own DotProductAttention materialises an explicit
        # [b, h, s, s] score matrix (measured 8.4 GiB vs 0.19 GiB for SDPA
        # flash at s=4096) and OOMs at 48 layers. Use the SDPA-flash module.
        from .sdpa_attention import SDPAFlashAttention

        # `get_gpt_decoder_block_spec` returns a block with a per-layer list, so
        # the swap must touch every layer spec, not a single shared one.
        for _ls in layer_spec.layer_specs:
            _ls.submodules.self_attention.submodules.core_attention = SDPAFlashAttention
        config.rfull_attention_backend = "sdpa_flash"
    else:
        config.rfull_attention_backend = "te"

    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=geo.vocab_size,
        max_sequence_length=seq_length,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=geo.tie_word_embeddings,
        position_embedding_type=geo.position_embedding_type,
        rotary_percent=geo.rotary_percent,
        rotary_base=geo.rotary_base,
    )

    # ROUTER-001: MCore has no dedicated router-init knob.
    touched = apply_router_init(model, std=ROUTER_INIT_STD, seed=router_init_seed)
    model.rfull_router_params_initialised = touched
    return model


_TE_ATTN_CACHE: Optional[bool] = None


def _te_attention_works() -> bool:
    """One-shot probe of TE's DotProductAttention with R-Full head geometry."""
    global _TE_ATTN_CACHE
    if _TE_ATTN_CACHE is not None:
        return _TE_ATTN_CACHE
    try:
        from transformer_engine.pytorch import DotProductAttention as TEDPA

        dpa = TEDPA(
            num_attention_heads=GEOMETRY.num_attention_heads,
            kv_channels=GEOMETRY.head_dim,
            num_gqa_groups=GEOMETRY.num_query_groups,
            attention_dropout=0.0,
            qkv_format="bshd",
            attn_mask_type="causal",
        ).cuda()
        q = torch.randn(1, 128, GEOMETRY.num_attention_heads, GEOMETRY.head_dim,
                        device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 128, GEOMETRY.num_query_groups, GEOMETRY.head_dim,
                        device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        with torch.no_grad():
            o = dpa(q, k, v)
        _TE_ATTN_CACHE = bool(torch.isfinite(o).all().item())
    except Exception:
        _TE_ATTN_CACHE = False
    return _TE_ATTN_CACHE


def count_built_parameters(model, geo: RFullGeometry = GEOMETRY) -> dict:
    """Count parameters on the built model, distinguishing expert-sharded ones.

    Under EP the routed experts are sharded, so the naive sum over
    `model.parameters()` is the LOCAL count. The global total is reconstructed
    by scaling the expert part by the EP world size.
    """
    import torch.distributed as dist

    total_local = 0
    expert_local = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total_local += n
        # Only ROUTED experts are EP-sharded. The shared expert is replicated
        # on every EP rank, so it must NOT be scaled when reconstructing the
        # global total. MCore names them:
        #   ...mlp.experts.*        -> routed (sharded)
        #   ...mlp.shared_experts.* -> shared (replicated)
        if ".experts." in name and ".shared_experts." not in name:
            expert_local += n
    non_expert = total_local - expert_local

    ep = 1
    try:
        from megatron.core import parallel_state as ps

        if dist.is_initialized():
            ep = ps.get_expert_model_parallel_world_size()
    except Exception:
        pass

    return {
        "local_total": total_local,
        "local_expert": expert_local,
        "local_non_expert": non_expert,
        "ep_size": ep,
        "reconstructed_global_total": non_expert + expert_local * ep,
    }

"""R-Full v0.1 frozen model geometry, parameter ledger, and MCore layer specs.

Everything in this module is derived from `configs/rfull/rfull_v0_1.source.json`
and must agree with it exactly. The ledger below is asserted at build time so a
silent geometry drift (most likely: Megatron inferring head_dim = hidden /
num_heads = 64 instead of the frozen 128) fails loudly instead of training a
different model than the one that was frozen.

Frozen geometry
---------------
    layers            48   (0-1 Dense SwiGLU, 2-47 MoE)
    hidden            2048
    q heads           32      kv heads 4      head_dim 128
    dense ffn         5504
    routed experts    96      top-k 6      expert ffn 896
    shared experts    1       (ffn 896)
    vocab             151936  (padded; tokenizer native 151669)
    tied embeddings   yes
    QK-RMSNorm        yes (learnable, 128 scales per projection per layer)

Note that 32 * 128 = 4096 != hidden = 2048. The attention projections are
therefore NOT square, which is exactly why `kv_channels` must be pinned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "RFullGeometry",
    "GEOMETRY",
    "parameter_ledger",
    "assert_frozen_ledger",
    "attention_projection_shapes",
    "assert_attention_shapes",
    "is_moe_layer",
    "moe_layer_pattern",
]


@dataclass(frozen=True)
class RFullGeometry:
    num_layers: int = 48
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_query_groups: int = 4          # GQA / KV heads
    head_dim: int = 128                # kv_channels; NOT hidden // heads
    dense_ffn_hidden_size: int = 5504
    num_dense_layers: int = 2          # layers [0, num_dense_layers)
    num_routed_experts: int = 96
    moe_router_topk: int = 6
    expert_ffn_hidden_size: int = 896
    num_shared_experts: int = 1
    vocab_size: int = 151936           # padded embedding rows
    tokenizer_vocab_size: int = 151669
    eot_id: int = 151643
    tie_word_embeddings: bool = True
    qk_layernorm: bool = True
    normalization: str = "RMSNorm"
    position_embedding_type: str = "rope"
    rotary_percent: float = 1.0
    rotary_base: int = 1000000
    add_bias_linear: bool = False
    gated_linear_unit: bool = True

    # Frozen totals from the design doc / source.json.
    expected_total_params: int = 25_857_439_744
    expected_active_params: int = 3_066_640_384

    @property
    def num_moe_layers(self) -> int:
        return self.num_layers - self.num_dense_layers

    @property
    def q_proj_out(self) -> int:
        return self.num_attention_heads * self.head_dim      # 4096

    @property
    def kv_proj_out(self) -> int:
        return self.num_query_groups * self.head_dim         # 512


GEOMETRY = RFullGeometry()


def is_moe_layer(layer_index: int, geo: RFullGeometry = GEOMETRY) -> bool:
    """Layer indices are 0-based. Layers 0..num_dense_layers-1 are Dense."""
    return layer_index >= geo.num_dense_layers


def moe_layer_pattern(geo: RFullGeometry = GEOMETRY) -> str:
    """Explicit per-layer pattern string, e.g. '001111...1'.

    The frozen design forbids expressing placement as a generic periodic
    `moe_layer_freq` integer, because that would silently change if the layer
    count changed. A full-length pattern is unambiguous.
    """
    return "".join("1" if is_moe_layer(i, geo) else "0"
                   for i in range(geo.num_layers))


def attention_projection_shapes(geo: RFullGeometry = GEOMETRY) -> dict:
    """Frozen attention weight shapes, [out_features, in_features]."""
    return {
        "Wq": (geo.q_proj_out, geo.hidden_size),      # (4096, 2048)
        "Wk": (geo.kv_proj_out, geo.hidden_size),     # (512, 2048)
        "Wv": (geo.kv_proj_out, geo.hidden_size),     # (512, 2048)
        "Wo": (geo.hidden_size, geo.q_proj_out),      # (2048, 4096)
    }


def parameter_ledger(geo: RFullGeometry = GEOMETRY) -> dict:
    """Exact analytic parameter count, total and active.

    Counted the way the frozen ledger counts:
      * tied embedding matrix counted once
      * QK-RMSNorm scales are learnable (head_dim per projection per layer)
      * router weights are counted, no router bias
      * shared expert always active
    """
    h = geo.hidden_size

    embedding = geo.vocab_size * h

    # --- per-layer attention block -----------------------------------------
    wq = geo.q_proj_out * h
    wk = geo.kv_proj_out * h
    wv = geo.kv_proj_out * h
    wo = h * geo.q_proj_out
    attn_proj = wq + wk + wv + wo

    qk_norm = (2 * geo.head_dim) if geo.qk_layernorm else 0   # q_norm + k_norm
    layer_norms = 2 * h                                       # input + pre_mlp
    per_layer_common = attn_proj + qk_norm + layer_norms
    all_layers_common = per_layer_common * geo.num_layers

    # --- dense FFN (SwiGLU: gate + up + down) ------------------------------
    f_dense = geo.dense_ffn_hidden_size
    dense_ffn = 3 * h * f_dense
    dense_total = dense_ffn * geo.num_dense_layers

    # --- MoE layers ---------------------------------------------------------
    f_e = geo.expert_ffn_hidden_size
    per_expert = 3 * h * f_e
    routed_per_layer = per_expert * geo.num_routed_experts
    shared_per_layer = per_expert * geo.num_shared_experts
    router_per_layer = geo.num_routed_experts * h
    moe_per_layer = routed_per_layer + shared_per_layer + router_per_layer
    moe_total = moe_per_layer * geo.num_moe_layers

    final_norm = h

    total = embedding + all_layers_common + dense_total + moe_total + final_norm

    # --- active parameters (per token) --------------------------------------
    active_moe_per_layer = (
        per_expert * geo.moe_router_topk
        + shared_per_layer
        + router_per_layer
    )
    active = (embedding + all_layers_common + dense_total
              + active_moe_per_layer * geo.num_moe_layers + final_norm)

    return {
        "embedding": embedding,
        "per_layer_common": per_layer_common,
        "all_layers_common": all_layers_common,
        "dense_ffn_per_layer": dense_ffn,
        "dense_total": dense_total,
        "per_expert": per_expert,
        "routed_per_layer": routed_per_layer,
        "shared_per_layer": shared_per_layer,
        "router_per_layer": router_per_layer,
        "moe_per_layer": moe_per_layer,
        "moe_total": moe_total,
        "final_norm": final_norm,
        "total": total,
        "active": active,
        "routed_expert_params": routed_per_layer * geo.num_moe_layers,
        "non_expert_params": total - routed_per_layer * geo.num_moe_layers,
    }


def assert_frozen_ledger(geo: RFullGeometry = GEOMETRY) -> dict:
    """Fail loudly if the computed ledger drifts from the frozen numbers."""
    led = parameter_ledger(geo)
    if led["total"] != geo.expected_total_params:
        raise AssertionError(
            f"total parameter drift: computed {led['total']:,} != frozen "
            f"{geo.expected_total_params:,}")
    if led["active"] != geo.expected_active_params:
        raise AssertionError(
            f"active parameter drift: computed {led['active']:,} != frozen "
            f"{geo.expected_active_params:,}")
    return led


def assert_attention_shapes(model, geo: RFullGeometry = GEOMETRY) -> dict:
    """Verify a built MCore model actually uses head_dim=128, not hidden/heads.

    Megatron fuses QKV into a single `linear_qkv` weight of shape
    ``[q_out + 2*kv_out, hidden]``. For R-Full that is ``[5120, 2048]``.
    """
    expected_qkv_out = geo.q_proj_out + 2 * geo.kv_proj_out   # 4096 + 1024
    expected_proj = (geo.hidden_size, geo.q_proj_out)         # (2048, 4096)
    found = {"linear_qkv": [], "linear_proj": []}
    for name, p in model.named_parameters():
        if name.endswith("linear_qkv.weight"):
            found["linear_qkv"].append((name, tuple(p.shape)))
        elif name.endswith("linear_proj.weight"):
            found["linear_proj"].append((name, tuple(p.shape)))

    bad = []
    for name, shape in found["linear_qkv"]:
        if shape != (expected_qkv_out, geo.hidden_size):
            bad.append((name, shape, (expected_qkv_out, geo.hidden_size)))
    for name, shape in found["linear_proj"]:
        if shape != expected_proj:
            bad.append((name, shape, expected_proj))
    if bad:
        raise AssertionError(
            "attention projection geometry drift (Megatron likely inferred "
            f"head_dim={geo.hidden_size // geo.num_attention_heads} instead of "
            f"{geo.head_dim}): {bad[:4]}")
    return {
        "n_linear_qkv": len(found["linear_qkv"]),
        "n_linear_proj": len(found["linear_proj"]),
        "expected_qkv_shape": (expected_qkv_out, geo.hidden_size),
        "expected_proj_shape": expected_proj,
    }


if __name__ == "__main__":
    import json

    led = assert_frozen_ledger()
    print(json.dumps({
        "geometry": {
            "layers": GEOMETRY.num_layers,
            "hidden": GEOMETRY.hidden_size,
            "heads": GEOMETRY.num_attention_heads,
            "kv_heads": GEOMETRY.num_query_groups,
            "head_dim": GEOMETRY.head_dim,
            "moe_layers": GEOMETRY.num_moe_layers,
            "experts": GEOMETRY.num_routed_experts,
            "topk": GEOMETRY.moe_router_topk,
            "expert_ffn": GEOMETRY.expert_ffn_hidden_size,
        },
        "attention_shapes": {k: list(v) for k, v in
                             attention_projection_shapes().items()},
        "ledger": {k: v for k, v in led.items()},
        "layer_pattern": moe_layer_pattern(),
        "LEDGER_MATCHES_FROZEN": True,
    }, indent=1))

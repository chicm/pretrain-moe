"""Memory-efficient causal attention for R-Full on ROCm/MI300X.

Why this module exists
----------------------
Three candidate attention paths were measured on this image at
b=1, h=32, kv=4, s=4096, d=128, bf16, with backward:

    torch SDPA FLASH backend .............. 0.188 GiB   OK
    torch SDPA mem-efficient .............. unavailable ("No available kernel")
    torch SDPA math / MCore DotProductAttention  8.406 GiB   OK but 45x memory
    flash_attn 2.8.3 package .............. 0.454 GiB   OK

MCore's local ``DotProductAttention`` materialises an explicit
``[b, h, s, s]`` score matrix and runs softmax over it. At 48 layers that is
~2 GiB of transient per layer and it OOMs a 191 GiB MI300X even though the
model's parameters and activations fit comfortably.

TE's fused attention is unusable here (C++ null-string crash), and TE refuses
the installed flash-attn 2.8.3 because it only accepts <= 2.8.1. So the
supported production path is torch SDPA pinned to the FLASH backend.

Contract
--------
* causal masking is delegated to the kernel (``is_causal=True``); we never
  build an explicit mask, which is what made the naive path expensive;
* GQA is handled by ``enable_gqa=True`` -- no manual key/value expansion, so
  KV heads stay at 4 rather than being broadcast to 32;
* layout follows MCore: input/output are ``[s, b, h, d]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide

# FLASH first; MATH only as a last resort so a kernel regression surfaces as a
# slowdown rather than a crash. EFFICIENT is omitted: measured unavailable.
_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]


class SDPAFlashAttention(MegatronModule):
    """Drop-in replacement for MCore ``DotProductAttention``."""

    def __init__(self, config: TransformerConfig, layer_number: int,
                 attn_mask_type: AttnMaskType, attention_type: str,
                 attention_dropout: float = None, softmax_scale: float = None,
                 cp_comm_type: str = None, model_comm_pgs=None, **kwargs):
        super().__init__(config=config)
        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        world = getattr(config, "tensor_model_parallel_size", 1) or 1
        self.hidden_size_per_partition = divide(
            config.kv_channels * config.num_attention_heads, world)
        self.num_attention_heads_per_partition = divide(
            config.num_attention_heads, world)
        self.num_query_groups_per_partition = divide(
            config.num_query_groups, world)

        self.softmax_scale = (
            softmax_scale if softmax_scale is not None
            else 1.0 / (config.kv_channels ** 0.5))
        p = (attention_dropout if attention_dropout is not None
             else config.attention_dropout)
        self.dropout_p = float(p or 0.0)

    def forward(self, query, key, value, attention_mask,
                attn_mask_type=None, attention_bias=None, packed_seq_params=None,
                **kwargs):
        if packed_seq_params is not None:
            raise NotImplementedError(
                "SDPAFlashAttention does not support packed sequences")
        if attention_bias is not None:
            raise NotImplementedError(
                "SDPAFlashAttention does not support attention_bias")

        # MCore layout [s, b, h, d] -> SDPA layout [b, h, s, d]
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)

        mask_type = attn_mask_type or self.attn_mask_type
        is_causal = mask_type in (AttnMaskType.causal, AttnMaskType.padding_causal)

        # An explicit mask is exactly what we are avoiding; only accept one if
        # the caller genuinely needs non-causal masking.
        mask = None
        if not is_causal and attention_mask is not None:
            mask = attention_mask

        with sdpa_kernel(_BACKENDS):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
                scale=self.softmax_scale,
                enable_gqa=(self.num_query_groups_per_partition
                            != self.num_attention_heads_per_partition),
            )

        # [b, h, s, d] -> [s, b, h*d], contiguous for the row-parallel proj
        s = out.shape[2]
        b = out.shape[0]
        return (out.permute(2, 0, 1, 3)
                   .reshape(s, b, self.hidden_size_per_partition)
                   .contiguous())

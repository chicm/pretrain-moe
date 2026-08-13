"""Project-owned semantic extensions for the frozen R-Full MoE model.

The package deliberately does not import Megatron-Core at module import time.  Pure
numerical contracts can therefore be tested without a Megatron checkout; the
pinned-MCore adapters live in :mod:`rfull_moe.mcore`.
"""

from .semantics import (
    limited_swiglu,
    limited_swiglu_from_fused,
    load_balancing_loss_from_statistics,
    selected_topk_softmax,
    z_loss_from_statistics,
)

__all__ = [
    "limited_swiglu",
    "limited_swiglu_from_fused",
    "load_balancing_loss_from_statistics",
    "selected_topk_softmax",
    "z_loss_from_statistics",
]

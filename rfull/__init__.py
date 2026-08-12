"""R-Full v0.1 production MoE package.

This package implements the frozen R-Full design on top of Megatron-Core.
It deliberately keeps patches local and auditable:

- `aux_loss`: EP-global, autograd-aware auxiliary load-balancing loss (AUX-001)
- `router_init`: Normal(0, 0.01) router initialisation (ROUTER-001)
- `limited_swiglu`: shared clamp semantics across all three FFN paths (ACT-001)
- `finite_consensus`: world-consensus finite checks + stop/replay (NUM-001)

Nothing here silently changes Megatron-Core behaviour; every patch is applied
explicitly by the training entrypoint and recorded in the run manifest.
"""

__all__ = [
    "aux_loss",
    "router_init",
    "limited_swiglu",
    "finite_consensus",
]

__version__ = "0.1.0"

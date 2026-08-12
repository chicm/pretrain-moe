"""Limited SwiGLU with identical clamp semantics on all FFN paths (ACT-001).

Frozen contract (design doc section 6.4):

    limit_gate = 7.0
    limit_up   = 7.0
    alpha      = 1.702
    y = clamp(up, -limit_up, +limit_up) * gate_clamped * sigmoid(alpha * gate_clamped)
    where gate_clamped = clamp(gate, -limit_gate, +limit_gate)

The same activation MUST be applied identically on all three FFN paths:
  1. the two Dense SwiGLU layers (layers 0-1),
  2. the 96 routed experts,
  3. the shared expert.

A mismatch between the dense path and the grouped-GEMM expert path is a silent
correctness bug that only shows up as a slow divergence, so this module is the
single source of truth and ships with a parity test.

Note on gradients: the clamp is a hard saturation, so tokens outside the limit
receive zero gradient through that branch. This is intended -- it is the
mechanism that bounds expert activation magnitude.
"""

from __future__ import annotations

import torch

__all__ = [
    "LIMIT_GATE",
    "LIMIT_UP",
    "ALPHA",
    "limited_swiglu",
    "limited_swiglu_chunked",
    "gate_activation",
    "make_mcore_activation",
    "patch_grouped_mlp_glu",
]

LIMIT_GATE = 7.0
LIMIT_UP = 7.0
ALPHA = 1.702


def limited_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    limit_gate: float = LIMIT_GATE,
    limit_up: float = LIMIT_UP,
    alpha: float = ALPHA,
) -> torch.Tensor:
    """Reference implementation, dtype-preserving.

    Args:
        gate: pre-activation of the gate projection.
        up: pre-activation of the up projection (same shape as ``gate``).
    """
    if gate.shape != up.shape:
        raise ValueError(f"gate {tuple(gate.shape)} != up {tuple(up.shape)}")
    g = torch.clamp(gate, -limit_gate, limit_gate)
    u = torch.clamp(up, -limit_up, limit_up)
    return u * g * torch.sigmoid(alpha * g)


def limited_swiglu_chunked(
    x: torch.Tensor,
    limit_gate: float = LIMIT_GATE,
    limit_up: float = LIMIT_UP,
    alpha: float = ALPHA,
) -> torch.Tensor:
    """Megatron-style entry point: input is ``[..., 2*ffn]`` = concat(gate, up).

    Megatron-Core's ``glu`` activations receive the fused projection output and
    split it in half along the last dimension. Keeping this wrapper next to the
    reference guarantees both paths share one definition.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            f"expected an even last dim for chunked GLU, got {x.shape[-1]}")
    gate, up = torch.chunk(x, 2, dim=-1)
    return limited_swiglu(gate, up, limit_gate, limit_up, alpha)


def _as_set(v) -> set:
    """Accept an int or any iterable of ints as a width specification."""
    if isinstance(v, int):
        return {v}
    return set(int(x) for x in v)


def make_mcore_activation(
    limit_gate: float = LIMIT_GATE,
    limit_up: float = LIMIT_UP,
    alpha: float = ALPHA,
    fused_width: int | None = None,
    gate_width: int | None = None,
):
    """Return a callable usable by EVERY MCore FFN path.

    MCore does not have a single calling convention for ``activation_func``:

    * ``MLP`` (dense layers and the shared expert) passes the FUSED
      ``[..., 2*ffn]`` projection output and expects the activation itself to
      split it.
    * ``GroupedMLP`` (the routed experts) splits first and then evaluates
      ``activation_func(gate) * up`` -- so it hands over a ``[..., ffn]``
      tensor and multiplies by ``up`` afterwards, OUTSIDE this function.

    Naively reusing one implementation therefore either explodes on shape
    (896 chunked into 448) or, worse, silently skips the ``up`` clamp on the
    expert path while applying it on the dense path -- exactly the silent
    dense/expert divergence ACT-001 exists to prevent.

    Disambiguation is by EXPLICIT WIDTH, never by parity of the last
    dimension: an expert FFN of 896 is itself even, so "even means fused" is
    wrong and would silently mis-split. Callers pass ``fused_width`` (e.g.
    2*5504) and/or ``gate_width`` (e.g. 896). When neither is supplied the
    function refuses to guess for ambiguous inputs.
    """

    def _act(x: torch.Tensor) -> torch.Tensor:
        last = x.shape[-1]
        fused_ok = fused_width is not None and last in _as_set(fused_width)
        gate_ok = gate_width is not None and last in _as_set(gate_width)
        if fused_ok and gate_ok:
            raise RuntimeError(
                f"ambiguous width {last}: listed as both fused and gate")
        if fused_ok:
            return limited_swiglu_chunked(x, limit_gate, limit_up, alpha)
        if gate_ok:
            return gate_activation(x, limit_gate, alpha)
        if fused_width is None and gate_width is None:
            raise RuntimeError(
                "limited_swiglu activation called without an explicit width "
                f"convention (last dim {last}). Pass fused_width and/or "
                "gate_width; do not rely on parity of the last dimension.")
        raise RuntimeError(
            f"limited_swiglu activation received last dim {last}, which "
            f"matches neither fused_width={fused_width} nor "
            f"gate_width={gate_width}.")

    _act.__name__ = "limited_swiglu"
    _act.rfull_limits = (limit_gate, limit_up, alpha)  # type: ignore[attr-defined]
    _act.rfull_widths = (fused_width, gate_width)      # type: ignore[attr-defined]
    return _act


def gate_activation(
    gate: torch.Tensor,
    limit_gate: float = LIMIT_GATE,
    alpha: float = ALPHA,
) -> torch.Tensor:
    """The gate-only factor: ``clamp(g)*sigmoid(alpha*clamp(g))``."""
    g = torch.clamp(gate, -limit_gate, limit_gate)
    return g * torch.sigmoid(alpha * g)


def patch_grouped_mlp_glu(
    limit_gate: float = LIMIT_GATE,
    limit_up: float = LIMIT_UP,
    alpha: float = ALPHA,
) -> dict:
    """Force every fused-GLU FFN path to clamp ``up`` as well (ACT-001).

    Empirically, on MCore 0.12 BOTH FFN implementations use the same
    "gate-only" convention -- they build

        glu(x) = activation_func(chunk0) * chunk1

    inside ``__init__`` (``mlp.py:141`` for the dense/shared-expert ``MLP``,
    and the corresponding line in ``moe/experts.py`` for ``GroupedMLP``). The
    ``up`` half therefore never passes through the configured activation and
    would keep its unbounded magnitude on EVERY path.

    Measured impact of leaving this unpatched: max |dense - expert| output
    divergence of ~168 on N(0,6) inputs, i.e. a silent numerical divergence
    rather than a crash.

    Both classes are patched so the dense layers, the shared expert and the
    routed experts are bit-for-bit the same function.
    """
    from megatron.core.transformer import mlp as _mlp
    from megatron.core.transformer.moe import experts as _experts

    def _make_glu():
        def glu(x):
            gate, up = torch.chunk(x, 2, dim=-1)
            return limited_swiglu(gate, up, limit_gate, limit_up, alpha)

        return glu

    patched = []

    def _patch(cls, label):
        original = getattr(cls, "_rfull_original_init", None)
        if original is None:
            original = cls.__init__
            cls._rfull_original_init = original  # type: ignore[attr-defined]

        def patched_init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            if getattr(self.config, "gated_linear_unit", False):
                self.activation_func = _make_glu()
                self._rfull_glu_patched = True

        cls.__init__ = patched_init  # type: ignore[assignment]
        patched.append(label)

    _patch(_experts.GroupedMLP, "moe.experts.GroupedMLP")
    _patch(_mlp.MLP, "transformer.mlp.MLP")
    if hasattr(_experts, "TEGroupedMLP"):
        _patch(_experts.TEGroupedMLP, "moe.experts.TEGroupedMLP")
    if hasattr(_experts, "SequentialMLP"):
        _patch(_experts.SequentialMLP, "moe.experts.SequentialMLP")

    return {"patched": patched, "limits": (limit_gate, limit_up, alpha)}

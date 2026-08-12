"""Router initialisation and forward-routing semantics (blocker ROUTER-001).

Frozen contract (design doc sections 6.2 / 12):

* router weight ~ Normal(mean=0, std=0.01), no bias, FP32 master weight
* router logits computed in FP32 regardless of the activation dtype
* forward gate = softmax over the SELECTED Top-k logits only
  (``topk_logits_then_selected_softmax``)
* the auxiliary objective uses a separate global N-way softmax
  (see :mod:`rfull.aux_loss`)

Megatron-Core 0.12's ``TopKRouter`` builds its weight with the generic
``init_method`` of the model config, which for R-Full would be the 0.02-scaled
normal used by the rest of the network. The frozen design mandates 0.01 for the
router specifically, so this module supplies both an explicit initialiser and a
verifier that can be run against a constructed model.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch

__all__ = [
    "ROUTER_INIT_STD",
    "init_router_weight_",
    "selected_logit_softmax_gates",
    "apply_router_init",
    "verify_router_init",
]

ROUTER_INIT_STD = 0.01


def init_router_weight_(weight: torch.Tensor, std: float = ROUTER_INIT_STD,
                        generator: torch.Generator | None = None) -> torch.Tensor:
    """In-place Normal(0, std) init for a router weight, done in FP32."""
    with torch.no_grad():
        if weight.dtype == torch.float32:
            weight.normal_(mean=0.0, std=std, generator=generator)
        else:
            tmp = torch.empty_like(weight, dtype=torch.float32)
            tmp.normal_(mean=0.0, std=std, generator=generator)
            weight.copy_(tmp)
    return weight


def selected_logit_softmax_gates(
    logits: torch.Tensor, topk: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Frozen forward routing.

    Returns ``(gates, indices)`` where gates are the softmax over ONLY the
    selected Top-k logits and therefore sum to 1 per token.

    This differs from "softmax over all N then renormalise the Top-k", which
    yields the same values mathematically but a different gradient path; the
    frozen contract names the selected-logit form explicitly, and it is the
    numerically stabler of the two.
    """
    if logits.dtype != torch.float32:
        logits = logits.float()
    topk_logits, indices = torch.topk(logits, topk, dim=-1)
    gates = torch.softmax(topk_logits, dim=-1)
    return gates, indices


def _iter_router_weights(model) -> Iterable[Tuple[str, torch.Tensor]]:
    for name, param in model.named_parameters():
        # MCore names the router projection "...mlp.router.weight"
        if name.endswith("router.weight"):
            yield name, param


def apply_router_init(model, std: float = ROUTER_INIT_STD, seed: int | None = None) -> list:
    """Re-initialise every router weight in ``model`` to Normal(0, std).

    Returns the list of parameter names touched, for the run manifest.
    """
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
    touched = []
    for name, param in _iter_router_weights(model):
        if gen is not None and param.is_cuda:
            tmp = torch.empty(param.shape, dtype=torch.float32)
            tmp.normal_(mean=0.0, std=std, generator=gen)
            with torch.no_grad():
                param.copy_(tmp.to(param.device))
        else:
            init_router_weight_(param, std=std, generator=gen)
        touched.append(name)
    return touched


def verify_router_init(model, std: float = ROUTER_INIT_STD,
                       tol_rel: float = 0.15) -> dict:
    """Check that router weights look like Normal(0, std).

    Uses a generous relative tolerance because each router is only
    ``num_experts x hidden`` elements; the point is to catch a wrong-scale
    initialiser (e.g. 0.02 or 1/sqrt(d)), not to test RNG quality.
    """
    report = {}
    ok = True
    for name, param in _iter_router_weights(model):
        w = param.detach().float()
        m, s = w.mean().item(), w.std().item()
        rel = abs(s - std) / std
        good = rel <= tol_rel and abs(m) <= 5.0 * std / math.sqrt(w.numel())
        ok = ok and good
        report[name] = {"mean": m, "std": s, "rel_err_std": rel,
                        "numel": w.numel(), "ok": good}
    return {"ok": ok, "expected_std": std, "routers": report}

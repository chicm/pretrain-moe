"""Capture per-layer router logits and top-k selections during forward.

MCore's ``TopKRouter`` computes routing internally and only returns the
dispatched activations, so the raw FP32 logits needed by the frozen auxiliary
objective are not otherwise reachable. Rather than fork the router, we attach a
forward hook to every router module and stash what it produced.

Contract:
  * hooks must NOT detach -- the auxiliary loss needs autograd back to the
    router weights;
  * the buffer is drained exactly once per micro-batch via ``pop_router_logits``
    so activation memory is not retained across steps;
  * if a step produced no router output (e.g. a dense-only debug model), the
    caller gets an empty list rather than a silent zero.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


class RouterTap:
    """Collects ``(logits, topk_indices)`` from every MoE layer of one model."""

    def __init__(self):
        self.records: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._handles = []
        self._routers = []
        self.n_taps = 0

    def _hook(self, module, inputs, output):
        # MCore TopKRouter.forward returns (scores, routing_map) where
        # routing_map is a BOOLEAN [T, num_experts] mask -- NOT [T, k] indices.
        # The frozen aux loss wants top-k indices, so convert here.
        logits = getattr(module, "_rfull_last_logits", None)
        # Clear immediately. The attribute keeps a live reference to a graph
        # tensor, and leaving it set makes the NEXT step's backward walk into
        # the previous step's freed graph:
        #   RuntimeError: Trying to backward through the graph a second time
        module._rfull_last_logits = None
        if logits is None or not isinstance(output, (tuple, list)):
            return
        routing_map = output[1]
        if routing_map is None:
            return
        if routing_map.dtype == torch.bool:
            # [T, N] mask -> [T, k] indices. Every row has exactly k set, so
            # the reshape is exact; an unexpected row count must raise rather
            # than silently truncate.
            k = int(routing_map.sum(dim=-1)[0].item())
            idx = routing_map.nonzero(as_tuple=False)[:, 1]
            if idx.numel() != routing_map.shape[0] * k:
                raise RuntimeError(
                    f"routing_map is not exactly {k}-hot per token: "
                    f"{idx.numel()} != {routing_map.shape[0]} * {k}")
            idx = idx.view(routing_map.shape[0], k)
        else:
            idx = routing_map
        self.records.append((logits.view(-1, logits.shape[-1]), idx))

    def attach(self, model) -> int:
        """Hook every router; also wrap ``gating`` to retain FP32 logits."""
        for name, mod in model.named_modules():
            if name.endswith("mlp.router"):
                self._wrap_gating(mod)
                self._handles.append(mod.register_forward_hook(self._hook))
                self._routers.append(mod)
                self.n_taps += 1
        return self.n_taps

    @staticmethod
    def _wrap_gating(router):
        """Make the router keep the logits tensor it computes (autograd intact).

        ``router.gating`` is a BOUND method, so the replacement must not expect
        an explicit ``self``.
        """
        if getattr(router, "_rfull_gating_wrapped", False):
            return
        original = router.gating

        def gating(input, *a, **kw):
            out = original(input, *a, **kw)
            router._rfull_last_logits = out
            return out

        router.gating = gating
        router._rfull_gating_wrapped = True

    def pop(self):
        """Drain the buffer. Must be called once per micro-batch.

        Anything still buffered here belongs to a graph that backward has
        already freed, so it must never survive into the next step.
        """
        recs, self.records = self.records, []
        return recs

    def clear(self):
        """Drop buffered records and any retained logits without consuming them.

        Used on the error path: if a step aborts between forward and backward,
        the stale graph references must go before the next forward runs.
        """
        self.records = []
        for mod in self._routers:
            mod._rfull_last_logits = None

    def detach_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def attach_router_tap(model) -> RouterTap:
    tap = RouterTap()
    n = tap.attach(model)
    if n == 0:
        raise RuntimeError(
            "no router modules found; the model is not MoE or the module "
            "naming changed (expected '*.mlp.router')")
    model._rfull_router_tap = tap
    return tap


def pop_router_logits(model):
    tap = getattr(model, "_rfull_router_tap", None)
    if tap is None:
        raise RuntimeError("attach_router_tap() was never called")
    return tap.pop()

"""Regression test: the router tap must not leak graphs across steps.

This reproduces a real failure. The tap stored FP32 router logits on the module
as ``_rfull_last_logits`` and never cleared them, so the reference outlived the
step. The second update then walked into the previous, already-freed graph:

    RuntimeError: Trying to backward through the graph a second time
    (or directly access saved tensors after they have already been freed).

It survived every single-update smoke test and only appeared on update 2 of the
first real two-node run, which is exactly the kind of bug worth pinning.

Runs on CPU with a tiny stand-in for MCore's TopKRouter -- no GPU, no
distributed init, so it can gate every commit.
"""
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rfull.router_tap import attach_router_tap, pop_router_logits


class FakeRouter(torch.nn.Module):
    """Mimics the parts of TopKRouter the tap depends on."""

    def __init__(self, d=8, n=4, k=2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n, d))
        self.k = k

    def gating(self, x):
        return torch.nn.functional.linear(x.float(), self.weight.float())

    def forward(self, x):
        logits = self.gating(x)
        topk = logits.topk(self.k, dim=-1).indices
        routing_map = torch.zeros_like(logits, dtype=torch.bool)
        routing_map.scatter_(-1, topk, True)
        return logits.softmax(-1), routing_map


class FakeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = FakeRouter()


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = FakeMLP()


class FakeModel(torch.nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer() for _ in range(layers)])
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, x):
        h = x
        for lyr in self.layers:
            scores, _ = lyr.mlp.router(h)
            h = self.proj(h) + scores.to(h.dtype) @ torch.eye(
                4, 8, dtype=h.dtype)
        return h


def main():
    torch.manual_seed(0)
    model = FakeModel()
    tap = attach_router_tap(model)
    assert tap.n_taps == 3, f"expected 3 taps, got {tap.n_taps}"

    # Several consecutive updates. Before the fix this raised on step 1.
    for step in range(4):
        x = torch.randn(5, 8)
        out = model(x)
        recs = pop_router_logits(model)
        assert len(recs) == 3, f"step {step}: expected 3 records, got {len(recs)}"
        for logits, idx in recs:
            assert logits.dtype == torch.float32
            assert idx.shape == (5, 2), f"bad topk shape {tuple(idx.shape)}"
            assert logits.requires_grad, "aux loss needs autograd to the router"
        aux = sum(lg.float().pow(2).mean() for lg, _ in recs)
        (out.pow(2).mean() + aux).backward()
        model.zero_grad(set_to_none=True)

        # The retained-logits attribute must not survive the step.
        for lyr in model.layers:
            leaked = getattr(lyr.mlp.router, "_rfull_last_logits", None)
            assert leaked is None, f"step {step}: router leaked a graph tensor"

    # Buffer must be empty after draining, and clear() must be safe.
    assert pop_router_logits(model) == []
    tap.clear()
    print("PASS: router tap survives 4 consecutive backward steps, no leak")


if __name__ == "__main__":
    main()

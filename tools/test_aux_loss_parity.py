"""AUX-001 acceptance oracle: EP1 reference vs EP8 distributed parity.

Run under torchrun with 8 ranks:

    torchrun --nproc_per_node=8 tools/test_aux_loss_parity.py

Contract verified (design doc section 6.3):
  Fix one identical set of logits/indices spanning the whole EP group.
  The EP8 computation must agree item-by-item with a single-process EP1
  reference on:
    * the scalar auxiliary loss,
    * per-expert selection counts n_i,
    * per-expert probability sums S_i,
    * the router-weight gradient dL/dW_r.

Also verifies that naive per-rank averaging (the stock Megatron-Core
behaviour) genuinely DIFFERS, so the patch is proven necessary rather than
cosmetic.

Exit code 0 == pass. Emits JSON evidence for the blocker register.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfull.aux_loss import ep_global_aux_loss, router_z_loss  # noqa: E402

NUM_EXPERTS = 96
TOPK = 6
HIDDEN = 2048
TOKENS_PER_RANK = 512
SEED = 20260806

# Tolerances: FP32 collectives reassociate sums, so bitwise equality is not
# expected. These bounds are far tighter than any training-relevant drift.
ATOL = 2e-5
RTOL = 2e-5


def _mk_inputs(device, world: int):
    """Deterministically build the FULL batch, identical on every rank."""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.randn(world * TOKENS_PER_RANK, HIDDEN, generator=g, dtype=torch.float32)
    w = torch.randn(NUM_EXPERTS, HIDDEN, generator=g, dtype=torch.float32) * 0.01
    return x.to(device), w.to(device)


def _route(x, w):
    """Frozen forward routing: FP32 logits -> Top-6 -> selected-logit softmax."""
    logits = torch.nn.functional.linear(x, w)          # [T, N] fp32
    _, idx = torch.topk(logits, TOPK, dim=-1)
    return logits, idx


def main() -> int:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda")

    x_full, w_init = _mk_inputs(device, world)

    # ---------------- EP1 reference: whole batch in one process --------------
    w_ref = w_init.clone().requires_grad_(True)
    logits_ref, idx_ref = _route(x_full, w_ref)
    loss_ref, stats_ref = ep_global_aux_loss(
        logits_ref, idx_ref, NUM_EXPERTS, TOPK,
        ep_group=None, scale_for_dp_average=False)
    loss_ref.backward()
    grad_ref = w_ref.grad.detach().clone()

    # ---------------- EP8: each rank owns a contiguous token slice ----------
    lo, hi = rank * TOKENS_PER_RANK, (rank + 1) * TOKENS_PER_RANK
    w_ep = w_init.clone().requires_grad_(True)
    logits_ep, idx_ep = _route(x_full[lo:hi], w_ep)
    loss_ep, stats_ep = ep_global_aux_loss(
        logits_ep, idx_ep, NUM_EXPERTS, TOPK,
        ep_group=dist.group.WORLD, scale_for_dp_average=False)
    loss_ep.backward()
    # Each rank holds the gradient of its own token shard; the true global
    # gradient is their sum (dense-DP would average, hence the ep_size factor
    # handled by scale_for_dp_average in production).
    grad_ep = w_ep.grad.detach().clone()
    dist.all_reduce(grad_ep, op=dist.ReduceOp.SUM)

    # ---------------- stock-style rank-local average (must differ) ----------
    w_local = w_init.clone().requires_grad_(True)
    logits_loc, idx_loc = _route(x_full[lo:hi], w_local)
    loss_loc, _ = ep_global_aux_loss(
        logits_loc, idx_loc, NUM_EXPERTS, TOPK,
        ep_group=None, scale_for_dp_average=False)
    loss_loc_avg = loss_loc.detach().clone()
    dist.all_reduce(loss_loc_avg, op=dist.ReduceOp.SUM)
    loss_loc_avg /= world

    # ---------------- comparisons ------------------------------------------
    d_loss = (loss_ep.detach() - loss_ref.detach()).abs().item()
    d_counts = (stats_ep["expert_counts"] - stats_ref["expert_counts"]).abs().max().item()
    d_probs = (stats_ep["expert_prob_sum"] - stats_ref["expert_prob_sum"]).abs().max().item()
    d_grad = (grad_ep - grad_ref).abs().max().item()
    grad_scale = grad_ref.abs().max().item()
    d_stock = (loss_loc_avg - loss_ref.detach()).abs().item()

    z_ref, zs_ref = router_z_loss(logits_ref)

    ok_loss = d_loss <= ATOL + RTOL * abs(loss_ref.item())
    ok_counts = d_counts == 0.0
    ok_probs = d_probs <= 1e-3          # sums over 4096 tokens, fp32 reassoc
    ok_grad = d_grad <= ATOL + RTOL * grad_scale
    stock_differs = d_stock > 1e-4      # proves the patch is necessary

    all_ok = ok_loss and ok_counts and ok_probs and ok_grad and stock_differs

    if rank == 0:
        evidence = {
            "test": "AUX-001 EP-global auxiliary loss parity",
            "world_size": world,
            "num_experts": NUM_EXPERTS,
            "topk": TOPK,
            "tokens_per_rank": TOKENS_PER_RANK,
            "tokens_total": world * TOKENS_PER_RANK,
            "seed": SEED,
            "loss_ep1_reference": loss_ref.item(),
            "loss_ep8": loss_ep.item(),
            "abs_diff_loss": d_loss,
            "max_abs_diff_counts": d_counts,
            "max_abs_diff_prob_sum": d_probs,
            "max_abs_diff_router_grad": d_grad,
            "router_grad_scale": grad_scale,
            "stock_rank_local_average_loss": loss_loc_avg.item(),
            "abs_diff_stock_vs_correct": d_stock,
            "z_loss_reference": z_ref.item(),
            "max_abs_logit": zs_ref["max_abs_logit"].item(),
            "checks": {
                "loss_matches": ok_loss,
                "counts_match": ok_counts,
                "prob_sums_match": ok_probs,
                "router_grad_matches": ok_grad,
                "stock_rank_local_differs": stock_differs,
            },
            "verdict": "PASS" if all_ok else "FAIL",
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
        }
        blob = json.dumps(evidence, indent=1, sort_keys=True)
        evidence["evidence_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
        print(json.dumps(evidence, indent=1, sort_keys=True))
        out = os.environ.get("RFULL_EVIDENCE_OUT")
        if out:
            with open(out, "w") as f:
                json.dump(evidence, f, indent=1, sort_keys=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

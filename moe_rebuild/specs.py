"""Concrete run specifications for the three rebuild phases.

Phase 1  dense_1b_*      -- validate Megatron multi-node transport & throughput
Phase 2  moe_1node_*     -- validate MoE (EP8, 96 experts, alltoall) on one node
Phase 3  rfull_moe_prod  -- 15-node / 120-GPU R-Full production run

Geometry for phase 3 is frozen by
docs/r_full_moe_production_training_design.md section 11.4.
"""

from __future__ import annotations

from .config import (
    CKPT_ROOT,
    Model,
    RunSpec,
    Schedule,
    Topology,
)

# --------------------------------------------------------------------------
# R-Full production geometry (design doc S11.4)
# --------------------------------------------------------------------------
RFULL_LAYERS = 48
RFULL_HIDDEN = 2048
RFULL_FFN = 5504            # dense-layer FFN
RFULL_EXPERTS = 96
RFULL_MOE_FFN = 896         # per-expert FFN
RFULL_TOPK = 6
# first 2 layers dense, remaining 46 MoE
RFULL_LAYER_FREQ = "[0]*2+[1]*46"


def _rfull_model(
    name: str,
    num_layers: int = RFULL_LAYERS,
    num_experts: int = RFULL_EXPERTS,
    layer_freq: str = RFULL_LAYER_FREQ,
    seq_length: int = 4096,
) -> Model:
    return Model(
        name=name,
        num_layers=num_layers,
        hidden_size=RFULL_HIDDEN,
        ffn_hidden_size=RFULL_FFN,
        num_attention_heads=32,
        num_query_groups=4,
        kv_channels=128,
        seq_length=seq_length,
        num_experts=num_experts,
        moe_layer_freq=layer_freq,
        moe_ffn_hidden_size=RFULL_MOE_FFN,
        moe_shared_expert_intermediate_size=RFULL_MOE_FFN,
        moe_router_topk=RFULL_TOPK,
    )


# --------------------------------------------------------------------------
# Phase 1 -- dense 1B, multi-node transport validation
# --------------------------------------------------------------------------
DENSE_1B = Model(
    name="dense_1b",
    num_layers=16,
    hidden_size=2048,
    ffn_hidden_size=5504,
    num_attention_heads=32,
    num_query_groups=4,
    kv_channels=128,
    seq_length=4096,
)


def dense_1b(nnodes: int, train_iters: int = 50, run_id: str | None = None) -> RunSpec:
    """Dense 1B smoke: proves rendezvous, RCCL, dataloader and throughput."""
    topo = Topology(nnodes=nnodes, expert_parallel=1)
    gbs = 8 * topo.data_parallel  # GA=8 per rank
    rid = run_id or f"p1_dense1b_{nnodes}n"
    return RunSpec(
        run_id=rid,
        model=DENSE_1B,
        topology=topo,
        schedule=Schedule(
            train_iters=train_iters,
            global_batch_size=gbs,
            lr_warmup_iters=max(2, train_iters // 10),
            save_interval=10_000,      # no checkpoint during smoke
            eval_interval=10_000,
            # Must stay >=1: Megatron always builds a valid dataloader and
            # MegatronPretrainingSampler asserts total_samples > 0, so
            # eval_iters=0 dies with "no sample to consume: 0".
            eval_iters=1,
        ),
        save=None,
    )


# --------------------------------------------------------------------------
# Phase 2 -- MoE on a single node (EP8)
# --------------------------------------------------------------------------
def moe_1node_mini(train_iters: int = 50, run_id: str | None = None) -> RunSpec:
    """8-layer / 96-expert MoE. Exercises the exact production MoE code path
    (EP8, alltoall dispatcher, grouped GEMM, 12 experts per GPU) at low cost."""
    topo = Topology(nnodes=1, expert_parallel=8)
    gbs = 8 * topo.data_parallel  # 64
    return RunSpec(
        run_id=run_id or "p2_moe_mini_1n",
        model=_rfull_model("rfull_mini", num_layers=8, layer_freq="[0]*2+[1]*6"),
        topology=topo,
        schedule=Schedule(
            train_iters=train_iters,
            global_batch_size=gbs,
            lr_warmup_iters=max(2, train_iters // 10),
            save_interval=10_000,
            eval_interval=10_000,
            eval_iters=1,   # 0 would empty the valid split; see dense_1b()
        ),
        save=None,
    )


def moe_1node_full(train_iters: int = 20, run_id: str | None = None) -> RunSpec:
    """Full 48-layer R-Full geometry on one node: proves HBM headroom
    (~4.59B logical params/GPU under EP8) before going to 15 nodes."""
    topo = Topology(nnodes=1, expert_parallel=8)
    gbs = 8 * topo.data_parallel
    return RunSpec(
        run_id=run_id or "p2_moe_full_1n",
        model=_rfull_model("rfull_full"),
        topology=topo,
        schedule=Schedule(
            train_iters=train_iters,
            global_batch_size=gbs,
            lr_warmup_iters=2,
            save_interval=10_000,
            eval_interval=10_000,
            eval_iters=1,   # 0 would empty the valid split; see dense_1b()
        ),
        save=None,
    )


# --------------------------------------------------------------------------
# Phase 3 -- production, 15 nodes / 120 GPUs
# --------------------------------------------------------------------------
# Design doc S10 stage A: seq 4096, 960 global sequences per update
# (= 3,932,160 target tokens), 203,451 updates.
PROD_STAGE_A_ITERS = 203_451
PROD_GBS = 960
PROD_MBS = 1


def rfull_moe_prod(
    nnodes: int = 15,
    train_iters: int = PROD_STAGE_A_ITERS,
    run_id: str = "rfull_moe_prod",
    save: str | None = None,
    load: str | None = None,
) -> RunSpec:
    topo = Topology(nnodes=nnodes, expert_parallel=8)
    # Production is 120 GPUs. Smaller worlds are allowed only for the
    # single-node bisect arms, which exist to vary DP width while holding the
    # EP group identical; anything in between is almost certainly a mistake.
    assert topo.world == 120 or nnodes == 1, (
        f"production expects 120 GPUs (or nnodes=1 for bisect), got {topo.world}")
    return RunSpec(
        run_id=run_id,
        model=_rfull_model("rfull"),
        topology=topo,
        schedule=Schedule(
            train_iters=train_iters,
            global_batch_size=PROD_GBS,
            # micro_batch_size is set by MEMORY, not by GEMM efficiency.
            #
            # Grouped-GEMM throughput does improve with larger micro-batches
            # (measured, MI300X, 12 experts/rank at EP8, H=2048, F=896):
            #     tok/expert   mbs   TFLOP/s
            #            256     1      48.0
            #           2048     8     139.2
            # but at DP=120 the global batch of 960 gives exactly 8 sequences
            # per rank, so mbs=8 collapses gradient accumulation to a single
            # micro-step and puts 8*4096 = 32768 tokens in flight at once.
            # That needs ~160 GiB of activations on top of 17.5 GiB of static
            # state and dies with:
            #   "HIP out of memory. Tried to allocate 1.01 GiB. GPU 4 has ...
            #    191.45 GiB of which 722.00 MiB is free"
            # mbs=1 with 8 accumulation steps keeps activations ~8x smaller.
            micro_batch_size=PROD_MBS,
            lr=2.0e-4,
            min_lr=2.0e-5,
            lr_warmup_iters=2543,
            lr_decay_style="cosine",
            lr_decay_iters=train_iters,
            save_interval=2000,
            eval_interval=2000,
            eval_iters=10,
            log_interval=1,
        ),
        save=save or f"{CKPT_ROOT}/{run_id}",
        load=load or f"{CKPT_ROOT}/{run_id}",
    )


def moe_prod_smoke(nnodes: int = 15, train_iters: int = 30) -> RunSpec:
    """Gate-4 style 120-GPU smoke: production topology, no checkpoint."""
    spec = rfull_moe_prod(
        nnodes=nnodes, train_iters=train_iters, run_id="p3_moe_smoke_15n"
    )
    spec.save = None
    spec.load = None
    spec.schedule.lr_warmup_iters = 5
    spec.schedule.eval_iters = 1   # 0 empties the valid split -> assert
    spec.schedule.eval_interval = 10_000
    spec.schedule.save_interval = 10_000
    return spec



def moe_bisect_1n(num_layers: int = 12, train_iters: int = 25,
                  timeout_min: int = 10) -> RunSpec:
    """Same model and EP group as the 15-node bisect, on a single node.

    EP=8 is intra-node either way (order tp-cp-ep-dp-pp with TP=CP=1), so the
    expert-parallel collectives are identical; only DP width changes, 120 -> 8.
    """
    spec = moe_prod_smoke(nnodes=1, train_iters=train_iters)
    spec.run_id = f"bisect_{num_layers}L_1n"
    # DP=8 here instead of 120, so keep 8 grad-accum steps per rank as in
    # production (gbs = dp * mbs * accum = 8 * 1 * 8).
    spec.schedule.global_batch_size = 64
    spec.model.num_layers = num_layers
    spec.model.moe_layer_freq = f"[0]*2+[1]*{num_layers - 2}"
    spec.schedule.lr_warmup_iters = 2
    spec.distributed_timeout_minutes = timeout_min
    return spec


def moe_bisect_15n(num_layers: int = 4, train_iters: int = 25,
                   timeout_min: int = 10) -> RunSpec:
    """Production topology at reduced depth, to isolate per-layer cost.

    Everything except `num_layers` matches `moe_prod_smoke`, so a comparison
    between depths changes exactly one variable.

    `timeout_min` defaults to 10 rather than the production 120. A hung
    collective costs exactly one timeout before it reports, so a long timeout
    makes each failed probe cost hours: the 48-layer run sat in a single
    ALLTOALL_BASE for the full 7 200 s before any rank said anything. Short
    timeouts are for bisection; production keeps a long one so that a slow
    iteration is never mistaken for a hang.
    """
    spec = moe_prod_smoke(nnodes=15, train_iters=train_iters)
    spec.run_id = f"bisect_{num_layers}L"
    spec.model.num_layers = num_layers
    spec.model.moe_layer_freq = f"[0]*2+[1]*{num_layers - 2}"
    spec.schedule.lr_warmup_iters = 2
    spec.distributed_timeout_minutes = timeout_min
    return spec

REGISTRY = {
    "dense_1b_1n": lambda: dense_1b(1),
    "dense_1b_2n": lambda: dense_1b(2),
    "dense_1b_15n": lambda: dense_1b(15),
    "moe_mini_1n": moe_1node_mini,
    "moe_full_1n": moe_1node_full,
    "moe_smoke_15n": moe_prod_smoke,
    "moe_prod_15n": rfull_moe_prod,
    # Single-node variants. EP=8 fits inside one node, so these keep the
    # expert-parallel dimension identical to production and change only the
    # data-parallel width (DP=8 instead of DP=120). That isolates "is the
    # stall in the EP alltoall?" from "is it in the 120-rank DP dimension?".
    "moe_bisect_12L_1n": lambda: moe_bisect_1n(12, 25),
    "moe_bisect_4L_1n": lambda: moe_bisect_1n(4, 25),
    "moe_bisect_4L": lambda: moe_bisect_15n(4, 25),
    "moe_bisect_12L": lambda: moe_bisect_15n(12, 25),
    "moe_bisect_24L": lambda: moe_bisect_15n(24, 12),
    "moe_bisect_36L": lambda: moe_bisect_15n(36, 12),
    "moe_bisect_48L": lambda: moe_bisect_15n(48, 12),
}

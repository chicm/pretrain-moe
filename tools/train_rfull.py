"""R-Full production training entry point.

Wires together the frozen pieces:
  * ``rfull.build_model``      -- 25.86B EP8 model with the ROCm-safe spec
  * ``rfull.data_scheduler``   -- O(1)-resume canonical sequence ordering
  * ``rfull.dataset``          -- real uint32 shard reads
  * ``rfull.aux_loss``         -- EP-global load balancing + z-loss
  * ``rfull.finite_consensus`` -- world-consensus finite check, no silent skip

Hard invariants enforced here (design sections 6, 10, 12):
  * a non-finite or absurd loss/grad aborts the update and the RUN; it is never
    silently skipped and never partially committed;
  * ``successful_updates`` advances only after a completed optimizer step;
  * data position is derived from ``successful_updates`` alone.

Smoke mode (``--smoke``) shrinks the world to a single node so MoE mechanics,
throughput and loss movement can be validated before the 120-GPU launch. It is
explicitly NOT production: EP8xDP1 is not EP8xEDP15.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import datetime
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rfull.aux_loss import ep_global_aux_loss, router_z_loss  # noqa: E402
from rfull.build_model import build_rfull_model, count_built_parameters  # noqa: E402
from rfull.data_scheduler import RFullScheduler  # noqa: E402
from rfull.dataset import (  # noqa: E402
    RFullBatchSource, ShardReader, build_plans,
)
from rfull.finite_consensus import check_update_or_die  # noqa: E402
from rfull.grad_reduce import allreduce_gradients  # noqa: E402
from megatron.core.transformer.moe.moe_utils import (  # noqa: E402
    clear_aux_losses_tracker,
)
from rfull.dispatcher_cleanup import clear_dispatcher_state
from rfull.checkpoint import (save_checkpoint, load_checkpoint,
                             latest_checkpoint, sweep_stale)  # noqa: E402
from rfull.model_spec import GEOMETRY  # noqa: E402
from rfull.router_tap import attach_router_tap, pop_router_logits  # noqa: E402

# Frozen coefficients (design section 6).
AUX_COEF = 0.001
Z_COEF = 0.0001

DEFAULT_DATA_ROOT = ("/scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore"
                     "/chec/pretrain/data")


def log0(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lr_at(update: int, cfg) -> float:
    """Warmup -> stable -> cosine decay, exactly as frozen."""
    if update < cfg.warmup_updates:
        return cfg.peak_lr * (update + 1) / cfg.warmup_updates
    if update < cfg.stable_until:
        return cfg.peak_lr
    prog = (update - cfg.stable_until) / max(1, cfg.decay_updates)
    prog = min(1.0, prog)
    return cfg.floor_lr + 0.5 * (cfg.peak_lr - cfg.floor_lr) * (
        1 + math.cos(math.pi * prog))


class Sched:
    warmup_updates = 2543
    stable_until = 228881
    decay_updates = 25432
    peak_lr = 2e-4
    floor_lr = 2e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="4k")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--global-seqs", type=int, default=None,
                    help="G; defaults to world_size * mbs * ga")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--reduce-bucket-mb", type=int, default=128,
                    help="coalesce gradients into buckets of this many MiB "
                         "before all-reduce; per-tensor reduction measured "
                         "212x slower on 2 nodes")
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument(
        "--no-a2a-serialise", action="store_true", dest="no_a2a_serialise",
        help="Disable the EP all-to-all serialisation workaround. Deadlocks "
             "on the current ROCm/RCCL build with a deep MoE stack; keep for "
             "re-testing after a runtime upgrade.")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--evidence", default="/scratch/rfull_stage/train_smoke_evidence.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--recompute", default="full",
                    choices=["none", "full", "selective"],
                    help="activation checkpointing; 'full' is required to fit "
                         "48 layers of 25.86B at seq 4096 with optimizer state")
    ap.add_argument("--ckpt-dir", type=str, default="/scratch/rfull_ckpt",
                    help="checkpoint root; committed state is named by "
                         "LATEST_COMMITTED inside it")
    ap.add_argument("--save-every", type=int, default=0,
                    help="commit a checkpoint every N updates; 0 disables")
    ap.add_argument("--keep-checkpoints", type=int, default=3)
    ap.add_argument("--resume", action="store_true",
                    help="resume from LATEST_COMMITTED if present")
    ap.add_argument("--coll-timeout", type=int, default=5400,
                    help="NCCL collective timeout in seconds; the default 600 "
                         "aborts healthy runs during latency spikes")
    args = ap.parse_args()

    # The default 600 s collective timeout is too tight for this stack. Deep MoE
    # steps on this ROCm/RCCL build have a ~1.7 s floor but intermittently take
    # 200-660 s, and a single spike past the watchdog aborts the whole world.
    # The spikes are a scheduling pathology, not a hang: split tables balance,
    # raw RCCL replays the same traffic cleanly, and progress always resumes.
    # Raise the ceiling so a slow step costs throughput instead of the run.
    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(seconds=args.coll_timeout))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % 8))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    ep = min(8, world)
    n_lanes = world // ep if world > ep else 1
    G = args.global_seqs or (n_lanes * args.micro_batch * args.grad_accum)
    log0(f"world={world} ep={ep} lanes={n_lanes} G={G} "
         f"mbs={args.micro_batch} ga={args.grad_accum} seq={args.seq_len}")

    # MCore parallel state must exist before any model construction: the MoE
    # layers query the EP group during __init__.
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep,
        context_parallel_size=1,
        # MCore creates the expert, data, and expert-data subgroups itself and
        # gives each its own timeout from this argument -- it does not inherit
        # the one passed to init_process_group. The stall happens inside
        # EXPERT_MODEL_PARALLEL_GROUP's all-to-all, so leaving this at the
        # default is exactly the case that matters, and a widened world timeout
        # alone changes nothing.
        distributed_timeout_minutes=max(1, args.coll_timeout // 60),
    )
    model_parallel_cuda_manual_seed(args.seed)

    # Eagerly build every NCCL communicator we will use. NCCL creates a
    # communicator on a group's first collective, and that creation is itself a
    # rendezvous: the first rank to arrive blocks until all peers arrive. When
    # ranks are skewed -- one still in a cold blobfuse read while peers finished
    # forward -- the early ranks burn the timeout waiting inside communicator
    # setup, which surfaces as the misleading
    #   "[15] is setting up NCCL communicator and retrieving ncclUniqueId
    #    from [0] ... store->get('0') got error: wait timeout after 600000ms"
    # That message names a store key, not a network fault; the real cause is a
    # peer that had not reached the same collective yet. Paying the rendezvous
    # here, while all ranks are provably at the same point, removes the ambiguity
    # from every later collective and makes a genuine hang mean what it says.
    warm = torch.ones(1, device=dev)
    for name, grp in (("world", None),
                      ("expert", ps.get_expert_model_parallel_group()),
                      ("data", ps.get_data_parallel_group()),
                      ("expert-data", ps.get_expert_data_parallel_group())):
        dist.all_reduce(warm, group=grp)
    torch.cuda.synchronize()
    dist.barrier()
    log0("all NCCL communicators established")

    # Deep MoE stacks on this ROCm/RCCL build suffer severe intermittent
    # latency spikes inside the EP all-to-all: a step whose floor is ~1.7 s can
    # take 200-660 s, and with the default 600 s collective timeout that
    # surfaces as an apparent deadlock. It is NOT a correctness problem --
    # 276 traced all_to_all_single calls had sum(in)==sum(out) and zero
    # pairwise split mismatches, and raw RCCL replaying the same tables ran
    # 200/200 iterations cleanly. Shape variation is also excluded: a
    # fixed-input run (identical GEMM shapes every step) spikes too.
    # Serialising the collective measured 12/12 steps across three runs where
    # the unserialised control died, so keep it on by default until the
    # underlying scheduling pathology is understood.
    if ep > 1 and not args.no_a2a_serialise:
        from rfull.a2a_serialise import enable_a2a_serialisation
        enable_a2a_serialisation()

    # ---- model ----------------------------------------------------------
    t0 = time.time()
    extra = {}
    if args.recompute == "full":
        extra.update(recompute_granularity="full", recompute_method="uniform",
                     recompute_num_layers=1)
    elif args.recompute == "selective":
        extra.update(recompute_granularity="selective")
    model = build_rfull_model(
        seq_length=args.seq_len, expert_model_parallel_size=ep, **extra)
    model = model.to(dev)
    cfg = model.config
    info = count_built_parameters(model)
    log0(f"model built in {time.time()-t0:.1f}s  "
         f"local_params={info['local_total']:,}  "
         f"global={info['reconstructed_global_total']:,}  "
         f"ledger={GEOMETRY.expected_total_params:,}  "
         f"attn={getattr(cfg,'rfull_attention_backend','?')}")
    if info["reconstructed_global_total"] != GEOMETRY.expected_total_params:
        raise SystemExit(
            f"parameter ledger mismatch: built "
            f"{info['reconstructed_global_total']:,} != frozen "
            f"{GEOMETRY.expected_total_params:,}")

    tap = attach_router_tap(model)
    n_moe_layers = tap.n_taps
    log0(f"router taps attached: {tap.n_taps} (expected {GEOMETRY.num_moe_layers})")
    if tap.n_taps != GEOMETRY.num_moe_layers:
        raise SystemExit(
            f"tapped {tap.n_taps} routers, expected {GEOMETRY.num_moe_layers}")

    # ---- data -----------------------------------------------------------
    t0 = time.time()
    plans = build_plans(args.data_root, args.seq_len)
    sch = RFullScheduler(args.seed, args.stage, plans, G)
    reader = ShardReader()
    lane = rank // ep if world > ep else 0
    src = RFullBatchSource(sch, reader, lane, n_lanes,
                           args.micro_batch, args.grad_accum)
    log0(f"corpus plans built in {time.time()-t0:.1f}s; "
         f"windows/source={[p.total_windows for p in plans.values()]}")

    opt = torch.optim.AdamW(model.parameters(), lr=Sched.peak_lr,
                            betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
    log0(f"post-optimizer-construct mem: "
         f"alloc {torch.cuda.memory_allocated()/2**30:.1f}GiB "
         f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB "
         f"(AdamW exp_avg+exp_avg_sq are allocated lazily on first step)")

    successful_updates = 0
    if args.resume:
        # The cursor is restored from the checkpoint payload, not from a
        # separate file, so weights and data position cannot drift apart. The
        # scheduler derives its position from this number alone.
        successful_updates = load_checkpoint(args.ckpt_dir, model, opt)
        if successful_updates:
            log0(f"resumed from update {successful_updates} "
                 f"({latest_checkpoint(args.ckpt_dir)})")
        else:
            log0("no committed checkpoint found; starting from scratch")
    hist = []
    for step in range(args.updates):
        lr = lr_at(successful_updates, Sched)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)

        # MCore's TopKRouter stashes its own aux/z losses in a module-level
        # tracker (save_to_aux_losses_tracker). Those entries hold live graph
        # tensors, and nothing in this loop consumes them -- R-Full computes its
        # own EP-global aux loss from the router tap instead. Left alone, the
        # next backward walks into the previous step's freed graph:
        #   RuntimeError: Trying to backward through the graph a second time
        # It passes update 1 and fails on update 2, which is exactly what the
        # first real two-node run did.
        # MCore's token dispatcher caches its per-step working set on the module
        # (probs, routing_map, split tables) and never releases it, leaving 332
        # live tensors per update with 46 MoE layers. Without this the process
        # is OOM-killed within a few updates.
        clear_dispatcher_state(model)
        clear_aux_losses_tracker()

        t_start = time.time()
        tot_loss = 0.0
        tot_ce = 0.0
        tot_aux = 0.0
        t_data = 0.0
        t_fetch0 = time.time()
        # The number of micro-batches is decided by the data source, which
        # derives it from the frozen curriculum (tokens per update / tokens per
        # micro-batch / lanes). --grad-accum is only a CLI override and defaults
        # to 1, so scaling the loss by it while looping src.ga times would both
        # mis-weight the average and, because the router tap is drained by the
        # first pop_router_logits(), leave every later micro-batch with an empty
        # aux term. Use the source's own count.
        n_micro = src.ga
        # Start every update from an empty tap. If a previous update aborted
        # between forward and backward -- a non-finite loss, a spike that tripped
        # a timeout, any raised exception -- its records are still buffered and
        # reference a graph that is already freed. They would then be appended to
        # by this update's forward, so the count comes out as a multiple of the
        # layer count (92 for 46 layers) and backward walks a dead graph. The
        # assertion below only catches that; this prevents it.
        tap.clear()
        for tokens, labels, wins in src.update_batches(successful_updates):

            t_data += time.time() - t_fetch0
            tokens = tokens.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)
            pos = torch.arange(tokens.size(1), device=dev).unsqueeze(0)
            # GPTModel(labels=...) returns per-token loss [b, s], not logits.
            tok_loss = model(input_ids=tokens, position_ids=pos,
                             attention_mask=None, labels=labels)
            ce = tok_loss.float().mean()

            # EP-global auxiliary balancing + router z-loss on this step's
            # router logits. Collected by the model spec during forward.
            aux = torch.zeros((), device=dev, dtype=torch.float32)
            recs = pop_router_logits(model)
            if len(recs) != n_moe_layers:
                raise RuntimeError(
                    f"router tap returned {len(recs)} records, expected "
                    f"{n_moe_layers} (one per MoE layer). An empty or short "
                    "list means the balancing loss is silently absent for this "
                    "micro-batch, which trains an unbalanced router while the "
                    "logged aux term still looks plausible.")
            for rlogits, ridx in recs:
                a, _ = ep_global_aux_loss(
                    rlogits, ridx,
                    num_experts=GEOMETRY.num_routed_experts,
                    topk=GEOMETRY.moe_router_topk,
                    ep_group=ps.get_expert_model_parallel_group(),
                    return_stats=False)
                z, _ = router_z_loss(rlogits, return_stats=False)
                aux = aux + AUX_COEF * a + Z_COEF * z
            if recs:
                aux = aux / len(recs)

            loss = ce + aux
            (loss / n_micro).backward()
            tot_loss += float(loss.detach()) / n_micro
            tot_ce += float(ce.detach()) / n_micro
            tot_aux += float(aux.detach()) / n_micro
            t_fetch0 = time.time()

        # Gradients must be averaged across the data-parallel groups BEFORE
        # clipping, otherwise every rank clips a different local gradient and
        # the ranks silently diverge.
        #
        # Reduce in coalesced buckets, not tensor by tensor. Measured on 2 nodes
        # with 1242 parameter tensors (3.301B elements): per-tensor all_reduce
        # cost 270.96 s per step, 128 MB buckets cost 1.28 s -- a 212x
        # difference, and per-tensor made communication 99.5% of the step.
        #
        # Routed experts are EP-sharded, so they are averaged over the
        # expert-data-parallel group; everything else over the regular
        # data-parallel group.
        t_r0 = time.time()
        allreduce_gradients(model, bucket_bytes=args.reduce_bucket_mb << 20)
        t_reduce = time.time() - t_r0

        # World-consensus finite check BEFORE any optimizer state is touched.
        # Rejection raises: no silent skip, no partial commit.
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        check_update_or_die(tot_loss, gnorm, dev, step=successful_updates)
        opt.step()
        successful_updates += 1

        # Commit AFTER the optimizer step and the increment, so the stored
        # cursor and the stored weights describe the same point in the run.
        if args.save_every and successful_updates % args.save_every == 0:
            t_c0 = time.time()
            save_checkpoint(args.ckpt_dir, successful_updates, model, opt,
                            extra={"seq_len": args.seq_len, "lr": lr})
            sweep_stale(args.ckpt_dir, keep=args.keep_checkpoints)
            log0(f"checkpoint committed at update {successful_updates} "
                 f"in {time.time() - t_c0:.1f}s")

        dt = time.time() - t_start
        toks = G * args.seq_len
        if step % args.log_every == 0:
            log0(f"upd {successful_updates:6d}  loss {tot_loss:8.4f}  "
                 f"ce {tot_ce:8.4f}  aux {tot_aux:.5f}  "
                 f"gnorm {float(gnorm):7.3f}  lr {lr:.3e}  "
                 f"{dt:6.2f}s  {toks/dt/1e3:8.1f}K tok/s  "
                 f"data {t_data:5.1f}s  redu {t_reduce:5.1f}s  "
                 f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}GiB")
        hist.append({"update": successful_updates, "loss": tot_loss,
                     "ce": tot_ce, "aux": tot_aux,
                     "grad_norm": float(gnorm), "lr": lr, "seconds": dt,
                     "tokens_per_s": toks / dt})

    if rank == 0:
        ev = {
            "mode": "smoke" if args.smoke else "run",
            "world": world, "ep": ep, "lanes": n_lanes, "G": G,
            "seq_len": args.seq_len, "updates": args.updates,
            "global_total": info["reconstructed_global_total"],
            "ledger_total": GEOMETRY.expected_total_params,
            "attention_backend": getattr(cfg, "rfull_attention_backend", None),
            "history": hist,
            "first_loss": hist[0]["loss"], "last_loss": hist[-1]["loss"],
            "loss_decreased": hist[-1]["loss"] < hist[0]["loss"],
            "all_finite": all(math.isfinite(h["loss"]) for h in hist),
            "peak_mem_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "batch_digest_0": sch.batch_digest(0),
        }
        ev["verdict"] = "PASS" if (ev["all_finite"] and ev["loss_decreased"]) else "FAIL"
        Path(args.evidence).write_text(json.dumps(ev, indent=1), encoding="utf-8")
        log0(f"EVIDENCE {args.evidence} verdict={ev['verdict']} "
             f"loss {ev['first_loss']:.4f} -> {ev['last_loss']:.4f}")

    reader.close()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

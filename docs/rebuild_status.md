# R-Full MoE rebuild — phase results

Every verdict records `n=` (number of runs). A single PASS/FAIL is anecdote,
not evidence — at a ~1/6 failure rate, n=1 carries no information.

## Environment (chec-mi300-7, verified 2026-08-29)

| item | value |
|---|---|
| nodes | 15 × 8 MI300X (192 GiB HBM) = 120 GPUs |
| torch | 2.7.0a0+git295f2ed, HIP 6.4 / ROCm 7.1 |
| Megatron-LM | `/scratch/rfull/megatron-lm` @ `5cb6dbb`, MCore **0.12.4**, clean |
| TE | 2.3.0.dev0 (`GroupedLinear` present) |
| flash-attn | 2.8.3 (TE gates at ≤2.8.1 → see shim) |
| grouped_gemm | MISSING → `--moe-grouped-gemm` uses the TE `GroupedLinear` path |
| corpus | 487 shards, int32, EOT=151643, **1.20 T tokens** |

## Phase 1 — dense 1B, multi-node transport  ✅ PASS (n=1)

`p1_dense1b_2n_0829_112656`, 2 nodes / 16 GPUs, TP1 PP1 DP16, gbs=128, seq=4096.

| metric | value |
|---|---|
| iterations | **30/30**, rc=0 |
| loss | 12.33 → 7.67 (monotone) |
| throughput | **209 TFLOP/s/GPU** steady |
| first step | 13.7 s (autotune) → **1.33 s** steady |
| nan / skipped | 0 / 0 |
| peak memory | 18.2 GiB allocated, 20.5 GiB reserved |

First-step cost is 10× steady state — any timeout budget must be
`first_step × 3 + steady × iters`, or a healthy run gets killed and
misfiled as a hang.

## Phase 1 at scale — 15 nodes / 120 GPUs  ✅ PASS (n=1)

`p1_dense1b_15n_0829_113957`, TP1 PP1 DP120, gbs=960, seq=4096.

| metric | value |
|---|---|
| iterations | **40/40**, rc=0 |
| loss | 12.33 → 7.67 |
| throughput | **206 TFLOP/s/GPU** (vs 209 on 2 nodes: −1.5% for 7.5× the GPUs) |
| first step | 498 s (autotune at 120 GPUs) → 6.4 s steady |
| nan / skipped | 0 / 0 |

Multi-node transport, rendezvous, RCCL over 8 IB HCAs, and the dataloader all
scale cleanly. **Phase 1 objective met.**

---

## Phase 2 — MoE  ✅ PASS (n=1 per config)

| run | config | result |
|---|---|---|
| `p2_moe_mini_1n` | 8 layers, 96 experts, EP8 | 30/30 iters, loss 10.24 → 7.78, 0 nan |
| `p2_moe_full_1n` | **48 layers, 96 experts, EP8** | builds + trains, 0 nan |

**The full R-Full model is 25.85 B total parameters / 4.586 B per rank**, which
matches the design doc's 4.59 B. It fits: 67 GiB of 192 GiB at mbs=1.

MoE mechanics verified working: alltoall dispatcher, TEGroupedMLP grouped GEMM,
aux-loss balancing, 12 experts per rank at EP8.

Note: `grouped_gemm` is **not installed**, but that is fine —
`--moe-grouped-gemm` with TE ≥ 1.7 selects `TEGroupedMLP`, which uses
`te.GroupedLinear` and never imports the missing package.

---

## Fixes required to get here

Each was a real defect, found and fixed with evidence.

### 1. `fused_kernels.load()` crashes on ROCm
`initialize.py` → `megatron/legacy/fused_kernels/__init__.py:load()` calls
`_get_cuda_bare_metal_version(CUDA_HOME)`, i.e. `None + "/bin/nvcc"` →
`TypeError`, killing rank 0 before training.
Upstream `load()` **builds nothing on any platform** — it defines
`_cpp_extention_load_helper` and never calls it. Shimmed to a no-op;
behaviour-preserving by inspection.

### 2. TE fused attention broken; unfused fallback is 2.6× slower
Measured (bf16, s=4096, b=2, h=32, GQA=4, d=128, fwd+bwd):

| backend | time/iter | peak mem | status |
|---|---|---|---|
| Fused (CK) | — | — | `RuntimeError: basic_string: construction from null` |
| Unfused | 14.0 ms | 6.76 GiB | works (TE's silent fallback) |
| **FlashAttention 2.8.3** | **5.4 ms** | **0.89 GiB** | works |

TE gates flash-attn at `≤2.8.1` **at import time**; when the gate fails the
`flash_attn_func` symbols stay `None`, so raising `max_version` later yields
`'NoneType' object is not callable`. Fix: patch
`importlib.metadata.version` to report 2.8.1 *before* TE is imported.

Correctness was **measured, not assumed** — vs an independent fp32 reference:

| backend | max_abs | mean_rel |
|---|---|---|
| unfused (bf16) | 0.01446 | 3.881e-03 |
| flash (bf16) | 0.01539 | **2.088e-03** |

Flash is *closer* to fp32 than the backend it replaces; the delta is bf16
rounding, not a defect. Selected via `--attention-backend flash` (Megatron
sets `NVTE_*` itself and asserts if the caller presets them).

### 3. Dataset cache assumed a shared filesystem
`GPTDataset` builds document/sample/shuffle indices on **global rank 0 only**;
all other ranks expect to read them off disk. The cache cannot live on
blobfuse, because it is read with `numpy.load(mmap_mode='r')` and a failed
mmap page fault on FUSE delivers SIGBUS/SIGSEGV to the faulting thread —
uncatchable, and it presents as a cluster-wide deadlock.
Fix: `tools/warm_cache.py` builds the cache once and rsyncs it to all nodes;
the cache key is a pure function of the run args, so it is bit-identical.

### 4. `.idx` is always mmap'd
`--no-mmap-bin-files` only covers `.bin`; `_IndexReader` mmaps every `.idx`
unconditionally (`indexed_dataset.py:253`). All 487 `.idx` files (16.4 GiB)
are therefore staged to node-local ext4, while the 3.8 TiB of `.bin`
payload stays on blobfuse and is read with pread.

### 5. `--eval-iters 0` empties the validation split
Megatron always builds a validation dataloader and
`MegatronPretrainingSampler` asserts `total_samples > 0`, so `eval_iters=0`
kills every rank with `no sample to consume: 0`. Minimum is 1.

### 6. Launcher bugs (mine, not Megatron's)
- A literal `\n` in the argv join corrupted the command line;
  argparse got `--tensor-model-parallel-size 'n'`, rank 0 exited 2 and every
  other rank died of SIGTERM **as a bystander**.
- `ssh` never closed its channel because the child held stdout, so only
  node-0 launched. The remote script now does `exec > log 2>&1 < /dev/null`.
- Process probes grepped `pretrain_gpt.py` after the entry point moved to
  `pretrain_entry.py`, so every check reported `trainers=0` while **91
  orphaned processes** held GPUs across the cluster.
- `pgrep -c … || echo 0` prints *two* lines when nothing matches.
- Node-0 lacked the compiled dataset helpers that nodes 1–14 had —
  fixed by compiling on all 15, not by special-casing node-0.

### 7. Megatron logs progress from the LAST rank
`print_rank_last` puts the `iteration N/M` line on the highest global rank,
which lives on the **last** node. Scanning only `node0.log` shows zero
iterations for a perfectly healthy run — indistinguishable from a hang.

### Non-fix
`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` is **rejected** by this
ROCm build (`option expandable_segments is unrecognized`). Removed. An env
var the runtime ignores cannot be credited with any effect.


### 8. Kernel autotune dominates short runs (the big one)

Per-round timing of a single MoE layer, identical shapes every round:

| round | time |
|---|---|
| 0 | **16 909 ms** |
| 1 | 27 ms |
| 2–13 | 22–25 ms |

A **676× first-call penalty**, paid per distinct kernel shape. With 48 layers
this is ~800 s of one-time cost, which is why 4–12 iteration runs looked
catastrophically slow (373 s/iter) and why "iteration 2 is slower than
iteration 1" kept appearing. It is warmup, not a defect.

Consequence for measurement discipline: **any MoE timing from fewer than ~15
iterations is measuring autotune, not throughput.** Timeout budgets must be
`first_step × 3 + steady × iters`.

This also invalidated an earlier hypothesis of mine: I suspected dropless
routing was forcing hipBLASLt to re-tune on every step because expert token
counts vary. Measured directly — fixed shapes 2.4 ms vs variable shapes 2.4 ms.
**Refuted.** Variable shapes are free; only the first call is expensive.

### 9. micro_batch_size is bounded by memory, not GEMM efficiency

Grouped-GEMM throughput improves a lot with more tokens per expert:

| tok/expert | mbs | TFLOP/s |
|---|---|---|
| 256 | 1 | 48.0 |
| 1024 | 4 | 79.2 |
| 2048 | 8 | 139.2 |
| 4096 | 16 | 218.5 |

So I set mbs=8 — and it OOMed at 120 GPUs:
`HIP out of memory ... 191.45 GiB of which 722.00 MiB is free`.

Reason: gbs=960 over DP=120 is exactly 8 sequences per rank, so mbs=8 collapses
gradient accumulation to a single micro-step with 32 768 tokens in flight,
needing ~160 GiB of activations. Reverted to **mbs=1 with 8 accumulation
steps**, plus `--recompute-granularity selective`. Peak VRAM now **82 GiB of
192**.

The 1-node test did *not* catch this: at DP=8 it had gbs=64/mbs=1 and therefore
8 accumulation steps already. **A 1-node test with a different accumulation
depth is not a memory test for the 120-GPU config.**

### 10. Diagnostic tooling defects (mine)

- The VRAM probe grepped `GPU memory use (%)`, but this `rocm-smi` prints
  `VRAM Total Used Memory (B)`. It therefore reported an **idle cluster while
  8 GPUs ran at 100% with 36 GB each**.
- `PYTHONFAULTHANDLER=1` does *not* make SIGUSR1 non-fatal — it only handles
  fatal signals. Sending SIGUSR1 to inspect a stalled rank **killed it**
  (exitcode −10) and every peer died of SIGTERM as a bystander. Fixed with
  `faulthandler.register(SIGUSR1, chain=False)`, which now yields a real
  stack (`custom_backward`) instead of a guess.
- `find /` to locate a binary walked the 3.8 TiB blobfuse mount and returned
  nothing useful. Ask `pip show -f` instead of searching a tree with a FUSE
  mount in it.


---

## Phase 3 — 120-GPU MoE  ✅ TRAINING

Production launched: `rfull_moe_prod_0829_174026`, 15 nodes / 120 GPUs,
4 585 502 720 params per rank (design doc: 4.59 B), peak **82 GiB of 192 GiB**.

Verified at 4 layers on the production topology before scaling up
(`bisect_4L_0829_172043`, 120 GPUs):

| iteration | 1 | 2 | 3 | … | 15 |
|---|---|---|---|---|---|
| lm loss | 12.33 | 11.52 | 10.90 | … | **7.97** |

Loss decreases monotonically, 0 NaN, 0 skipped. Best iteration 3.2 s
(36.6 TFLOP/s/GPU); iteration time is still variable (3–83 s), which is a
throughput question, not a correctness one.

### Two throughput defects found by bisection

Both were *my* configuration choices, not Megatron bugs, and both were found by
reducing the model to 4 layers and changing one variable at a time.

### Established by measurement

| finding | evidence |
|---|---|
| Model builds and fits at 120 GPUs | 4.586 B params/rank (design doc says 4.59 B); peak 82 GiB of 192 GiB at mbs=1 |
| EP topology is optimal | order `tp-cp-ep-dp-pp`, TP=CP=1 → EP group = ranks 0–7 = one node (xGMI, not IB) |
| Not a data-loading problem | aggregate `rchar` delta = **0 MiB/s** during a stall; 0 processes in FUSE wait |
| Not a compile problem | Triton cache static (419 → 419 files); load 17 of 96 cores |
| Not the DP collective | 8.5 GiB reduce-scatter ≈ 3 s at IB speed, and `--overlap-grad-reduce` is on |
| Not the optimizer | `optimizer` timer = **49 ms** |
| Not expert imbalance | all 15 nodes identical: 42 % GPU, ~2 cores, no laggard |
| No watchdogs, no divergence | all 120 ranks aligned on the same frame |

### Root cause found and fixed: timers insert global barriers

`--timing-log-level 1` was added to get a phase breakdown. Megatron passes
`barrier=True` for several timers, and `Timer.start` then calls
`torch.distributed.barrier()` + `torch.cuda.synchronize()`
(`core/timers.py:135`). At level 0 these are `DummyTimer` no-ops; at level ≥ 1
they become real global sync points across all 120 ranks and destroy
compute/communication overlap.

With timers **on**, 4 layers never completed an iteration.
With timers **off**, the same config produced `it1 = 50.7 s`, `it2 = 10.0 s`.

**Never leave `--timing-log-level ≥ 1` on for a production run.** Use it only
for a short, deliberate profiling run, and read the numbers knowing they are
inflated by the barriers they introduce.

### Still open: erratic iteration time

Even with timers off, the 4-layer run went `50.7 s → 10.0 s → >9 min`. During
the long iteration:

* every rank is pinned at `moe_utils.py:296 permute`
  (`tokens.index_select(0, sorted_indices)`), reached via
  `token_dispatcher.py:499 token_permutation`;
* GPUs sit at **21 %** (busy, but far from saturated);
* the same `permute` call benchmarks at **0.19 ms** in isolation at production
  shapes — about 0.1 s per iteration in total.

A Python frame that is 0.19 ms in isolation but blocks for minutes in situ is
almost certainly **not** the real consumer: it is the first synchronising point
after a large amount of queued asynchronous GPU work. The dispatcher calls
`_maybe_dtoh_and_synchronize` immediately before this line, which forces a
device-to-host copy of `tokens_per_expert` — the natural place for previously
queued work to come due.

**Next step:** attribute the queued work with a GPU-side profile
(`rocprofv3 --kernel-trace` on one rank for a few iterations, or
`torch.profiler` with `activities=[CUDA]` limited to 2 iterations), rather than
inferring from Python frames. The Python stack has now been sampled repeatedly
and always points at the same synchronisation boundary, which is exactly what
that boundary would look like regardless of the true cause.

### Deliberately reverted

* `--recompute-granularity selective` — added for an OOM whose real cause was
  `mbs=8`; with mbs=1 peak is 82 GiB, so it was removed as an extra variable.
  A controlled A/B (recompute on vs off, all else equal) showed no difference
  in the stall.
* `micro_batch_size=8` — 2.9× better grouped-GEMM throughput, but needs
  ~160 GiB of activations at DP=120 and OOMs. Memory is the binding constraint.


### GPU-side attribution (torch.profiler, 1 production MoE layer, EP8, fwd+bwd)

Self CUDA time total **64.4 ms for 3 iterations = 21.5 ms/layer**, matching the
isolated benchmark. Where it goes:

| item | CUDA time | share |
|---|---|---|
| `ncclDevKernel_Generic` | 15.3 ms | 23.7 % |
| `_GroupedLinearBackward` | 14.8 ms | 23.0 % |
| `nccl:all_to_all` (18 calls, 786 µs each) | 14.1 ms | 22.0 % |
| `_GroupedLinear` | 9.2 ms | 14.3 % |
| `FusedAttnFuncBackward` | 8.5 ms | 13.2 % |

**~45 % of GPU time is RCCL**, even though EP=8 keeps the alltoall inside a
single node. The expert GEMMs are only ~37 %. This is the structural cost of
dropless alltoall MoE at 12 experts/rank with ~256 tokens per expert, and it
bounds what any tuning can achieve at this geometry.

Note the profile also shows `FusedAttnFunc` / `ck_tile FmhaBwd` kernels: in a
bare microbenchmark TE picks its own backend, whereas the training runs force
`--attention-backend flash`. The two are not directly comparable.


### Defect: `--moe-per-layer-logging` with `--log-interval 1`

The flag makes `track_moe_metrics` all-reduce the aux-loss tracker **per MoE
layer**, so at `--log-interval 1` every layer adds a collective on every step.

| config | result at 4 layers / 120 GPUs |
|---|---|
| per-layer logging ON | 2 iterations in ~20 min, then stalls of 4–10 min |
| per-layer logging OFF | 15 iterations in ~9 min, loss 12.33 → 7.97 |

Removed from the default path. Enable only for a short diagnostic run.

### Defect: `--timing-log-level 1`

Megatron passes `barrier=True` for timers on hot paths, and `Timer.start` then
calls `torch.distributed.barrier()` + `torch.cuda.synchronize()`
(`core/timers.py:135`). At level 0 these are `DummyTimer` no-ops; at level ≥ 1
they become global sync points across all 120 ranks.

With timers **on**, 4 layers never completed an iteration. With timers **off**,
the same config gave `it1 = 50.7 s`, `it2 = 10.0 s`.

Ironic and worth remembering: **the instrumentation added to find the slowness
was itself a large part of the slowness.** Profile with a short, deliberate run
and read the numbers knowing the barriers inflate them.

### Ruled out by measurement (do not re-investigate)

| hypothesis | evidence against |
|---|---|
| Data loading over blobfuse | aggregate `rchar` = **0 MiB/s** during stalls; 0 procs in FUSE wait |
| torch.compile / Triton JIT | Triton cache static (419 → 419 files); load 17 of 96 cores |
| DP gradient collective | 8.5 GiB reduce-scatter ≈ 3 s; `--overlap-grad-reduce` on |
| Optimizer step | `optimizer` timer = **49 ms** |
| Expert imbalance / a slow node | all 15 nodes identical: 42 % GPU, ~2 cores, no laggard |
| EP alltoall crossing nodes | order `tp-cp-ep-dp-pp`, TP=CP=1 → EP group = ranks 0–7 = one node |
| Dropless variable shapes re-tuning | fixed shapes 2.4 ms vs variable 2.4 ms — identical |
| Activation recompute | A/B with all else equal showed no difference |
| `check_grads` NaN scan | disabled via `--no-check-for-nan-in-loss-and-grad`; stall persisted until the logging fix |

### Remaining work

Iteration time still varies 3–83 s. The GPU profile shows ~45 % of layer time
in RCCL (`ncclDevKernel_Generic` 23.7 %, `nccl:all_to_all` 22.0 %) against
~37 % in the expert GEMMs, so the alltoall is the structural ceiling at this
geometry. Next step is a `rocprofv3 --kernel-trace` on one rank across several
iterations to see whether the slow iterations differ in kernel mix or purely in
collective wait time.


## The 120-GPU stall: what it is not (2026-08-30)

Symptom: training produces iterations, then freezes for minutes at a time.
Deeper models freeze longer. Not a crash -- it recovers.

### The measurement that actually discriminates

`rocm-smi --showuse` reports **100 % busy while RCCL spins waiting**, so it
cannot tell "computing" from "waiting". Power draw can:

| state | MI300X power | variance |
|---|---|---|
| idle | ~130-150 W | - |
| real GEMM work | 400-700 W | large, sd > 40 W |
| **RCCL spin-wait** | **250-270 W** | **flat, sd < 1 W** |

During a stall, 20 samples gave `mean=268 W sd=0.2 W` at 100 % "use". That is
the signature: the GPUs are burning power in a spin loop, not computing.

### Ruled out, each by direct measurement

| hypothesis | evidence against |
|---|---|
| node-5 is a bad node | bare 8-GPU variable-split alltoall: 300 iters in 10.6 s, same as node-0 |
| the interconnect is unhealthy | bare 120-rank collectives: all_reduce(1) **0.3 ms**, reduce_scatter(480 MiB) **4.6 ms**, all_gather **2.8 ms**, barrier **0.33 ms** |
| EP alltoall with data-dependent sizes | bare 120-rank EP probe (dispatch+combine, uneven splits, DP all_reduce every 8): **400 iters in 15.3 s**, 2 stalls, both at init |
| the training env block (RCCL/HSA vars) | same probe **with** the full env block: 15.7 s vs 15.3 s. No effect |
| grad-norm apex fallback | `amp_C` is genuinely missing, so Megatron uses `local_multi_tensor_l2_norm` -- but that costs **40.4 ms** for 1104 large tensors, and `torch._foreach_norm` is no faster (44 ms) |
| a depth threshold | wrong: 4L stalls too (iteration 4 took 201 s). Depth changes duration, not presence |

### Two process errors worth not repeating

**The stack frame is a synchronisation point, not the cost.** Every rank sat in
`clip_grads.py get_grad_norm_fp`, so I attributed the stall to grad-norm --
then measured it at 40 ms. It is simply the first `.item()` after the backward,
where all queued async work settles. A frame that is milliseconds in isolation
but minutes in situ is a barrier, not a culprit; switch to GPU-side attribution
instead of sampling more Python stacks.

**A probe that reports nothing yet is not a probe that hung.** I declared the
env-block arm "stuck -- 12 procs, zero output" and was about to conclude the
env block caused the stall. It finished in 15.7 s; I had polled too early.
Both arms passed. Poll until the process exits, then read the result.


## Three fixes tried against the stall, all refuted (2026-08-30)

Each was plausible, each was reverted, and the reverts matter more than the
attempts: two of them made things measurably worse.

| change | theory | result |
|---|---|---|
| `--moe-expert-capacity-factor 1.25` + `--moe-pad-expert-input-to-capacity` | dropless routing varies the grouped-GEMM shape every step, so ROCm re-picks hipBLASLt heuristics per shape | 36 layers: still 0 iterations at T+14 min, `cu_occ=0` |
| `--ddp-bucket-size 200M` + `--overlap-grad-reduce` | one unbucketed ~8.5 GiB reduce-scatter after backward, cost scaling with depth | see below |
| both together | - | **regression**: 12 layers reached iteration 1 in 252 s without them, and produced nothing in 10 min with them |

The 12-layer control is the important measurement. It is the configuration
known to work, so running it with the new flags isolated their effect --
and they were harmful, not neutral. Reverted; a rerun then reproduced
iteration 1 at **233 s** (baseline 252 s), confirming the revert restored the
prior behaviour rather than merely appearing to.

**Process failure worth naming:** I changed three things at once (capacity
factor, DDP bucketing, TensorBoard) and launched at 36 layers, where a single
attempt costs ~15 minutes. When it failed I could not attribute the failure to
any one change. The fix was to go back to the cheap 12-layer control and change
one variable. That is the same lesson as the earlier `--timing-log-level`
episode, and I did not apply it the second time.

### Current state, stated honestly

- The stall is **not** explained. It is present in the original configuration
  too: 12 layers reaches iteration 1, then iteration 2 has not appeared after
  4 minutes.
- It is **not** a bad node, the interconnect, the EP alltoall, the env block,
  grad-norm, DDP bucketing, or expert capacity -- each ruled out by direct
  measurement (see the table above and the previous section).
- It scales with depth: 4 layers stalls intermittently and recovers, 12 layers
  stalls after iteration 1, 36 and 48 layers have never produced an iteration.
- During a stall, **CU occupancy is 0 on every GPU on every node** while power
  sits flat at 248-268 W and `rocm-smi --showuse` reports 100 %. No kernel is
  running anywhere; every rank is waiting.

### Next step

Get GPU-side attribution rather than more Python stacks. `rocprofv3` is not on
PATH; find it in the ROCm install or use `torch.profiler` with
`ProfilerActivity.CUDA` around iterations 1-3 on one rank, dumping a trace on
first stall. The question to answer is which collective is outstanding when
occupancy hits zero -- the Python frame has pointed at three different
synchronisation points so far, and none of them was the cost.


## The stall is in the DP dimension, not the model (2026-08-30)

The experiment that finally isolated it: **the same 12-layer model, the same
EP=8 group, changing only data-parallel width.**

EP=8 is intra-node under the `tp-cp-ep-dp-pp` order with TP=CP=1, so the
expert-parallel collectives are byte-identical between the two arms. The only
difference is DP=8 versus DP=120.

| arm | result |
|---|---|
| **1 node, DP=8** | **25/25 iterations, loss 12.346 -> 7.743, median 5.5 s/iter** |
| 15 nodes, DP=120 | iteration 1 (107 s), then nothing for 7+ minutes |

So:

- The MoE model, router, dispatcher, grouped GEMM, expert parallelism and the
  whole training loop are **correct** -- they converge cleanly.
- The failure is specific to the 120-rank data-parallel dimension.
- It is not the interconnect per se: bare 120-rank collectives are healthy
  (reduce_scatter 4.6 ms, all_gather 2.8 ms, barrier 0.33 ms) and a bare
  120-rank EP alltoall probe ran 400 iterations in 15 s.

That combination -- healthy collectives in isolation, healthy model at DP=8,
stall only when both are combined -- points at the interaction between the
distributed optimizer's DP-wide operations and the MoE parameter structure,
rather than at any single component.

Single-node throughput is also a useful reference point that did not exist
before: **median 5.5 s/iter at 12 layers**, with one 47.7 s outlier in 24
iterations, so the intermittent stall exists at DP=8 too -- just rarely enough
to make progress. Depth and DP width both increase its frequency.

### Also fixed here

- TensorBoard is confirmed working end to end: 73 KB of events written by the
  last rank, no `--log-timers-to-tensorboard` (which would re-enable the
  barrier-inserting timers).
- `pack.py` now writes both `_deploy.tar.gz` and the `_deploy.b64` that
  `_deploy.py` consumes, and `_deploy.py` refuses a b64 older than its sources.
  Before this, deploys silently shipped hours-old code while reporting success.


## DP-width scan: the stall is a continuum, not a 120-rank cliff

Same 12-layer model, same EP=8 group, 8 grad-accum steps per rank in every arm.
Only the data-parallel width changes.

| DP width | nodes | result |
|---|---|---|
| 8 | 1 | **25/25 iterations**, loss 12.346 -> 7.743, median 5.5 s/iter (one 47.7 s outlier) |
| 16 | 2 | **14 iterations**, loss 12.345 -> 8.044, 3.5-8.6 s/iter, then stalled at iteration 15 for 6+ min |
| 120 | 15 | iteration 1 (107 s), then nothing |

So the stall exists at every width; what changes is how many iterations run
before it hits. That rules out a threshold effect and, with it, the whole class
of "something breaks above N ranks" explanations.

Two more single-variable arms, both refuted:

| variable | result |
|---|---|
| `--use-distributed-optimizer` off (15 nodes) | still 0 iterations at T+9 min |
| global batch 960 -> 120, no grad accumulation (15 nodes) | still 0 iterations at T+11 min |

The second matters because the 1-node arm had necessarily changed *two* things
(DP width and global batch, since gbs must divide by DP width). Holding DP at
120 and moving only the batch showed the batch was not the variable.

### Where it stalls, and why that frame is a red herring

At DP=16 both nodes sit in `clip_grads.py:103 get_grad_norm_fp32`, inside
`multi_tensor_applier(l2_norm_impl, ...)`. Benchmarked on the real MoE tensor
mix (many small per-expert grads, not the uniform large tensors I used the
first time):

| shape mix | apex fallback | `torch._foreach_norm` |
|---|---|---|
| 12 layers, 270 tensors | 28.2 ms | 14.5 ms |
| 48 layers, 1242 tensors | 35.3 ms | 3.3 ms |

35 ms cannot produce a 6-minute stall. `get_grad_norm_fp32` ends in a DP-wide
`all_reduce` of the norm, so it is simply the first place a rank must wait for
every other rank. It is the *rendezvous*, not the cost -- the third distinct
synchronisation point this stall has hidden behind (`token_permutation`,
`custom_backward`, now grad-norm).

The `amp_C` module genuinely is missing, so Megatron uses its Python fallback.
That is worth fixing for throughput (14.5 -> 3.3 ms at 48 layers) but it is not
the stall.

### The useful outcome

There is now a **2-node reproducer** that stalls within ~15 iterations and
costs 3 minutes per attempt, instead of a 15-node one costing 15 minutes. Any
further work on this should use it.


## Grad-norm ruled out, and the stall's real character

`clip_grad > 0` is the only caller of `get_grad_norm_fp32`
(`optimizer.py:483`), so setting `--clip-grad 0.0` removes the DP-wide norm
all-reduce entirely. On the 2-node reproducer:

| iteration | 7 | 8 | 9 | **10** | 11 | 12 | **13** | 14 |
|---|---|---|---|---|---|---|---|---|
| seconds | 5.9 | 10.6 | 9.3 | **40.4** | 6.7 | 3.9 | **72.4** | 6.8 |

The stalls survive with clipping off. Grad-norm was never the cost -- it was
just where ranks happened to meet. That is now three synchronisation points
this stall has hidden behind (`token_permutation`, `custom_backward`,
`get_grad_norm_fp32`), which is exactly what a straggler looks like from the
Python level.

### The important property: stalls always recover

22 of 25 iterations, loss **12.345 -> 7.900**, median **7.8 s**, P90 25 s,
worst 72.4 s, and **10 % of iterations exceed 30 s**. Nothing hangs
permanently. Everything that looked like a deadlock earlier was this same
intermittent stall, observed for too short a window -- at 48 layers the first
iteration is long enough that a 10-minute poll never saw the far side of it.

So the correct characterisation is: **training is functional and converges;
roughly one iteration in ten pays a large latency penalty.** That is a
throughput problem, not a correctness or liveness one, and it should not block
production -- it should be budgeted for and investigated separately.

Also worth noting from the DP=8 arm: it showed one 47.7 s outlier in 24
iterations (~4 %). The stall frequency grows with DP width but is present at
every scale, which again argues for a straggler/jitter mechanism rather than a
scale threshold.

### Networking definitively cleared

- 8 IB HCAs per node, all `mlx5_ib0..7`, all **400 Gb/s NDR**, all ACTIVE.
- `packet_seq_err`, `local_ack_timeout_err`, `implied_nak_seq_err` all **0**,
  and not growing.
- RCCL uses all 8 evenly (12 references each in a 2-node NET trace), and does
  not touch the 100 Gb/s `mlx5_an0`.
- Bare 120-rank collectives: reduce_scatter 4.6 ms, all_gather 2.8 ms,
  barrier 0.33 ms.


## Production 48L attempt: genuine deadlock, and a corrected verdict

Launched `moe_prod_15n` on 15/15 nodes at 06:16. Model built correctly
(4,585,502,720 params/rank, 25.86B total), reached `training ...`, TensorBoard
opened. Then **0 iterations for 2h20m**, killed at 08:33.

This was a real deadlock, and it means the optimistic reading in the previous
section was wrong for 48 layers.

### Evidence (both liveness criteria met)

| probe | result |
|---|---|
| iterations | 0 for 129 min |
| TensorBoard | frozen at 48,633 bytes since 06:17 |
| `/proc/<pid>/io` over 45 s | node-0 **+178 KB**, node-7 **+0 bytes** |
| CPU jiffies over 20 s | utime 1024 -> 1025, stime 577 -> 578 (static) |
| GPU power | **245-260 W, sd < 8 W** = RCCL spin-wait, not compute |
| stack, nodes 0/7/14 | all in `schedules.py:151 custom_backward` |
| log sizes, 12 of 15 nodes | identical to the byte (15,338) |

Progress has stopped *and* ranks are misaligned (node-0 still trickling io,
node-7 completely still), which is the deadlock definition, unlike the 2-node
case where every stall recovered.

### The 120-minute watchdog never fired

`distributed_timeout_minutes = 120`, yet at T+129 min **all 15 nodes reported
zero watchdog lines**. The hang is therefore somewhere the RCCL watchdog does
not police -- consistent with the backward pass blocking inside a kernel or an
autograd wait rather than inside a monitored collective. Worth noting that a
timeout which does not fire is itself a bug: it turns a crash into an
indefinite occupancy of 120 GPUs.

`node-0`'s extra 36 KB was only rank-0's argument dump, and node-7/14's extra
bytes were my own SIGUSR1 traces. No genuine outlier node -- everything is
stuck together.

### What this changes

The scan now reads:

| layers | DP width | result |
|---|---|---|
| 12 | 8 | 25/25, converges |
| 12 | 16 | 22/25, converges, ~10 % of iterations stall but recover |
| 12 | 120 | 1 iteration, then stall |
| **48** | **120** | **0 iterations, hard deadlock** |

So **depth is a second independent factor**, not just DP width. The earlier
"stalls always recover" conclusion holds for 12 layers and does not generalise
to 48. I should not have launched production on the strength of a 12-layer
result -- the honest reading of the 15-node 12L arm (1 iteration then stall)
was already a warning that 120-wide is qualitatively different.

### Next

The next experiment must change depth alone at fixed DP=16 on the cheap 2-node
reproducer: 12L (known good) -> 24L -> 48L. If 48L/DP=16 deadlocks, the
reproducer costs 2 nodes instead of 15 and the DP-width dimension drops out of
the problem entirely.


## Depth isolated: 48L reproduces on 2 nodes

Ran `moe_bisect_48L_2n` -- 48 layers at DP=16, the same width where 12 layers
ran 22/25 and converged. Depth is the only variable.

**It reproduces the production failure exactly.**

| arm | layers | DP | iteration 1 | after that |
|---|---|---|---|---|
| `bisect_12L_2n_noclip` | 12 | 16 | ~6 s | 22/25, loss 12.345 -> 7.900 |
| `bisect_48L_2n` | 48 | 16 | **310.3 s @ 2.4 TFLOP/s** | iteration 2 unfinished after **12 min** |
| `rfull_moe_prod` | 48 | 120 | never | 0 iterations in 2h20m |

So the 120-GPU deadlock is **not about DP width at all**. Everything I attributed
to scale -- the DP=8/16/120 gradient, "the stall grows with DP width" -- was
depth doing the work, since the only 48-layer arm I had ever run was also the
only 120-wide one. Width and depth had been confounded in every production
attempt.

### The 2.4 TFLOP/s number is the real finding

Dense 1B sustains 206 TFLOP/s/GPU on this cluster. This MoE run does **2.4**,
roughly 1 % of that, on an iteration that did complete correctly. That is not a
hang -- it is arithmetic running at a pathological rate, and the "deadlock" is
just that rate applied to a 120-wide gradient reduction.

Supporting evidence at the moment of the stall:

- Power **246-268 W** on both nodes, sd < 8 W -- RCCL spin-wait.
- node-0 in `custom_backward`, node-1 in `token_permutation` **at the same
  instant** -- the ranks are in different phases, i.e. genuinely diverged rather
  than symmetrically waiting on one collective. That is the straggler shape,
  and it explains why the stall frame kept moving between sync points.

### Cost of the correction

The 2-node arm reproduces in **9 minutes** what the 15-node production run took
**140 minutes** to show, at 1/7 the hardware. This should have been the first
experiment after 12L passed; instead depth was only ever changed together with
width, which is the same one-variable discipline failure as the earlier
capacity-factor/DDP-flags episode.

### Next

Profile the 48L forward/backward directly on 2 nodes and find what costs 100x.
The earlier single-layer profile (RCCL 45 %, `_GroupedLinearBackward` 23 %,
21.5 ms/layer) predicts ~1 s for 48 layers, not 310 s, so the profile does not
yet explain the run -- something that does not appear at one layer is dominating
at 48. Candidates in order: per-layer RCCL group count scaling with depth,
`--moe-token-dispatcher-type allgather` as a single-variable swap, and the
missing `amp_C` fused kernels.


## ROOT CAUSE: `--moe-grouped-gemm`

Single-variable test on 48L/1n -- same config, same node, same data cache, only
this flag differs:

| `--moe-grouped-gemm` | iteration 1 | after that | TFLOP/s | iterations > 30 s |
|---|---|---|---|---|
| **on** | never completed | 0 iterations, all 8 ranks spinning in `backward_step` at ~250 W | 2.4 | 100 % |
| **off** | 52.3 s | **22/25, loss 12.338 -> 7.797** | **82.5** | **0/21** |

A 34x throughput difference and the difference between running and not running.

### Why every earlier hypothesis missed it

The stall reproduces on **one node with no inter-node traffic**, which retired
the entire networking line of enquiry in a single run. The reason it looked
like a distributed problem for so long is that the failure only manifests at
depth, and depth had never been varied independently -- the sole 48-layer
config was also the sole 120-GPU config.

The moving stack frames (`permute`, `custom_backward`, `get_grad_norm_fp32`)
were all downstream: with one rank stuck inside a fused per-expert GEMM, the
other 119 pile up at whichever collective comes next, and which one that is
depends on when each rank arrived.

These were tested and cleared along the way, each as a single variable:
capacity factor (twice), DDP bucket size, overlap-grad-reduce, distributed
optimizer, global batch size, gradient clipping, token dispatcher
(`alltoall` -> `allgather`, still stalls), IB fabric, and GPU memory
(65 GB of 192 GB used, zero allocator retries).

### Production run

`rfull_moe_prod_0830_101828`, 48 layers, 25.85B parameters, 120 GPUs:

- **33 iterations in 8 minutes**, loss **12.3330 -> 12.1911**, decreasing
  monotonically
- median **9.9 s/iter**, max 16.3 s, **0 of 30 iterations over 30 s**
- **76.5 TFLOP/s/GPU** median, peak 79.6
- TensorBoard growing (80,938 bytes)
- ETA for the full 203,451-step schedule: **~23 days**

The grouped path stays switchable (`moe_grouped_gemm`) so it can be retested on
a future ROCm build.


## Production running

`rfull_moe_prod_0830_103353` -- 48 layers, 25.85B parameters, 120 GPUs
(15 nodes x 8 MI300X), grouped GEMM off.

| metric | value |
|---|---|
| iterations | 33 and climbing |
| loss | **12.3330 -> 12.1936**, monotone (first-half mean 12.3258, second-half 12.2566) |
| step time | median **9.9 s**, max 11.1 s, **0 of 30 over 30 s** |
| throughput | **76.7 TFLOP/s/GPU** median |
| TensorBoard | serving HTTP 200 on node-14:6006, events file growing |
| ETA | **~23 days** for the full 203,451-step schedule |

Compare with the previous production attempt on the same hardware and config,
differing only in `--moe-grouped-gemm`: 0 iterations in 2 h 20 min.

### Self-inflicted kill, worth recording

An earlier launch of this same fixed config ran 37 clean iterations at
75.5 TFLOP/s and then died -- because I ran `pkill -f '[t]ensorboard'` to
restart the TensorBoard *server*. The trainer's argv contains
`--tensorboard-dir /scratch/.../tensorboard`, so `-f` matched all 120 ranks and
SIGTERMed them.

This is the third variant of the same footgun (after a stop script matching
itself, and SIGUSR1 hitting the torchrun agent). The rule: **kill by absolute
executable path, and always `pgrep -af <pattern>` first to see what a pattern
would match.** Flags like `--tensorboard-dir`, `--save`, `--load` and
`--data-path` drag directory names into argv, so tool names make dangerous
patterns.

`pkill -f '/opt/venv/bin/tensorboard'` is safe and was verified: TensorBoard
restarted while `trainers still up: 41`.

### TensorBoard events live on the last rank's node

Megatron writes them from node-14, not node-0, and `/scratch` is node-local, so
the server has to run on node-14. `ls -la <run>/tensorboard/` across nodes
finds the owner.

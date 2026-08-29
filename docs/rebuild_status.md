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

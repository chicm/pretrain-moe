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

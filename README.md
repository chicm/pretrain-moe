# R-Full MoE — rebuild on stock Megatron-LM

A clean, from-scratch MoE training setup for `chec-mi300-7`
(15 × 8 MI300X = 120 GPUs), built on **unmodified** Megatron-LM.

The previous hand-rolled implementation was removed. Everything here drives
stock `pretrain_gpt.py`; ROCm-specific defects are handled by a small, tested
overlay (`moe_rebuild/rocm_shim.py`) rather than by patching the Megatron tree,
so `git -C /scratch/rfull/megatron-lm rev-parse HEAD` stays a truthful record of
what actually ran.

## Layout

```
moe_rebuild/
  config.py       geometry/topology/schedule dataclasses -> Megatron argv
  specs.py        the three phases as concrete, named run specs
  rocm_shim.py    ROCm + node-local-cache fixes, each with evidence
tools/
  stage_data.py   copy .idx to local ext4, symlink .bin on blob
  warm_cache.py   pre-build the GPTDataset index cache, fan out to all nodes
  launch.py       preflight -> write per-node script -> detached launch -> verify
  monitor.py      PASS / SLOW / WEDGED classification with evidence
  stop.py         kill trainers, then prove GPUs are actually free
  pack.py         build the deploy tarball (rejects CR bytes)
tests/            28 tests, all runnable offline
docs/
  rebuild_status.md   phase results + every fix, with measurements
```

## The three phases

| spec | what it proves |
|---|---|
| `dense_1b_2n` / `dense_1b_15n` | multi-node transport, RCCL, dataloader, throughput |
| `moe_mini_1n` / `moe_full_1n` | MoE mechanics; full 25.85 B geometry fits in HBM |
| `moe_smoke_15n` → `moe_prod_15n` | production topology at 120 GPUs |

## Usage

```bash
export PYTHONPATH=/scratch/rfull/moe
cd /scratch/rfull/moe

python tools/stage_data.py --corpora fineweb_edu_240bt_tok --verify   # once per node
python tools/warm_cache.py  moe_smoke_15n --iters 60                  # per (spec, gbs, iters)
python tools/launch.py      moe_smoke_15n --iters 60
python tools/monitor.py     /scratch/rfull/runs/<run_dir>
python tools/stop.py --nnodes 15
```

`--dry-run` prints the resolved argv without launching. Every run directory
contains `argv.json`, `spec.json` and `meta.json`, so acceptance checks read
structured data instead of grepping shell strings.

## Things that will bite you

**Be patient with the first iteration.** ROCm autotunes every distinct kernel
shape on first use: ~16.9 s per MoE layer versus ~25 ms once warm — a 676×
penalty. With 48 layers and 8 accumulation steps the first iteration can take
tens of minutes. *Any MoE timing from fewer than ~15 iterations is measuring
autotune, not throughput.* Budget timeouts as `first_step × 3 + steady × iters`.

**Progress is printed by the LAST rank.** Megatron uses `print_rank_last`, so
`iteration N/M` appears in `node14.log`, not `node0.log`. Scanning node-0 shows
zero iterations for a perfectly healthy run.

**Nothing mmap'd may live on blobfuse.** `.idx` files are mmap'd
unconditionally (`--no-mmap-bin-files` only covers `.bin`), and the dataset
cache is read with `numpy.load(mmap_mode='r')`. A failed page fault on FUSE
raises SIGBUS/SIGSEGV in the faulting thread — no errno, no exception, no retry
— and looks exactly like a cluster-wide deadlock.

**A timeout is not a hang.** Declare a wedge only when a progress counter is
frozen *and* ranks are misaligned. Use `kill -USR1 <pid>` for a non-fatal
all-thread stack dump (the shim registers it; plain `PYTHONFAULTHANDLER` would
make SIGUSR1 *kill* the rank).

**In a collective timeout, the loud ranks are victims.** Count watchdog reports
against world size and go straight to the nodes that said nothing.

See `docs/rebuild_status.md` for the full list of defects found and fixed, each
with the measurement that identified it.

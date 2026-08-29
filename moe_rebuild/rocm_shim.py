"""ROCm + node-local-cache compatibility shims for stock Megatron-LM.

Kept as an explicit, testable overlay instead of patching the Megatron tree, so
`git -C /scratch/rfull/megatron-lm status` stays clean and the provenance of the
upstream commit (5cb6dbb, MCore 0.12.4) remains verifiable.

Each shim documents what upstream does, why it breaks here, and why the
replacement is semantically equivalent.
"""

from __future__ import annotations

import os
import sys

import torch


def _is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _noop_load(args=None):  # noqa: ARG001
    """Replacement for `megatron.legacy.fused_kernels.load`.

    Upstream `load()` does exactly two things:
      1. calls `_get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)`, which
         shells out to `CUDA_HOME + "/bin/nvcc" -V`; and
      2. defines a nested `_cpp_extention_load_helper` -- **which it never
         calls**. No extension is ever built or imported.

    On CUDA the only lasting effect is computing `cc_flag`, which is then
    discarded. On ROCm `cpp_extension.CUDA_HOME` is None, so step 1 raises
    `TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'` and
    kills rank 0 inside `initialize_megatron`, before training starts.

    Replacing it with a no-op is behaviour-preserving: it removes a crash and
    nothing else. Attention/softmax kernels come from Transformer Engine
    (`--transformer-impl transformer_engine`), not from this legacy path.
    """
    return None



def _enable_flash_attention() -> str | None:
    """Let Transformer Engine use the installed flash-attn 2.8.3 on ROCm.

    The problem
    -----------
    Three attention backends were measured on this MI300X image
    (bf16, s=4096, b=2, 32 heads, GQA=4, d=128, fwd+bwd):

        backend                       time/iter   peak mem   status
        FusedAttention (CK)               --         --      RuntimeError:
                                                             "basic_string:
                                                             construction from
                                                             null is not valid"
        UnfusedDotProductAttention      14.0 ms    6.76 GiB   works
        FlashAttention 2.8.3             5.4 ms    0.89 GiB   works

    TE's fused path is broken in this build, so TE falls back to the *unfused*
    backend -- 2.6x slower and 7.6x more memory, which materialises the full
    s x s attention matrix and would dominate activation memory at s=4096.

    TE would happily use flash-attn, but it gates on
    `2.1.1 <= version <= 2.8.1` and the image ships 2.8.3. That check runs at
    IMPORT time in `dot_product_attention/backends.py`; when it fails, the
    module-level `from flash_attn... import flash_attn_func` block is skipped
    and the symbols stay None. Raising `FlashAttentionUtils.max_version`
    afterwards therefore does nothing except make TE select a backend whose
    functions are None -- "TypeError: 'NoneType' object is not callable".

    The fix
    -------
    Patch `importlib.metadata.version` to report 2.8.1 for flash-attn *before*
    transformer_engine is imported, so the gate passes and the real symbols get
    bound. 2.8.3 is a patch release over 2.8.1 with an identical Python API.

    Correctness check (not assumed -- measured)
    -------------------------------------------
    Against an independent fp32 reference implementation of causal GQA SDPA:

        unfused (bf16)  max_abs 0.01446   mean_rel 3.881e-03
        flash   (bf16)  max_abs 0.01539   mean_rel 2.088e-03

    Flash is *closer* to the fp32 reference than the backend it replaces. The
    flash-vs-unfused delta (max_abs 0.0156) is ordinary bf16 rounding, not a
    numerical defect.
    """
    import importlib.metadata as md

    if "transformer_engine" in sys.modules:
        return ("WARNING: transformer_engine already imported; "
                "flash-attn gate NOT lifted")

    real_version = md.version

    def _patched(name: str) -> str:
        v = real_version(name)
        if name.replace("_", "-") == "flash-attn" and v == "2.8.3":
            return "2.8.1"
        return v

    md.version = _patched
    return ("flash-attn 2.8.3 reported to TE as 2.8.1 -> FlashAttention backend "
            "(5.4ms/0.89GiB vs unfused 14.0ms/6.76GiB; fused CK is broken here)")


def _install_ep_group_timeout_fix() -> str:
    """Give EXPERT_MODEL_PARALLEL_GROUP the configured distributed timeout.

    Upstream bug (MCore 0.12.4, parallel_state.py:1133). Every process group
    built by `initialize_model_parallel` is created with `timeout=timeout`,
    derived from `--distributed-timeout-minutes` -- except one:

        for ranks in generator_wrapper('ep', is_expert=True):
            group = create_group(
                ranks,
                pg_options=get_nccl_options('ep', nccl_comm_cfgs),
                group_desc='EXPERT_MODEL_PARALLEL_GROUP',
            )                      # <-- no timeout=timeout

    Its immediate neighbours (TENSOR_AND_CONTEXT_PARALLEL_GROUP above,
    EXPERT_TENSOR_PARALLEL_GROUP below) both pass it, so this is an omission,
    not a design choice. The EP group therefore silently keeps PyTorch's
    10-minute default no matter what the user asks for.

    That matters here because the FIRST iteration of this model takes ~16
    minutes: ROCm autotunes each distinct kernel shape on first use, at
    ~16.9 s per MoE layer, across 48 layers and 8 gradient-accumulation steps.
    While rank N is autotuning, its EP peers sit in an alltoall and hit the
    10-minute watchdog:

        [PG ID 12 ... (EXPERT_MODEL_PARALLEL_GROUP) Rank 0]
        Process group watchdog thread terminated with exception:
        Watchdog caught collective operation timeout

    Nodes 9, 11 and 14 aborted at 12.5 minutes; the rest were SIGTERM'd as
    bystanders. Nothing was actually wrong -- the job was still warming up.

    We wrap `create_group` so that a call naming the EP group inherits the same
    timeout as its siblings. This changes only how long a rank waits before
    declaring failure; it cannot mask a real hang, because the timeout still
    fires, just at the value the operator asked for.
    """
    import megatron.core.parallel_state as ps

    original = ps.create_group

    def _patched(ranks, timeout=None, **kw):
        if timeout is None and kw.get("group_desc") == "EXPERT_MODEL_PARALLEL_GROUP":
            timeout = _ep_timeout()
        return original(ranks, timeout=timeout, **kw)

    ps.create_group = _patched
    mins = _ep_timeout().total_seconds() / 60
    return (f"EXPERT_MODEL_PARALLEL_GROUP timeout -> {mins:.0f} min "
            f"(upstream omits timeout= for this one group; parallel_state.py:1133)")


def _ep_timeout():
    from datetime import timedelta

    return timedelta(minutes=int(os.environ.get("MOE_EP_TIMEOUT_MINUTES", "120")))

def apply() -> list[str]:
    """Install the shims. Returns human-readable descriptions of what changed."""
    applied: list[str] = []

    if _is_rocm():
        # Must run before transformer_engine is imported anywhere.
        note = _enable_flash_attention()
        if note:
            applied.append(note)

        applied.append(_install_ep_group_timeout_fix())

        import megatron.legacy.fused_kernels as fused_kernels

        if fused_kernels.load is not _noop_load:
            fused_kernels.load = _noop_load
            applied.append(
                "megatron.legacy.fused_kernels.load -> no-op "
                "(upstream needs nvcc; the function builds nothing on any platform)"
            )

    if os.environ.get("MOE_FAULTHANDLER_SIGUSR1") == "1":
        applied.append(_install_sigusr1_dump())

    if os.environ.get("MOE_CACHE_ONLY") == "1":
        applied.append(_install_cache_only_exit())

    return applied


def _install_cache_only_exit() -> str:
    """Build the dataset index cache, then exit -- without building the model.

    Why this is needed
    ------------------
    `GPTDataset._build_document_sample_shuffle_indices` builds the
    document/sample/shuffle indices on **global rank 0 only**; every other rank
    expects to find the resulting `.npy` files already on disk
    (gpt_dataset.py:352-354). That assumes a shared filesystem.

    The cache cannot live on the shared blobfuse mount, because it is read back
    with `numpy.load(..., mmap_mode='r')` and a failed mmap page fault on FUSE
    delivers SIGBUS/SIGSEGV to the faulting thread -- no errno, no exception, no
    retry. The rank dies instantly and every other rank then hangs in the next
    collective, which is indistinguishable from a deadlock.

    So the cache lives on node-local ext4, is built once here, and is fanned out
    to every node. Its key is a pure function of (dataset paths, num_samples,
    random_seed, sequence_length, split, tokenizer), so a cache built by this
    single-process run is bit-identical to what the distributed job expects.

    Why we must skip the model
    --------------------------
    `pretrain()` calls `setup_model_and_optimizer` BEFORE
    `build_train_valid_test_data_iterators`. With world_size=1 the whole model
    lands on one GPU and DDP tries to allocate a single 96.33 GiB gradient
    buffer for the 48-layer / 96-expert config -> `torch.OutOfMemoryError`
    before the dataset code is ever reached. Since we only want the dataset
    indices, we stub out model setup entirely.
    """
    import megatron.training.training as T

    def _fake_setup(model_provider_func, model_type, *a, **kw):  # noqa: ARG001
        """Return a model-shaped stub that owns a real TransformerConfig.

        `pretrain()` immediately calls `get_model_config(model[0])`, so an empty
        list raises IndexError. The stub only has to expose `.config`; nothing
        downstream of the dataset build is reached, because we exit inside
        `build_train_valid_test_data_iterators`.
        """
        print("[rocm_shim] MOE_CACHE_ONLY: skipping model/optimizer setup",
              flush=True)
        from megatron.training.arguments import core_transformer_config_from_args
        from megatron.training import get_args

        class _StubModel:
            def __init__(self, cfg):
                self.config = cfg

        class _Sched:
            def step(self, *a, **kw):
                pass

        args = get_args()
        # `load_checkpoint` normally sets these; we skipped it, and
        # build_train_valid_test_data_loaders reads them.
        for name, default in (("iteration", 0),
                              ("consumed_train_samples", 0),
                              ("skipped_train_samples", 0),
                              ("consumed_valid_samples", 0)):
            if not hasattr(args, name):
                setattr(args, name, default)
        cfg = core_transformer_config_from_args(args)
        return [_StubModel(cfg)], None, _Sched()

    T.setup_model_and_optimizer = _fake_setup

    original = T.build_train_valid_test_data_iterators

    def _build_then_exit(*a, **kw):
        original(*a, **kw)
        print("[rocm_shim] dataset cache built; exiting before training "
              "(MOE_CACHE_ONLY=1)", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    T.build_train_valid_test_data_iterators = _build_then_exit
    return ("MOE_CACHE_ONLY: model setup stubbed + "
            "build_train_valid_test_data_iterators -> build cache then exit(0)")


def _install_sigusr1_dump() -> str:
    """Dump all thread stacks on SIGUSR1 without terminating the process.

    `PYTHONFAULTHANDLER=1` only installs handlers for fatal signals
    (SIGSEGV/SIGFPE/SIGABRT/SIGBUS/SIGILL). SIGUSR1 keeps its default
    disposition, which is *terminate* -- so using it to inspect a stalled rank
    kills that rank (exitcode -10) and every peer then dies of SIGTERM as a
    bystander. `faulthandler.register` makes it non-fatal and gives the one
    thing that actually identifies a stall: the current Python frame of every
    thread.

    **This protects workers only.** The torchrun agent
    (`python -m torch.distributed.run ... pretrain_entry.py ...`) never imports
    this module, so SIGUSR1 stays fatal for it -- and the training script path
    appears in the agent's argv, so a probe matching that path selects the agent
    as well as the workers. Signalling it kills the agent, which tears down all
    8 local workers; if that agent owns rendezvous (node rank 0) it takes the
    TCPStore with it and every other node fails with `Broken pipe`.

    Probes must therefore exclude the agent explicitly, e.g.

        pgrep -f '[p]retrain_entry[.]py' | while read pid; do
            tr '\0' ' ' < /proc/$pid/cmdline | grep -q distributed.run || echo $pid
        done

    A probe must never be able to change the state of the thing it observes.
    """
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    return "SIGUSR1 -> non-fatal all-thread stack dump (faulthandler.register)"

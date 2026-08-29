"""R-Full MoE rebuild: model/topology/data specs compiled to Megatron-LM argv.

Design rules enforced here (learned the hard way):
  * Nothing that Megatron mmaps may live on blobfuse. `.idx` files are ALWAYS
    mmap'd by `IndexedDataset._IndexReader` (indexed_dataset.py:253) even with
    `--no-mmap-bin-files`, and the dataset cache `.npy` files are opened with
    `numpy.load(mmap_mode='r')` (gpt_dataset.py:495/505/515). A page fault on a
    FUSE mount raises SIGBUS/SIGSEGV in the faulting thread -- no errno, no
    Python exception, no retry. Both therefore go on node-local ext4.
  * `.bin` payload (3.8 TiB) stays on blobfuse but is read with pread, never
    mapped, via `--no-mmap-bin-files`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

# --------------------------------------------------------------------------
# Frozen environment facts for chec-mi300-7 (verified 2026-08-29).
# --------------------------------------------------------------------------
NNODES = 15
GPUS_PER_NODE = 8
WORLD = NNODES * GPUS_PER_NODE  # 120

MEGATRON_DIR = "/scratch/rfull/megatron-lm"       # node-local, commit 5cb6dbb, MCore 0.12.4
PROJECT_DIR = "/scratch/rfull/moe"                # node-local copy of this repo
PYTHON_BIN = "/opt/venv/bin/python"

BLOB = "/scratch/workspaceblobstore/chec"
BLOB_DATA = f"{BLOB}/pretrain/data"               # .bin payload (read-only, pread)
LOCAL_DATA = "/scratch/rfull/data"                # real .idx + symlinked .bin
LOCAL_CACHE = "/scratch/rfull/data-cache"         # mmap'd .npy -> MUST be local
CKPT_ROOT = f"{BLOB}/pretrain-moe/runs"           # checkpoints (streamed, not mapped)

# Tokenizer: corpus is pre-tokenized with a Qwen2-family tokenizer.
# corpus_manifest.json: eot=151643, global_max_token=151668, native_vocab_size=151669.
# The design doc freezes the padded embedding at 151936 (= 151669 rounded up to
# a multiple of 128 * ... -> the Qwen2 standard padded size). We pass it
# explicitly and never let Megatron derive it.
EOD_ID = 151643
PADDED_VOCAB = 151936


@dataclass
class Model:
    """Transformer geometry. Dense when num_experts is None."""

    name: str
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int = 32
    num_query_groups: int = 4
    kv_channels: int = 128
    seq_length: int = 4096
    max_position_embeddings: int = 32768
    rotary_base: int = 1_000_000
    norm_epsilon: float = 1e-6
    init_method_std: float = 0.02

    # MoE (None => dense model)
    num_experts: int | None = None
    moe_layer_freq: str | None = None
    moe_ffn_hidden_size: int | None = None
    moe_shared_expert_intermediate_size: int | None = None
    moe_router_topk: int = 6
    moe_aux_loss_coeff: float = 1e-3
    moe_z_loss_coeff: float = 1e-4

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None


@dataclass
class Topology:
    nnodes: int = NNODES
    gpus_per_node: int = GPUS_PER_NODE
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    context_parallel: int = 1
    expert_parallel: int = 1

    @property
    def world(self) -> int:
        return self.nnodes * self.gpus_per_node

    @property
    def data_parallel(self) -> int:
        return self.world // (
            self.tensor_parallel * self.pipeline_parallel * self.context_parallel
        )

    def validate(self) -> None:
        d = self.tensor_parallel * self.pipeline_parallel * self.context_parallel
        assert self.world % d == 0, f"world {self.world} not divisible by TP*PP*CP={d}"
        assert self.data_parallel % self.expert_parallel == 0, (
            f"DP {self.data_parallel} not divisible by EP {self.expert_parallel}"
        )


@dataclass
class Schedule:
    train_iters: int
    global_batch_size: int
    micro_batch_size: int = 1
    lr: float = 2.0e-4
    min_lr: float = 2.0e-5
    lr_warmup_iters: int = 2543
    lr_decay_style: str = "cosine"
    lr_decay_iters: int | None = None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    clip_grad: float = 1.0
    save_interval: int = 2000
    eval_interval: int = 1000
    eval_iters: int = 10
    log_interval: int = 1


@dataclass
class RunSpec:
    run_id: str
    model: Model
    topology: Topology
    schedule: Schedule
    data_blend: list[str] = field(default_factory=list)
    data_split: str = "999,1,0"
    save: str | None = None
    load: str | None = None
    extra_args: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def build_argv(spec: RunSpec) -> list[str]:
    """Compile a RunSpec into Megatron-LM `pretrain_gpt.py` arguments."""
    m, t, s = spec.model, spec.topology, spec.schedule
    t.validate()

    a: list[str] = []

    # ---- parallelism -----------------------------------------------------
    a += ["--tensor-model-parallel-size", str(t.tensor_parallel)]
    a += ["--pipeline-model-parallel-size", str(t.pipeline_parallel)]
    if t.context_parallel > 1:
        a += ["--context-parallel-size", str(t.context_parallel)]
    if m.is_moe:
        a += ["--expert-model-parallel-size", str(t.expert_parallel)]

    # ---- geometry --------------------------------------------------------
    a += ["--num-layers", str(m.num_layers)]
    a += ["--hidden-size", str(m.hidden_size)]
    a += ["--ffn-hidden-size", str(m.ffn_hidden_size)]
    a += ["--num-attention-heads", str(m.num_attention_heads)]
    a += ["--group-query-attention", "--num-query-groups", str(m.num_query_groups)]
    a += ["--kv-channels", str(m.kv_channels)]
    a += ["--seq-length", str(m.seq_length)]
    a += ["--max-position-embeddings", str(m.max_position_embeddings)]

    # ---- architecture (Llama/Qwen-style decoder) -------------------------
    a += ["--position-embedding-type", "rope"]
    a += ["--rotary-base", str(m.rotary_base)]
    a += ["--rotary-percent", "1.0"]
    a += ["--normalization", "RMSNorm"]
    a += ["--norm-epsilon", str(m.norm_epsilon)]
    a += ["--swiglu"]
    a += ["--qk-layernorm"]
    a += ["--disable-bias-linear"]
    a += ["--attention-softmax-in-fp32"]
    a += ["--init-method-std", str(m.init_method_std)]
    # Tied embedding / output head: do NOT pass
    # --untie-embeddings-and-output-weights.

    # ---- MoE -------------------------------------------------------------
    if m.is_moe:
        a += ["--num-experts", str(m.num_experts)]
        if m.moe_layer_freq:
            a += ["--moe-layer-freq", m.moe_layer_freq]
        if m.moe_ffn_hidden_size:
            a += ["--moe-ffn-hidden-size", str(m.moe_ffn_hidden_size)]
        if m.moe_shared_expert_intermediate_size:
            a += [
                "--moe-shared-expert-intermediate-size",
                str(m.moe_shared_expert_intermediate_size),
            ]
        a += ["--moe-router-topk", str(m.moe_router_topk)]
        a += ["--moe-router-score-function", "softmax"]
        # pre_softmax=False -> softmax over the selected top-k logits.
        a += ["--moe-router-dtype", "fp32"]
        a += ["--moe-router-load-balancing-type", "aux_loss"]
        a += ["--moe-aux-loss-coeff", str(m.moe_aux_loss_coeff)]
        a += ["--moe-z-loss-coeff", str(m.moe_z_loss_coeff)]
        a += ["--moe-token-dispatcher-type", "alltoall"]
        a += ["--moe-grouped-gemm"]
        # Dropless: no --moe-expert-capacity-factor, no pad-to-capacity.
        # NOTE: --moe-per-layer-logging is deliberately OFF.
        # It calls track_moe_metrics, which does a per-layer all-reduce of
        # the aux-loss tracker. Combined with --log-interval 1 that is one
        # extra collective per MoE layer per step. Enable it only for a
        # short diagnostic run, and expect it to cost throughput.
        # NOTE: no activation recompute.
        # It was added to fix an OOM that turned out to be caused by
        # micro_batch_size=8 (32768 tokens in flight, ~160 GiB of activations).
        # With mbs=1 and 8 accumulation steps the peak is 82 GiB of 192 GiB, so
        # recompute buys nothing and only adds a variable. If activation memory
        # ever becomes binding, prefer '--recompute-granularity selective'
        # (attention core only) and re-measure before keeping it.

    # ---- batch / schedule ------------------------------------------------
    a += ["--micro-batch-size", str(s.micro_batch_size)]
    a += ["--global-batch-size", str(s.global_batch_size)]
    a += ["--train-iters", str(s.train_iters)]
    a += ["--lr", repr(s.lr)]
    a += ["--min-lr", repr(s.min_lr)]
    a += ["--lr-decay-style", s.lr_decay_style]
    a += ["--lr-warmup-iters", str(s.lr_warmup_iters)]
    a += ["--lr-decay-iters", str(s.lr_decay_iters or s.train_iters)]
    a += ["--optimizer", "adam"]
    a += ["--adam-beta1", repr(s.adam_beta1)]
    a += ["--adam-beta2", repr(s.adam_beta2)]
    a += ["--adam-eps", repr(s.adam_eps)]
    a += ["--weight-decay", repr(s.weight_decay)]
    a += ["--clip-grad", repr(s.clip_grad)]

    # ---- precision -------------------------------------------------------
    a += ["--bf16"]
    a += ["--accumulate-allreduce-grads-in-fp32"]
    a += ["--use-distributed-optimizer"]
    # Disable the per-bucket NaN/Inf grad check.
    #
    # `_ParamAndGradBuffer.check_grads` (param_and_grad_buffer.py:155) loops over
    # every bucket, computes `bucket.grad_data.norm(p=2)`, and hands each result
    # to `RerunStateMachine.validate_result`, which forces a device-to-host
    # synchronisation. A dense model has a handful of buckets; this MoE has one
    # per expert shard, so the loop becomes thousands of tiny norm+sync pairs
    # that serialise the whole backward pass.
    #
    # Measured at 4 layers on 120 GPUs: iterations ran 37.2 s, 4.1 s, then
    # **234.0 s** (0.5 TFLOP/s/GPU), with every rank parked in
    # `check_grads -> validate_result` and GPUs at 21%.
    #
    # Loss scaling is not in play (bf16, loss scale fixed at 1.0), and grad
    # clipping still computes a global grad norm every step -- a NaN would show
    # up there as `grad norm: nan` and in the reported loss. So the safety net
    # this removes is redundant, while its cost is not.
    a += ["--no-check-for-nan-in-loss-and-grad"]

    # ---- tokenizer / vocab ----------------------------------------------
    # NullTokenizer: ids are already in the .bin; it only needs a vocab size and
    # supplies eod = vocab_size. We pass vocab_size = EOD_ID so that
    # tokenizer.eod == 151643, matching corpus_manifest.eot, then override the
    # padded embedding size explicitly.
    a += ["--tokenizer-type", "NullTokenizer"]
    a += ["--vocab-size", str(EOD_ID)]
    a += ["--make-vocab-size-divisible-by", "128"]

    # ---- data ------------------------------------------------------------
    if spec.data_blend:
        a += ["--data-path"] + spec.data_blend
    a += ["--split", spec.data_split]
    a += ["--data-cache-path", LOCAL_CACHE]
    a += ["--no-mmap-bin-files"]          # .bin via pread; .idx is local so mmap is safe
    a += ["--num-workers", "2"]
    a += ["--dataloader-type", "single"]

    # ---- kernels / perf --------------------------------------------------
    a += ["--transformer-impl", "transformer_engine"]
    # Pick the attention backend explicitly. On this MI300X image TE's fused
    # (CK) kernel raises "basic_string: construction from null is not valid",
    # and 'auto' would silently fall back to the unfused path, which costs
    # 14.0 ms / 6.76 GiB per iteration versus flash at 5.4 ms / 0.89 GiB
    # (bf16, s=4096, b=2, 32 heads, GQA=4). Megatron sets the NVTE_* env vars
    # itself from this flag and asserts if we preset them, so this is the only
    # supported way to choose.
    a += ["--attention-backend", "flash"]
    a += ["--cross-entropy-loss-fusion"]
    a += ["--no-gradient-accumulation-fusion"]
    a += ["--manual-gc"]
    a += ["--manual-gc-interval", "100"]

    # ---- checkpoint / logging -------------------------------------------
    if spec.save:
        a += ["--save", spec.save, "--save-interval", str(s.save_interval)]
        a += ["--ckpt-format", "torch_dist"]
    if spec.load:
        a += ["--load", spec.load]
    a += ["--eval-interval", str(s.eval_interval)]
    a += ["--eval-iters", str(s.eval_iters)]
    a += ["--log-interval", str(s.log_interval)]
    a += ["--log-throughput"]
    # Keep timers OFF in production. Megatron's timers take `barrier=True`
    # in several hot paths (training.py:744/758/1650, and the ones
    # finalize_model_grads starts), and `Timer.start` then does
    # `torch.distributed.barrier()` + `torch.cuda.synchronize()`
    # (core/timers.py:135). At log-level 0 those become DummyTimer no-ops;
    # at level >=1 they become real global sync points that serialise all
    # 120 ranks and destroy compute/communication overlap.
    # Raise this only for a short, deliberate profiling run.
    a += ["--timing-log-level", "0"]
    # 30 min is not enough: iteration 1 alone takes ~16 min of kernel
    # autotune on ROCm, and a slow checkpoint write can add more.
    a += ["--distributed-timeout-minutes", "120"]

    a += spec.extra_args
    return a

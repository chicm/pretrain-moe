"""Local sanity tests for the argv compiler. Run: python -m pytest tests -q"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe_rebuild.config import EOD_ID, build_argv  # noqa: E402
from moe_rebuild.specs import (  # noqa: E402
    RFULL_EXPERTS,
    RFULL_LAYERS,
    dense_1b,
    moe_1node_full,
    moe_1node_mini,
    rfull_moe_prod,
)


def val(argv, flag):
    return argv[argv.index(flag) + 1]


def test_dense_has_no_moe_flags():
    a = build_argv(dense_1b(2))
    assert not [x for x in a if x.startswith("--moe")]
    assert "--num-experts" not in a
    assert "--expert-model-parallel-size" not in a


def test_dense_topology():
    s = dense_1b(2)
    assert s.topology.world == 16
    assert s.topology.data_parallel == 16
    assert s.schedule.global_batch_size == 128


def test_moe_mini_flags():
    a = build_argv(moe_1node_mini())
    assert val(a, "--num-experts") == str(RFULL_EXPERTS)
    assert val(a, "--expert-model-parallel-size") == "8"
    assert val(a, "--moe-token-dispatcher-type") == "alltoall"
    assert val(a, "--moe-router-topk") == "6"
    assert val(a, "--moe-router-dtype") == "fp32"
    assert "--moe-grouped-gemm" in a
    # dropless: capacity factor must NOT be set
    assert "--moe-expert-capacity-factor" not in a


def test_moe_layer_freq_counts_match_layers():
    for spec in (moe_1node_mini(), moe_1node_full(), rfull_moe_prod()):
        a = build_argv(spec)
        freq = val(a, "--moe-layer-freq")
        n = eval(freq)  # noqa: S307 - same expression megatron evaluates
        assert len(n) == spec.model.num_layers, (spec.model.name, len(n))
        assert n[0] == 0 and n[-1] == 1


def test_production_geometry_and_parallelism():
    s = rfull_moe_prod()
    a = build_argv(s)
    assert s.topology.world == 120
    assert s.topology.data_parallel == 120
    assert s.topology.expert_parallel == 8
    assert s.topology.data_parallel % s.topology.expert_parallel == 0
    assert val(a, "--num-layers") == str(RFULL_LAYERS)
    assert val(a, "--global-batch-size") == "960"
    assert val(a, "--seq-length") == "4096"


def test_experts_divide_expert_parallel():
    s = rfull_moe_prod()
    assert s.model.num_experts % s.topology.expert_parallel == 0
    assert s.model.num_experts // s.topology.expert_parallel == 12


def test_global_batch_divisible_by_dp():
    for spec in (dense_1b(15), rfull_moe_prod(), moe_1node_mini()):
        dp = spec.topology.data_parallel
        gbs = spec.schedule.global_batch_size
        assert gbs % (dp * spec.schedule.micro_batch_size) == 0, spec.run_id


def test_data_cache_and_mmap_safety():
    """Nothing mmap'd may sit on blobfuse."""
    a = build_argv(rfull_moe_prod())
    assert "--no-mmap-bin-files" in a
    cache = val(a, "--data-cache-path")
    assert cache.startswith("/scratch/rfull"), cache
    assert "workspaceblobstore" not in cache


def test_tokenizer_eod_matches_corpus():
    a = build_argv(rfull_moe_prod())
    assert val(a, "--tokenizer-type") == "NullTokenizer"
    # NullTokenizer sets eod == vocab_size; corpus EOT is 151643.
    assert val(a, "--vocab-size") == str(EOD_ID) == "151643"


def test_bf16_and_optimizer():
    a = build_argv(rfull_moe_prod())
    assert "--bf16" in a
    assert "--accumulate-allreduce-grads-in-fp32" in a
    assert "--use-distributed-optimizer" in a
    assert val(a, "--clip-grad") == "1.0"
    assert val(a, "--adam-beta2") == "0.95"


def test_tied_embeddings():
    a = build_argv(rfull_moe_prod())
    assert "--untie-embeddings-and-output-weights" not in a


def test_smoke_specs_do_not_save():
    from moe_rebuild.specs import moe_prod_smoke
    for spec in (dense_1b(15), moe_1node_mini(), moe_prod_smoke()):
        a = build_argv(spec)
        assert "--save" not in a, spec.run_id


def test_eval_iters_never_zero():
    """eval_iters=0 empties the validation split and Megatron asserts.

    `build_train_valid_test_data_loaders` always constructs a validation
    dataloader, and `MegatronPretrainingSampler.__init__` asserts
    `total_samples > 0`. With eval_iters=0 the valid split receives zero
    samples and every rank dies with "AssertionError: no sample to consume: 0"
    during dataset construction -- before a single training step.
    """
    from moe_rebuild.specs import REGISTRY
    for name, factory in REGISTRY.items():
        spec = factory()
        assert spec.schedule.eval_iters >= 1, name
        a = build_argv(spec)
        assert int(val(a, "--eval-iters")) >= 1, name


def test_shim_is_applied_via_entry_point():
    """The launcher must go through the wrapper, not stock pretrain_gpt.py."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.launch import build_launch_script
    from moe_rebuild.specs import dense_1b
    s = build_launch_script(0, 2, 29999, Path("/tmp/x"), build_argv(dense_1b(2)))
    assert "tools/pretrain_entry.py" in s
    assert "megatron-lm/pretrain_gpt.py" not in s


def test_production_keeps_gradient_accumulation():
    """mbs must leave >1 accumulation step, or activations blow the 192 GiB HBM.

    With gbs=960 over DP=120 there are exactly 8 sequences per rank. Setting
    mbs=8 makes that a single micro-step with 8*4096 = 32768 tokens in flight,
    which needs ~160 GiB of activations on top of 17.5 GiB of static state and
    dies with "HIP out of memory ... 191.45 GiB of which 722.00 MiB is free".
    """
    from moe_rebuild.specs import rfull_moe_prod
    spec = rfull_moe_prod()
    dp = spec.topology.data_parallel
    accum = spec.schedule.global_batch_size // (spec.schedule.micro_batch_size * dp)
    assert accum >= 8, f"accumulation steps {accum} too low -> activation OOM"


def test_no_recompute_anywhere():
    """No activation recompute is configured.

    It was briefly added to fix an OOM whose real cause was micro_batch_size=8.
    With mbs=1 the peak is 82 GiB of 192 GiB, so recompute is dead weight and an
    extra variable. This test exists so it cannot be reintroduced silently.
    """
    from moe_rebuild.specs import REGISTRY
    for name, factory in REGISTRY.items():
        assert "--recompute-granularity" not in build_argv(factory()), name

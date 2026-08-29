"""Tests for the ROCm shims (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_shim_module_imports_without_torch_cuda():
    from moe_rebuild import rocm_shim
    assert hasattr(rocm_shim, "apply")
    assert hasattr(rocm_shim, "_enable_flash_attention")
    assert hasattr(rocm_shim, "_noop_load")


def test_flash_gate_patch_rewrites_only_flash_attn():
    """The metadata patch must alter flash-attn 2.8.3 and nothing else."""
    import importlib.metadata as md
    from moe_rebuild import rocm_shim

    real = md.version
    saved = sys.modules.pop("transformer_engine", None)
    try:
        fake = {"flash-attn": "2.8.3", "torch": "2.7.0", "numpy": "1.26.4"}
        md.version = lambda n: fake[n.replace("_", "-")]
        note = rocm_shim._enable_flash_attention()
        assert note and "2.8.1" in note
        assert md.version("flash-attn") == "2.8.1"     # gated down
        assert md.version("flash_attn") == "2.8.1"     # underscore form too
        assert md.version("torch") == "2.7.0"          # untouched
        assert md.version("numpy") == "1.26.4"         # untouched
    finally:
        md.version = real
        if saved is not None:
            sys.modules["transformer_engine"] = saved


def test_flash_gate_refuses_after_te_import():
    """Patching after TE is imported is a no-op; the shim must say so loudly."""
    import importlib.metadata as md
    from moe_rebuild import rocm_shim

    real = md.version
    sys.modules["transformer_engine"] = object()  # simulate prior import
    try:
        note = rocm_shim._enable_flash_attention()
        assert note is not None and "WARNING" in note
        assert md.version is real, "must not patch when TE already loaded"
    finally:
        sys.modules.pop("transformer_engine", None)
        md.version = real


def test_noop_load_returns_none():
    """Upstream fused_kernels.load builds nothing; the replacement must match."""
    from moe_rebuild import rocm_shim
    assert rocm_shim._noop_load() is None
    assert rocm_shim._noop_load(object()) is None


def test_ep_group_gets_a_timeout():
    """EXPERT_MODEL_PARALLEL_GROUP must not keep PyTorch's 10-minute default.

    Upstream (MCore 0.12.4, parallel_state.py:1133) creates every process group
    with timeout=timeout except this one. Since iteration 1 of this model takes
    ~16 minutes of ROCm kernel autotune, EP peers hit the 10-minute watchdog
    mid-warmup and abort a perfectly healthy job.
    """
    import sys
    import types
    from datetime import timedelta
    from moe_rebuild import rocm_shim

    calls = []

    fake = types.ModuleType("megatron.core.parallel_state")

    def create_group(ranks, timeout=None, **kw):
        calls.append((kw.get("group_desc"), timeout))
        return object()

    fake.create_group = create_group

    # `import megatron.core.parallel_state` needs the parent packages present
    # too, otherwise the import machinery raises ModuleNotFoundError before it
    # ever consults sys.modules for the leaf.
    saved = {k: sys.modules.get(k) for k in
             ("megatron", "megatron.core", "megatron.core.parallel_state")}
    megatron = sys.modules.get("megatron") or types.ModuleType("megatron")
    core = sys.modules.get("megatron.core") or types.ModuleType("megatron.core")
    core.parallel_state = fake
    megatron.core = core
    sys.modules["megatron"] = megatron
    sys.modules["megatron.core"] = core
    sys.modules["megatron.core.parallel_state"] = fake
    try:
        note = rocm_shim._install_ep_group_timeout_fix()
        assert "EXPERT_MODEL_PARALLEL_GROUP" in note

        fake.create_group([0, 1], group_desc="EXPERT_MODEL_PARALLEL_GROUP")
        desc, timeout = calls[-1]
        assert timeout is not None, "EP group still has no timeout"
        assert timeout >= timedelta(minutes=30), timeout

        # an explicit timeout must be preserved, not overridden
        fake.create_group([0, 1], timeout=timedelta(minutes=7),
                          group_desc="EXPERT_MODEL_PARALLEL_GROUP")
        assert calls[-1][1] == timedelta(minutes=7)

        # other groups must be left exactly as upstream had them
        fake.create_group([0, 1], group_desc="DATA_PARALLEL_GROUP")
        assert calls[-1][1] is None
    finally:
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)

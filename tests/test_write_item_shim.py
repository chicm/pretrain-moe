"""Tests for the _write_item arity shim.

torch 2.10-dev added a 6th positional parameter to
torch.distributed.checkpoint.filesystem._write_item; MCore 0.12.4 calls it
with five. The shim binds the new argument to a default so the old call works.
"""

import inspect
import sys
import types

import pytest

from moe_rebuild import rocm_shim


def _fake_torch_filesystem(with_new_arg: bool, default=inspect.Parameter.empty):
    """Build a stand-in for torch.distributed.checkpoint.filesystem."""
    mod = types.ModuleType("filesystem")

    class SerializationFormat:
        TORCH_SAVE = "torch_save"

    mod.SerializationFormat = SerializationFormat
    calls = []

    if with_new_arg:
        if default is inspect.Parameter.empty:

            def _write_item(transforms, stream, data, write_item, storage_key,
                            serialization_format):
                calls.append(serialization_format)
                return "ok"
        else:

            def _write_item(transforms, stream, data, write_item, storage_key,
                            serialization_format=default):
                calls.append(serialization_format)
                return "ok"
    else:

        def _write_item(transforms, stream, data, write_item, storage_key):
            calls.append(None)
            return "ok"

    mod._write_item = _write_item
    mod.calls = calls
    return mod


@pytest.fixture
def torch_ns(monkeypatch):
    """Install a fake torch.distributed.checkpoint package tree."""

    def install(mod):
        for name in ("torch", "torch.distributed", "torch.distributed.checkpoint"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        pkg = sys.modules["torch.distributed.checkpoint"]
        pkg.filesystem = mod
        monkeypatch.setitem(
            sys.modules, "torch.distributed.checkpoint.filesystem", mod
        )
        return mod

    return install


def test_binds_default_when_torch_requires_new_arg(torch_ns):
    mod = torch_ns(_fake_torch_filesystem(with_new_arg=True))

    note = rocm_shim._install_write_item_arity_fix()

    assert note is not None
    assert "serialization_format" in note
    # The 5-argument call that MCore makes must now succeed.
    assert mod._write_item(None, None, None, None, "key") == "ok"
    assert mod.calls == ["torch_save"]


def test_noop_on_older_torch_without_the_arg(torch_ns):
    mod = torch_ns(_fake_torch_filesystem(with_new_arg=False))
    before = mod._write_item

    assert rocm_shim._install_write_item_arity_fix() is None
    assert mod._write_item is before


def test_noop_when_arg_already_has_a_default(torch_ns):
    mod = torch_ns(_fake_torch_filesystem(with_new_arg=True, default="torch_save"))
    before = mod._write_item

    assert rocm_shim._install_write_item_arity_fix() is None
    assert mod._write_item is before


def test_six_argument_calls_still_pass_through(torch_ns):
    mod = torch_ns(_fake_torch_filesystem(with_new_arg=True))
    rocm_shim._install_write_item_arity_fix()

    mod._write_item(None, None, None, None, "key", "explicit")

    assert mod.calls == ["explicit"]


def test_patches_mcores_imported_reference(torch_ns, monkeypatch):
    """MCore did `from ... import _write_item`, so it holds its own binding."""
    mod = torch_ns(_fake_torch_filesystem(with_new_arg=True))
    original = mod._write_item

    mcore = types.ModuleType("filesystem_async")
    mcore._write_item = original
    for name in (
        "megatron",
        "megatron.core",
        "megatron.core.dist_checkpointing",
        "megatron.core.dist_checkpointing.strategies",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    strategies = sys.modules["megatron.core.dist_checkpointing.strategies"]
    strategies.filesystem_async = mcore
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.dist_checkpointing.strategies.filesystem_async",
        mcore,
    )

    note = rocm_shim._install_write_item_arity_fix()

    assert "filesystem_async" in note
    assert mcore._write_item is not original
    assert mcore._write_item(None, None, None, None, "key") == "ok"

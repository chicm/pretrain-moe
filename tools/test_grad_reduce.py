"""Correctness tests for bucketed gradient reduction.

Value/shape/dtype preservation across contiguous, non-contiguous, and mixed
buckets. These were written while chasing a two-node GPU memory access fault;
they pass against both the old and the tightened flattening, so they document
required behaviour rather than pinning that specific fault.

Single-process gloo, CPU only, so it can gate every commit.
"""
import os
import sys
import pathlib

import torch
import torch.distributed as dist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rfull.grad_reduce import _flush  # noqa: E402


def setup():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group("gloo")


def test_contiguous():
    g = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    want = g.clone()
    _flush([g], dist.group.WORLD, 1)
    assert torch.equal(g, want), "contiguous single-tensor bucket corrupted"


def test_non_contiguous():
    """A transposed gradient is not contiguous; values must survive."""
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    g = base.t()                      # 4x3, non-contiguous
    assert not g.is_contiguous()
    want = g.clone()
    _flush([g], dist.group.WORLD, 1)
    assert torch.equal(g, want), "non-contiguous gradient corrupted by flush"


def test_mixed_bucket():
    """Several tensors, mixed contiguity and shapes, in one bucket."""
    a = torch.randn(5, 7)
    b = torch.randn(3, 9).t()          # non-contiguous
    c = torch.randn(11)
    d = torch.randn(2, 3, 4)
    tensors = [a, b, c, d]
    want = [t.clone() for t in tensors]
    _flush(tensors, dist.group.WORLD, 1)
    for i, (got, exp) in enumerate(zip(tensors, want)):
        assert torch.allclose(got, exp, atol=1e-6), f"tensor {i} corrupted"
        assert got.shape == exp.shape, f"tensor {i} shape changed"


def test_averaging():
    """world>1 is simulated by dividing: check the divide actually happens."""
    g = torch.full((4, 4), 8.0)
    _flush([g], dist.group.WORLD, 4)   # world=4 with 1 real rank -> /4
    assert torch.allclose(g, torch.full((4, 4), 2.0)), "averaging wrong"


def test_dtype_preserved():
    g = torch.randn(6, 6, dtype=torch.bfloat16)
    want = g.clone()
    _flush([g], dist.group.WORLD, 1)
    assert g.dtype == torch.bfloat16, "dtype not preserved"
    assert torch.allclose(g.float(), want.float(), atol=1e-2)


def main():
    setup()
    for fn in [test_contiguous, test_non_contiguous, test_mixed_bucket,
               test_averaging, test_dtype_preserved]:
        fn()
        print(f"  ok  {fn.__name__}")
    print("PASS: bucketed gradient reduction preserves values, shapes, dtypes, "
          "and handles non-contiguous gradients")


if __name__ == "__main__":
    main()

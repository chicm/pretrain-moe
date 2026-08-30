"""Tests for the generated launch script.

These exist because a mangled line-continuation ("\n" as a literal backslash
plus 'n' instead of a newline) silently corrupted the argv, argparse rejected
`--tensor-model-parallel-size 'n'`, rank 0 exited with code 2, and every other
rank was SIGTERM'd as a bystander. The log then showed 15 "victim" ranks and
one real cause -- exactly the pattern that is expensive to misread.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe_rebuild.config import build_argv  # noqa: E402
from moe_rebuild.specs import dense_1b, moe_1node_mini, rfull_moe_prod  # noqa: E402
from tools.launch import MASTER, build_launch_script, hosts  # noqa: E402


def _script(spec, rank=0, nnodes=2, port=29999):
    return build_launch_script(rank, nnodes, port, Path("/tmp/run"), build_argv(spec))


def test_script_has_no_backslash_continuations():
    s = _script(dense_1b(2))
    assert "\\" not in s, "backslash in generated script -- continuation risk"


def test_torchrun_line_is_single_line_and_parses():
    s = _script(dense_1b(2))
    line = [l for l in s.splitlines() if "torch.distributed.run" in l]
    assert len(line) == 1
    toks = shlex.split(line[0])
    # every flag that takes a value must be followed by a non-flag token
    for i, t in enumerate(toks):
        if t == "--tensor-model-parallel-size":
            assert toks[i + 1].isdigit(), toks[i + 1]
        if t == "--num-layers":
            assert toks[i + 1].isdigit(), toks[i + 1]


def test_all_numeric_flags_have_numeric_values():
    numeric = {
        "--tensor-model-parallel-size", "--pipeline-model-parallel-size",
        "--expert-model-parallel-size", "--num-layers", "--hidden-size",
        "--ffn-hidden-size", "--num-attention-heads", "--num-query-groups",
        "--kv-channels", "--seq-length", "--max-position-embeddings",
        "--micro-batch-size", "--global-batch-size", "--train-iters",
        "--num-experts", "--moe-router-topk", "--moe-ffn-hidden-size",
    }
    for spec in (dense_1b(2), moe_1node_mini(), rfull_moe_prod()):
        argv = build_argv(spec)
        for i, a in enumerate(argv):
            if a in numeric:
                v = argv[i + 1]
                assert v.lstrip("-").isdigit(), f"{spec.run_id}: {a}={v!r}"


def test_master_is_node0_not_hostlist_head():
    """The rendezvous master must never be derived from the host list.

    Changing the host set for an A/B arm once also changed the master, so all
    ranks dialled node-7, nothing registered, and it looked like a bad node.
    """
    assert MASTER == "node-0"
    for n in (1, 2, 8, 15):
        assert hosts(n)[0] == "node-0"
    s = _script(dense_1b(15), rank=3, nnodes=15)
    assert "--master-addr=node-0" in s


def test_faulthandler_and_local_cache_env():
    s = _script(rfull_moe_prod())
    assert "export PYTHONFAULTHANDLER=1" in s   # SIGSEGV has no Python traceback
    assert "workspaceblobstore" not in s.split("--data-cache-path")[1].split()[0]


def test_log_redirect_lets_ssh_detach():
    """The script must reopen its own fds or ssh never closes the channel."""
    s = _script(dense_1b(2))
    assert "exec > /tmp/run/node0.log 2>&1 < /dev/null" in s


class TestTensorBoardReachesArgv:
    """The launcher must put --tensorboard-dir into the argv it actually runs.

    Regression: build_argv() was called before the launcher assigned
    spec.tensorboard_dir (the run dir is only known later), so every launched
    run had no --tensorboard-dir at all. config.py was correct and its unit
    test passed -- the bug lived in the ordering between the two, which only
    an end-to-end assertion on the emitted argv can catch.
    """

    def test_launcher_orders_assignment_before_build(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "tools" / "launch.py").read_text(encoding="utf-8")
        assign = src.index("spec.tensorboard_dir =")
        build = src.index("argv = build_argv(spec)")
        assert assign < build, (
            "spec.tensorboard_dir must be set before build_argv(), "
            "otherwise the flag never reaches the training process")

    def test_emitted_argv_has_tensorboard_dir(self):
        from moe_rebuild.specs import rfull_moe_prod
        from moe_rebuild.config import build_argv

        spec = rfull_moe_prod()
        spec.tensorboard_dir = "/scratch/rfull/runs/r_0101_000000/tensorboard"
        argv = build_argv(spec)
        assert "--tensorboard-dir" in argv
        got = argv[argv.index("--tensorboard-dir") + 1]
        assert got.endswith("/tensorboard"), got
        assert "\\" not in got, "run dir must use forward slashes on the cluster"

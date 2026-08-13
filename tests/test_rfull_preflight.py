"""Tests for the R-Full multi-node preflight verifier."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "verify_rfull_preflight.py"

CHECKS = ["PREFLIGHT_RANK_UP", "PREFLIGHT_ALLREDUCE", "PREFLIGHT_CROSS_NODE_P2P",
          "PREFLIGHT_EP_ALL_TO_ALL", "PREFLIGHT_EDP_ALL_REDUCE",
          "PREFLIGHT_RANK_RESULT", "PREFLIGHT_RANK_COMPLETE"]


def write_logs(d: pathlib.Path, nodes: int, gpn: int = 8, *,
               drop: tuple[str, int] | None = None,
               fail: tuple[str, int] | None = None,
               ib: bool = True, newlines: bool = True) -> None:
    for n in range(nodes):
        parts = ["NCCL INFO NET/IB : Using [0]mlx5_ib0\n" if ib else "NCCL INFO NET/Socket : eth0\n"]
        for local in range(gpn):
            rank = n * gpn + local
            for marker in CHECKS:
                if drop and drop[0] == marker and drop[1] == rank:
                    continue
                obj = {"marker": marker, "rank": rank, "host": f"node-{n}", "ok": True}
                if fail and fail[0] == marker and fail[1] == rank:
                    obj["ok"] = False
                parts.append(json.dumps(obj) + ("\n" if newlines else ""))
        (d / f"node-{n}.log").write_text("".join(parts))
    (d / "probe.py").write_text("# probe\n")


def run(d: pathlib.Path, world: int, nodes: int, out: pathlib.Path | None = None):
    cmd = [sys.executable, str(TOOL), "--preflight-dir", str(d),
           "--expected-world", str(world), "--expected-nodes", str(nodes)]
    if out:
        cmd += ["--acceptance-output", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True)


class PreflightVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_complete_two_node_run_passes(self) -> None:
        write_logs(self.d, 2)
        proc = run(self.d, 16, 2)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PREFLIGHT_RESULT=PASS", proc.stdout)

    def test_complete_fifteen_node_run_passes(self) -> None:
        write_logs(self.d, 15)
        proc = run(self.d, 120, 15)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PREFLIGHT_RESULT=PASS", proc.stdout)

    def test_missing_rank_marker_fails(self) -> None:
        write_logs(self.d, 2, drop=("PREFLIGHT_EP_ALL_TO_ALL", 9))
        proc = run(self.d, 16, 2)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("PREFLIGHT_RESULT=FAIL", proc.stdout)
        self.assertIn("EP_ALL_TO_ALL", proc.stderr)

    def test_failed_check_fails(self) -> None:
        write_logs(self.d, 2, fail=("PREFLIGHT_EDP_ALL_REDUCE", 3))
        proc = run(self.d, 16, 2)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("failed on [3]", proc.stderr)

    def test_missing_node_log_fails(self) -> None:
        write_logs(self.d, 1)
        proc = run(self.d, 16, 2)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("wrong number of node logs", proc.stderr)

    def test_socket_only_transport_fails(self) -> None:
        write_logs(self.d, 2, ib=False)
        proc = run(self.d, 16, 2)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("NET/IB", proc.stderr)

    def test_concatenated_markers_without_newlines_are_parsed(self) -> None:
        write_logs(self.d, 2, newlines=False)
        proc = run(self.d, 16, 2)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_acceptance_file_is_written(self) -> None:
        write_logs(self.d, 2)
        out = self.d / "acceptance.json"
        proc = run(self.d, 16, 2, out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(out.read_text())
        self.assertEqual(payload["result"], "PASS")
        self.assertEqual(payload["expected_world"], 16)
        self.assertEqual(len(payload["hosts"]), 2)
        self.assertIn("ACCEPTANCE_SHA256=", proc.stdout)

    def test_wrong_world_size_fails(self) -> None:
        write_logs(self.d, 2)
        proc = run(self.d, 120, 2)
        self.assertEqual(proc.returncode, 1)


if __name__ == "__main__":
    unittest.main()

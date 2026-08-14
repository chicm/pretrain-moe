"""Tests for the qualification-config deriver."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.derive_rfull_qualification_config import derive, main


def _production() -> dict:
    return {
        "cluster": {"nnodes": 15},
        "model": {
            "sequence_length": 4096,
            "native_vocab_size": 151669,
            "num_layers": 48,
        },
        "training": {
            "train_iters": 256856,
            "global_batch_size": 960,
            "micro_batch_size": 1,
            "eval_interval": 1000,
            "eval_iters": 10,
            "lr_warmup_iters": 2543,
            "lr_decay_iters": 25432,
            "lr_stable_iters": 228881,
        },
        "data": {
            "split": "998,2,0",
            "blend": [{"prefix": "/a/b", "weight": 1.0}],
        },
        "runtime": {"distributed_timeout_minutes": 240},
    }


class DeriveTest(unittest.TestCase):
    def test_scale_knobs_are_overridden(self) -> None:
        out = derive(_production(), nnodes=2, train_iters=4, gbs=16, mbs=1)
        self.assertEqual(out["cluster"]["nnodes"], 2)
        self.assertEqual(out["training"]["train_iters"], 4)
        self.assertEqual(out["training"]["global_batch_size"], 16)

    def test_data_path_semantics_are_preserved(self) -> None:
        prod = _production()
        out = derive(prod, nnodes=2, train_iters=4, gbs=16, mbs=1)
        # These are exactly the fields whose divergence broke Gate 4a v2.
        self.assertEqual(out["model"]["sequence_length"], 4096)
        self.assertEqual(out["model"]["native_vocab_size"], 151669)
        self.assertEqual(out["data"]["split"], prod["data"]["split"])
        self.assertEqual(out["data"]["blend"], prod["data"]["blend"])

    def test_timeout_is_preserved(self) -> None:
        out = derive(_production(), nnodes=2, train_iters=4, gbs=16, mbs=1)
        self.assertEqual(out["runtime"]["distributed_timeout_minutes"], 240)

    def test_lr_schedule_stays_consistent(self) -> None:
        out = derive(_production(), nnodes=2, train_iters=4, gbs=16, mbs=1)
        self.assertLessEqual(out["training"]["lr_warmup_iters"], 4)
        self.assertLessEqual(out["training"]["lr_decay_iters"], 4)
        self.assertNotIn("lr_stable_iters", out["training"])

    def test_production_config_is_not_mutated(self) -> None:
        prod = _production()
        derive(prod, nnodes=2, train_iters=4, gbs=16, mbs=1)
        self.assertEqual(prod["cluster"]["nnodes"], 15)
        self.assertEqual(prod["training"]["train_iters"], 256856)

    def test_cli_writes_config_and_reports_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            src = root / "prod.json"
            src.write_text(json.dumps(_production()))
            dst = root / "mini.json"
            rc = main([
                "--production-config", str(src),
                "--output", str(dst),
                "--nnodes", "2",
                "--train-iters", "4",
            ])
            self.assertEqual(rc, 0)
            doc = json.loads(dst.read_text())
            self.assertEqual(doc["model"]["sequence_length"], 4096)
            self.assertEqual(doc["cluster"]["nnodes"], 2)


if __name__ == "__main__":
    unittest.main()

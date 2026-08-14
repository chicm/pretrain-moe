"""Tests for the offline GPTDataset cache pre-builder."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.prebuild_rfull_data_cache import _sample_counts, build


def _config(**overrides) -> dict:
    config = {
        "runtime": {"mock_data": False, "seed": 1234, "distributed_timeout_minutes": 180},
        "model": {"seq_length": 4096, "eot_token_id": 151643},
        "training": {
            "train_iters": 1000,
            "global_batch_size": 960,
            "eval_interval": 500,
            "eval_iters": 10,
        },
        "data": {
            "split": "990,9,1",
            "blend": [{"weight": 1.0, "prefix": "/data/shard_0000"}],
        },
    }
    for key, value in overrides.items():
        config[key] = value
    return config


class SampleCountTests(unittest.TestCase):
    def test_train_samples_are_iters_times_global_batch(self) -> None:
        train, _, _ = _sample_counts(_config())
        self.assertEqual(train, 1000 * 960)

    def test_valid_samples_cover_every_eval_round(self) -> None:
        _, valid, _ = _sample_counts(_config())
        # 1000/500 + 1 = 3 rounds of 10 iters at gbs 960
        self.assertEqual(valid, 3 * 10 * 960)

    def test_production_scale_counts_do_not_overflow(self) -> None:
        config = _config()
        config["training"]["train_iters"] = 256856
        train, valid, test = _sample_counts(config)
        self.assertEqual(train, 256856 * 960)
        self.assertGreater(valid, 0)
        self.assertGreater(test, 0)


class GuardTests(unittest.TestCase):
    def test_mock_data_config_is_rejected(self) -> None:
        config = _config()
        config["runtime"]["mock_data"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "c.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(SystemExit):
                build(path, pathlib.Path(tmp) / "cache",
                      upstream_root=pathlib.Path(tmp), dry_run=True)

    def test_missing_blend_is_rejected(self) -> None:
        config = _config()
        config["data"]["blend"] = []
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "c.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(SystemExit):
                build(path, pathlib.Path(tmp) / "cache",
                      upstream_root=pathlib.Path(tmp), dry_run=True)

    def test_malformed_split_is_rejected(self) -> None:
        config = _config()
        config["data"]["split"] = "990,9"
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "c.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(SystemExit):
                build(path, pathlib.Path(tmp) / "cache",
                      upstream_root=pathlib.Path(tmp), dry_run=True)

    def test_dry_run_reports_the_plan_without_touching_megatron(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "c.json"
            path.write_text(json.dumps(_config()))
            plan = build(path, pathlib.Path(tmp) / "cache",
                         upstream_root=pathlib.Path(tmp), dry_run=True)
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["blend_entries"], 1)
        self.assertEqual(plan["seq_length"], 4096)
        self.assertEqual(plan["samples"]["train"], 1000 * 960)


if __name__ == "__main__":
    unittest.main()

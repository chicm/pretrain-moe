from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest

from tools.megatron_smoke import (
    ConfigError,
    build_megatron_args,
    build_torchrun_command,
    estimate_dense_parameters,
    load_config,
    topology,
    validate_config,
)
from tools.verify_megatron_smoke import verify


ROOT = pathlib.Path(__file__).resolve().parents[1]
DENSE_CONFIG = ROOT / "configs" / "smoke" / "dense_1b.json"


class DenseSmokeConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(DENSE_CONFIG)

    def test_all_committed_smoke_profiles_validate(self) -> None:
        profiles = sorted((ROOT / "configs" / "smoke").glob("*.json"))
        self.assertGreaterEqual(len(profiles), 4)
        for profile in profiles:
            with self.subTest(profile=profile.name):
                load_config(profile)

    def test_topology_is_two_node_tp2_dp8(self) -> None:
        self.assertEqual(
            topology(self.config),
            {"nnodes": 2, "gpus_per_node": 8, "world_size": 16, "tp": 2, "pp": 1, "cp": 1, "dp": 8},
        )

    def test_model_is_in_one_billion_parameter_class(self) -> None:
        estimate = estimate_dense_parameters(self.config)
        self.assertGreater(estimate, 950_000_000)
        self.assertLess(estimate, 1_050_000_000)

    def test_profile_renders_stock_dense_megatron_arguments(self) -> None:
        args = build_megatron_args(self.config, "/tmp/run")
        self.assertIn("--use-mcore-models", args)
        self.assertIn("--mock-data", args)
        self.assertIn("--use-distributed-optimizer", args)
        self.assertIn("--sequence-parallel", args)
        self.assertIn("--no-masked-softmax-fusion", args)
        self.assertNotIn("--num-experts", args)
        self.assertEqual(args[args.index("--attention-backend") + 1], "unfused")
        self.assertEqual(args[args.index("--num-layers") + 1], "18")
        self.assertEqual(args[args.index("--tensor-model-parallel-size") + 1], "2")

    def test_real_data_blend_renders_weighted_data_path(self) -> None:
        # The dense path must be able to read the SAME real corpus as the MoE
        # path, so it can serve as a one-variable control for crash triage.
        config = copy.deepcopy(self.config)
        config["runtime"]["mock_data"] = False
        config["data"] = {
            "split": "990,9,1",
            "blend": [
                {"weight": 2.0, "prefix": "/corpus/shard_a"},
                {"weight": 0.5, "prefix": "/corpus/shard_b"},
            ],
        }
        args = build_megatron_args(config, "/tmp/run")
        self.assertNotIn("--mock-data", args)
        start = args.index("--data-path")
        self.assertEqual(
            args[start + 1 : start + 5],
            ["2.0", "/corpus/shard_a", "0.5", "/corpus/shard_b"],
        )
        self.assertEqual(args[args.index("--split") + 1], "990,9,1")

    def test_real_data_and_mock_data_are_mutually_exclusive(self) -> None:
        config = copy.deepcopy(self.config)
        config["runtime"]["mock_data"] = True
        config["data"] = {"blend": [{"weight": 1.0, "prefix": "/corpus/a"}]}
        with self.assertRaises(ConfigError):
            build_megatron_args(config, "/tmp/run")

    def test_disabling_mock_data_without_a_blend_fails_closed(self) -> None:
        # Never silently fall back to mock tokens on a real run.
        config = copy.deepcopy(self.config)
        config["runtime"]["mock_data"] = False
        config.pop("data", None)
        with self.assertRaises(ConfigError):
            build_megatron_args(config, "/tmp/run")

    def test_blend_entry_requires_absolute_prefix_and_positive_weight(self) -> None:
        config = copy.deepcopy(self.config)
        config["runtime"]["mock_data"] = False
        for bad in (
            {"weight": 1.0, "prefix": "relative/path"},
            {"weight": 0, "prefix": "/corpus/a"},
            {"weight": -1.0, "prefix": "/corpus/a"},
            {"prefix": "/corpus/a"},
        ):
            with self.subTest(entry=bad):
                config["data"] = {"blend": [bad]}
                with self.assertRaises(ConfigError):
                    build_megatron_args(config, "/tmp/run")

    def test_torchrun_command_is_one_agent_for_the_requested_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            upstream = pathlib.Path(directory)
            (upstream / "pretrain_gpt.py").write_text("# placeholder\n", encoding="utf-8")
            command = build_torchrun_command(
                self.config,
                python="/opt/venv/bin/python",
                megatron_dir=str(upstream),
                run_dir="/tmp/run",
                node_rank=1,
                master_addr="node-0",
                master_port=29601,
            )
        self.assertEqual(command[:3], ["/opt/venv/bin/python", "-m", "torch.distributed.run"])
        self.assertTrue(any(item.endswith("megatron_rocm_entrypoint.py") for item in command))
        self.assertIn("--upstream-entrypoint", command)
        self.assertEqual(command[command.index("--nnodes") + 1], "2")
        self.assertEqual(command[command.index("--nproc-per-node") + 1], "8")
        self.assertEqual(command[command.index("--node-rank") + 1], "1")

    def test_invalid_world_partition_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["parallel"]["tensor_model_parallel_size"] = 3
        with self.assertRaisesRegex(ConfigError, "not divisible"):
            validate_config(config)

    def test_invalid_global_batch_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["training"]["global_batch_size"] = 15
        with self.assertRaisesRegex(ConfigError, "must be divisible"):
            validate_config(config)

    def test_all_attention_backends_disabled_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        for name in ("te_fused_attention", "te_flash_attention", "te_unfused_attention"):
            config["runtime"][name] = False
        with self.assertRaisesRegex(ConfigError, "do not match"):
            validate_config(config)


class SmokeLogVerifierTests(unittest.TestCase):
    def test_complete_finite_run_passes(self) -> None:
        config = {
            "name": "fixture",
            "upstream": {"commit": "a" * 40},
            "cluster": {"nnodes": 2, "gpus_per_node": 2},
            "training": {"train_iters": 3},
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            config_path = run_dir / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            for node in range(2):
                lines = []
                node_ranks = range(node * 2, node * 2 + 2)
                for rank in node_ranks:
                    lines.append(
                        json.dumps({"marker": "DISTRIBUTED_PROBE_RANK_OK", "rank": rank})
                    )
                # torchrun can concatenate simultaneous rank writes without a
                # newline; exercise the exact multiplexed-log shape seen in a
                # 2x8 run rather than only one JSON record per line.
                for marker in (
                    "EARLY_DEVICE_BIND",
                    "PROCESS_GROUP_DEVICE_BIND",
                    "DISTRIBUTED_PROBE_DEVICE_BIND",
                ):
                    lines.append(
                        "".join(
                            json.dumps({"marker": marker, "rank": rank})
                            for rank in node_ranks
                        )
                    )
                if node == 0:
                    lines.append(json.dumps({"marker": "DISTRIBUTED_PROBE_WORLD_OK"}))
                    for iteration, loss in ((1, 9.0), (2, 8.5), (3, 8.0)):
                        lines.append(
                            f"iteration {iteration:8d}/       3 | lm loss: {loss:.6E} |"
                        )
                lines.append('{"marker":"NODE_RUN_COMPLETE"}')
                lines.append('{"marker":"GPU_TELEMETRY_COMPLETE"}')
                (run_dir / f"node-{node}-fixture.log").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
                (run_dir / f"gpu-node-{node}.csv").write_text(
                    "sampled_utc,device,GPU use (%),GFX Activity,"
                    "GPU Memory Allocated (VRAM%),GPU Memory Read/Write Activity (%),"
                    "Memory Activity,Avg. Memory Bandwidth\n"
                    "2026-01-01T00:00:00Z,card0,90,0,10,0,0,0\n"
                    "2026-01-01T00:00:00Z,card1,80,0,11,0,0,0\n",
                    encoding="utf-8",
                )
            summary = verify(run_dir, config_path)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["probe_ranks"], [0, 1, 2, 3])
        self.assertEqual(summary["iterations"], [1, 2, 3])

    def test_runtime_error_fails(self) -> None:
        config = {
            "name": "fixture",
            "upstream": {"commit": "a" * 40},
            "cluster": {"nnodes": 1, "gpus_per_node": 1},
            "training": {"train_iters": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            config_path = run_dir / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            (run_dir / "node-0-fixture.log").write_text(
                '\n'.join(
                    [
                        '{"marker":"DISTRIBUTED_PROBE_RANK_OK","rank":0}',
                        '{"marker":"DISTRIBUTED_PROBE_WORLD_OK"}',
                        '{"marker":"EARLY_DEVICE_BIND","rank":0}',
                        '{"marker":"PROCESS_GROUP_DEVICE_BIND","rank":0}',
                        '{"marker":"DISTRIBUTED_PROBE_DEVICE_BIND","rank":0}',
                        "iteration 1/ 1 | lm loss: 1.0E+00 |",
                        "RuntimeError: synthetic failure",
                        '{"marker":"NODE_RUN_COMPLETE"}',
                        '{"marker":"GPU_TELEMETRY_COMPLETE"}',
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AssertionError, "fatal log signatures"):
                verify(run_dir, config_path)


if __name__ == "__main__":
    unittest.main()

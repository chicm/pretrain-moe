from __future__ import annotations

import argparse
import ast
import copy
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

from tools.pretrain_rfull_moe import (
    _batch_fingerprint,
    _completed_iteration,
    _validate_runtime_args,
    _wrap_resume_rng_neutral_iterator_builder,
    extra_args_provider,
)
from tools.rfull_gate2 import (
    ConfigError,
    PINNED_MEGATRON_COMMIT,
    build_megatron_args,
    build_torchrun_command,
    load_config,
    validate_config,
    validate_launch_environment,
)
from tools.rfull_rocm_entrypoint import (
    EXPECTED_NUMPY_VERSION_FOR_PRODUCT_ALIAS,
    _adapt_dcp_write_item_api,
    _ensure_numpy_product_api,
    _infer_same_geometry_reformulation_metadata,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
MINI_CONFIG = ROOT / "configs" / "gate2" / "rfull_ep8_mini.json"
FULL_CONFIG = ROOT / "configs" / "gate2" / "rfull_ep8_full_geometry.json"


class RFullGate2ConfigTests(unittest.TestCase):
    def test_launch_environment_rejects_detail_process_group_wrapper(self) -> None:
        with self.assertRaisesRegex(ConfigError, "_ProcessGroupWrapper"):
            validate_launch_environment({"TORCH_DISTRIBUTED_DEBUG": "detail"})
        validate_launch_environment({"TORCH_DISTRIBUTED_DEBUG": "INFO"})

    def test_launch_environment_rejects_unconsumed_extra_mcore_args(self) -> None:
        with self.assertRaisesRegex(ConfigError, "not consumed"):
            validate_launch_environment({"EXTRA_MCORE_ARGS": "--load /checkpoint"})
        validate_launch_environment({"EXTRA_MCORE_ARGS": "  "})

    def test_profiles_validate_with_frozen_local_parameter_counts(self) -> None:
        mini = load_config(MINI_CONFIG)
        full = load_config(FULL_CONFIG)
        self.assertEqual(mini["model"]["expected_local_parameters"], 19_371_008)
        self.assertEqual(full["model"]["expected_local_parameters"], 4_586_027_008)
        self.assertEqual(full["upstream"]["commit"], PINNED_MEGATRON_COMMIT)
        self.assertEqual(full["model"]["moe_layer_frequency"], [0, 0] + [1] * 46)

    def test_production_cli_preserves_native_and_padded_vocab_contract(self) -> None:
        full = load_config(FULL_CONFIG)
        args = build_megatron_args(full, data_cache_path="/shared/cache")
        native_index = args.index("--vocab-size")
        divisor_index = args.index("--make-vocab-size-divisible-by")
        self.assertEqual(args[native_index + 1], "151669")
        self.assertEqual(args[divisor_index + 1], "1187")
        self.assertIn("--rfull-qualification-only", args)
        self.assertEqual(args[args.index("--rfull-profile") + 1], "production")
        self.assertEqual(
            args[args.index("--rfull-expected-local-parameters") + 1],
            "4586027008",
        )

    def test_cli_uses_exact_grouped_dropless_selected_softmax_contract(self) -> None:
        mini = load_config(MINI_CONFIG)
        args = build_megatron_args(mini, data_cache_path="/shared/cache")
        required = {
            "--moe-grouped-gemm",
            "--moe-use-legacy-grouped-gemm",
            "--moe-token-dispatcher-type",
            "--moe-router-score-function",
            "--moe-router-dtype",
            "--moe-shared-expert-intermediate-size",
            "--disable-bias-linear",
            "--qk-layernorm",
        }
        self.assertTrue(required.issubset(args))
        for forbidden in (
            "--moe-router-pre-softmax",
            "--moe-expert-capacity-factor",
            "--moe-token-drop-policy",
            "--moe-shared-expert-overlap",
            "--moe-enable-deepep",
            "--moe-permute-fusion",
        ):
            self.assertNotIn(forbidden, args)
        self.assertEqual(args[args.index("--moe-token-dispatcher-type") + 1], "alltoall")
        self.assertEqual(args[args.index("--moe-router-score-function") + 1], "softmax")
        self.assertEqual(args[args.index("--moe-router-dtype") + 1], "fp32")

    def test_checkpoint_arguments_are_explicit_and_validated(self) -> None:
        mini = load_config(MINI_CONFIG)
        args = build_megatron_args(
            mini,
            data_cache_path="/shared/cache",
            train_iters=6,
            save_dir="/shared/checkpoints/run-a",
            save_interval=3,
            load_dir="/shared/checkpoints/run-a",
            exit_interval=3,
        )
        self.assertEqual(args[args.index("--train-iters") + 1], "6")
        self.assertEqual(args[args.index("--save-interval") + 1], "3")
        self.assertEqual(args[args.index("--exit-interval") + 1], "3")
        self.assertEqual(args[args.index("--load") + 1], "/shared/checkpoints/run-a")
        # Saving must disable the fully-parallel ("fully_sharded_model_space")
        # optimizer sharding strategy.  Upstream MCore's GLU GroupedMLP produces
        # flattened ranges that do not tile the shard when a distributed-optimizer
        # slice straddles a gate/up chunk boundary, which is what broke the
        # 120-rank save (expert-DP 15 -> 1.6 chunks per shard).  The 16-rank gates
        # never caught it because expert-DP 2 gives a whole 12.0 chunks per shard.
        self.assertIn("--no-ckpt-fully-parallel-save", args)
        # It is only meaningful when actually saving.
        no_save = build_megatron_args(mini, data_cache_path="/shared/cache", train_iters=6)
        self.assertNotIn("--no-ckpt-fully-parallel-save", no_save)
        with self.assertRaisesRegex(ConfigError, "save_interval"):
            build_megatron_args(
                mini, data_cache_path="/shared/cache", save_dir="/shared/checkpoints/run-b"
            )
        with self.assertRaisesRegex(ConfigError, "absolute"):
            build_megatron_args(mini, data_cache_path="relative/cache")

    def test_geometry_and_attention_drift_fail_closed(self) -> None:
        mini = load_config(MINI_CONFIG)
        for path, value in (
            (("model", "hidden_size"), 768),
            (("parallel", "expert_model_parallel_size"), 4),
            (("runtime", "te_flash_attention"), True),
        ):
            drifted = copy.deepcopy(mini)
            drifted[path[0]][path[1]] = value
            with self.assertRaises(ConfigError):
                validate_config(drifted)

    def test_native_entrypoint_is_lazy_and_qualification_flag_is_required(self) -> None:
        parser = extra_args_provider(argparse.ArgumentParser())
        parsed = parser.parse_args(
            [
                "--rfull-profile",
                "ep8-mini",
                "--rfull-expected-local-parameters",
                "19371008",
            ]
        )
        self.assertFalse(parsed.rfull_qualification_only)
        self.assertFalse(parsed.rfull_production_launch)
        args = SimpleNamespace(
            rfull_qualification_only=False,
            rfull_production_launch=False,
            use_legacy_models=False,
            transformer_impl="transformer_engine",
            rfull_profile="ep8-mini",
            vocab_size=4000,
            padded_vocab_size=4096,
            make_vocab_size_divisible_by=128,
        )
        # Neither intent declared -> refuse to run at all.
        with self.assertRaisesRegex(RuntimeError, "declare run intent"):
            _validate_runtime_args(args)

    def test_runtime_args_reject_conflicting_or_unsound_production_intent(self) -> None:
        def make(**overrides: object) -> SimpleNamespace:
            base = dict(
                rfull_qualification_only=False,
                rfull_production_launch=True,
                use_legacy_models=False,
                transformer_impl="transformer_engine",
                rfull_profile="production",
                mock_data=False,
                vocab_size=151669,
                padded_vocab_size=151936,
                make_vocab_size_divisible_by=1187,
            )
            base.update(overrides)
            return SimpleNamespace(**base)

        # Both intents at once is a contradiction.
        with self.assertRaisesRegex(RuntimeError, "mutually exclusive"):
            _validate_runtime_args(make(rfull_qualification_only=True))
        # "Production" on gate geometry would silently train the wrong model.
        with self.assertRaisesRegex(RuntimeError, "requires the production profile"):
            _validate_runtime_args(
                make(rfull_profile="ep8-mini", vocab_size=4000, padded_vocab_size=4096,
                     make_vocab_size_divisible_by=128)
            )
        # "Production" on mock data would burn the cluster on random tokens.
        with self.assertRaisesRegex(RuntimeError, "refuses mock data"):
            _validate_runtime_args(make(mock_data=True))
        # A correctly declared production run passes.
        _validate_runtime_args(make())

    def test_native_entrypoint_uses_only_pinned_pretrain_keywords(self) -> None:
        tree = ast.parse(
            (ROOT / "tools" / "pretrain_rfull_moe.py").read_text(encoding="utf-8")
        )
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "pretrain"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 4)
        self.assertEqual(
            {keyword.arg for keyword in calls[0].keywords},
            {
                "args_defaults",
                "extra_args_provider",
                "get_embedding_ranks",
                "get_position_embedding_ranks",
                "non_loss_data_func",
                "process_non_loss_data_func",
            },
        )

    def test_native_entrypoint_uses_only_pinned_gpt_model_keywords(self) -> None:
        tree = ast.parse(
            (ROOT / "tools" / "pretrain_rfull_moe.py").read_text(encoding="utf-8")
        )
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "GPTModel"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 0)
        self.assertEqual(
            {keyword.arg for keyword in calls[0].keywords},
            {
                "config",
                "transformer_layer_spec",
                "vocab_size",
                "max_sequence_length",
                "pre_process",
                "post_process",
                "fp16_lm_cross_entropy",
                "parallel_output",
                "share_embeddings_and_output_weights",
                "position_embedding_type",
                "rotary_percent",
                "rotary_base",
                "rope_scaling",
            },
        )

    def test_completion_marker_uses_pinned_final_train_iters_contract(self) -> None:
        self.assertEqual(
            _completed_iteration(SimpleNamespace(train_iters=4, iteration=0)),
            4,
        )
        for invalid in (None, 0, -1, True, 4.0):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(RuntimeError, "train_iters"):
                    _completed_iteration(SimpleNamespace(train_iters=invalid))

    def test_resume_iterator_builder_restores_only_global_cpu_rng(self) -> None:
        import torch

        runtime_args = SimpleNamespace(
            load="/checkpoint",
            iteration=2,
            no_load_rng=False,
            rank=0,
        )
        observed_draws = []

        def builder():
            observed_draws.append(torch.empty((), dtype=torch.int64).random_().item())
            return "iterators"

        wrapped = _wrap_resume_rng_neutral_iterator_builder(
            builder, lambda: runtime_args
        )
        torch.manual_seed(1234)
        before = torch.get_rng_state().clone()
        self.assertEqual(wrapped(), "iterators")
        self.assertEqual(len(observed_draws), 1)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))

        runtime_args.load = None
        runtime_args.iteration = 0
        before = torch.get_rng_state().clone()
        wrapped()
        self.assertFalse(torch.equal(torch.get_rng_state(), before))

    def test_resume_iterator_builder_rejects_no_load_rng(self) -> None:
        import torch

        runtime_args = SimpleNamespace(
            load="/checkpoint",
            iteration=2,
            no_load_rng=True,
            rank=0,
        )

        def builder():
            torch.empty((), dtype=torch.int64).random_().item()
            return None

        wrapped = _wrap_resume_rng_neutral_iterator_builder(
            builder, lambda: runtime_args
        )
        with self.assertRaisesRegex(RuntimeError, "--no-load-rng"):
            wrapped()

    def test_batch_fingerprint_is_order_stable_and_value_sensitive(self) -> None:
        import numpy as np
        import torch

        left = {
            "tokens": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            "mask": np.array([[True, False, True]], dtype=np.bool_),
        }
        right = {"mask": left["mask"].copy(), "tokens": left["tokens"].clone()}
        self.assertEqual(_batch_fingerprint(left), _batch_fingerprint(right))
        right["tokens"][0, 2] = 4
        self.assertNotEqual(_batch_fingerprint(left), _batch_fingerprint(right))

    def test_native_entrypoint_accepts_exact_mini_vocab_contract(self) -> None:
        args = SimpleNamespace(
            rfull_qualification_only=True,
            use_legacy_models=False,
            transformer_impl="transformer_engine",
            rfull_profile="ep8-mini",
            vocab_size=4000,
            padded_vocab_size=4096,
            make_vocab_size_divisible_by=128,
        )
        _validate_runtime_args(args)

    def test_native_entrypoint_rejects_resume_without_rng_loading(self) -> None:
        args = SimpleNamespace(
            rfull_qualification_only=True,
            use_legacy_models=False,
            transformer_impl="transformer_engine",
            rfull_profile="ep8-mini",
            vocab_size=4000,
            padded_vocab_size=4096,
            make_vocab_size_divisible_by=128,
            load="/checkpoint",
            no_load_rng=True,
        )
        with self.assertRaisesRegex(RuntimeError, "requires RNG state loading"):
            _validate_runtime_args(args)

    def test_native_entrypoint_rejects_vocab_drift_before_model_import(self) -> None:
        args = SimpleNamespace(
            rfull_qualification_only=True,
            use_legacy_models=False,
            transformer_impl="transformer_engine",
            rfull_profile="production",
            vocab_size=151669,
            padded_vocab_size=151936,
            make_vocab_size_divisible_by=128,
        )
        with self.assertRaisesRegex(RuntimeError, "vocab contract drift"):
            _validate_runtime_args(args)

    def test_torchrun_command_routes_through_project_rocm_entrypoint(self) -> None:
        mini = load_config(MINI_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            project = root / "project"
            megatron = root / "Megatron-LM"
            (project / "tools").mkdir(parents=True)
            megatron.mkdir()
            for path in (
                project / "tools" / "rfull_rocm_entrypoint.py",
                project / "tools" / "pretrain_rfull_moe.py",
                megatron / "pretrain_gpt.py",
            ):
                path.write_text("# fixture\n", encoding="utf-8")
            command = build_torchrun_command(
                mini,
                python="/opt/venv/bin/python",
                project_dir=str(project),
                megatron_dir=str(megatron),
                master_addr="127.0.0.1",
                master_port=29710,
                data_cache_path="/shared/cache",
            )
        self.assertIn("torch.distributed.run", command)
        self.assertEqual(command[command.index("--nproc-per-node") + 1], "8")
        self.assertIn("rfull_rocm_entrypoint.py", " ".join(command))
        self.assertIn("pretrain_rfull_moe.py", " ".join(command))
        self.assertNotIn("pretrain_gpt.py", " ".join(command))


class RFullProductionDataBlendTests(unittest.TestCase):
    """The data source is a fail-closed choice between mock and a real blend."""

    def _real_data_config(self) -> dict:
        config = copy.deepcopy(load_config(MINI_CONFIG))
        config["runtime"]["mock_data"] = False
        config["runtime"]["distributed_timeout_minutes"] = 180
        config["data"] = {
            "split": "990,9,1",
            "blend": [
                {"weight": 0.6, "prefix": "/shared/data/dclm_tok/part_00/shard_0000"},
                {"weight": 0.4, "prefix": "/shared/data/math_tok/shard_0001"},
            ],
        }
        return config

    def test_real_data_requires_an_explicit_distributed_timeout(self) -> None:
        # Regression: the 10 minute default aborted first-time index
        # construction over a 487 shard blend with a watchdog SIGABRT on a
        # 1-element ALLREDUCE after 600013 ms.
        config = self._real_data_config()
        del config["runtime"]["distributed_timeout_minutes"]
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_distributed_timeout_must_be_a_positive_int(self) -> None:
        for bad in (0, -5, "180", 1.5, None):
            config = self._real_data_config()
            config["runtime"]["distributed_timeout_minutes"] = bad
            with self.assertRaises(ConfigError):
                validate_config(config)

    def test_real_blend_emits_the_configured_timeout(self) -> None:
        args = build_megatron_args(self._real_data_config(), data_cache_path="/shared/cache")
        self.assertIn("--distributed-timeout-minutes", args)
        self.assertEqual(args[args.index("--distributed-timeout-minutes") + 1], "180")

    def test_mock_data_does_not_require_a_timeout(self) -> None:
        # Qualification runs keep upstream's default; only real-data runs pay
        # the index construction cost.
        args = build_megatron_args(load_config(MINI_CONFIG), data_cache_path="/shared/cache")
        self.assertNotIn("--distributed-timeout-minutes", args)

    def test_mock_data_still_emits_mock_flags(self) -> None:
        args = build_megatron_args(load_config(MINI_CONFIG), data_cache_path="/shared/cache")
        self.assertIn("--mock-data", args)
        self.assertNotIn("--data-path", args)
        self.assertEqual(args[args.index("--split") + 1], "949,50,1")

    def test_default_schedule_is_cosine_with_unit_warmup(self) -> None:
        # Gates are only a handful of iterations; a real warmup would keep LR at
        # ~0 for the whole run and the optimizer would never be exercised.
        args = build_megatron_args(load_config(MINI_CONFIG), data_cache_path="/shared/cache")
        self.assertEqual(args[args.index("--lr-decay-style") + 1], "cosine")
        self.assertEqual(args[args.index("--lr-warmup-iters") + 1], "1")
        self.assertNotIn("--lr-wsd-decay-iters", args)

    def test_warmup_stable_decay_schedule_emits_wsd_flags(self) -> None:
        config = self._real_data_config()
        config["training"]["train_iters"] = 254313
        config["training"]["lr_schedule"] = {
            "style": "warmup_stable_decay",
            "warmup_iters": 2543,
            "decay_iters": 25432,
            "stable_end_iter": 228881,
        }
        args = build_megatron_args(config, data_cache_path="/shared/cache")
        self.assertEqual(args[args.index("--lr-decay-style") + 1], "WSD")
        self.assertEqual(args[args.index("--lr-wsd-decay-style") + 1], "cosine")
        self.assertEqual(args[args.index("--lr-wsd-decay-iters") + 1], "25432")
        self.assertEqual(args[args.index("--lr-warmup-iters") + 1], "2543")
        self.assertEqual(args[args.index("--lr-decay-iters") + 1], "254313")

    def test_warmup_stable_decay_rejects_inconsistent_phases(self) -> None:
        def config_with(**schedule: object) -> dict:
            config = self._real_data_config()
            config["training"]["train_iters"] = 254313
            base = {
                "style": "warmup_stable_decay",
                "warmup_iters": 2543,
                "decay_iters": 25432,
                "stable_end_iter": 228881,
            }
            base.update(schedule)
            config["training"]["lr_schedule"] = base
            return config

        # stable_end_iter that disagrees with train_iters - decay_iters means one
        # of the three numbers is a typo; refuse rather than reshape the run.
        with self.assertRaises(ConfigError):
            build_megatron_args(config_with(stable_end_iter=230000),
                                data_cache_path="/shared/cache")
        # Phases that do not fit leave no stable phase at all.
        with self.assertRaises(ConfigError):
            build_megatron_args(config_with(warmup_iters=240000, stable_end_iter=None),
                                data_cache_path="/shared/cache")
        for bad in (0, -1, "2543", None):
            with self.assertRaises(ConfigError):
                build_megatron_args(config_with(warmup_iters=bad, stable_end_iter=None),
                                    data_cache_path="/shared/cache")
        with self.assertRaises(ConfigError):
            build_megatron_args(config_with(style="linear"), data_cache_path="/shared/cache")

    def test_rotary_base_comes_from_the_config(self) -> None:
        # The frozen design pins theta=1e6; MCore's flag defaults to 10000, so a
        # missing passthrough silently trains the wrong positional encoding.
        config = self._real_data_config()
        config["model"]["rotary_base"] = 1000000
        args = build_megatron_args(config, data_cache_path="/shared/cache")
        self.assertEqual(args[args.index("--rotary-base") + 1], "1000000")
        default = build_megatron_args(load_config(MINI_CONFIG), data_cache_path="/shared/cache")
        self.assertEqual(default[default.index("--rotary-base") + 1], "10000")

    def test_gate_configs_declare_qualification_intent(self) -> None:
        args = build_megatron_args(load_config(MINI_CONFIG), data_cache_path="/shared/cache")
        self.assertIn("--rfull-qualification-only", args)
        self.assertNotIn("--rfull-production-launch", args)

    def test_production_launch_flag_is_opt_in_via_config(self) -> None:
        config = self._real_data_config()
        config["production_launch"] = True
        args = build_megatron_args(config, data_cache_path="/shared/cache")
        self.assertIn("--rfull-production-launch", args)
        self.assertNotIn("--rfull-qualification-only", args)

    def test_real_blend_emits_weighted_data_path(self) -> None:
        args = build_megatron_args(self._real_data_config(), data_cache_path="/shared/cache")
        self.assertNotIn("--mock-data", args)
        self.assertEqual(args[args.index("--split") + 1], "990,9,1")
        start = args.index("--data-path")
        self.assertEqual(
            args[start + 1 : start + 5],
            [
                "0.6",
                "/shared/data/dclm_tok/part_00/shard_0000",
                "0.4",
                "/shared/data/math_tok/shard_0001",
            ],
        )

    def test_mock_data_forbids_a_blend(self) -> None:
        config = self._real_data_config()
        config["runtime"]["mock_data"] = True
        with self.assertRaises(ConfigError):
            build_megatron_args(config, data_cache_path="/shared/cache")

    def test_real_data_requires_a_blend(self) -> None:
        config = copy.deepcopy(load_config(MINI_CONFIG))
        config["runtime"]["mock_data"] = False
        with self.assertRaises(ConfigError):
            build_megatron_args(config, data_cache_path="/shared/cache")

    def test_blend_entries_are_validated(self) -> None:
        for bad in (
            {"weight": 0.0, "prefix": "/shared/a"},
            {"weight": -1.0, "prefix": "/shared/a"},
            {"weight": 1.0, "prefix": "relative/path"},
            {"weight": 1.0},
            {"prefix": "/shared/a"},
        ):
            config = self._real_data_config()
            config["data"]["blend"] = [bad]
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                build_megatron_args(config, data_cache_path="/shared/cache")

    def test_empty_blend_is_rejected(self) -> None:
        config = self._real_data_config()
        config["data"]["blend"] = []
        with self.assertRaises(ConfigError):
            build_megatron_args(config, data_cache_path="/shared/cache")

    def test_split_must_have_three_fields(self) -> None:
        config = self._real_data_config()
        config["data"]["split"] = "99,1"
        with self.assertRaises(ConfigError):
            build_megatron_args(config, data_cache_path="/shared/cache")


class RFullGate3MultiNodeTests(unittest.TestCase):
    MINI_2NODE = ROOT / "configs" / "gate3" / "rfull_ep8_mini_2node.json"
    FULL_2NODE = ROOT / "configs" / "gate3" / "rfull_ep8_full_geometry_2node.json"

    def _fixture_dirs(self, root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
        project = root / "project"
        megatron = root / "Megatron-LM"
        (project / "tools").mkdir(parents=True)
        megatron.mkdir()
        for path in (
            project / "tools" / "rfull_rocm_entrypoint.py",
            project / "tools" / "pretrain_rfull_moe.py",
            megatron / "pretrain_gpt.py",
        ):
            path.write_text("# fixture\n", encoding="utf-8")
        return project, megatron

    def test_two_node_profiles_declare_world_16_with_ep8_and_edp2(self) -> None:
        for path in (self.MINI_2NODE, self.FULL_2NODE):
            config = load_config(path)
            self.assertEqual(config["cluster"]["nnodes"], 2)
            self.assertEqual(config["cluster"]["gpus_per_node"], 8)
            world = config["cluster"]["nnodes"] * config["cluster"]["gpus_per_node"]
            self.assertEqual(world, 16)
            ep = config["parallel"]["expert_model_parallel_size"]
            self.assertEqual(ep, 8)
            # Expert-data parallel replicas = world / EP.
            self.assertEqual(world // ep, 2)
            self.assertEqual(config["training"]["global_batch_size"] % world, 0)
            self.assertEqual(config["upstream"]["commit"], PINNED_MEGATRON_COMMIT)

    def test_two_node_full_geometry_keeps_frozen_ledger(self) -> None:
        full = load_config(self.FULL_2NODE)
        self.assertEqual(full["model"]["expected_local_parameters"], 4_586_027_008)
        self.assertEqual(full["model"]["num_layers"], 48)
        self.assertEqual(full["model"]["moe_layer_frequency"], [0, 0] + [1] * 46)

    def test_global_batch_must_divide_world_size(self) -> None:
        config = copy.deepcopy(load_config(self.MINI_2NODE))
        config["training"]["global_batch_size"] = 12
        with self.assertRaisesRegex(ConfigError, "divisible"):
            validate_config(config)

    def test_expert_parallel_group_must_fit_inside_one_node(self) -> None:
        config = copy.deepcopy(load_config(self.MINI_2NODE))
        config["parallel"]["expert_model_parallel_size"] = 16
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_node_count_must_be_positive_int(self) -> None:
        config = copy.deepcopy(load_config(self.MINI_2NODE))
        config["cluster"]["nnodes"] = 0
        with self.assertRaisesRegex(ConfigError, "positive int"):
            validate_config(config)

    def test_torchrun_command_carries_node_rank_and_nnodes(self) -> None:
        config = load_config(self.MINI_2NODE)
        with tempfile.TemporaryDirectory() as temporary:
            project, megatron = self._fixture_dirs(pathlib.Path(temporary))
            for node_rank in (0, 1):
                command = build_torchrun_command(
                    config,
                    python="/opt/venv/bin/python",
                    project_dir=str(project),
                    megatron_dir=str(megatron),
                    master_addr="100.64.142.47",
                    master_port=29710,
                    data_cache_path="/shared/cache",
                    node_rank=node_rank,
                )
                self.assertEqual(command[command.index("--nnodes") + 1], "2")
                self.assertEqual(command[command.index("--node-rank") + 1], str(node_rank))
                self.assertEqual(command[command.index("--nproc-per-node") + 1], "8")
                self.assertEqual(
                    command[command.index("--master-addr") + 1], "100.64.142.47"
                )

    def test_torchrun_rejects_out_of_range_node_rank(self) -> None:
        config = load_config(self.MINI_2NODE)
        with tempfile.TemporaryDirectory() as temporary:
            project, megatron = self._fixture_dirs(pathlib.Path(temporary))
            for bad_rank in (-1, 2, 99):
                with self.assertRaisesRegex(ConfigError, "outside"):
                    build_torchrun_command(
                        config,
                        python="/opt/venv/bin/python",
                        project_dir=str(project),
                        megatron_dir=str(megatron),
                        master_addr="100.64.142.47",
                        master_port=29710,
                        data_cache_path="/shared/cache",
                        node_rank=bad_rank,
                    )

    def test_torchrun_rejects_loopback_master_for_multi_node(self) -> None:
        config = load_config(self.MINI_2NODE)
        with tempfile.TemporaryDirectory() as temporary:
            project, megatron = self._fixture_dirs(pathlib.Path(temporary))
            for loopback in ("127.0.0.1", "localhost"):
                with self.assertRaisesRegex(ConfigError, "routable"):
                    build_torchrun_command(
                        config,
                        python="/opt/venv/bin/python",
                        project_dir=str(project),
                        megatron_dir=str(megatron),
                        master_addr=loopback,
                        master_port=29710,
                        data_cache_path="/shared/cache",
                        node_rank=0,
                    )

    def test_single_node_gate2_profiles_remain_valid(self) -> None:
        # Gate 3 generalisation must not regress the sealed Gate 2 contract.
        for path in (MINI_CONFIG, FULL_CONFIG):
            config = load_config(path)
            self.assertEqual(config["cluster"]["nnodes"], 1)
        with tempfile.TemporaryDirectory() as temporary:
            project, megatron = self._fixture_dirs(pathlib.Path(temporary))
            command = build_torchrun_command(
                load_config(MINI_CONFIG),
                python="/opt/venv/bin/python",
                project_dir=str(project),
                megatron_dir=str(megatron),
                master_addr="127.0.0.1",
                master_port=29710,
                data_cache_path="/shared/cache",
            )
            self.assertEqual(command[command.index("--nnodes") + 1], "1")
            self.assertEqual(command[command.index("--node-rank") + 1], "0")


class RFullCheckpointCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _product(values: tuple[int, ...]) -> int:
        result = 1
        for value in values:
            result *= value
        return result

    def test_qualified_numpy_two_installs_exact_prod_alias(self) -> None:
        fake_numpy = SimpleNamespace(
            __version__=EXPECTED_NUMPY_VERSION_FOR_PRODUCT_ALIAS,
            prod=self._product,
        )
        mode = _ensure_numpy_product_api(fake_numpy)
        self.assertEqual(mode, "alias_to_prod")
        self.assertIs(fake_numpy.product, fake_numpy.prod)
        self.assertEqual(fake_numpy.product((2, 3, 4)), 24)

    def test_unknown_numpy_without_product_fails_closed(self) -> None:
        fake_numpy = SimpleNamespace(__version__="2.3.0", prod=self._product)
        with self.assertRaisesRegex(RuntimeError, "not the qualified compatibility target"):
            _ensure_numpy_product_api(fake_numpy)
        self.assertFalse(hasattr(fake_numpy, "product"))

    def test_native_product_is_not_replaced(self) -> None:
        native = self._product
        fake_numpy = SimpleNamespace(__version__="1.26.4", prod=native, product=native)
        mode = _ensure_numpy_product_api(fake_numpy)
        self.assertEqual(mode, "native")
        self.assertIs(fake_numpy.product, native)

    def test_dcp_write_item_with_optional_sixth_argument_is_not_replaced(self) -> None:
        def native_write_item(
            transforms, stream, data, write_item, storage_key, safe_tensors=False
        ):
            return safe_tensors

        adapted, mode = _adapt_dcp_write_item_api(
            native_write_item,
            SimpleNamespace(TORCH_SAVE=SimpleNamespace(value="torch_save")),
        )
        self.assertEqual(mode, "native_five_arg_compatible")
        self.assertIs(adapted, native_write_item)
        self.assertFalse(adapted(None, None, None, None, "key"))

    def test_dcp_write_item_adapter_passes_exact_torch_save_format(self) -> None:
        torch_save = SimpleNamespace(value="torch_save")
        observed = {}

        def native_write_item(
            transforms, stream, data, write_item, storage_key, serialization_format
        ):
            observed["arguments"] = (
                transforms,
                stream,
                data,
                write_item,
                storage_key,
                serialization_format,
            )
            return "written"

        adapted, mode = _adapt_dcp_write_item_api(
            native_write_item, SimpleNamespace(TORCH_SAVE=torch_save)
        )
        self.assertEqual(mode, "append_torch_save_serialization_format")
        self.assertIsNot(adapted, native_write_item)
        self.assertEqual(adapted(1, 2, 3, 4, "key"), "written")
        self.assertEqual(observed["arguments"], (1, 2, 3, 4, "key", torch_save))

    def test_unknown_dcp_write_item_signature_fails_closed(self) -> None:
        def unknown_write_item(
            transforms, stream, data, write_item, storage_key, mystery
        ):
            return mystery

        with self.assertRaisesRegex(RuntimeError, "unknown torch DCP"):
            _adapt_dcp_write_item_api(
                unknown_write_item,
                SimpleNamespace(TORCH_SAVE=SimpleNamespace(value="torch_save")),
            )

    @staticmethod
    def _fake_reformulation_strategy() -> SimpleNamespace:
        return SimpleNamespace(
            nested_values=lambda state: state,
            is_nd_flattened_tensor=lambda tensor: tensor.is_nd_flattened,
            TensorReformulationMetadata=lambda original, stored: (original, stored),
        )

    def test_missing_mcore_metadata_is_inferred_only_for_same_geometry(self) -> None:
        tensor = SimpleNamespace(
            key="optimizer.state.weight",
            is_nd_flattened=True,
            global_shape=(2, 3, 4),
        )
        checkpoint_metadata = SimpleNamespace(
            state_dict_metadata={tensor.key: SimpleNamespace(size=(24,))}
        )
        inferred = _infer_same_geometry_reformulation_metadata(
            self._fake_reformulation_strategy(), checkpoint_metadata, [tensor]
        )
        self.assertEqual(inferred, {tensor.key: ((2, 3, 4), (24,))})

    def test_missing_mcore_metadata_rejects_geometry_mismatch(self) -> None:
        tensor = SimpleNamespace(
            key="optimizer.state.weight",
            is_nd_flattened=True,
            global_shape=(2, 3, 4),
        )
        checkpoint_metadata = SimpleNamespace(
            state_dict_metadata={tensor.key: SimpleNamespace(size=(23,))}
        )
        with self.assertRaisesRegex(RuntimeError, "element-count mismatch"):
            _infer_same_geometry_reformulation_metadata(
                self._fake_reformulation_strategy(), checkpoint_metadata, [tensor]
            )

    def test_missing_mcore_metadata_without_qualified_tensors_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no N-D flattened tensors"):
            _infer_same_geometry_reformulation_metadata(
                self._fake_reformulation_strategy(),
                SimpleNamespace(state_dict_metadata={}),
                [],
            )


if __name__ == "__main__":
    unittest.main()

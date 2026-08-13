import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import torch
import torch.nn.functional as F


_MCORE_AVAILABLE = importlib.util.find_spec("megatron") is not None


@unittest.skipUnless(_MCORE_AVAILABLE, "set PYTHONPATH to the pinned Megatron checkout")
class RFullMCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from megatron.core.transformer.transformer_config import TransformerConfig
        from rfull_moe import mcore

        cls.TransformerConfig = TransformerConfig
        cls.mcore = mcore

    def test_source_guard_accepts_duplicate_entries_for_one_physical_root(self):
        import megatron
        from rfull_moe import pinned_mcore

        roots = list(megatron.__path__)
        self.assertGreaterEqual(len(roots), 1)
        with mock.patch.object(megatron, "__path__", [roots[0], roots[0]]):
            self.assertEqual(
                pinned_mcore._megatron_package_root(),
                Path(roots[0]).resolve(),
            )

    def make_config(self, **overrides):
        values = dict(
            num_layers=48,
            hidden_size=2048,
            num_attention_heads=32,
            num_query_groups=4,
            kv_channels=128,
            ffn_hidden_size=5504,
            num_moe_experts=96,
            moe_ffn_hidden_size=896,
            moe_shared_expert_intermediate_size=896,
            moe_router_topk=6,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=1.0e-3,
            moe_z_loss_coeff=1.0e-4,
            moe_router_dtype="fp32",
            moe_router_score_function="softmax",
            moe_router_pre_softmax=False,
            moe_token_dispatcher_type="alltoall",
            moe_expert_capacity_factor=None,
            moe_pad_expert_input_to_capacity=False,
            moe_grouped_gemm=True,
            moe_use_legacy_grouped_gemm=True,
            moe_layer_freq=[0, 0] + [1] * 46,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=8,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=False,
            add_bias_linear=False,
            moe_shared_expert_overlap=False,
            normalization="RMSNorm",
            layernorm_epsilon=1.0e-6,
            qk_layernorm=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
        )
        values.update(overrides)
        return self.TransformerConfig(**values)

    def test_router_init_is_bf16_normal_point_zero_one_and_does_not_mutate_config(self):
        torch.manual_seed(1234)
        config = self.make_config()
        original_init = config.init_method
        router = self.mcore.RFullTopKRouter(config)
        self.assertIs(router.config, config)
        self.assertIs(config.init_method, original_init)
        self.assertEqual(router.weight.dtype, torch.bfloat16)
        self.assertEqual(tuple(router.weight.shape), (96, 2048))
        weights = router.weight.detach().float()
        self.assertLess(abs(float(weights.mean())), 1.0e-4)
        self.assertAlmostEqual(float(weights.std()), 0.01, delta=8.0e-5)

    def test_router_state_dict_is_stock_key_compatible(self):
        from megatron.core.transformer.moe.router import TopKRouter

        config = self.make_config()
        stock = TopKRouter(config)
        custom = self.mcore.RFullTopKRouter(config)
        self.assertEqual(list(stock.state_dict()), ["weight"])
        self.assertEqual(list(custom.state_dict()), ["weight"])
        custom.load_state_dict(stock.state_dict(), strict=True)
        stock.load_state_dict(custom.state_dict(), strict=True)

    def test_router_forward_is_selected_softmax_and_attaches_finite_losses(self):
        from megatron.core import parallel_state
        from megatron.core.transformer.moe.moe_utils import (
            MoEAuxLossAutoScaler,
            clear_aux_losses_tracker,
        )

        clear_aux_losses_tracker()
        torch.manual_seed(9)
        router = self.mcore.RFullTopKRouter(self.make_config())
        router.layer_number = 1
        MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1.0))
        logits = torch.randn(7, 96, dtype=torch.float32, requires_grad=True)
        with mock.patch.object(
            self.mcore,
            "save_to_aux_losses_tracker",
            wraps=self.mcore.save_to_aux_losses_tracker,
        ) as save_metric:
            probabilities, routing_map = router.routing(logits)
        self.assertEqual(tuple(probabilities.shape), (7, 96))
        self.assertEqual(tuple(routing_map.shape), (7, 96))
        self.assertTrue(torch.all(routing_map.sum(dim=-1) == 6))
        self.assertTrue(torch.all(probabilities[~routing_map] == 0))
        torch.testing.assert_close(
            probabilities.float().sum(dim=-1),
            torch.ones(7),
            rtol=2e-3,
            atol=2e-3,
        )
        probabilities.float().sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        tracker = parallel_state.get_moe_layer_wise_logging_tracker()
        self.assertEqual(set(tracker), {"load_balancing_loss", "z_loss"})
        self.assertEqual(save_metric.call_count, 2)
        for call in save_metric.call_args_list:
            self.assertIn("avg_group", call.kwargs)
            self.assertNotIn("reduce_group", call.kwargs)
        clear_aux_losses_tracker()

    def test_router_config_rejects_semantic_drift(self):
        bad_cases = [
            {"moe_router_topk": 5},
            {"moe_router_pre_softmax": True},
            {"moe_router_dtype": "fp64"},
            {"moe_expert_capacity_factor": 1.25},
            {"moe_aux_loss_coeff": 2.0e-3},
            {"moe_grouped_gemm": False},
            {"moe_use_legacy_grouped_gemm": False},
            {"tensor_model_parallel_size": 2},
            {"moe_shared_expert_overlap": True},
        ]
        for drift in bad_cases:
            with self.subTest(drift=drift), self.assertRaises((ValueError, AssertionError)):
                self.mcore.RFullTopKRouter(self.make_config(**drift))

    def test_rfull_mlp_forward_uses_both_limited_branches(self):
        class Dummy:
            def linear_fc1(self, hidden):
                del hidden
                return torch.tensor([[11.0, -20.0, 15.0, -12.0]]), None

            def linear_fc2(self, intermediate):
                return intermediate, None

        output, bias = self.mcore.RFullMLP.forward(Dummy(), torch.zeros(1, 1))
        expected = F.silu(torch.tensor([[10.0, -20.0]])) * torch.tensor(
            [[10.0, -10.0]]
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertIsNone(bias)

    def test_block_spec_has_exact_mixed_pattern_and_custom_modules(self):
        config = self.make_config()
        with mock.patch(
            "megatron.core.models.gpt.gpt_layer_specs.get_transformer_layer_offset",
            return_value=0,
        ), mock.patch(
            "megatron.core.models.gpt.gpt_layer_specs.get_num_layers_to_build",
            return_value=48,
        ):
            block = self.mcore.get_rfull_decoder_block_spec(
                config, use_transformer_engine=False
            )
        self.assertEqual(len(block.layer_specs), 48)
        dense = block.layer_specs[:2]
        moe = block.layer_specs[2:]
        self.assertTrue(
            all(layer.submodules.mlp.module is self.mcore.RFullMLP for layer in dense)
        )
        self.assertTrue(
            all(layer.submodules.mlp.module is self.mcore.RFullMoELayer for layer in moe)
        )
        for layer in moe:
            submodules = layer.submodules.mlp.submodules
            self.assertIs(submodules.router.module, self.mcore.RFullTopKRouter)
            self.assertIs(submodules.experts.module, self.mcore.RFullGroupedMLP)
            self.assertIs(
                submodules.shared_experts.module,
                self.mcore.RFullSharedExpertMLP,
            )

    def test_ep8_mini_profile_is_explicit_and_preserves_custom_semantics(self):
        config = self.make_config(
            num_layers=4,
            hidden_size=512,
            num_attention_heads=8,
            num_query_groups=2,
            kv_channels=64,
            ffn_hidden_size=1408,
            moe_ffn_hidden_size=256,
            moe_shared_expert_intermediate_size=256,
            moe_layer_freq=[0, 0, 1, 1],
        )
        with mock.patch(
            "megatron.core.models.gpt.gpt_layer_specs.get_transformer_layer_offset",
            return_value=0,
        ), mock.patch(
            "megatron.core.models.gpt.gpt_layer_specs.get_num_layers_to_build",
            return_value=4,
        ):
            block = self.mcore.get_rfull_decoder_block_spec(
                config,
                use_transformer_engine=False,
                profile=self.mcore.RFULL_EP8_MINI_PROFILE,
            )
        self.assertEqual(config.rfull_profile, "ep8-mini")
        self.assertEqual(len(block.layer_specs), 4)
        self.assertTrue(
            all(
                layer.submodules.mlp.module is self.mcore.RFullMLP
                for layer in block.layer_specs[:2]
            )
        )
        self.assertTrue(
            all(
                layer.submodules.mlp.module is self.mcore.RFullMoELayer
                for layer in block.layer_specs[2:]
            )
        )
        with self.assertRaisesRegex(ValueError, "conflicting R-Full profile"):
            self.mcore.get_rfull_decoder_block_spec(
                config,
                use_transformer_engine=False,
                profile=self.mcore.RFULL_PRODUCTION_PROFILE,
            )

    def test_te_grouped_experts_fail_closed_until_branch_clamps_exist(self):
        from megatron.core.transformer.moe.experts import TEGroupedMLP
        from megatron.core.transformer.spec_utils import ModuleSpec

        with self.assertRaisesRegex(ValueError, "TEGroupedMLP"):
            self.mcore._customize_expert_spec(ModuleSpec(module=TEGroupedMLP))


if __name__ == "__main__":
    unittest.main()

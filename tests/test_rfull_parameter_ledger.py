from __future__ import annotations

import copy
import unittest
from pathlib import Path

from tools.rfull_parameter_ledger import (
    LedgerValidationError,
    SOURCE_CONFIG_PATH,
    calculate_parameter_ledger,
    frozen_rfull_parameter_ledger,
    load_source_config,
    validate_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NORMATIVE_SOURCE = REPOSITORY_ROOT / "configs" / "rfull" / "rfull_v0_1.source.json"

EXPECTED_TOTAL_COMPONENTS = {
    "embedding": 311_164_928,
    "lm_head": 0,
    "attention_q_projection": 402_653_184,
    "attention_k_projection": 50_331_648,
    "attention_v_projection": 50_331_648,
    "attention_output_projection": 402_653_184,
    "dense_ffn": 67_633_152,
    "routed_experts": 24_310_185_984,
    "shared_experts": 253_231_104,
    "routers": 9_043_968,
    "block_norms": 196_608,
    "final_norm": 2_048,
    "qk_norms": 12_288,
}

EXPECTED_ACTIVE_COMPONENTS = {
    "embedding": 311_164_928,
    "lm_head": 0,
    "attention_q_projection": 402_653_184,
    "attention_k_projection": 50_331_648,
    "attention_v_projection": 50_331_648,
    "attention_output_projection": 402_653_184,
    "dense_ffn": 67_633_152,
    "routed_experts": 1_519_386_624,
    "shared_experts": 253_231_104,
    "routers": 9_043_968,
    "block_norms": 196_608,
    "final_norm": 2_048,
    "qk_norms": 12_288,
}

EXPECTED_TOTAL_SUBTOTALS = {
    "embedding_and_lm_head": 311_164_928,
    "attention": 905_969_664,
    "dense_ffn": 67_633_152,
    "routed_experts": 24_310_185_984,
    "shared_experts": 253_231_104,
    "routers": 9_043_968,
    "norms": 210_944,
}

EXPECTED_ACTIVE_SUBTOTALS = {
    "embedding_and_lm_head": 311_164_928,
    "attention": 905_969_664,
    "dense_ffn": 67_633_152,
    "routed_experts": 1_519_386_624,
    "shared_experts": 253_231_104,
    "routers": 9_043_968,
    "norms": 210_944,
}


class RFullParameterLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertEqual(SOURCE_CONFIG_PATH, NORMATIVE_SOURCE)
        self.config = load_source_config(NORMATIVE_SOURCE)
        self.ledger = calculate_parameter_ledger(self.config)

    def mutated_ledger(self, section: str, key: str, value: object) -> dict[str, object]:
        config = copy.deepcopy(self.config)
        config[section][key] = value
        return calculate_parameter_ledger(config)

    def assert_parameter_delta(
        self,
        mutated: dict[str, object],
        *,
        total: int,
        active: int,
    ) -> None:
        self.assertEqual(mutated["total"] - self.ledger["total"], total)
        self.assertEqual(mutated["active"] - self.ledger["active"], active)

    def test_frozen_source_has_every_exact_component_and_golden_total(self) -> None:
        self.assertEqual(self.ledger["total_components"], EXPECTED_TOTAL_COMPONENTS)
        self.assertEqual(self.ledger["active_components"], EXPECTED_ACTIVE_COMPONENTS)
        self.assertEqual(self.ledger["total_subtotals"], EXPECTED_TOTAL_SUBTOTALS)
        self.assertEqual(self.ledger["active_subtotals"], EXPECTED_ACTIVE_SUBTOTALS)
        self.assertEqual(sum(EXPECTED_TOTAL_COMPONENTS.values()), 25_857_439_744)
        self.assertEqual(sum(EXPECTED_ACTIVE_COMPONENTS.values()), 3_066_640_384)
        self.assertEqual(self.ledger["total"], 25_857_439_744)
        self.assertEqual(self.ledger["active"], 3_066_640_384)
        self.assertEqual(self.ledger["norm_subtotal"], 210_944)
        self.assertEqual(frozen_rfull_parameter_ledger(NORMATIVE_SOURCE), self.ledger)

    def test_attention_dimensions_are_derived_from_heads_groups_and_channels(self) -> None:
        dimensions = self.ledger["dimensions"]
        self.assertEqual(dimensions["query_projection_width"], 32 * 128)
        self.assertEqual(dimensions["key_projection_width"], 4 * 128)
        self.assertEqual(dimensions["value_projection_width"], 4 * 128)

        config = copy.deepcopy(self.config)
        config["model"]["num_attention_heads"] = 16
        config["model"]["num_query_groups"] = 2
        config["model"]["kv_channels"] = 64
        config["model"]["qk_norm_scale_size"] = 64
        mutated = calculate_parameter_ledger(config)

        self.assertEqual(mutated["dimensions"]["query_projection_width"], 1_024)
        self.assertEqual(mutated["dimensions"]["key_projection_width"], 128)
        self.assertEqual(mutated["dimensions"]["value_projection_width"], 128)

        expected_total_components = dict(EXPECTED_TOTAL_COMPONENTS)
        expected_total_components.update(
            {
                "attention_q_projection": 100_663_296,
                "attention_k_projection": 12_582_912,
                "attention_v_projection": 12_582_912,
                "attention_output_projection": 100_663_296,
                "qk_norms": 6_144,
            }
        )
        expected_active_components = dict(EXPECTED_ACTIVE_COMPONENTS)
        expected_active_components.update(
            {
                "attention_q_projection": 100_663_296,
                "attention_k_projection": 12_582_912,
                "attention_v_projection": 12_582_912,
                "attention_output_projection": 100_663_296,
                "qk_norms": 6_144,
            }
        )
        self.assertEqual(mutated["total_components"], expected_total_components)
        self.assertEqual(mutated["active_components"], expected_active_components)
        self.assertEqual(mutated["total_subtotals"]["attention"], 226_492_416)
        self.assertEqual(mutated["norm_subtotal"], 204_800)
        self.assert_parameter_delta(mutated, total=-679_483_392, active=-679_483_392)

    def test_disabling_qk_norm_removes_exactly_the_q_and_k_scales(self) -> None:
        mutated = self.mutated_ledger("model", "qk_rmsnorm", False)
        expected_total_components = dict(EXPECTED_TOTAL_COMPONENTS, qk_norms=0)
        expected_active_components = dict(EXPECTED_ACTIVE_COMPONENTS, qk_norms=0)
        self.assertEqual(mutated["total_components"], expected_total_components)
        self.assertEqual(mutated["active_components"], expected_active_components)
        self.assertEqual(mutated["norm_subtotal"], 198_656)
        self.assertEqual(mutated["total"], 25_857_427_456)
        self.assertEqual(mutated["active"], 3_066_628_096)
        self.assert_parameter_delta(mutated, total=-12_288, active=-12_288)

    def test_untied_lm_head_adds_one_padded_vocabulary_matrix(self) -> None:
        mutated = self.mutated_ledger("model", "tie_word_embeddings", False)
        expected_total_components = dict(EXPECTED_TOTAL_COMPONENTS, lm_head=311_164_928)
        expected_active_components = dict(EXPECTED_ACTIVE_COMPONENTS, lm_head=311_164_928)
        self.assertEqual(mutated["total_components"], expected_total_components)
        self.assertEqual(mutated["active_components"], expected_active_components)
        self.assertEqual(mutated["total"], 26_168_604_672)
        self.assertEqual(mutated["active"], 3_377_805_312)
        self.assert_parameter_delta(mutated, total=311_164_928, active=311_164_928)

    def test_native_vocabulary_rows_replace_padded_rows_exactly(self) -> None:
        native_vocab = self.config["model"]["tokenizer_native_vocab_size"]
        mutated = self.mutated_ledger("model", "padded_vocab_size", native_vocab)
        expected_total_components = dict(EXPECTED_TOTAL_COMPONENTS, embedding=310_618_112)
        expected_active_components = dict(EXPECTED_ACTIVE_COMPONENTS, embedding=310_618_112)
        self.assertEqual(mutated["total_components"], expected_total_components)
        self.assertEqual(mutated["active_components"], expected_active_components)
        self.assertEqual(mutated["dimensions"]["vocab_size"], 151_669)
        self.assertEqual(mutated["total"], 25_856_892_928)
        self.assertEqual(mutated["active"], 3_066_093_568)
        self.assert_parameter_delta(mutated, total=-546_816, active=-546_816)

    def test_top_k_plus_or_minus_one_changes_only_active_routed_experts(self) -> None:
        cases = (
            (5, -253_231_104, 1_266_155_520, 2_813_409_280),
            (7, 253_231_104, 1_772_617_728, 3_319_871_488),
        )
        for top_k, delta, active_routed, active_total in cases:
            with self.subTest(top_k=top_k):
                mutated = self.mutated_ledger("moe", "top_k", top_k)
                expected_active_components = dict(
                    EXPECTED_ACTIVE_COMPONENTS,
                    routed_experts=active_routed,
                )
                self.assertEqual(mutated["total_components"], EXPECTED_TOTAL_COMPONENTS)
                self.assertEqual(mutated["active_components"], expected_active_components)
                self.assertEqual(mutated["total"], 25_857_439_744)
                self.assertEqual(mutated["active"], active_total)
                self.assert_parameter_delta(mutated, total=0, active=delta)

    def test_two_shared_experts_are_both_total_and_active(self) -> None:
        mutated = self.mutated_ledger("moe", "shared_experts", 2)
        expected_total_components = dict(
            EXPECTED_TOTAL_COMPONENTS,
            shared_experts=506_462_208,
        )
        expected_active_components = dict(
            EXPECTED_ACTIVE_COMPONENTS,
            shared_experts=506_462_208,
        )
        self.assertEqual(mutated["total_components"], expected_total_components)
        self.assertEqual(mutated["active_components"], expected_active_components)
        self.assertEqual(mutated["total"], 26_110_670_848)
        self.assertEqual(mutated["active"], 3_319_871_488)
        self.assert_parameter_delta(mutated, total=253_231_104, active=253_231_104)

    def test_dense_and_moe_layer_ids_must_partition_every_layer(self) -> None:
        config = copy.deepcopy(self.config)
        config["model"]["moe_layer_ids"] = config["model"]["moe_layer_ids"][:-1]
        with self.assertRaisesRegex(
            LedgerValidationError,
            r"must partition all 48 layers exactly once: missing=\[47\]",
        ):
            validate_config(config)

        config = copy.deepcopy(self.config)
        config["model"]["dense_layer_ids"] = [0, 2]
        with self.assertRaisesRegex(LedgerValidationError, r"overlap=\[2\].*missing=\[1\]"):
            validate_config(config)

    def test_hidden_and_qk_scale_width_mismatches_are_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["moe"]["dispatch_width"] = 2_049
        with self.assertRaisesRegex(
            LedgerValidationError,
            r"dispatch_width must match model.hidden_size: 2049 != 2048",
        ):
            validate_config(config)

        config = copy.deepcopy(self.config)
        config["model"]["qk_norm_scale_size"] = 64
        with self.assertRaisesRegex(
            LedgerValidationError,
            r"qk_norm_scale_size must match model.kv_channels.*64 != 128",
        ):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()

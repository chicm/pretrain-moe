from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from tools.verify_rfull_gate2 import (
    FATAL_PATTERNS,
    _DCP_MCORE_METADATA_SOURCE_SHA256,
    _json_markers,
    _require_mcore_metadata_preservation,
    _verify_resume_contract,
    _verify_resume_cpu_rng_guard,
    verify,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
MINI_CONFIG = ROOT / "configs" / "gate2" / "rfull_ep8_mini.json"
COMMIT = "5cb6dbb3ed04e6fa11a862d90fe898ac2d3ddfad"


class VerifyRFullGate2Tests(unittest.TestCase):
    def test_resume_cpu_rng_guard_requires_exact_restore_on_every_rank(self) -> None:
        markers = {
            "RFULL_RESUME_CPU_RNG_GUARD": [
                {
                    "marker": "RFULL_RESUME_CPU_RNG_GUARD",
                    "rank": rank,
                    "is_resume": True,
                    "loaded_iteration": 2,
                    "restored": True,
                    "builder_changed_cpu_rng": True,
                    "changed_byte_count": 2,
                    "before_sha256": f"{rank + 1:064x}",
                    "after_build_sha256": f"{rank + 9:064x}",
                    "after_guard_sha256": f"{rank + 1:064x}",
                }
                for rank in range(8)
            ]
        }
        payloads = _verify_resume_cpu_rng_guard(markers, list(range(8)), 2)
        self.assertEqual([item["rank"] for item in payloads], list(range(8)))
        markers["RFULL_RESUME_CPU_RNG_GUARD"][3]["after_guard_sha256"] = "f" * 64
        with self.assertRaisesRegex(AssertionError, "not restored exactly"):
            _verify_resume_cpu_rng_guard(markers, list(range(8)), 2)

    def test_resume_contract_requires_argument_table_and_actual_load_marker(self) -> None:
        load_dir = "/checkpoints/source"
        save_dir = "/checkpoints/output"
        text = (
            f"  load ........................................... {load_dir}\n"
            f"  save ........................................... {save_dir}\n"
            f" successfully loaded checkpoint from {load_dir} "
            "[ t 2026-08-13 12:45:22 ] at iteration 2\n"
        )
        self.assertEqual(
            _verify_resume_contract(text, load_dir, save_dir, 2),
            {"load_dir": load_dir, "save_dir": save_dir, "loaded_iteration": 2},
        )

    def test_resume_contract_rejects_false_resume_with_none_arguments(self) -> None:
        text = (
            "  load ........................................... None\n"
            "  save ........................................... None\n"
        )
        with self.assertRaisesRegex(AssertionError, "trainer load argument mismatch"):
            _verify_resume_contract(text, "/checkpoints/source", "/checkpoints/output", 2)

    def test_resume_contract_rejects_missing_actual_load_marker(self) -> None:
        text = (
            "  load ........................................... /checkpoints/source\n"
            "  save ........................................... /checkpoints/output\n"
        )
        with self.assertRaisesRegex(AssertionError, "checkpoint load marker mismatch"):
            _verify_resume_contract(text, "/checkpoints/source", "/checkpoints/output", 2)

    def test_non_finite_pattern_only_matches_the_lm_loss_value(self) -> None:
        pattern = FATAL_PATTERNS["non_finite"]
        finite = (
            "iteration 1/4 | lm loss: 8.473267E+00 | "
            "number of skipped iterations: 0 | number of nan iterations: 0"
        )
        self.assertIsNone(pattern.search(finite))
        self.assertIsNotNone(pattern.search("iteration 1/4 | lm loss: nan |"))
        self.assertIsNotNone(pattern.search("iteration 1/4 | lm loss: -inf |"))
        self.assertIsNotNone(pattern.search("optimizer found NaN in gradient"))

    def test_multiple_checkpoint_metadata_preservation_events_are_valid(self) -> None:
        payloads = [
            {
                "marker": "DCP_MCORE_METADATA_PRESERVED",
                "rank": 0,
                "entries": 160,
            },
            {
                "marker": "DCP_MCORE_METADATA_PRESERVED",
                "rank": 0,
                "entries": 160,
            },
        ]
        markers = {"DCP_MCORE_METADATA_PRESERVED": payloads}
        self.assertEqual(_require_mcore_metadata_preservation(markers), payloads)
        with self.assertRaisesRegex(AssertionError, "preservation evidence"):
            _require_mcore_metadata_preservation(
                {"DCP_MCORE_METADATA_PRESERVED": [{"rank": 0, "entries": 0}]}
            )

    def test_concatenated_json_objects_recover_every_rank(self) -> None:
        text = "".join(
            json.dumps(
                {
                    "marker": "EARLY_DEVICE_BIND",
                    "rank": rank,
                    "source_sha256": {"nested.py": f"hash-{rank}"},
                }
            )
            for rank in range(8)
        )
        markers = _json_markers(text)
        self.assertEqual(
            sorted(item["rank"] for item in markers["EARLY_DEVICE_BIND"]),
            list(range(8)),
        )
        self.assertEqual(
            markers["EARLY_DEVICE_BIND"][7]["source_sha256"],
            {"nested.py": "hash-7"},
        )

    def test_synthetic_complete_run_passes_and_zero_expert_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = pathlib.Path(temporary)
            records = []
            for rank in range(8):
                for marker in (
                    "EARLY_DEVICE_BIND",
                    "PROCESS_GROUP_DEVICE_BIND",
                    "ROCM_LEGACY_FUSED_KERNEL_LOADER_SKIPPED",
                ):
                    records.append({"marker": marker, "rank": rank})
                records.extend(
                    [
                        {
                            "marker": "DCP_MCORE_METADATA_COMPAT_READY",
                            "rank": rank,
                            "save_mode": "preserve_mcore_data_after_dataclasses_replace",
                            "load_mode": "infer_same_geometry_if_absent",
                            "source_sha256": _DCP_MCORE_METADATA_SOURCE_SHA256,
                        },
                        {
                            "marker": "DCP_MCORE_METADATA_FALLBACK_APPLIED",
                            "rank": rank,
                            "entries": 3,
                            "shape_sha256": f"{rank + 1:064x}",
                        },
                        {
                            "marker": "RFULL_MODEL_BUILT",
                            "rank": rank,
                            "local_parameters": 19_371_008,
                            "trainable_parameters": 19_371_008,
                            "source_guard_file_count": 12,
                            "source_guard_megatron_commit": COMMIT,
                        },
                        {
                            "marker": "RFULL_GROUPED_GEMM_FORWARD",
                            "rank": rank,
                            "num_local_experts": 12,
                            "assigned_tokens": 48,
                            "zero_token_experts": 1 if rank == 0 else 0,
                            "hidden_size": 512,
                        },
                        {
                            "marker": "RFULL_EP_GLOBAL_AUX_LOSS",
                            "rank": rank,
                            "raw_aux_loss": 1.2,
                            "ep_world_size": 8,
                            "tracker_group": "expert_parallel_avg",
                        },
                        {
                            "marker": "RFULL_EP_GLOBAL_Z_LOSS",
                            "rank": rank,
                            "raw_z_loss": 3.4,
                            "ep_world_size": 8,
                            "tracker_group": "expert_parallel_avg",
                        },
                        {
                            "marker": "RFULL_TRAINING_COMPLETE",
                            "rank": rank,
                            "iteration": 4,
                            "consumed_train_samples": 32,
                            "consumed_train_tokens": 256,
                        },
                    ]
                )
                for iteration in range(1, 5):
                    records.append(
                        {
                            "marker": "RFULL_BATCH_FINGERPRINT",
                            "rank": rank,
                            "iteration": iteration,
                            "microbatch": 0,
                            "sha256": f"{rank * 4 + iteration:064x}",
                        }
                    )
                # Final validation and test forwards use the same final iteration but
                # continue the per-iteration microbatch counter above the training
                # range. They are valid evidence, not duplicate training batches.
                for microbatch in (1, 2):
                    records.append(
                        {
                            "marker": "RFULL_BATCH_FINGERPRINT",
                            "rank": rank,
                            "iteration": 4,
                            "microbatch": microbatch,
                            "sha256": f"{1000 + rank * 2 + microbatch:064x}",
                        }
                    )
            lines = ["".join(json.dumps(record) for record in records)]
            lines.extend(
                (
                    f"iteration {iteration} / 4 | lm loss: {9.0 - iteration:.6E} | "
                    "number of skipped iterations: 0 | number of nan iterations: 0"
                )
                for iteration in range(1, 5)
            )
            lines.extend(
                [
                    "  attention_backend ............................... AttnBackend.unfused",
                    "RFULL_NODE_COMPLETE=now host=test rc=0",
                ]
            )
            (run_dir / "train.console.log").write_text("\n".join(lines), encoding="utf-8")
            (run_dir / "gpu.telemetry.status.log").write_text(
                "GPU_TELEMETRY_START,now,test\nGPU_TELEMETRY_COMPLETE,now,test,rc=0\n",
                encoding="utf-8",
            )
            telemetry = ["device,GPU use (%),GPU Memory Allocated (VRAM%)"]
            telemetry.extend(f"card{rank},90,4" for rank in range(8))
            (run_dir / "gpu.telemetry.csv").write_text(
                "\n".join(telemetry) + "\n", encoding="utf-8"
            )
            summary = verify(
                run_dir,
                MINI_CONFIG,
                require_batch_fingerprints=True,
                require_dcp_mcore_metadata_compat=True,
                require_dcp_mcore_metadata_fallback=True,
            )
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(len(summary["batch_fingerprints"]), 48)
            self.assertEqual(len(summary["training_batch_fingerprints"]), 32)
            self.assertEqual(len(summary["additional_batch_fingerprints"]), 16)
            self.assertEqual(summary["zero_token_expert_ranks"], [0])

            phase_records = [
                record
                for record in records
                if record["marker"] != "RFULL_TRAINING_COMPLETE"
            ]
            phase_lines = ["".join(json.dumps(record) for record in phase_records)]
            phase_lines.extend(lines[-6:-4])
            phase_lines.extend(lines[-2:])
            (run_dir / "train.console.log").write_text(
                "\n".join(phase_lines), encoding="utf-8"
            )
            phase_summary = verify(
                run_dir,
                MINI_CONFIG,
                expected_first_iteration=1,
                expected_final_iteration=2,
                expect_training_complete=False,
            )
            self.assertEqual(phase_summary["iterations"], [1, 2])
            self.assertFalse(phase_summary["expect_training_complete"])

            (run_dir / "train.console.log").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            text = (run_dir / "train.console.log").read_text(encoding="utf-8")
            text = text.replace('"zero_token_experts": 1', '"zero_token_experts": 0')
            (run_dir / "train.console.log").write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "zero-token"):
                verify(run_dir, MINI_CONFIG)


if __name__ == "__main__":
    unittest.main()

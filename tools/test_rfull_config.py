#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compiler = load_module("compile_rfull_config", ROOT / "tools/compile_rfull_config.py")
orchestrator = load_module("run_rfull_stages", ROOT / "tools/run_rfull_stages.py")


class RFullConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = compiler.read_json(ROOT / "configs/rfull/rfull_v0_1.source.json")
        cls.schema = compiler.read_json(ROOT / "configs/rfull/rfull_v0_1.schema.json")

    def test_schema_and_invariants(self):
        Draft202012Validator(self.schema).validate(self.source)
        derived = compiler.validate_invariants(self.source)
        self.assertEqual(derived["total_sequences"], 216370080)
        self.assertEqual(derived["source_target_tokens"][1], 179999895552)

    def test_unknown_key_rejected(self):
        changed = copy.deepcopy(self.source)
        changed["model"]["unknown_option"] = True
        self.assertTrue(list(Draft202012Validator(self.schema).iter_errors(changed)))

    def test_default_padded_vocab_rejected(self):
        changed = copy.deepcopy(self.source)
        changed["model"]["padded_vocab_size"] = 151680
        with self.assertRaisesRegex(ValueError, "model vocab"):
            compiler.validate_invariants(changed)

    def test_threshold_seven_rejected(self):
        changed = copy.deepcopy(self.source)
        changed["model"]["limited_swiglu_threshold"] = 7
        with self.assertRaisesRegex(ValueError, "threshold"):
            compiler.validate_invariants(changed)

    def test_committed_skip_rejected(self):
        changed = copy.deepcopy(self.source)
        changed["run"]["max_committed_skips"] = 1
        with self.assertRaisesRegex(ValueError, "skips"):
            compiler.validate_invariants(changed)

    def test_fineweb_aggregate_mutation_rejected(self):
        changed = copy.deepcopy(self.source)
        changed["stages"][3]["source_sequences"][1] -= 1
        with self.assertRaisesRegex(ValueError, "source quota"):
            compiler.validate_invariants(changed)

    def test_communication_mutation_rejected(self):
        changed = copy.deepcopy(self.source)
        changed["communication"]["a2a_forward_bytes_per_token"] = 946864128
        with self.assertRaisesRegex(ValueError, "communication"):
            compiler.validate_invariants(changed)

    def test_storage_mutation_rejected(self):
        changed = copy.deepcopy(self.source)
        changed["storage"]["required_co_resident_usable_bytes"] = 5000000000000
        with self.assertRaisesRegex(ValueError, "storage"):
            compiler.validate_invariants(changed)

    def test_holdout_disjointness_mutation_rejected(self):
        changed = copy.deepcopy(self.source)
        changed["evaluation"]["holdout_candidate_selection"] = "token_disjoint_only"
        with self.assertRaisesRegex(ValueError, "holdout_candidate_selection mismatch"):
            compiler.validate_invariants(changed)

    def test_checkpoint_resume_policy_mutation_rejected(self):
        changed = copy.deepcopy(self.source)
        changed["checkpoint"]["stage_resume_policy"] = "advance_to_next_stage"
        with self.assertRaisesRegex(ValueError, "checkpoint contract mismatch"):
            compiler.validate_invariants(changed)

    def test_generated_hash_chain(self):
        manifest, stages = orchestrator.verify(ROOT / "configs/rfull/generated/manifest.json")
        self.assertFalse(manifest["launch_allowed"])
        self.assertEqual(len(manifest["unresolved_blockers"]), 29)
        self.assertEqual(len(stages), 4)
        self.assertEqual(stages[-1][1]["stage"]["end_update_tokens"], 999999406080)

    def test_self_consistent_stage_tamper_rejected(self):
        generated = ROOT / "configs/rfull/generated"
        with tempfile.TemporaryDirectory(prefix="rfull_tamper_") as temp:
            copied = Path(temp)
            manifest = compiler.read_json(generated / "manifest.json")
            for record in manifest["stage_artifacts"]:
                compiler.write_json(copied / record["path"], compiler.read_json(generated / record["path"]))
            first = manifest["stage_artifacts"][0]
            stage_path = copied / first["path"]
            stage = compiler.read_json(stage_path)
            stage["model"]["hidden_size"] = 7
            stage["artifact_sha256"] = orchestrator.object_hash(stage, True)
            first["artifact_sha256"] = stage["artifact_sha256"]
            manifest["artifact_sha256"] = orchestrator.object_hash(manifest, True)
            compiler.write_json(stage_path, stage)
            compiler.write_json(copied / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "pinned compiler reproduction"):
                orchestrator.verify(copied / "manifest.json")

    def test_checkpoint_head_and_resume_contract(self):
        manifest, stages = orchestrator.verify(ROOT / "configs/rfull/generated/manifest.json")
        self.assertEqual(orchestrator.stage_for_update(stages, 203451)[1]["stage"]["id"], "4k")
        self.assertEqual(orchestrator.stage_for_update(stages, 203452)[1]["stage"]["id"], "8k")
        with tempfile.TemporaryDirectory(prefix="rfull_checkpoint_") as temp:
            checkpoint_root = Path(temp)
            checkpoint = checkpoint_root / "step_100000"
            checkpoint.mkdir()
            root_manifest = checkpoint / "root_manifest.json"
            root_manifest.write_bytes(b"{}\n")
            stage = stages[0][1]
            marker = {
                "schema_version": "rfull-commit-v1",
                "run_name": stage["run"]["name"],
                "config_manifest_sha256": manifest["artifact_sha256"],
                "stage_id": "4k",
                "stage_artifact_sha256": stage["artifact_sha256"],
                "successful_updates": 100000,
                "update_tokens": 100000 * stage["run"]["target_tokens_per_update"],
                "root_manifest_path": "root_manifest.json",
                "root_manifest_sha256": orchestrator.file_hash(root_manifest),
                "parent_commit_sha256": "GENESIS",
                "lineage_id": "lineage-test",
            }
            marker_path = checkpoint / "COMMITTED"
            compiler.write_json(marker_path, marker)
            pointer = {
                "schema_version": "rfull-latest-v1",
                "commit_marker_path": "step_100000/COMMITTED",
                "commit_marker_sha256": orchestrator.file_hash(marker_path),
            }
            pointer_path = checkpoint_root / "LATEST_COMMITTED"
            compiler.write_json(pointer_path, pointer)
            latest = orchestrator.latest_committed_checkpoint(checkpoint_root, manifest, stages)
            self.assertIsNotNone(latest)
            self.assertEqual(latest[1]["successful_updates"], 100000)

            marker["update_tokens"] += 1
            compiler.write_json(marker_path, marker)
            pointer["commit_marker_sha256"] = orchestrator.file_hash(marker_path)
            compiler.write_json(pointer_path, pointer)
            with self.assertRaisesRegex(ValueError, "token counter mismatch"):
                orchestrator.latest_committed_checkpoint(checkpoint_root, manifest, stages)

            pointer_path.unlink()
            with self.assertRaisesRegex(ValueError, "LATEST_COMMITTED is missing"):
                orchestrator.latest_committed_checkpoint(checkpoint_root, manifest, stages)

    def test_document_binds_generated_artifacts_and_blockers(self):
        document_path = ROOT / "docs/r_full_moe_production_training_design.md"
        raw = document_path.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r\n", raw)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(any(line.endswith((b" ", b"\t")) for line in raw.splitlines()))
        text = raw.decode("utf-8")
        manifest, stages = orchestrator.verify(ROOT / "configs/rfull/generated/manifest.json")
        for digest in [
            manifest["source_config_sha256"], manifest["source_schema_sha256"],
            manifest["compiler_sha256"], manifest["artifact_sha256"],
            *[stage["artifact_sha256"] for _, stage in stages],
        ]:
            self.assertEqual(text.count(digest), 1, digest)
        for blocker in manifest["unresolved_blockers"]:
            self.assertIn(f"{blocker}: TBD-BLOCKER", text)
        self.assertIn("| 32K | 107,984 | 54,907 | 15,252 | 61,008 | 45,756 | 9,517 | 6,589 | 4,027 | 305,040 |", text)
        self.assertIn("| **all** | **76,595,008** | **38,946,615** | **10,818,504** | **43,274,016** | **32,455,512** | **6,750,746** | **4,673,594** | **2,856,085** | **216,370,080** |", text)
        for value in ["353,999,781,888", "179,999,895,552", "49,999,970,304", "199,999,881,216", "149,999,910,912", "31,199,969,280", "21,599,993,856", "13,200,003,072"]:
            self.assertIn(value, text)
        for value in ["8.542141", "17.084282", "2.264063", "19.91 GB/序列", "48.845 GiB/rank/update"]:
            self.assertIn(value, text)
        self.assertIn("L_{\\mathrm{aux}}=N\\sum_{i=1}^{N}f_iP_i", text)
        self.assertIn("token-disjoint 且 document-disjoint", text)
        self.assertIn("rolled_back_uncheckpointed_tokens", text)
        self.assertIn("`LATEST_COMMITTED`（`rfull-latest-v1`）", text)
        self.assertIn("`rfull-commit-v1` canonical-JSON envelope", text)
        self.assertNotIn("L_{\\mathrm{aux}}=\\alpha N", text)
        self.assertNotIn("最后健康 step、丢失 tokens", text)
        for stale in ["38,946,614", "6,750,744", "4,673,595", "2,856,086", "348,445,999,104", "| 9,515 | 6,590 | 4,028 |", "8.541322", "17.082644", "2.263131"]:
            self.assertNotIn(stale, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

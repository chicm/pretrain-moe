"""Tests for the R-Full production data blend generator."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "build_rfull_data_blend.py"

CORPORA = {
    "dclm_tok": (4, 400_000),
    "fineweb_edu_240bt_tok": (3, 200_000),
    "finephrase_tok": (2, 150_000),
    "starcoder_tok": (2, 100_000),
    "fineweb_edu_100bt_tok": (1, 90_000),
    "finepdfs_edu_tok": (1, 60_000),
    "math_tok": (2, 30_000),
    "infimath_tok": (1, 20_000),
    "owm_tok": (1, 12_000),
    "fineweb_tok": (1, 10_000),
}


def _manifest(path: pathlib.Path) -> int:
    entries, total = [], 0
    for corpus, (shards, per) in CORPORA.items():
        for i in range(shards):
            entries.append({
                "bin": f"/mnt/data/{corpus}/shard_{i:04d}.bin",
                "idx": f"/mnt/data/{corpus}/shard_{i:04d}.idx",
                "tokens": per, "documents": per // 100, "max_token": 151668,
            })
            total += per
    path.write_text(json.dumps({"schema_version": 1, "entries": entries, "total_tokens": total}))
    return total


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True)


class DataBlendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.manifest = self.dir / "corpus_manifest.json"
        self.total = _manifest(self.manifest)
        self.out = self.dir / "blend.json"
        self.addCleanup(self.tmp.cleanup)

    def _blend(self, *extra: str, budget: int | None = None) -> subprocess.CompletedProcess:
        return _run("--manifest", str(self.manifest), "--output", str(self.out),
                    "--budget-tokens", str(budget if budget is not None else int(self.total * 0.95)),
                    *extra)

    def test_natural_mixture_covers_every_shard_and_sums_to_one(self) -> None:
        proc = self._blend()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(self.out.read_text())
        self.assertEqual(len(data["blend"]), sum(s for s, _ in CORPORA.values()))
        self.assertAlmostEqual(sum(b["weight"] for b in data["blend"]), 1.0, places=6)
        self.assertIn("BLEND_RESULT=PASS", proc.stdout)
        self.assertIn("BLEND_MIXTURE=natural", proc.stdout)

    def test_natural_mixture_never_repeats_below_full_corpus(self) -> None:
        proc = self._blend()
        self.assertNotIn("REPEATS", proc.stdout)
        for line in proc.stdout.splitlines():
            if line.startswith("BLEND_MAX_CORPUS_EPOCHS="):
                self.assertLessEqual(float(line.split("=")[1]), 1.0)
                break
        else:
            self.fail("no BLEND_MAX_CORPUS_EPOCHS line")

    def test_natural_weights_match_token_share(self) -> None:
        self._blend()
        data = json.loads(self.out.read_text())
        got = 0.0
        for entry in data["blend"]:
            if "/dclm_tok/" in entry["prefix"]:
                got += entry["weight"]
        shards, per = CORPORA["dclm_tok"]
        self.assertAlmostEqual(got, shards * per / self.total, places=9)

    def test_prefix_has_no_bin_suffix(self) -> None:
        self._blend()
        data = json.loads(self.out.read_text())
        for entry in data["blend"]:
            self.assertFalse(entry["prefix"].endswith(".bin"))
            self.assertTrue(entry["prefix"].startswith("/"))

    def test_budget_above_corpus_fails_closed(self) -> None:
        proc = self._blend(budget=self.total * 2)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("exceeds corpus", proc.stderr)

    def test_budget_above_corpus_allowed_explicitly(self) -> None:
        proc = self._blend("--allow-repeats", budget=self.total * 2)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PASS_WITH_REPEATS", proc.stdout)

    def test_quality_weighted_repeats_require_explicit_opt_in(self) -> None:
        proc = self._blend("--mixture", "quality-weighted")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("repeats data", proc.stderr)

    def test_quality_weighted_with_opt_in_succeeds(self) -> None:
        proc = self._blend("--mixture", "quality-weighted", "--allow-repeats")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("BLEND_MIXTURE=quality-weighted", proc.stdout)

    def test_unknown_corpus_fails_quality_weighted(self) -> None:
        payload = json.loads(self.manifest.read_text())
        payload["entries"].append({"bin": "/mnt/data/surprise_tok/shard_0000.bin",
                                   "idx": "/mnt/data/surprise_tok/shard_0000.idx",
                                   "tokens": 5, "documents": 1, "max_token": 10})
        self.manifest.write_text(json.dumps(payload))
        proc = self._blend("--mixture", "quality-weighted", "--allow-repeats")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("absent from mixture", proc.stderr)

    def test_split_is_passed_through(self) -> None:
        self._blend("--split", "990,9,1")
        self.assertEqual(json.loads(self.out.read_text())["split"], "990,9,1")

    def test_output_is_deterministic(self) -> None:
        self._blend()
        first = self.out.read_bytes()
        self._blend()
        self.assertEqual(first, self.out.read_bytes())


if __name__ == "__main__":
    unittest.main()

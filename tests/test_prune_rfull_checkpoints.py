"""Tests for the checkpoint retention tool.

Deleting a checkpoint is irreversible and each one costs hours of compute, so
the selection logic is tested directly rather than through the filesystem
walk.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prune_rfull_checkpoints import (  # noqa: E402
    Checkpoint,
    select_for_deletion,
)


def ck(iteration: int, *, complete: bool = True, nbytes: int = 1) -> Checkpoint:
    return Checkpoint(
        iteration=iteration,
        path=Path(f"/tmp/iter_{iteration:07d}"),
        complete=complete,
        nbytes=nbytes,
    )


class SelectForDeletionTest(unittest.TestCase):
    def test_keeps_the_newest_n(self) -> None:
        checkpoints = [ck(i) for i in (1000, 2000, 3000, 4000, 5000)]
        kept, deletable = select_for_deletion(checkpoints, keep=3, protected=None)
        self.assertEqual([c.iteration for c in kept], [3000, 4000, 5000])
        self.assertEqual([c.iteration for c in deletable], [1000, 2000])

    def test_never_deletes_the_tracked_iteration(self) -> None:
        # The tracker points at an old checkpoint -- a resume needs it, so it
        # must survive even though it falls outside the newest N.
        checkpoints = [ck(i) for i in (1000, 2000, 3000, 4000, 5000)]
        kept, deletable = select_for_deletion(checkpoints, keep=2, protected=1000)
        self.assertIn(1000, [c.iteration for c in kept])
        self.assertNotIn(1000, [c.iteration for c in deletable])
        self.assertEqual([c.iteration for c in deletable], [2000, 3000])

    def test_incomplete_checkpoint_does_not_consume_the_quota(self) -> None:
        # A partial save must not push a good checkpoint out of the keep set,
        # otherwise a crash mid-save would silently cost us a restore point.
        checkpoints = [ck(1000), ck(2000), ck(3000, complete=False)]
        kept, deletable = select_for_deletion(checkpoints, keep=2, protected=None)
        kept_iters = [c.iteration for c in kept]
        self.assertIn(1000, kept_iters)
        self.assertIn(2000, kept_iters)
        self.assertEqual(deletable, [])

    def test_in_flight_save_is_left_alone(self) -> None:
        # An incomplete directory newer than everything kept is probably being
        # written right now; deleting it would corrupt a live save.
        checkpoints = [ck(1000), ck(2000), ck(3000), ck(4000, complete=False)]
        kept, _ = select_for_deletion(checkpoints, keep=2, protected=None)
        self.assertIn(4000, [c.iteration for c in kept])

    def test_stale_partial_save_is_removed(self) -> None:
        # An incomplete directory older than the newest kept checkpoint is
        # abandoned garbage and should be reclaimed.
        checkpoints = [ck(1000, complete=False), ck(2000), ck(3000)]
        _, deletable = select_for_deletion(checkpoints, keep=2, protected=None)
        self.assertEqual([c.iteration for c in deletable], [1000])

    def test_keep_larger_than_available_deletes_nothing(self) -> None:
        checkpoints = [ck(1000), ck(2000)]
        kept, deletable = select_for_deletion(checkpoints, keep=5, protected=None)
        self.assertEqual(len(kept), 2)
        self.assertEqual(deletable, [])

    def test_keep_zero_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            select_for_deletion([ck(1000)], keep=0, protected=None)

    def test_production_retention_budget(self) -> None:
        # The reason this tool exists: keeping every checkpoint of the real run
        # would need far more storage than exists.
        checkpoint_bytes = 362_020_918_299
        saves = 256_856 // 1000
        free_bytes = 7.3 * 1000**4
        self.assertGreater(saves * checkpoint_bytes, free_bytes)
        self.assertLess(3 * checkpoint_bytes, free_bytes)


if __name__ == "__main__":
    unittest.main()

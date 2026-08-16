#!/usr/bin/env python3
"""Prune old distributed checkpoints, keeping only the most recent N.

Upstream Megatron-LM has no checkpoint retention: `--save-interval` writes an
`iter_XXXXXXX` directory every interval and never deletes one.  At the
production geometry each checkpoint is ~337 GiB (14 bytes/param for bf16
weights + fp32 master + two fp32 Adam moments over 25,857,439,744 params), so
a 256856-iteration run saving every 1000 steps would need ~86 TB against 7.3 TB
of free shared storage.  The run would die of a full disk, not of a bug.

This tool is deliberately conservative, because deleting the wrong directory
destroys hours of compute:

  * It only ever removes directories matching ``iter_<digits>`` exactly.
  * It never removes the iteration recorded in
    ``latest_checkpointed_iteration.txt`` -- that is the one a resume needs.
  * It refuses to run with ``keep < 1``.
  * ``--dry-run`` prints the plan without touching anything, and is the
    default mode for the caller to inspect first.

Only complete checkpoints are eligible for retention.  A directory that is
missing ``.metadata`` is treated as a partial/aborted save: it is never
counted towards the keep quota, and it is only deleted when it is older than
the newest kept checkpoint (so an in-flight save is never removed).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, NamedTuple

ITER_DIR = re.compile(r"^iter_(\d+)$")
TRACKER = "latest_checkpointed_iteration.txt"


class Checkpoint(NamedTuple):
    iteration: int
    path: Path
    complete: bool
    nbytes: int


def _size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def discover(root: Path, *, measure: bool = True) -> list[Checkpoint]:
    """Return every ``iter_<n>`` checkpoint under *root*, oldest first."""
    found: list[Checkpoint] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = ITER_DIR.match(child.name)
        if match is None:
            continue
        complete = (child / ".metadata").is_file()
        found.append(
            Checkpoint(
                iteration=int(match.group(1)),
                path=child,
                complete=complete,
                nbytes=_size(child) if measure else 0,
            )
        )
    found.sort(key=lambda c: c.iteration)
    return found


def read_tracker(root: Path) -> int | None:
    tracker = root / TRACKER
    try:
        text = tracker.read_text().strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def select_for_deletion(
    checkpoints: Iterable[Checkpoint], keep: int, protected: int | None
) -> tuple[list[Checkpoint], list[Checkpoint]]:
    """Split *checkpoints* into (kept, deletable).

    Completeness matters: a checkpoint without ``.metadata`` never satisfies
    the keep quota, because restoring from it would fail.
    """
    if keep < 1:
        raise ValueError("keep must be >= 1")

    ordered = sorted(checkpoints, key=lambda c: c.iteration, reverse=True)
    complete = [c for c in ordered if c.complete]

    kept_iters = {c.iteration for c in complete[:keep]}
    if protected is not None:
        kept_iters.add(protected)

    newest_kept = max(kept_iters) if kept_iters else None

    kept: list[Checkpoint] = []
    deletable: list[Checkpoint] = []
    for candidate in ordered:
        if candidate.iteration in kept_iters:
            kept.append(candidate)
        elif not candidate.complete and (
            newest_kept is None or candidate.iteration >= newest_kept
        ):
            # A partial save at or beyond the newest kept checkpoint may still
            # be in flight; leave it alone.
            kept.append(candidate)
        else:
            deletable.append(candidate)

    kept.sort(key=lambda c: c.iteration)
    deletable.sort(key=lambda c: c.iteration)
    return kept, deletable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    if args.keep < 1:
        print("REFUSE: --keep must be >= 1", file=sys.stderr)
        return 2
    if args.apply == args.dry_run:
        print("REFUSE: pass exactly one of --dry-run or --apply", file=sys.stderr)
        return 2

    root = Path(args.checkpoint_dir)
    if not root.is_dir():
        print(f"REFUSE: not a directory: {root}", file=sys.stderr)
        return 2

    checkpoints = discover(root)
    protected = read_tracker(root)
    kept, deletable = select_for_deletion(checkpoints, args.keep, protected)

    print(f"checkpoint_dir={root}")
    print(f"keep={args.keep} tracker={protected}")
    for entry in kept:
        flag = "" if entry.complete else "  (INCOMPLETE, left alone)"
        star = " *tracker" if entry.iteration == protected else ""
        print(f"  KEEP   iter_{entry.iteration:07d}  {entry.nbytes / 2**30:8.1f} GiB{star}{flag}")
    for entry in deletable:
        print(f"  DELETE iter_{entry.iteration:07d}  {entry.nbytes / 2**30:8.1f} GiB")

    reclaim = sum(entry.nbytes for entry in deletable)
    print(f"reclaimable_bytes={reclaim} ({reclaim / 2**30:.1f} GiB)")

    if args.dry_run:
        print("DRY_RUN (nothing removed)")
        return 0

    removed = 0
    for entry in deletable:
        shutil.rmtree(entry.path, ignore_errors=False)
        removed += 1
        print(f"removed iter_{entry.iteration:07d}")
    print(f"PRUNE_COMPLETE removed={removed} reclaimed_bytes={reclaim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

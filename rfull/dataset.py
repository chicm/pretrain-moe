"""Real-corpus shard reader + Megatron batch adapter for R-Full.

Turns the scheduler's abstract ``WindowRef`` into actual token tensors read
from the tokenized uint32 shards on shared storage.

Storage facts this file depends on (verified during the corpus scan):
  * payload is little-endian uint32, one token id per 4 bytes;
  * each source directory carries an index json describing its shards;
  * the legacy ``"vocab_size": 151643`` field in those indexes is an EOT-id
    mislabel, NOT the padded model vocabulary (151936). We therefore validate
    against the payload bound and never against that field.

Reads are sequential and bounded. Random-offset probing across blobfuse is
what made an earlier inventory scan pathological, so each window is fetched as
one contiguous ``pread`` of (seq_len+1) tokens.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data_scheduler import (
    SOURCE_IDS, RFullScheduler, ShardRef, SourcePlan, WindowRef,
)

TOKEN_DTYPE = np.uint32
TOKEN_BYTES = 4
EOT_ID = 151643
PADDED_VOCAB = 151936
MAX_PAYLOAD_ID_EXCLUSIVE = 151669

# Physical directory names of the eight selected sources, in frozen order.
SOURCE_DIRS = {
    "dclm": "dclm_tok",
    "fineweb_edu": "fineweb_edu_240bt_tok",
    "finepdfs": "finepdfs_edu_tok",
    "finephrase": "finephrase_tok",
    "code": "starcoder_tok",
    "finemath": "math_tok",
    "infimath": "infimath_tok",
    "owm": "owm_tok",
}


@dataclass(frozen=True)
class ShardFile:
    path: str
    n_tokens: int


def _index_candidates(d: Path):
    """Deprecated: kept only so callers get a clear error, not a silent path."""
    raise NotImplementedError(
        "top-level source indexes are stale; use discover_source()")


def discover_source(root: Path, source_id: str) -> list:
    """List every physically present shard of one source, across all parts.

    Layout (verified on the shared corpus):

        <source>_tok/
            index.json           <- legacy top-level index, MOSTLY STALE
            shard_*.bin          <- partial merged copies from the old dense run
            part_0/ .. part_5/
                index.json / index_v2.json
                shard_*.bin      <- THE REAL CORPUS

    Two traps this function is written to avoid:

    1. The TOP-LEVEL ``index.json`` declares shards that no longer exist (e.g.
       dclm declares 5, only 2 remain). Trusting it silently yields a
       FileNotFoundError mid-training -- or worse, a corpus 58x smaller than
       intended. The real data lives under ``part_*``.
    2. Declared ``tokens`` fields are nominal. We always use the real file size
       so a truncated shard cannot desynchronise window arithmetic.

    Shards that are declared but absent are reported, never silently skipped.
    """
    d = root / SOURCE_DIRS[source_id]
    if not d.is_dir():
        raise FileNotFoundError(f"source dir missing: {d}")

    parts = sorted(p for p in d.iterdir()
                   if p.is_dir() and p.name.startswith("part_"))
    if not parts:
        raise FileNotFoundError(
            f"{d} has no part_* directories; refusing to fall back to the "
            f"stale top-level index")

    out, missing = [], []
    for pd in parts:
        idx = pd / "index_v2.json"
        if not idx.is_file():
            idx = pd / "index.json"
        if not idx.is_file():
            raise FileNotFoundError(f"no index in {pd}")
        meta = json.loads(idx.read_text())
        if meta.get("dtype") != "uint32":
            raise ValueError(f"{idx}: unexpected dtype {meta.get('dtype')!r}")
        if int(meta.get("eot", -1)) != EOT_ID:
            raise ValueError(f"{idx}: unexpected eot {meta.get('eot')!r}")
        for s in meta.get("shards", []):
            p = pd / s["path"]
            if not p.is_file():
                missing.append(str(p))
                continue
            out.append(ShardFile(str(p), p.stat().st_size // TOKEN_BYTES))

    if missing:
        raise FileNotFoundError(
            f"source {source_id}: {len(missing)} declared shards absent, "
            f"first={missing[0]}")
    if not out:
        raise FileNotFoundError(f"source {source_id}: no shards present")
    return sorted(out, key=lambda x: x.path)


def build_plans(root, seq_len: int, holdout_tail_tokens: int = 0) -> dict:
    """Build the per-source window plans for one stage.

    ``holdout_tail_tokens`` reserves a contiguous tail of every shard for
    evaluation, so training windows can never reach it. Reserving a tail (not a
    random subset) keeps the exclusion checkable by a single inequality.
    """
    root = Path(root)
    plans = {}
    for idx, sid in enumerate(SOURCE_IDS):
        refs = []
        for shard_id, sf in enumerate(discover_source(root, sid)):
            usable = sf.n_tokens - holdout_tail_tokens
            # need seq_len+1 tokens per window (inputs + shifted labels)
            n_win = (usable - 1) // seq_len
            if n_win <= 0:
                continue
            refs.append(ShardRef(shard_id=shard_id, path=sf.path, n_win=n_win,
                                 win_stride=seq_len, seq_len=seq_len,
                                 token_offset=0))
        if not refs:
            raise ValueError(f"source {sid} has no usable shards at seq_len={seq_len}")
        plans[idx] = SourcePlan.build(sid, idx, refs)
    return plans


class ShardReader:
    """Bounded sequential reader with a small fd cache."""

    def __init__(self, max_open: int = 64, validate: bool = True):
        self._fds = {}
        self.max_open = max_open
        self.validate = validate
        self.n_reads = 0

    def _fd(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            if len(self._fds) >= self.max_open:
                old, ofd = next(iter(self._fds.items()))
                os.close(ofd)
                del self._fds[old]
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def read_window(self, w: WindowRef) -> np.ndarray:
        need = w.byte_len
        buf = os.pread(self._fd(w.shard_path), need, w.byte_start)
        if len(buf) != need:
            raise IOError(
                f"short read {len(buf)}/{need} at {w.shard_path}:{w.byte_start} "
                f"(window {w.window_index} of shard {w.shard_id})")
        arr = np.frombuffer(buf, dtype=TOKEN_DTYPE)
        self.n_reads += 1
        if self.validate:
            mx = int(arr.max())
            if mx >= MAX_PAYLOAD_ID_EXCLUSIVE:
                raise ValueError(
                    f"token id {mx} >= payload bound "
                    f"{MAX_PAYLOAD_ID_EXCLUSIVE} in {w.shard_path}")
        return arr

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


class RFullBatchSource:
    """Produces one rank's micro-batches for a given committed update."""

    def __init__(self, scheduler: RFullScheduler, reader: ShardReader,
                 lane: int, n_lanes: int, micro_batch_size: int,
                 grad_accum: int):
        self.sch = scheduler
        self.reader = reader
        self.lane = lane
        self.n_lanes = n_lanes
        self.mbs = micro_batch_size
        self.ga = grad_accum
        per_lane = scheduler.G // n_lanes
        if per_lane != micro_batch_size * grad_accum:
            raise ValueError(
                f"lane share {per_lane} != mbs {micro_batch_size} x ga {grad_accum}")

    def update_batches(self, successful_updates: int):
        """Yield ``grad_accum`` micro-batches of (tokens, labels) for this rank."""
        wins = self.sch.lane_windows(successful_updates, self.lane, self.n_lanes)
        for step in range(self.ga):
            chunk = wins[step * self.mbs:(step + 1) * self.mbs]
            raw = np.stack([self.reader.read_window(w) for w in chunk])
            t = torch.from_numpy(raw.astype(np.int64))
            yield t[:, :-1].contiguous(), t[:, 1:].contiguous(), chunk

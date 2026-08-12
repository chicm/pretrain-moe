"""R-Full deterministic data scheduler (``rfull-hash-affine-v1``).

Design reference: ``docs/r_full_moe_production_training_design.md`` section 9.

Everything here is a pure function of
``(seed, stage_id, successful_updates, G, manifest)``. There is no mutable
cursor and no history replay: locating global sequence ordinal ``j`` costs one
10,000-entry cycle rebuild (cached) plus an ``O(log n_shards)`` binary search,
independent of how many sequences were already consumed.

This is the property that kills the old multi-hour fast-forward on resume.

Determinism rules (hard requirements):
  * SHA-256 only. Never Python ``hash()`` (PYTHONHASHSEED-dependent) and never
    an unpinned library RNG.
  * No modulo-biased shuffles. Ordering is by ``(digest, original_position)``.
  * All integers are serialised little-endian with a fixed width, and every
    hash input starts with a domain-separation string so that two different
    uses can never collide.
"""

from __future__ import annotations

import hashlib
import struct
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

# Frozen source order and quotas (section 9.2). The order is load-bearing:
# it defines source_index, which is part of every hash domain.
SOURCE_IDS = (
    "dclm", "fineweb_edu", "finepdfs", "finephrase",
    "code", "finemath", "infimath", "owm",
)
SOURCE_QUOTAS = (3540, 1800, 500, 2000, 1500, 312, 216, 132)
SOURCE_CYCLE_SEQUENCES = 10000

assert len(SOURCE_IDS) == len(SOURCE_QUOTAS)
assert sum(SOURCE_QUOTAS) == SOURCE_CYCLE_SEQUENCES

DOMAIN_SOURCE_CYCLE = b"rfull-source-cycle-v1\0"
DOMAIN_SHARD_ORDER = b"rfull-shard-order-v1\0"
DOMAIN_PERM_A = b"rfull-perm-a-v1\0"
DOMAIN_PERM_B = b"rfull-perm-b-v1\0"

STAGE_IDS = ("4k", "8k", "16k", "32k")


def _u64(x: int) -> bytes:
    return struct.pack("<Q", x)


def _u32(x: int) -> bytes:
    return struct.pack("<I", x)


def _digest(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def stage_index(stage_id: str) -> int:
    try:
        return STAGE_IDS.index(stage_id)
    except ValueError:
        raise ValueError(
            f"unknown stage_id {stage_id!r}; expected one of {STAGE_IDS}")


# --------------------------------------------------------------------------
# 9.2 source cycle
# --------------------------------------------------------------------------

@lru_cache(maxsize=64)
def build_source_cycle(seed: int, stage_id: str, cycle_id: int) -> tuple:
    """Return the canonical 10,000-entry source-id sequence for one cycle.

    The multiset is fixed by ``SOURCE_QUOTAS``; only the ORDER varies per
    cycle. Ordering key is ``SHA256(domain || seed || stage || cycle ||
    position)`` sorted by ``(digest, original_position)``, so the result is
    fully reproducible across processes, machines and Python versions.
    """
    st = stage_index(stage_id)
    base = []
    for src_idx, quota in enumerate(SOURCE_QUOTAS):
        base.extend([src_idx] * quota)
    assert len(base) == SOURCE_CYCLE_SEQUENCES

    prefix = DOMAIN_SOURCE_CYCLE + _u64(seed) + _u32(st) + _u64(cycle_id)
    keyed = [
        (_digest(prefix + _u32(pos)), pos, src_idx)
        for pos, src_idx in enumerate(base)
    ]
    keyed.sort(key=lambda t: (t[0], t[1]))
    return tuple(src_idx for _, _, src_idx in keyed)


@lru_cache(maxsize=64)
def _cycle_prefix_counts(seed: int, stage_id: str, cycle_id: int) -> tuple:
    """Per-position running count of each source within one cycle.

    Without this, ``source_for_ordinal`` would do an O(10,000) ``list.count``
    per sequence, i.e. ~10^7 ops per 960-sequence update. Here it is O(1) after
    one cached build.
    """
    cycle = build_source_cycle(seed, stage_id, cycle_id)
    n_src = len(SOURCE_IDS)
    running = [0] * n_src
    out = []
    for src_idx in cycle:
        out.append(running[src_idx])
        running[src_idx] += 1
    return tuple(out)


def source_for_ordinal(seed: int, stage_id: str, j: int) -> tuple:
    """Map global sequence ordinal ``j`` -> (source_index, occurrence_ordinal).

    ``occurrence_ordinal`` counts, across all cycles, how many sequences of
    this specific source were emitted before ``j``. It is what indexes into the
    per-source window permutation.
    """
    if j < 0:
        raise ValueError(f"negative ordinal {j}")
    cycle_id, pos = divmod(j, SOURCE_CYCLE_SEQUENCES)
    cycle = build_source_cycle(seed, stage_id, cycle_id)
    src_idx = cycle[pos]

    # Occurrences contributed by all previous complete cycles are exactly the
    # quota (the multiset is cycle-invariant), so no scan over history.
    full_cycles = cycle_id * SOURCE_QUOTAS[src_idx]
    within = _cycle_prefix_counts(seed, stage_id, cycle_id)[pos]
    return src_idx, full_cycles + within


# --------------------------------------------------------------------------
# 9.3 source-local window permutation
# --------------------------------------------------------------------------

def _coprime_multiplier(seed_bytes: bytes, n: int) -> int:
    """Derive ``a`` with ``gcd(a, n) == 1`` from a digest, incrementing on hit.

    Deterministic and terminating: at most n steps, and for n>1 a coprime
    always exists (a=1 works).
    """
    if n <= 1:
        return 1
    from math import gcd
    a = int.from_bytes(seed_bytes[:8], "little") % n
    if a == 0:
        a = 1
    for _ in range(n):
        if gcd(a, n) == 1:
            return a
        a += 1
        if a >= n:
            a = 1
    raise RuntimeError(f"no coprime multiplier found for n={n}")


@dataclass(frozen=True)
class ShardRef:
    """One physical shard's eligible window range within a stage/source."""
    shard_id: int
    path: str
    n_win: int          # number of eligible sequence windows in this shard
    win_stride: int     # token stride between window starts
    seq_len: int
    token_offset: int   # first eligible token index (post-holdout exclusion)


@dataclass(frozen=True)
class SourcePlan:
    """Eligible shards of one source in one stage, with prefix counts."""
    source_id: str
    source_index: int
    shards: tuple
    prefix: tuple       # len == len(shards)+1, prefix[i] = windows before i
    total_windows: int

    @staticmethod
    def build(source_id: str, source_index: int, shards: Sequence[ShardRef]):
        prefix = [0]
        for s in shards:
            prefix.append(prefix[-1] + s.n_win)
        return SourcePlan(source_id, source_index, tuple(shards),
                          tuple(prefix), prefix[-1])


@lru_cache(maxsize=256)
def _shard_order(seed: int, stage_id: str, source_index: int, pass_id: int,
                 n_shards: int) -> tuple:
    """Deterministic shard visiting order for one pass over a source."""
    st = stage_index(stage_id)
    prefix = (DOMAIN_SHARD_ORDER + _u64(seed) + _u32(st)
              + _u32(source_index) + _u64(pass_id))
    keyed = [(_digest(prefix + _u32(sid)), sid) for sid in range(n_shards)]
    keyed.sort()
    return tuple(sid for _, sid in keyed)


@dataclass(frozen=True)
class WindowRef:
    """A fully resolved read location: which file, which byte range."""
    source_id: str
    shard_path: str
    shard_id: int
    window_index: int
    token_start: int
    seq_len: int
    pass_id: int

    @property
    def byte_start(self) -> int:
        return self.token_start * 4          # uint32-le payload

    @property
    def byte_len(self) -> int:
        # +1 token so the loader can build inputs/labels by shifting.
        return (self.seq_len + 1) * 4


def resolve_window(plan: SourcePlan, seed: int, stage_id: str,
                   occurrence: int) -> WindowRef:
    """Map a source occurrence ordinal to a concrete window. O(log n_shards).

    ``pass_id`` counts wraps over the source. The design explicitly requires
    repeats to be visible rather than hidden behind the word "epoch", so the
    pass id is carried into WindowRef and into telemetry.
    """
    if plan.total_windows <= 0:
        raise ValueError(f"source {plan.source_id} has no eligible windows")

    pass_id, offset = divmod(occurrence, plan.total_windows)
    order = _shard_order(seed, stage_id, plan.source_index, pass_id,
                         len(plan.shards))

    # Prefix counts in *visit* order, so binary search finds the shard directly
    # instead of walking shards one by one.
    acc, bounds = 0, []
    for sid in order:
        acc += plan.shards[sid].n_win
        bounds.append(acc)
    k = bisect_right(bounds, offset)
    shard = plan.shards[order[k]]
    local = offset - (bounds[k - 1] if k else 0)

    st = stage_index(stage_id)
    dom = (_u64(seed) + _u32(st) + _u32(plan.source_index)
           + _u64(pass_id) + _u32(shard.shard_id))
    a = _coprime_multiplier(_digest(DOMAIN_PERM_A + dom), shard.n_win)
    b = int.from_bytes(_digest(DOMAIN_PERM_B + dom)[:8], "little") % shard.n_win
    win = (a * local + b) % shard.n_win

    return WindowRef(
        source_id=plan.source_id,
        shard_path=shard.path,
        shard_id=shard.shard_id,
        window_index=win,
        token_start=shard.token_offset + win * shard.win_stride,
        seq_len=shard.seq_len,
        pass_id=pass_id,
    )


# --------------------------------------------------------------------------
# 9.1 canonical global sequence ordinal -> concrete windows
# --------------------------------------------------------------------------

class RFullScheduler:
    """Stateless O(1)-resume scheduler over the frozen corpus plan."""

    def __init__(self, seed: int, stage_id: str, plans: dict,
                 global_sequences_per_update: int):
        self.seed = int(seed)
        # Fail fast on an unknown stage: stage_id feeds every hash domain, so a
        # typo would silently produce a valid-looking but wrong stream.
        stage_index(stage_id)
        self.stage_id = stage_id
        self.plans = plans                       # source_index -> SourcePlan
        self.G = int(global_sequences_per_update)
        missing = [i for i in range(len(SOURCE_IDS)) if i not in plans]
        if missing:
            raise ValueError(
                f"missing SourcePlan for source indices {missing}")

    def ordinal(self, successful_updates: int, slot: int) -> int:
        """j = vG + q  (section 9.1). Position depends only on committed state."""
        if not 0 <= slot < self.G:
            raise ValueError(f"slot {slot} outside [0,{self.G})")
        return successful_updates * self.G + slot

    def window_for_ordinal(self, j: int) -> WindowRef:
        src_idx, occurrence = source_for_ordinal(self.seed, self.stage_id, j)
        return resolve_window(self.plans[src_idx], self.seed, self.stage_id,
                              occurrence)

    def update_windows(self, successful_updates: int) -> list:
        """All G canonical windows of one update, in canonical slot order.

        Note this is world-size and GA independent: runtime maps slot -> (lane,
        microbatch, accumulation step) afterwards. Changing world size while
        holding G fixed must not change this multiset.
        """
        base = successful_updates * self.G
        return [self.window_for_ordinal(base + q) for q in range(self.G)]

    def lane_windows(self, successful_updates: int, lane: int,
                     n_lanes: int) -> list:
        """The subset of an update assigned to one data-parallel lane."""
        if self.G % n_lanes:
            raise ValueError(
                f"G={self.G} not divisible by n_lanes={n_lanes}")
        per = self.G // n_lanes
        base = successful_updates * self.G + lane * per
        return [self.window_for_ordinal(base + i) for i in range(per)]

    def batch_digest(self, successful_updates: int) -> str:
        """Stable digest of an update's canonical batch (resume oracle).

        Saved in the checkpoint and re-asserted after restore: if the restored
        scheduler would feed different data, the run stops instead of silently
        training on a different stream.
        """
        h = hashlib.sha256()
        h.update(b"rfull-batch-digest-v1\0")
        h.update(_u64(self.seed))
        h.update(_u32(stage_index(self.stage_id)))
        h.update(_u64(successful_updates))
        h.update(_u32(self.G))
        for w in self.update_windows(successful_updates):
            h.update(w.source_id.encode())
            h.update(_u32(w.shard_id))
            h.update(_u64(w.token_start))
            h.update(_u32(w.seq_len))
        return h.hexdigest()

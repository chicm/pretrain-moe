"""DATA-003 acceptance: rfull-hash-affine-v1 scheduler.

Directly encodes the five oracles of design section 9.4, plus the O(1) resume
property and quota exactness. Runs on CPU; no GPU required.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rfull.data_scheduler import (  # noqa: E402
    SOURCE_IDS, SOURCE_QUOTAS, SOURCE_CYCLE_SEQUENCES, RFullScheduler,
    ShardRef, SourcePlan, build_source_cycle, source_for_ordinal,
    resolve_window,
)

SEED = 20260809
R = {}


def synthetic_plans(n_shards=17, n_win=9973, seq_len=4096):
    """Deliberately non-round shard/window counts to expose modulo bias.

    ``seq_len`` also drives ``win_stride``: in the real corpus a 4K stage
    strides 4096 tokens and an 8K stage strides 8192, so windows of different
    stages live on different grids. Using a single grid for every stage (the
    first version of this fixture) made stage-boundary collisions ~20x more
    likely than chance and produced a false failure.
    """
    plans = {}
    for idx, sid in enumerate(SOURCE_IDS):
        shards = [
            ShardRef(shard_id=s, path=f"/fake/{sid}/shard_{s:05d}.bin",
                     n_win=n_win + (s * 37) % 101, win_stride=seq_len,
                     seq_len=seq_len, token_offset=0)
            for s in range(n_shards + idx)
        ]
        plans[idx] = SourcePlan.build(sid, idx, shards)
    return plans


def main():
    plans = synthetic_plans()
    G = 960
    sch = RFullScheduler(SEED, "4k", plans, G)

    # ---- 1. cycle multiset is EXACTLY the frozen quota -------------------
    cyc = build_source_cycle(SEED, "4k", 0)
    c = Counter(cyc)
    R["cycle_len"] = len(cyc)
    R["quota_exact"] = all(c[i] == q for i, q in enumerate(SOURCE_QUOTAS))
    R["cycle_len_ok"] = len(cyc) == SOURCE_CYCLE_SEQUENCES

    # different cycles must reorder, never re-weight
    cyc1 = build_source_cycle(SEED, "4k", 1)
    R["cycle_reorders"] = cyc1 != cyc
    R["cycle_multiset_invariant"] = Counter(cyc1) == c

    # stage separation: same seed, different stage -> different order
    R["stage_separated"] = build_source_cycle(SEED, "8k", 0) != cyc
    R["seed_separated"] = build_source_cycle(SEED + 1, "4k", 0) != cyc

    # ---- 2. long-run source proportions track the quota ------------------
    N = 200_000
    seen = Counter(source_for_ordinal(SEED, "4k", j)[0] for j in range(N))
    worst = max(abs(seen[i] / N - q / SOURCE_CYCLE_SEQUENCES)
                for i, q in enumerate(SOURCE_QUOTAS))
    R["quota_drift_max"] = worst
    R["quota_drift_ok"] = worst < 1e-9   # exact by construction, not statistical

    # ---- 3. occurrence ordinals are dense and gapless --------------------
    occ = {i: [] for i in range(len(SOURCE_IDS))}
    for j in range(50_000):
        s, o = source_for_ordinal(SEED, "4k", j)
        occ[s].append(o)
    R["occurrence_contiguous"] = all(
        v == list(range(len(v))) for v in occ.values())

    # ---- 4. window permutation is a BIJECTION (no dup, no hole) ----------
    p = plans[0]
    sh = p.shards[3]
    # collect windows landing in that shard over a full pass
    hits = []
    for m in range(p.total_windows):
        w = resolve_window(p, SEED, "4k", m)
        if w.shard_id == sh.shard_id:
            hits.append(w.window_index)
    R["perm_covers_shard"] = sorted(hits) == list(range(sh.n_win))
    R["perm_no_dupes"] = len(hits) == len(set(hits))

    # second pass must differ from first (re-permuted), same coverage
    hits2 = []
    for m in range(p.total_windows, 2 * p.total_windows):
        w = resolve_window(p, SEED, "4k", m)
        if w.shard_id == sh.shard_id:
            hits2.append(w.window_index)
    R["pass2_covers_shard"] = sorted(hits2) == list(range(sh.n_win))
    R["pass2_differs"] = hits2 != hits
    R["pass_id_advances"] = resolve_window(
        p, SEED, "4k", p.total_windows).pass_id == 1

    # ---- 5. ORACLE: world-size / GA invariance (9.4 #3) ------------------
    v = 12345
    canon = sch.update_windows(v)
    key = lambda w: (w.source_id, w.shard_id, w.token_start)
    for n_lanes in (120, 240, 480, 960):
        got = []
        for lane in range(n_lanes):
            got.extend(sch.lane_windows(v, lane, n_lanes))
        R[f"lane_split_{n_lanes}_ok"] = (
            Counter(map(key, got)) == Counter(map(key, canon)))

    # ---- 6. ORACLE: O(1) resume near 1M / 100M / final (9.4 #2) ----------
    timings = {}
    for label, upd in (("1M", 1_000_000 // G), ("100M", 100_000_000 // G),
                       ("final", 203_450)):
        t0 = time.perf_counter()
        d = sch.batch_digest(upd)
        timings[label] = round(time.perf_counter() - t0, 4)
        R[f"digest_{label}"] = d[:16]
    R["resume_timings_s"] = timings
    # late updates must not cost more than early ones (no history walk)
    t_early = timings["1M"]
    t_late = timings["final"]
    R["resume_is_O1"] = t_late < max(3 * t_early, t_early + 0.5)

    # ---- 7. ORACLE: cross-process reproducibility (9.4 #1) ---------------
    child = subprocess.run(
        [sys.executable, __file__, "--child", str(SEED)],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONHASHSEED": "1"},
    )
    child_out = child.stdout.strip().splitlines()[-1] if child.stdout else ""
    parent_sig = sch.batch_digest(0) + "|" + sch.batch_digest(777)
    R["child_matches_parent"] = (child_out == parent_sig)
    R["child_sig"] = child_out[:16]
    R["parent_sig"] = parent_sig[:16]

    # ---- 8. ORACLE: stage boundaries (9.4 #5) ---------------------------
    # Two distinct properties, tested separately:
    #  (a) WITHIN a stage the schedule must never repeat a window before the
    #      source wraps -- this is the permutation-correctness property.
    #  (b) ACROSS stages, disjointness cannot come from hashing alone: an 8K
    #      window starting at a multiple of 8192 is also a legal 4K start, so
    #      collisions are possible by construction. The design therefore makes
    #      the stage manifest carve disjoint eligible ranges. We assert the
    #      manifest mechanism (token_offset partitioning) actually works.
    s4 = RFullScheduler(SEED, "4k", plans, 960)
    last4 = list(map(key, s4.update_windows(203_450)))
    R["intra_stage_no_repeat"] = len(set(last4)) == len(last4)
    adj = set(map(key, s4.update_windows(10)))
    R["intra_stage_adjacent_disjoint"] = not (
        adj & set(map(key, s4.update_windows(11))))
    R["intra_stage_far_disjoint"] = not (
        adj & set(map(key, s4.update_windows(50_000))))

    # (b) partitioned ranges -> provable disjointness, no reliance on luck
    part8 = {}
    for idx, sid in enumerate(SOURCE_IDS):
        shards = [
            ShardRef(shard_id=s.shard_id, path=s.path,
                     n_win=s.n_win // 4, win_stride=8192, seq_len=8192,
                     token_offset=s.n_win * 4096 + 8192)  # beyond 4K region
            for s in plans[idx].shards
        ]
        part8[idx] = SourcePlan.build(sid, idx, shards)
    s8p = RFullScheduler(SEED, "8k", part8, 480)
    first8 = set(map(key, s8p.update_windows(0)))
    R["stage_boundary_disjoint"] = not (set(last4) & first8)

    # ---- 9. guards --------------------------------------------------------
    def raises(fn):
        try:
            fn()
            return False
        except Exception:
            return True

    R["rejects_bad_slot"] = raises(lambda: sch.ordinal(0, G))
    R["rejects_neg_ordinal"] = raises(lambda: sch.window_for_ordinal(-1))
    R["rejects_bad_stage"] = raises(
        lambda: RFullScheduler(SEED, "64k", plans, G).ordinal(0, 0))
    R["rejects_missing_source"] = raises(
        lambda: RFullScheduler(SEED, "4k", {0: plans[0]}, G))
    R["rejects_indivisible_lanes"] = raises(
        lambda: sch.lane_windows(0, 0, 7))

    checks = {k: v for k, v in R.items() if isinstance(v, bool)}
    R["n_checks"] = len(checks)
    R["failed"] = sorted(k for k, v in checks.items() if not v)
    R["verdict"] = "PASS" if not R["failed"] else "FAIL"

    body = json.dumps(R, indent=1, sort_keys=True)
    R["evidence_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    print(json.dumps(R, indent=1, sort_keys=True))

    out = Path(os.environ.get("RFULL_EVIDENCE_DIR", ".")) / "data003_evidence.json"
    out.write_text(json.dumps(R, indent=1, sort_keys=True), encoding="utf-8")
    print(f"EVIDENCE {out} sha256={R['evidence_sha256']}")
    return 0 if R["verdict"] == "PASS" else 1


def child(seed):
    plans = synthetic_plans()
    sch = RFullScheduler(int(seed), "4k", plans, 960)
    print(sch.batch_digest(0) + "|" + sch.batch_digest(777))


if __name__ == "__main__":
    if "--child" in sys.argv:
        child(sys.argv[sys.argv.index("--child") + 1])
    else:
        raise SystemExit(main())

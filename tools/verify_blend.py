#!/usr/bin/env python3
"""Verify a Megatron data blend index before committing it to a long run.

Checks, in order:
  1. every shard prefix has BOTH a .bin and a .idx sidecar
  2. weights are positive and sum to ~1.0
  3. prints a per-corpus rollup (shard count + aggregate weight)

Existence is checked with os.path.exists() on exact paths rather than find(1):
these live on a blobfuse mount, where a timed-out find silently returns fewer
results and a "no matches" answer is indistinguishable from "does not exist".

Exit codes: 0 ok, 2 missing files, 3 weight problem, 4 bad/unreadable index.
"""
import argparse
import collections
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", help="blend json with {'blend':[{'prefix','weight'}...]}")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="allowed deviation of the weight sum from 1.0")
    ap.add_argument("--max-report", type=int, default=10)
    args = ap.parse_args()

    try:
        with open(args.index) as fh:
            doc = json.load(fh)
        blend = doc["blend"]
    except (OSError, ValueError, KeyError) as exc:
        print(f"FAIL unreadable index: {exc}")
        return 4

    if not blend:
        print("FAIL blend is empty")
        return 4

    print(f"index      {args.index}")
    print(f"split      {doc.get('split', '<none>')}")
    print(f"entries    {len(blend)}")

    missing, bad_weight = [], []
    per_corpus_n = collections.Counter()
    per_corpus_w = collections.defaultdict(float)
    total_bytes = 0

    for entry in blend:
        prefix, weight = entry["prefix"], entry["weight"]
        bin_path, idx_path = prefix + ".bin", prefix + ".idx"
        if not (os.path.exists(bin_path) and os.path.exists(idx_path)):
            missing.append(prefix)
        else:
            try:
                total_bytes += os.path.getsize(bin_path)
            except OSError:
                pass
        if not (isinstance(weight, (int, float)) and weight > 0):
            bad_weight.append((prefix, weight))
        # ".../data/<corpus>/part_N/shard_XXXX" -> <corpus>
        corpus = prefix.split("/data/")[-1].split("/")[0] if "/data/" in prefix else "?"
        per_corpus_n[corpus] += 1
        per_corpus_w[corpus] += float(weight) if isinstance(weight, (int, float)) else 0.0

    weight_sum = sum(e["weight"] for e in blend if isinstance(e["weight"], (int, float)))
    print(f"weight_sum {weight_sum:.9f}")
    print(f"bytes      {total_bytes:,} ({total_bytes / 1024**4:.2f} TiB)")
    print()
    print(f"{'corpus':<26}{'shards':>8}{'weight':>10}")
    print("-" * 44)
    for corpus in sorted(per_corpus_w, key=lambda k: -per_corpus_w[k]):
        print(f"{corpus:<26}{per_corpus_n[corpus]:>8}{per_corpus_w[corpus]:>10.4f}")
    print()

    rc = 0
    if missing:
        print(f"FAIL {len(missing)} prefixes missing .bin/.idx")
        for m in missing[:args.max_report]:
            print(f"  MISSING {m}")
        rc = 2
    if bad_weight:
        print(f"FAIL {len(bad_weight)} non-positive weights")
        for prefix, weight in bad_weight[:args.max_report]:
            print(f"  BADWEIGHT {weight!r} {prefix}")
        rc = rc or 3
    if abs(weight_sum - 1.0) > args.tol:
        print(f"FAIL weight sum {weight_sum!r} deviates from 1.0 by more than {args.tol}")
        rc = rc or 3

    print("BLEND OK" if rc == 0 else f"BLEND FAIL rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

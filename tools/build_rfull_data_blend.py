#!/usr/bin/env python3
"""Generate the R-Full production data blend from the built corpus manifest.

Emits the `data` block for the production config: one weighted entry per shard
prefix, with weights chosen so each corpus contributes its intended share of
the 1T token budget.

Two mixtures are available:

  natural (default)
      Each corpus contributes in proportion to the tokens actually collected.
      The prior project already curated this corpus deliberately, so natural
      share reflects that intent, needs no extra justification, and -- because
      the budget is below the corpus size -- repeats no document anywhere.

  quality-weighted
      Up-weights code / math / reasoning sources relative to their size.  This
      is a real training decision that forces repetition of the small math
      corpora, so it must be requested explicitly and never becomes the default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

# Explicit, opt-in mixture.  Sums to 1.0.
QUALITY_WEIGHTED_MIX = {
    "dclm_tok":               0.29,
    "fineweb_edu_240bt_tok":  0.17,
    "finephrase_tok":         0.15,
    "starcoder_tok":          0.13,
    "fineweb_edu_100bt_tok":  0.09,
    "finepdfs_edu_tok":       0.06,
    "math_tok":               0.05,
    "infimath_tok":           0.03,
    "owm_tok":                0.02,
    "fineweb_tok":            0.01,
}


def corpus_of(path: str) -> str:
    parts = path.split("/")
    return parts[parts.index("data") + 1] if "data" in parts else "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--budget-tokens", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", default="998,2,0")
    ap.add_argument("--mixture", choices=("natural", "quality-weighted"), default="natural")
    ap.add_argument(
        "--allow-repeats",
        action="store_true",
        help="permit a corpus to be consumed for more than one epoch",
    )
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    entries = manifest["entries"]

    by_corpus: dict[str, list[dict]] = {}
    for e in entries:
        by_corpus.setdefault(corpus_of(e["bin"]), []).append(e)

    total_tokens = sum(e["tokens"] for e in entries)
    if args.mixture == "natural":
        mix = {c: sum(e["tokens"] for e in v) / total_tokens for c, v in by_corpus.items()}
    else:
        mix = dict(QUALITY_WEIGHTED_MIX)
        unknown = sorted(set(by_corpus) - set(mix))
        if unknown:
            print(f"FAIL: corpus present but absent from mixture: {unknown}", file=sys.stderr)
            return 1
        absent = sorted(set(mix) - set(by_corpus))
        if absent:
            print(f"FAIL: mixture names a corpus with no shards: {absent}", file=sys.stderr)
            return 1
    if abs(sum(mix.values()) - 1.0) > 1e-9:
        print(f"FAIL: mixture sums to {sum(mix.values())}", file=sys.stderr)
        return 1

    if args.budget_tokens > total_tokens and not args.allow_repeats:
        print(f"FAIL: budget {args.budget_tokens:,} exceeds corpus {total_tokens:,}", file=sys.stderr)
        return 1

    print(f"mixture: {args.mixture}   budget: {args.budget_tokens:,} tokens")
    print(f"{'corpus':24} {'shards':>6} {'avail(B)':>10} {'natural%':>9} "
          f"{'target%':>8} {'need(B)':>9} {'epochs':>7}")
    blend, warnings = [], []
    for corpus in sorted(mix):
        shards = sorted(by_corpus[corpus], key=lambda e: e["bin"])
        avail = sum(e["tokens"] for e in shards)
        target = mix[corpus]
        need = target * args.budget_tokens
        epochs = need / avail
        natural = avail / total_tokens
        flag = ""
        if epochs > 1.0:
            flag = "  <-- REPEATS"
            warnings.append(f"{corpus}: needs {epochs:.2f} epochs")
        print(f"{corpus:24} {len(shards):>6} {avail/1e9:>9.1f} {100*natural:>8.2f}% "
              f"{100*target:>7.2f}% {need/1e9:>8.1f} {epochs:>7.3f}{flag}")
        # Weight each shard by its token count so a corpus's share is spread
        # evenly over its shards regardless of how unevenly they are sized.
        for e in shards:
            blend.append({
                "weight": round(target * e["tokens"] / avail, 12),
                "prefix": e["bin"][:-4],
            })

    wsum = sum(b["weight"] for b in blend)
    print(f"\nblend entries: {len(blend)}   weight sum: {wsum:.9f}")
    if abs(wsum - 1.0) > 1e-6:
        print(f"FAIL: blend weights sum to {wsum}", file=sys.stderr)
        return 1
    for w in warnings:
        print(f"WARNING: {w}")
    if warnings and not args.allow_repeats:
        print("FAIL: mixture repeats data; pass --allow-repeats to accept", file=sys.stderr)
        return 1

    data_block = {"split": args.split, "blend": blend}
    blob = json.dumps(data_block, indent=1, sort_keys=True).encode()
    with open(args.output, "wb") as fh:
        fh.write(blob)
    max_epochs = max(mix[c] * args.budget_tokens / sum(e["tokens"] for e in by_corpus[c]) for c in mix)
    print(f"\nBLEND_OUTPUT={args.output}")
    print(f"BLEND_SHA256={hashlib.sha256(blob).hexdigest()}")
    print(f"BLEND_MIXTURE={args.mixture}")
    print(f"BLEND_TOTAL_AVAILABLE_TOKENS={total_tokens}")
    print(f"BLEND_BUDGET_TOKENS={args.budget_tokens}")
    print(f"BLEND_BUDGET_EPOCHS={args.budget_tokens/total_tokens:.4f}")
    print(f"BLEND_MAX_CORPUS_EPOCHS={max_epochs:.3f}")
    print("BLEND_RESULT=" + ("PASS" if not warnings else "PASS_WITH_REPEATS"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

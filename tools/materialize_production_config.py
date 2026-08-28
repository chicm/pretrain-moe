#!/usr/bin/env python3
"""Materialise the production config by inlining a verified blend index.

Kept as a script (rather than hand-editing JSON) because the blend is 487
entries: pasting that by hand is how a weight or a prefix silently goes wrong.
The config is the launch contract, so everything it asserts is checked here.

Run tools/verify_blend.py on the index first -- this script trusts that the
shards exist and only re-checks internal consistency.
"""
import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--blend-index", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-iters", type=int, required=True)
    ap.add_argument("--warmup-iters", type=int, required=True)
    ap.add_argument("--decay-iters", type=int, required=True)
    ap.add_argument("--stable-end-iter", type=int, required=True)
    ap.add_argument("--rotary-base", type=int, required=True)
    ap.add_argument("--timeout-minutes", type=int, required=True)
    ap.add_argument("--save-interval", type=int, required=True)
    ap.add_argument("--eval-interval", type=int, required=True)
    ap.add_argument("--eval-iters", type=int, required=True)
    ap.add_argument("--target-tokens", type=int, required=True)
    args = ap.parse_args()

    with open(args.config) as fh:
        config = json.load(fh)
    with open(args.blend_index) as fh:
        index = json.load(fh)

    blend = index["blend"]
    weight_sum = sum(e["weight"] for e in blend)
    if abs(weight_sum - 1.0) > 1e-6:
        print(f"FAIL blend weights sum to {weight_sum!r}, expected 1.0")
        return 2

    tokens_per_step = (
        config["training"]["global_batch_size"] * config["model"]["sequence_length"]
    )
    actual_tokens = tokens_per_step * args.train_iters
    if actual_tokens != args.target_tokens:
        print(
            f"FAIL token budget mismatch: {args.train_iters} iters x {tokens_per_step} "
            f"= {actual_tokens:,}, expected {args.target_tokens:,}"
        )
        return 3
    if args.stable_end_iter != args.train_iters - args.decay_iters:
        print(
            f"FAIL stable_end_iter {args.stable_end_iter} != train_iters - decay_iters "
            f"({args.train_iters - args.decay_iters})"
        )
        return 3

    config["training"]["train_iters"] = args.train_iters
    config["training"]["lr_schedule"] = {
        "style": "warmup_stable_decay",
        "warmup_iters": args.warmup_iters,
        "decay_iters": args.decay_iters,
        "stable_end_iter": args.stable_end_iter,
    }
    config["model"]["rotary_base"] = args.rotary_base
    config["runtime"]["distributed_timeout_minutes"] = args.timeout_minutes
    config["runtime"]["save_interval"] = args.save_interval
    config["runtime"]["eval_interval"] = args.eval_interval
    config["runtime"]["eval_iters"] = args.eval_iters
    config["production_launch"] = True
    config["data"]["split"] = index.get("split", config["data"]["split"])
    config["data"]["blend"] = blend

    with open(args.output, "w", newline="\n") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")

    print(f"wrote            {args.output}")
    print(f"blend entries    {len(blend)}")
    print(f"weight sum       {weight_sum:.9f}")
    print(f"split            {config['data']['split']}")
    print(f"train_iters      {args.train_iters:,}")
    print(f"tokens/step      {tokens_per_step:,}")
    print(f"total tokens     {actual_tokens:,}")
    print(f"lr schedule      warmup {args.warmup_iters:,} -> stable to "
          f"{args.stable_end_iter:,} -> cosine decay {args.decay_iters:,}")
    print(f"rotary_base      {args.rotary_base}")
    print(f"save_interval    {args.save_interval}")
    print("CONFIG OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

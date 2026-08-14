"""Derive a Gate 4a mini config that is a VALID proxy for production.

The previous mini config (v2) had two defects that made it both ruinously slow
and guaranteed to crash:

  seq_length = 8      The GPTDataset sample index has ~tokens/seq_len entries,
                      so seq 8 produced a 1.98 GB index per shard (~1 TB, ~15 h
                      over blobfuse).  Production seq 4096 is 512x smaller.

  vocab_size = 4000   The real corpus contains tokens up to 151668, so the very
                      first embedding lookup would have been out of range.

A qualification run must differ from production ONLY in scale (iterations,
batch, nodes) -- never in semantics that change the data path.  This tool
copies the production config and overrides only the scale knobs, so the cache
it builds is byte-identical in key structure to what production needs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


SCALE_ONLY_KEYS = {
    "train_iters",
    "global_batch_size",
    "micro_batch_size",
    "eval_interval",
    "eval_iters",
    "save_interval",
    "lr_warmup_iters",
    "lr_decay_iters",
    "lr_stable_iters",
}


def derive(production: dict, *, nnodes: int, train_iters: int, gbs: int, mbs: int) -> dict:
    config = json.loads(json.dumps(production))  # deep copy

    cluster = config.setdefault("cluster", {})
    cluster["nnodes"] = nnodes

    training = config["training"]
    training["train_iters"] = train_iters
    training["global_batch_size"] = gbs
    training["micro_batch_size"] = mbs
    training["eval_interval"] = max(train_iters, 1)
    training["eval_iters"] = 1

    # Keep the LR schedule self-consistent at the reduced iteration count so
    # the trainer's own validation does not reject it.
    for key, value in (
        ("lr_warmup_iters", max(1, train_iters // 10)),
        ("lr_decay_iters", max(1, train_iters)),
    ):
        if key in training:
            training[key] = value
    training.pop("lr_stable_iters", None)

    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--nnodes", type=int, default=2)
    parser.add_argument("--train-iters", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    args = parser.parse_args(argv)

    production = json.loads(args.production_config.read_text())
    derived = derive(
        production,
        nnodes=args.nnodes,
        train_iters=args.train_iters,
        gbs=args.global_batch_size,
        mbs=args.micro_batch_size,
    )

    # Fail closed: the data path must be semantically identical to production.
    pm, dm = production["model"], derived["model"]
    pd, dd = production["data"], derived["data"]
    for field in ("sequence_length", "seq_length", "native_vocab_size"):
        if pm.get(field) != dm.get(field):
            raise SystemExit(f"data-path divergence in model.{field}")
    for field in ("split", "blend"):
        if json.dumps(pd.get(field)) != json.dumps(dd.get(field)):
            raise SystemExit(f"data-path divergence in data.{field}")

    text = json.dumps(derived, indent=1, sort_keys=True)
    args.output.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()

    seq = dm.get("sequence_length", dm.get("seq_length"))
    print(json.dumps({
        "output": str(args.output),
        "sha256": digest,
        "nnodes": derived["cluster"]["nnodes"],
        "train_iters": derived["training"]["train_iters"],
        "global_batch_size": derived["training"]["global_batch_size"],
        "sequence_length": seq,
        "native_vocab_size": dm.get("native_vocab_size"),
        "split": dd.get("split"),
        "blend_entries": len(dd.get("blend", [])),
        "data_path_matches_production": True,
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

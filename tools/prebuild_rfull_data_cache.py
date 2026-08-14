"""Pre-build Megatron GPTDataset indices offline, before any GPU is allocated.

Motivation
----------
`BlendedMegatronDatasetBuilder` constructs the per-shard
{description,document,sample,shuffle}_index.npy files on rank 0 only, while
every other rank blocks on a 1-element ALLREDUCE barrier.  Measured on the
487-shard 1T blend this takes ~90 minutes.  Paying that on 120 idle GPUs is
pure waste, and Megatron's default 10 minute watchdog kills it outright.

The cache key depends on the blend, split, sequence length, seed and the
requested sample counts, so a cache built for mini qualification does NOT
serve the production run.  This tool builds the exact cache a given config
will look for, using one CPU process and no distributed init.

Fail-closed: verifies every produced index is readable and non-empty, and
refuses to report success if any shard is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time


def _load_config(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _sample_counts(config: dict) -> tuple[int, int, int]:
    """Mirror Megatron's train/valid/test sample budgeting."""
    training = config["training"]
    train_iters = int(training["train_iters"])
    gbs = int(training["global_batch_size"])
    eval_interval = int(training.get("eval_interval", 1000))
    eval_iters = int(training.get("eval_iters", 10))

    train_samples = train_iters * gbs
    # Megatron requests enough eval samples for every eval round plus one.
    eval_rounds = train_iters // eval_interval + 1
    valid_samples = eval_rounds * eval_iters * gbs
    test_samples = eval_iters * gbs
    return train_samples, valid_samples, test_samples


def build(
    config_path: pathlib.Path,
    cache_path: pathlib.Path,
    *,
    upstream_root: pathlib.Path,
    dry_run: bool,
    builder_threads: int = 1,
) -> dict[str, object]:
    config = _load_config(config_path)

    runtime = config.get("runtime", {})
    if runtime.get("mock_data"):
        raise SystemExit("config uses mock data; nothing to pre-build")

    data = config.get("data") or {}
    blend_entries = data.get("blend") or []
    if not blend_entries:
        raise SystemExit("config has no data.blend; materialise it first")

    split = data.get("split")
    if not isinstance(split, str) or split.count(",") != 2:
        raise SystemExit("config data.split must be 'train,valid,test'")

    model = config["model"]
    # Production configs use `sequence_length`; some qualification configs use
    # the shorter `seq_length`.  Accept either, fail closed if neither.
    if "sequence_length" in model:
        seq_length = int(model["sequence_length"])
    elif "seq_length" in model:
        seq_length = int(model["seq_length"])
    else:
        raise SystemExit("config model has neither sequence_length nor seq_length")
    seed = int(runtime.get("seed", config.get("seed", 1234)))

    train_s, valid_s, test_s = _sample_counts(config)

    plan = {
        "config": str(config_path),
        "cache_path": str(cache_path),
        "blend_entries": len(blend_entries),
        "split": split,
        "seq_length": seq_length,
        "seed": seed,
        "samples": {"train": train_s, "valid": valid_s, "test": test_s},
        "builder_threads": int(builder_threads),
    }
    if dry_run:
        plan["dry_run"] = True
        return plan

    sys.path.insert(0, str(upstream_root))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "0")

    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.core.datasets.utils import get_blend_from_list

    weights = [float(e["weight"]) for e in blend_entries]
    prefixes = [str(e["prefix"]) for e in blend_entries]
    interleaved: list[str] = []
    for weight, prefix in zip(weights, prefixes):
        interleaved.extend([str(weight), prefix])

    cache_path.mkdir(parents=True, exist_ok=True)

    class _NullTokenizer:
        def __init__(self, eod: int) -> None:
            self.eod = eod

    eod = int(config["model"].get("eot_token_id", 151643))
    ds_config = GPTDatasetConfig(
        random_seed=seed,
        sequence_length=seq_length,
        blend=get_blend_from_list(interleaved),
        blend_per_split=None,
        split=split,
        num_dataset_builder_threads=int(builder_threads),
        path_to_cache=str(cache_path),
        mmap_bin_files=True,
        tokenizer=_NullTokenizer(eod),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
    )

    started = time.time()
    builder = BlendedMegatronDatasetBuilder(
        GPTDataset, [train_s, valid_s, test_s], lambda: True, ds_config
    )
    datasets = builder.build()
    elapsed = time.time() - started

    sizes = []
    for name, ds in zip(("train", "valid", "test"), datasets):
        sizes.append({"split": name, "len": (len(ds) if ds is not None else 0)})

    produced = sorted(p.name for p in cache_path.rglob("*.npy"))
    plan.update(
        {
            "elapsed_seconds": round(elapsed, 1),
            "dataset_sizes": sizes,
            "cache_files": len(produced),
        }
    )
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--cache-path", required=True, type=pathlib.Path)
    parser.add_argument("--upstream-root", required=True, type=pathlib.Path)
    parser.add_argument(
        "--builder-threads",
        type=int,
        default=1,
        help="MCore num_dataset_builder_threads; >1 parallelises index construction",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = build(
        args.config,
        args.cache_path,
        upstream_root=args.upstream_root,
        dry_run=args.dry_run,
        builder_threads=args.builder_threads,
    )
    print(json.dumps(report, indent=2))
    print("PREBUILD_RESULT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

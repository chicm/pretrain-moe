#!/usr/bin/env python3
"""Validate, render, and launch reproducible Megatron-LM smoke profiles.

The training loop and model are intentionally upstream Megatron-LM.  This file
only translates an audited JSON profile into ``pretrain_gpt.py`` arguments and
launches one torchrun agent per node.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any


class ConfigError(ValueError):
    """Raised when a smoke profile is internally inconsistent."""


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")
    return value


def topology(config: dict[str, Any]) -> dict[str, int]:
    cluster = config["cluster"]
    parallel = config["parallel"]
    nnodes = _positive_int(cluster["nnodes"], "cluster.nnodes")
    gpus_per_node = _positive_int(cluster["gpus_per_node"], "cluster.gpus_per_node")
    tp = _positive_int(parallel["tensor_model_parallel_size"], "parallel.tp")
    pp = _positive_int(parallel["pipeline_model_parallel_size"], "parallel.pp")
    cp = _positive_int(parallel.get("context_parallel_size", 1), "parallel.cp")
    world_size = nnodes * gpus_per_node
    model_parallel = tp * pp * cp
    if world_size % model_parallel:
        raise ConfigError(
            f"world_size={world_size} is not divisible by TP*PP*CP={model_parallel}"
        )
    return {
        "nnodes": nnodes,
        "gpus_per_node": gpus_per_node,
        "world_size": world_size,
        "tp": tp,
        "pp": pp,
        "cp": cp,
        "dp": world_size // model_parallel,
    }


def estimate_dense_parameters(config: dict[str, Any]) -> int:
    """Return an auditable transformer-only dense parameter estimate.

    The estimate includes token embeddings/output, attention projections,
    SwiGLU/standard MLP projections, and two norm vectors per layer.  It is a
    sizing guard, not a replacement for Megatron's exact parameter report.
    """

    model = config["model"]
    layers = int(model["num_layers"])
    hidden = int(model["hidden_size"])
    ffn = int(model["ffn_hidden_size"])
    heads = int(model["num_attention_heads"])
    query_groups = int(model.get("num_query_groups", heads))
    vocab = int(model["vocab_size"])
    if heads <= 0 or hidden % heads:
        raise ConfigError("model.hidden_size must be divisible by num_attention_heads")
    if query_groups <= 0 or heads % query_groups:
        raise ConfigError("num_attention_heads must be divisible by num_query_groups")

    head_dim = hidden // heads
    # Q has `heads`; K and V have `query_groups`; output projection is hidden^2.
    attention = hidden * (heads + 2 * query_groups) * head_dim + hidden * hidden
    mlp_multiplier = 3 if model.get("swiglu", False) else 2
    mlp = mlp_multiplier * hidden * ffn
    norms = 2 * hidden
    embeddings = vocab * hidden
    if not model.get("tie_embeddings", True):
        embeddings *= 2
    return layers * (attention + mlp + norms) + embeddings + hidden


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    for section in ("upstream", "cluster", "model", "parallel", "training", "runtime"):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"missing object section: {section}")
    commit = config["upstream"].get("commit", "")
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ConfigError("upstream.commit must be a full lowercase SHA-1")

    topo = topology(config)
    train = config["training"]
    micro = _positive_int(train["micro_batch_size"], "training.micro_batch_size")
    global_batch = _positive_int(train["global_batch_size"], "training.global_batch_size")
    one_micro_round = micro * topo["dp"]
    if global_batch % one_micro_round:
        raise ConfigError(
            f"global_batch_size={global_batch} must be divisible by "
            f"micro_batch_size*DP={one_micro_round}"
        )
    model = config["model"]
    for name in (
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
        "sequence_length",
        "max_position_embeddings",
        "vocab_size",
    ):
        _positive_int(model[name], f"model.{name}")
    if model["sequence_length"] > model["max_position_embeddings"]:
        raise ConfigError("sequence_length cannot exceed max_position_embeddings")
    runtime = config["runtime"]
    for name in ("te_fused_attention", "te_flash_attention", "te_unfused_attention"):
        if not isinstance(runtime.get(name), bool):
            raise ConfigError(f"runtime.{name} must be an explicit boolean")
    backend = runtime.get("attention_backend")
    expected_backend_flags = {
        "flash": (False, True, False),
        "fused": (True, False, False),
        "unfused": (False, False, True),
        "auto": (True, True, True),
    }
    if backend not in expected_backend_flags:
        raise ConfigError("runtime.attention_backend must be flash, fused, unfused, or auto")
    actual_backend_flags = (
        runtime["te_fused_attention"],
        runtime["te_flash_attention"],
        runtime["te_unfused_attention"],
    )
    if actual_backend_flags != expected_backend_flags[backend]:
        raise ConfigError(
            "explicit NVTE attention flags do not match runtime.attention_backend"
        )
    estimate_dense_parameters(config)


def _flag(args: list[str], enabled: bool, name: str) -> None:
    if enabled:
        args.append(name)


def build_megatron_args(
    config: dict[str, Any], run_dir: str, data_cache_path: str | None = None
) -> list[str]:
    validate_config(config)
    model = config["model"]
    parallel = config["parallel"]
    train = config["training"]
    runtime = config["runtime"]
    train_iters = int(train["train_iters"])

    args = [
        "--use-mcore-models",
        "--transformer-impl", str(runtime["transformer_impl"]),
        "--attention-backend", str(runtime["attention_backend"]),
        "--num-layers", str(model["num_layers"]),
        "--hidden-size", str(model["hidden_size"]),
        "--ffn-hidden-size", str(model["ffn_hidden_size"]),
        "--num-attention-heads", str(model["num_attention_heads"]),
        "--seq-length", str(model["sequence_length"]),
        "--max-position-embeddings", str(model["max_position_embeddings"]),
        "--normalization", str(model["normalization"]),
        "--position-embedding-type", str(model["position_embedding_type"]),
        "--rotary-base", str(model["rotary_base"]),
        "--tokenizer-type", "NullTokenizer",
        "--vocab-size", str(model["vocab_size"]),
        "--split", str((config.get("data") or {}).get("split", "949,50,1")),
        "--data-cache-path", str(data_cache_path or pathlib.Path(run_dir) / "data-cache"),
        "--micro-batch-size", str(train["micro_batch_size"]),
        "--global-batch-size", str(train["global_batch_size"]),
        "--train-iters", str(train_iters),
        "--lr", str(train["learning_rate"]),
        "--min-lr", str(train["min_learning_rate"]),
        "--lr-decay-style", "cosine",
        "--lr-decay-iters", str(train_iters),
        "--lr-warmup-iters", "1",
        "--weight-decay", str(train["weight_decay"]),
        "--adam-beta1", str(train["adam_beta1"]),
        "--adam-beta2", str(train["adam_beta2"]),
        "--clip-grad", str(train["clip_grad"]),
        "--seed", str(train["seed"]),
        "--init-method-std", "0.02",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--tensor-model-parallel-size", str(parallel["tensor_model_parallel_size"]),
        "--pipeline-model-parallel-size", str(parallel["pipeline_model_parallel_size"]),
        "--context-parallel-size", str(parallel.get("context_parallel_size", 1)),
        "--distributed-backend", "nccl",
        "--no-masked-softmax-fusion",
        "--log-interval", str(runtime["log_interval"]),
        "--eval-interval", str(max(1000, train_iters + 1)),
        "--eval-iters", "1",
    ]
    if model.get("num_query_groups", model["num_attention_heads"]) != model["num_attention_heads"]:
        args += ["--group-query-attention", "--num-query-groups", str(model["num_query_groups"])]
    _flag(args, model.get("swiglu", False), "--swiglu")
    _flag(args, not model.get("linear_bias", True), "--disable-bias-linear")
    _flag(args, not model.get("tie_embeddings", True), "--untie-embeddings-and-output-weights")
    _flag(args, model.get("position_embedding_type") == "rope", "--no-position-embedding")
    _flag(args, train.get("bf16", False), "--bf16")
    # Data source is a fail-closed choice, mirroring the MoE path: either mock
    # tokens or a real corpus blend, never a silent fallback. The dense path
    # needs real data so it can serve as a CONTROL for the MoE crash
    # investigation -- one of the observed crash stacks was inside GPTDataset's
    # numpy mmap, so a mock-data dense run would not exercise the same code and
    # could not falsify a data-path hypothesis.
    data = config.get("data")
    if runtime.get("mock_data", False):
        if data is not None:
            raise ConfigError("runtime.mock_data=true forbids a data blend")
        args.append("--mock-data")
    else:
        if not isinstance(data, dict):
            raise ConfigError(
                "runtime.mock_data must be true, or config.data must define a real blend"
            )
        blend = data.get("blend")
        if not isinstance(blend, list) or not blend:
            raise ConfigError("data.blend must be a non-empty list")
        weighted: list[str] = []
        for entry in blend:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("weight"), (int, float))
                or float(entry["weight"]) <= 0
                or not isinstance(entry.get("prefix"), str)
                or not entry["prefix"].startswith("/")
            ):
                raise ConfigError(
                    "each data.blend entry needs a positive weight and an absolute prefix"
                )
            weighted.extend([repr(float(entry["weight"])), entry["prefix"]])
        args.extend(["--data-path", *weighted])
    _flag(args, runtime.get("distributed_optimizer", False), "--use-distributed-optimizer")
    _flag(args, runtime.get("sequence_parallel", False), "--sequence-parallel")
    _flag(args, runtime.get("overlap_grad_reduce", False), "--overlap-grad-reduce")
    _flag(args, runtime.get("overlap_param_gather", False), "--overlap-param-gather")
    return args


def build_torchrun_command(
    config: dict[str, Any],
    *,
    python: str,
    megatron_dir: str,
    run_dir: str,
    node_rank: int,
    master_addr: str,
    master_port: int,
    data_cache_path: str | None = None,
) -> list[str]:
    topo = topology(config)
    if not 0 <= node_rank < topo["nnodes"]:
        raise ConfigError(f"node_rank={node_rank} outside [0, {topo['nnodes']})")
    pretrain = pathlib.Path(megatron_dir) / "pretrain_gpt.py"
    if not pretrain.is_file():
        raise ConfigError(f"missing upstream entrypoint: {pretrain}")
    rocm_entrypoint = pathlib.Path(__file__).resolve().with_name(
        "megatron_rocm_entrypoint.py"
    )
    if not rocm_entrypoint.is_file():
        raise ConfigError(f"missing ROCm compatibility entrypoint: {rocm_entrypoint}")
    return [
        python,
        "-m",
        "torch.distributed.run",
        "--nnodes", str(topo["nnodes"]),
        "--nproc-per-node", str(topo["gpus_per_node"]),
        "--node-rank", str(node_rank),
        "--master-addr", master_addr,
        "--master-port", str(master_port),
        str(rocm_entrypoint),
        "--upstream-entrypoint", str(pretrain),
        "--",
        *build_megatron_args(config, run_dir, data_cache_path),
    ]


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="validate and summarize a profile")
    inspect_parser.add_argument("--config", required=True)
    inspect_parser.add_argument("--run-dir", default="/tmp/megatron-smoke")
    inspect_parser.add_argument("--json", action="store_true")

    launch_parser = subparsers.add_parser("launch-node", help="launch one node's torchrun agent")
    launch_parser.add_argument("--config", required=True)
    launch_parser.add_argument("--megatron-dir", required=True)
    launch_parser.add_argument("--run-dir", required=True)
    launch_parser.add_argument("--data-cache-path")
    launch_parser.add_argument("--node-rank", required=True, type=int)
    launch_parser.add_argument("--master-addr", required=True)
    launch_parser.add_argument("--master-port", required=True, type=int)
    launch_parser.add_argument("--python", default=sys.executable)
    launch_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "inspect":
        summary = {
            "name": config["name"],
            "topology": topology(config),
            "estimated_dense_parameters": estimate_dense_parameters(config),
            "upstream_commit": config["upstream"]["commit"],
            "megatron_args": build_megatron_args(config, args.run_dir),
        }
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"profile: {summary['name']}")
            print(f"topology: {summary['topology']}")
            print(f"estimated dense parameters: {summary['estimated_dense_parameters']:,}")
            print(f"upstream: {summary['upstream_commit']}")
        return 0

    pathlib.Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    command = build_torchrun_command(
        config,
        python=args.python,
        megatron_dir=args.megatron_dir,
        run_dir=args.run_dir,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        data_cache_path=args.data_cache_path,
    )
    print("MEGATRON_TORCHRUN=" + json.dumps(command), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [args.megatron_dir, str(pathlib.Path(__file__).resolve().parents[1]), env.get("PYTHONPATH")])
    )
    os.chdir(args.megatron_dir)
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(_main())

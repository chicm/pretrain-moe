#!/usr/bin/env python3
"""Validate R-Full Gate 2 profiles and launch the pinned native Megatron path."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from collections.abc import Mapping
from typing import Any


PINNED_MEGATRON_COMMIT = "5cb6dbb3ed04e6fa11a862d90fe898ac2d3ddfad"
PROFILE_GEOMETRY = {
    "ep8-mini": {
        "num_layers": 4,
        "hidden_size": 512,
        "dense_ffn_hidden_size": 1408,
        "expert_ffn_hidden_size": 256,
        "num_attention_heads": 8,
        "num_query_groups": 2,
        "head_dim": 64,
        "moe_layer_frequency": [0, 0, 1, 1],
        "shared_expert_ffn_hidden_size": 256,
        "native_vocab_size": 4000,
        "padded_vocab_size": 4096,
        "make_vocab_size_divisible_by": 128,
        "expected_local_parameters": 19_371_008,
    },
    "production": {
        "num_layers": 48,
        "hidden_size": 2048,
        "dense_ffn_hidden_size": 5504,
        "expert_ffn_hidden_size": 896,
        "num_attention_heads": 32,
        "num_query_groups": 4,
        "head_dim": 128,
        "moe_layer_frequency": [0, 0] + [1] * 46,
        "shared_expert_ffn_hidden_size": 896,
        "native_vocab_size": 151_669,
        "padded_vocab_size": 151_936,
        "make_vocab_size_divisible_by": 1187,
        "expected_local_parameters": 4_586_027_008,
    },
}


class ConfigError(ValueError):
    pass


def validate_launch_environment(env: Mapping[str, str] | None = None) -> None:
    """Reject environment settings known to alter or bypass the qualified launch path."""
    values = os.environ if env is None else env
    distributed_debug = values.get("TORCH_DISTRIBUTED_DEBUG", "").strip().upper()
    if distributed_debug == "DETAIL":
        raise ConfigError(
            "TORCH_DISTRIBUTED_DEBUG=DETAIL is incompatible with the qualified "
            "distributed-optimizer path because it installs _ProcessGroupWrapper"
        )
    if values.get("EXTRA_MCORE_ARGS", "").strip():
        raise ConfigError(
            "EXTRA_MCORE_ARGS is not consumed; pass checkpoint and iteration options "
            "through the explicit launch-node arguments"
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with pathlib.Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    _require(config.get("schema_version") == 1, "unsupported schema_version")
    _require(
        config.get("upstream", {}).get("commit") == PINNED_MEGATRON_COMMIT,
        "unpinned Megatron commit",
    )
    profile = config.get("rfull_profile")
    _require(profile in PROFILE_GEOMETRY, f"unknown R-Full profile: {profile!r}")
    model = config.get("model", {})
    for key, expected in PROFILE_GEOMETRY[profile].items():
        _require(model.get(key) == expected, f"{profile} model.{key} drift")
    # Pinned NullTokenizer reserves one synthetic EOD id above --vocab-size.
    # Certify the model rows produced by tokenizer.py rather than assuming that
    # an already-divisible CLI value remains unchanged.
    null_vocab_size = model["native_vocab_size"] + 1
    multiple = model["make_vocab_size_divisible_by"]
    null_padded_vocab_size = ((null_vocab_size + multiple - 1) // multiple) * multiple
    _require(
        null_padded_vocab_size == model["padded_vocab_size"],
        "NullTokenizer synthetic EOD changes padded vocab size",
    )
    _require(model.get("num_experts") == 96, "R-Full requires 96 routed experts")
    _require(model.get("top_k") == 6, "R-Full requires Top-6")
    _require(model.get("sequence_length", 0) > 0, "sequence_length must be positive")
    _require(
        model["sequence_length"] <= model.get("max_position_embeddings", 0),
        "sequence_length exceeds max_position_embeddings",
    )
    if profile == "production":
        _require(
            model["max_position_embeddings"] == 4096,
            "production max_position_embeddings must be 4096",
        )

    cluster = config.get("cluster", {})
    parallel = config.get("parallel", {})
    _require(
        set(cluster) == {"nnodes", "gpus_per_node"},
        "cluster must declare exactly nnodes and gpus_per_node",
    )
    nnodes = cluster.get("nnodes")
    gpus_per_node = cluster.get("gpus_per_node")
    _require(isinstance(nnodes, int) and nnodes >= 1, "cluster.nnodes must be a positive int")
    _require(gpus_per_node == 8, "each node contributes exactly 8 GPUs")
    world_size = nnodes * gpus_per_node

    # EP is confined inside a node so that all-to-all stays on the intra-node
    # fabric; the remaining data parallelism (EDP) spans nodes.
    expected_parallel = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
    }
    _require(parallel == expected_parallel, "R-Full topology must be TP1/PP1/CP1/EP8/ETP1")
    _require(
        world_size % parallel["expert_model_parallel_size"] == 0,
        "world size must be divisible by expert_model_parallel_size",
    )
    _require(
        parallel["expert_model_parallel_size"] <= gpus_per_node,
        "expert parallel group must fit inside a single node",
    )

    training = config.get("training", {})
    for key in ("micro_batch_size", "global_batch_size", "train_iters"):
        _require(isinstance(training.get(key), int) and training[key] > 0, f"invalid training.{key}")
    _require(
        training["global_batch_size"] % world_size == 0,
        "global batch must be divisible by the dense data-parallel size",
    )

    runtime = config.get("runtime", {})
    _require(runtime.get("attention_backend") == "unfused", "Gate 2 baseline must use unfused attention")
    _require(
        (
            runtime.get("te_fused_attention"),
            runtime.get("te_flash_attention"),
            runtime.get("te_unfused_attention"),
        )
        == (False, False, True),
        "NVTE attention flags do not match unfused baseline",
    )
    for required_true in (
        "mock_data",
        "distributed_optimizer",
        "overlap_grad_reduce",
        "overlap_param_gather",
    ):
        _require(runtime.get(required_true) is True, f"runtime.{required_true} must be true")


def build_megatron_args(
    config: dict[str, Any],
    *,
    data_cache_path: str,
    train_iters: int | None = None,
    save_dir: str | None = None,
    load_dir: str | None = None,
    save_interval: int | None = None,
    exit_interval: int | None = None,
) -> list[str]:
    validate_config(config)
    model = config["model"]
    parallel = config["parallel"]
    training = config["training"]
    runtime = config["runtime"]
    iterations = train_iters or training["train_iters"]
    if not pathlib.PurePosixPath(data_cache_path).is_absolute():
        raise ConfigError("data_cache_path must be absolute")
    args = [
        "--transformer-impl", "transformer_engine",
        "--attention-backend", runtime["attention_backend"],
        "--rfull-profile", config["rfull_profile"],
        "--rfull-qualification-only",
        "--rfull-expected-local-parameters", str(model["expected_local_parameters"]),
        "--num-layers", str(model["num_layers"]),
        "--hidden-size", str(model["hidden_size"]),
        "--ffn-hidden-size", str(model["dense_ffn_hidden_size"]),
        "--moe-ffn-hidden-size", str(model["expert_ffn_hidden_size"]),
        "--num-attention-heads", str(model["num_attention_heads"]),
        "--group-query-attention",
        "--num-query-groups", str(model["num_query_groups"]),
        "--kv-channels", str(model["head_dim"]),
        "--seq-length", str(model["sequence_length"]),
        "--max-position-embeddings", str(model["max_position_embeddings"]),
        "--normalization", "RMSNorm",
        "--norm-epsilon", "1e-6",
        "--qk-layernorm",
        "--position-embedding-type", "rope",
        "--no-position-embedding",
        "--rotary-percent", "1.0",
        "--rotary-base", "10000",
        "--tokenizer-type", "NullTokenizer",
        "--vocab-size", str(model["native_vocab_size"]),
        "--make-vocab-size-divisible-by", str(model["make_vocab_size_divisible_by"]),
        "--split", "949,50,1",
        "--mock-data",
        "--data-cache-path", data_cache_path,
        "--swiglu",
        "--disable-bias-linear",
        "--num-experts", str(model["num_experts"]),
        "--moe-router-topk", str(model["top_k"]),
        "--moe-router-score-function", "softmax",
        "--moe-router-dtype", "fp32",
        "--moe-router-load-balancing-type", "aux_loss",
        "--moe-aux-loss-coeff", "0.001",
        "--moe-z-loss-coeff", "0.0001",
        "--moe-token-dispatcher-type", "alltoall",
        "--moe-grouped-gemm",
        "--moe-use-legacy-grouped-gemm",
        "--moe-shared-expert-intermediate-size", str(model["shared_expert_ffn_hidden_size"]),
        "--moe-layer-freq", json.dumps(model["moe_layer_frequency"], separators=(",", ":")),
        "--tensor-model-parallel-size", str(parallel["tensor_model_parallel_size"]),
        "--pipeline-model-parallel-size", str(parallel["pipeline_model_parallel_size"]),
        "--context-parallel-size", str(parallel["context_parallel_size"]),
        "--expert-model-parallel-size", str(parallel["expert_model_parallel_size"]),
        "--expert-tensor-parallel-size", str(parallel["expert_tensor_parallel_size"]),
        "--micro-batch-size", str(training["micro_batch_size"]),
        "--global-batch-size", str(training["global_batch_size"]),
        "--train-iters", str(iterations),
        "--lr", str(training["learning_rate"]),
        "--min-lr", str(training["min_learning_rate"]),
        "--lr-decay-style", "cosine",
        "--lr-decay-iters", str(iterations),
        "--lr-warmup-iters", "1",
        "--weight-decay", str(training["weight_decay"]),
        "--adam-beta1", str(training["adam_beta1"]),
        "--adam-beta2", str(training["adam_beta2"]),
        "--clip-grad", str(training["clip_grad"]),
        "--seed", str(training["seed"]),
        "--init-method-std", "0.02",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--bf16",
        "--use-distributed-optimizer",
        "--overlap-grad-reduce",
        "--overlap-param-gather",
        "--distributed-backend", "nccl",
        "--no-masked-softmax-fusion",
        "--no-bias-swiglu-fusion",
        "--no-bias-dropout-fusion",
        "--no-rope-fusion",
        "--log-interval", str(runtime["log_interval"]),
        "--eval-interval", str(max(1000, iterations + 1)),
        "--eval-iters", "1",
        "--ckpt-format", "torch_dist",
    ]
    if save_dir:
        if not save_interval or save_interval <= 0:
            raise ConfigError("save_interval must be positive when save_dir is set")
        args.extend(["--save", save_dir, "--save-interval", str(save_interval)])
    elif save_interval is not None:
        raise ConfigError("save_interval requires save_dir")
    if load_dir:
        args.extend(["--load", load_dir])
    if exit_interval is not None:
        if exit_interval <= 0 or exit_interval > iterations:
            raise ConfigError("exit_interval must be in [1, train_iters]")
        args.extend(["--exit-interval", str(exit_interval)])
    return args


def build_torchrun_command(
    config: dict[str, Any],
    *,
    python: str,
    project_dir: str,
    megatron_dir: str,
    master_addr: str,
    master_port: int,
    data_cache_path: str,
    node_rank: int = 0,
    train_iters: int | None = None,
    save_dir: str | None = None,
    load_dir: str | None = None,
    save_interval: int | None = None,
    exit_interval: int | None = None,
) -> list[str]:
    project = pathlib.Path(project_dir).resolve()
    megatron = pathlib.Path(megatron_dir).resolve()
    compatibility_entrypoint = project / "tools" / "rfull_rocm_entrypoint.py"
    training_entrypoint = project / "tools" / "pretrain_rfull_moe.py"
    if not compatibility_entrypoint.is_file() or not training_entrypoint.is_file():
        raise ConfigError("deployed R-Full entry points are missing")
    if not (megatron / "pretrain_gpt.py").is_file():
        raise ConfigError("pinned Megatron entry point is missing")
    nnodes = config["cluster"]["nnodes"]
    if not 0 <= node_rank < nnodes:
        raise ConfigError(f"node_rank {node_rank} outside [0, {nnodes})")
    if nnodes > 1 and master_addr in ("127.0.0.1", "localhost"):
        raise ConfigError("multi-node launch requires a routable master address")
    return [
        python,
        "-m", "torch.distributed.run",
        "--nnodes", str(nnodes),
        "--nproc-per-node", "8",
        "--node-rank", str(node_rank),
        "--master-addr", master_addr,
        "--master-port", str(master_port),
        str(compatibility_entrypoint),
        "--upstream-root", str(megatron),
        "--project-entrypoint", str(training_entrypoint),
        "--",
        *build_megatron_args(
            config,
            data_cache_path=data_cache_path,
            train_iters=train_iters,
            save_dir=save_dir,
            load_dir=load_dir,
            save_interval=save_interval,
            exit_interval=exit_interval,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--config", required=True)
    inspect_parser.add_argument("--data-cache-path", default="/tmp/rfull-cache")

    launch_parser = subparsers.add_parser("launch-node")
    launch_parser.add_argument("--config", required=True)
    launch_parser.add_argument("--project-dir", required=True)
    launch_parser.add_argument("--megatron-dir", required=True)
    launch_parser.add_argument("--master-addr", required=True)
    launch_parser.add_argument("--master-port", required=True, type=int)
    launch_parser.add_argument("--data-cache-path", required=True)
    launch_parser.add_argument("--python", default=sys.executable)
    launch_parser.add_argument("--node-rank", type=int, default=0)
    launch_parser.add_argument("--train-iters", type=int)
    launch_parser.add_argument("--save-dir")
    launch_parser.add_argument("--load-dir")
    launch_parser.add_argument("--save-interval", type=int)
    launch_parser.add_argument("--exit-interval", type=int)
    launch_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "inspect":
        print(
            json.dumps(
                {
                    "name": config["name"],
                    "profile": config["rfull_profile"],
                    "expected_local_parameters": config["model"]["expected_local_parameters"],
                    "megatron_args": build_megatron_args(
                        config, data_cache_path=args.data_cache_path
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    validate_launch_environment()
    command = build_torchrun_command(
        config,
        python=args.python,
        project_dir=args.project_dir,
        megatron_dir=args.megatron_dir,
        master_addr=args.master_addr,
        master_port=args.master_port,
        data_cache_path=args.data_cache_path,
        node_rank=args.node_rank,
        train_iters=args.train_iters,
        save_dir=args.save_dir,
        load_dir=args.load_dir,
        save_interval=args.save_interval,
        exit_interval=args.exit_interval,
    )
    print("RFULL_TORCHRUN=" + json.dumps(command), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [args.megatron_dir, args.project_dir, env.get("PYTHONPATH")])
    )
    return subprocess.call(command, cwd=args.megatron_dir, env=env)


if __name__ == "__main__":
    raise SystemExit(main())

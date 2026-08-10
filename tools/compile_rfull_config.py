#!/usr/bin/env python3
"""Validate R-Full v0.1 source config and emit four standalone stage artifacts.

This compiler intentionally permits TBD-BLOCKER values so the design can be audited,
but it marks every generated artifact launch_allowed=false until all blockers close.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError as exc:  # fail closed
    raise SystemExit("jsonschema is required: python -m pip install jsonschema") from exc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "configs/rfull/rfull_v0_1.source.json"
DEFAULT_SCHEMA = ROOT / "configs/rfull/rfull_v0_1.schema.json"
DEFAULT_OUTPUT = ROOT / "configs/rfull/generated"
SOURCE_IDS = ["dclm", "fineweb_edu", "finepdfs", "finephrase", "code", "finemath", "infimath", "owm"]
EXPECTED_STAGE_TOKENS = [800001884160, 150000107520, 40001863680, 9995550720]
EXPECTED_STAGE_IDS = ["4k", "8k", "16k", "32k"]
EXPECTED_SEQUENCE_LENGTHS = [4096, 8192, 16384, 32768]
EXPECTED_GLOBAL_SEQUENCE_BATCHES = [960, 480, 240, 120]
EXPECTED_GRADIENT_ACCUMULATION = [8, 4, 2, 1]
EXPECTED_STAGE_UPDATES = [203451, 38147, 10173, 2542]
EXPECTED_STAGE_SOURCE_SEQUENCES = [
    [69140788, 35156333, 9765648, 39062592, 29296944, 6093764, 4218760, 2578131],
    [6481938, 3295901, 915528, 3662112, 2746584, 571290, 395508, 241699],
    [864298, 439474, 122076, 488304, 366228, 76175, 52737, 32228],
    [107984, 54907, 15252, 61008, 45756, 9517, 6589, 4027],
]
EXPECTED_SOURCE_TOTAL_TOKENS = [353999781888, 179999895552, 49999970304, 199999881216, 149999910912, 31199969280, 21599993856, 13200003072]
EXPECTED_SOURCE_WEIGHTS = [3540, 1800, 500, 2000, 1500, 312, 216, 132]
EXPECTED_INDEXED_PAYLOAD_TOKENS = [320239478022, 376937913261, 70201495725, 101050225690, 39527688034, 19734484408, 7790815324, 2526172306]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_float=lambda text: (_ for _ in ()).throw(ValueError(f"floats forbidden in canonical config: {text}")))


def canonical_bytes(value: Any, *, omit_artifact_hash: bool = False) -> bytes:
    value = copy.deepcopy(value)
    if omit_artifact_hash and isinstance(value, dict):
        value.pop("artifact_sha256", None)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_object(value: Any, *, omit_artifact_hash: bool = False) -> str:
    return hashlib.sha256(canonical_bytes(value, omit_artifact_hash=omit_artifact_hash)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be inside repository: {path}") from exc


def blocker_paths(value: Any, prefix: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            result.extend(blocker_paths(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(blocker_paths(child, f"{prefix}[{index}]"))
    elif value == "TBD-BLOCKER":
        result.append(prefix)
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_invariants(cfg: dict[str, Any]) -> dict[str, Any]:
    run, model, moe = cfg["run"], cfg["model"], cfg["moe"]
    parallel, precision = cfg["parallel"], cfg["precision"]
    optimizer, data, stages = cfg["optimizer"], cfg["data"], cfg["stages"]

    require(run["failure_policy"] == "stop_replay" and run["max_committed_skips"] == 0, "committed numerical skips are forbidden")
    require(run == {
        "name": "rfull-25p857b-1t-v0.1", "seed": 1337,
        "successful_updates_target": 254313, "target_tokens_per_update": 3932160,
        "target_tokens_total": 999999406080, "failure_policy": "stop_replay",
        "max_committed_skips": 0,
    }, "run contract mismatch")
    require(model["tokenizer_native_vocab_size"] == 151669, "native vocab mismatch")
    require(model["padded_vocab_size"] == 151936 and model["explicit_padded_vocab_override"], "explicit model vocab override mismatch")
    require(model["padded_vocab_size"] % 128 == 0, "model vocab rows must be divisible by 128")
    require(((model["tokenizer_native_vocab_size"] + 127) // 128) * 128 == 151680, "default padding oracle changed")
    require(model["dense_layer_ids"] == [0, 1], "dense layer IDs mismatch")
    require(model["moe_layer_ids"] == list(range(2, 48)), "MoE layer IDs must be explicit 2..47")
    require(model["max_position_embeddings"] == 32768, "baseline position ceiling mismatch")
    require(model["limited_swiglu_threshold"] == 10, "limited-SwiGLU threshold mismatch")
    fixed_model = {
        "type": "gpt_rfull_moe", "hidden_size": 2048, "num_layers": 48,
        "num_attention_heads": 32, "num_query_groups": 4, "kv_channels": 128,
        "attention_type": "causal_full_gqa", "qk_rmsnorm": True,
        "qk_norm_scale_size": 128, "normalization": "rmsnorm",
        "norm_epsilon": "0.000001", "dense_ffn_hidden_size": 5504,
        "position_embedding": "rope", "rotary_percent": "1.0", "rotary_base": 1000000,
        "tie_word_embeddings": True, "add_bias_linear": False,
        "hidden_dropout": "0.0", "attention_dropout": "0.0", "mtp_layers": 0,
    }
    for key, expected in fixed_model.items():
        require(model[key] == expected, f"model.{key} mismatch")
    fixed_moe = {
        "num_routed_experts": 96, "top_k": 6, "expert_ffn_hidden_size": 896,
        "shared_experts": 1, "shared_expert_hidden_size": 896, "shared_expert_gate": False,
        "aux_aggregation": "autograd_aware", "aux_loss_granularity": "per_microstep_per_layer",
        "aux_loss_coefficient": "0.001", "z_loss_coefficient": "0.0001",
        "router_init_std": "0.01", "dropless": True, "capacity_factor": None,
        "token_drop": False, "token_dispatcher": "alltoall", "grouped_gemm": True,
        "shared_expert_overlap": False,
    }
    for key, expected in fixed_moe.items():
        require(moe[key] == expected, f"moe.{key} mismatch")
    require(moe["dispatch_width"] == model["hidden_size"] == 2048, "A2A dispatch width must equal d_model")
    require(moe["forward_gate"] == "topk_logits_then_selected_softmax", "forward routing mismatch")
    require(moe["aux_probability"] == "global_96way_softmax" and moe["aux_statistics_scope"] == "ep_group", "aux objective mismatch")
    require(moe["router_parameter_dtype"] == "bf16" and moe["router_compute_dtype"] == "fp32", "router dtype split mismatch")

    d, n_layers = model["hidden_size"], model["num_layers"]
    vocab = model["padded_vocab_size"]
    dense_layers, moe_layers = len(model["dense_layer_ids"]), len(model["moe_layer_ids"])
    dense_ffn, expert_ffn = model["dense_ffn_hidden_size"], moe["expert_ffn_hidden_size"]
    num_experts, top_k = moe["num_routed_experts"], moe["top_k"]
    embedding = vocab * d
    attention_per_layer = d * 4096 + d * 512 + d * 512 + 4096 * d
    attention = attention_per_layer * n_layers
    dense = dense_layers * 3 * d * dense_ffn
    routed_per_expert = 3 * d * expert_ffn
    routed = moe_layers * num_experts * routed_per_expert
    active_routed = moe_layers * top_k * routed_per_expert
    shared = moe_layers * 3 * d * moe["shared_expert_hidden_size"]
    routers = moe_layers * d * num_experts
    block_norms = n_layers * 2 * d
    final_norm = d
    qk_norms = n_layers * 2 * model["qk_norm_scale_size"]
    total = embedding + attention + dense + routed + shared + routers + block_norms + final_norm + qk_norms
    active = embedding + attention + dense + active_routed + shared + routers + block_norms + final_norm + qk_norms
    require(total == model["expected_total_parameters"] == 25857439744, f"total parameter mismatch: {total}")
    require(active == model["expected_active_parameters"] == 3066640384, f"active parameter mismatch: {active}")

    require(parallel == {
        "world_size": 120, "nodes": 15, "gpus_per_node": 8, "rank_order": "tp-cp-ep-dp-pp",
        "tensor_parallel": 1, "pipeline_parallel": 1, "context_parallel": 1, "expert_parallel": 8,
        "batch_data_parallel": 120, "expert_data_parallel": 15, "experts_per_rank_per_moe_layer": 12,
        "expert_parallel_node_local": True, "distributed_optimizer": True,
    }, "baseline topology mismatch")
    require(cfg["initialization"] == {
        "base_std": "0.02", "residual_output_std": "0.0020412414523193153", "norm_scale": "1.0",
    }, "initialization contract mismatch")
    require(precision == {
        "model_parameters": "bf16", "router_compute": "fp32", "loss_reduction": "fp32",
        "gradient_buffer": "fp32", "reduce_scatter_wire": "fp32",
        "parameter_all_gather_wire": "bf16", "optimizer_state": "fp32",
        "loss_scale": "none", "tf32": False, "fp8": False, "fp4": False,
    }, "precision contract mismatch")
    require(optimizer == {
        "name": "adamw", "beta1": "0.9", "beta2": "0.95", "epsilon": "0.00000001",
        "weight_decay": "0.1", "peak_lr": "0.0002", "min_lr": "0.00002",
        "warmup_updates": 2543, "stable_through_update": 228881,
        "cosine_decay_updates": 25432, "grad_clip": "1.0", "label_smoothing": "0.0",
        "schedule_version": "rfull-token-wsd-v1",
    }, "optimizer/LR contract mismatch")
    require(cfg["numerical"] == {
        "finite_consensus_scope": "world", "scan_forward": True, "scan_gradients": True,
        "scan_post_update_parameters": True, "scan_post_update_optimizer_state": True,
        "on_failure": "terminate_without_commit_and_replay",
        "large_finite_norm_action": "terminate_without_commit_and_replay",
    }, "numerical failure contract mismatch")
    communication = cfg["communication"]
    expected_forward_per_token = 2 * len(model["moe_layer_ids"]) * moe["top_k"] * model["hidden_size"] * communication["activation_dtype_bytes"]
    expected_remote_per_token = expected_forward_per_token * (parallel["expert_parallel"] - 1) // parallel["expert_parallel"]
    expected_forward_per_update = expected_remote_per_token * communication["rank_target_tokens_per_update"]
    require(communication == {
        "activation_dtype_bytes": 2,
        "a2a_forward_bytes_per_token": expected_forward_per_token,
        "a2a_ep8_remote_forward_bytes_per_token": expected_remote_per_token,
        "rank_target_tokens_per_update": 32768,
        "a2a_remote_forward_sent_bytes_per_rank_update": expected_forward_per_update,
        "a2a_remote_forward_backward_sent_bytes_per_rank_update": 2 * expected_forward_per_update,
        "a2a_nic_tx_rx_bytes_per_rank_update": 4 * expected_forward_per_update,
        "moe_recompute_enabled": False,
        "common_raw_fp32_gradient_bytes": 4 * (total - routed),
        "common_raw_bf16_parameter_bytes": 2 * (total - routed),
        "local_routed_raw_fp32_gradient_bytes": 4 * (routed // parallel["expert_parallel"]),
        "local_routed_raw_bf16_parameter_bytes": 2 * (routed // parallel["expert_parallel"]),
    }, "communication ledger mismatch")

    require([item["id"] for item in data["sources"]] == SOURCE_IDS, "source order mismatch")
    require([item["weight_numerator"] for item in data["sources"]] == EXPECTED_SOURCE_WEIGHTS, "source weights mismatch")
    require([item["indexed_payload_tokens"] for item in data["sources"]] == EXPECTED_INDEXED_PAYLOAD_TOKENS, "reported indexed inventory mismatch")
    require(sum(item["weight_numerator"] for item in data["sources"]) == 10000, "source weights do not sum to 10000")
    fixed_data = {
        "tokenizer_identifier": "Qwen/Qwen3-8B", "eot_id": 151643,
        "payload_dtype": "uint32-le", "add_special_tokens": False, "add_bos": False,
        "append_eos": False, "max_payload_id_exclusive": 151669,
        "window_algorithm": "physical-shard-aligned-stride-s-v1",
        "source_scheduler": "rfull-hash-affine-v1", "source_cycle_sequences": 10000,
    }
    for key, expected in fixed_data.items():
        require(data[key] == expected, f"data.{key} mismatch")
    holdout_contract = {
        "holdout_algorithm": "rfull-holdout-v1",
        "master_windows_per_source": 256,
        "holdout_candidate_selection": "token_disjoint_and_document_disjoint_greedy",
        "holdout_document_exclusion": "exclude_all_fragments_of_selected_doc_hashes",
        "holdout_insufficient_candidates": "fail_closed",
        "holdout_membership_unit": "physical_shard_document_sidecar",
    }
    for key, expected in holdout_contract.items():
        require(cfg["evaluation"][key] == expected, f"evaluation.{key} mismatch")
    require(cfg["launch"]["allow_unresolved_blockers"] is False, "unresolved blockers may never be bypassed")
    require(cfg["launch"]["long_context_continuation_enabled"] is False, "long-context continuation is outside baseline")

    total_updates = total_tokens = total_sequences = 0
    source_token_totals = [0] * len(SOURCE_IDS)
    stage_derived: list[dict[str, Any]] = []
    start_update = start_token = 0
    require(len(stages) == 4, "exactly four stages are required")
    for index, stage in enumerate(stages):
        seq = stage["sequence_length"]
        updates = stage["successful_updates"]
        global_sequences = stage["global_sequence_batch"]
        require(stage["id"] == EXPECTED_STAGE_IDS[index], f"stage {index} ID mismatch")
        require(seq == EXPECTED_SEQUENCE_LENGTHS[index], f"stage {stage['id']} sequence length mismatch")
        require(global_sequences == EXPECTED_GLOBAL_SEQUENCE_BATCHES[index], f"stage {stage['id']} global batch mismatch")
        require(stage["micro_batch_size"] == 1, f"stage {stage['id']} MBS mismatch")
        require(stage["gradient_accumulation"] == EXPECTED_GRADIENT_ACCUMULATION[index], f"stage {stage['id']} GA mismatch")
        require(updates == EXPECTED_STAGE_UPDATES[index], f"stage {stage['id']} update count mismatch")
        require(stage["source_sequences"] == EXPECTED_STAGE_SOURCE_SEQUENCES[index], f"stage {stage['id']} source quota mismatch")
        require(seq * global_sequences == run["target_tokens_per_update"], f"stage {stage['id']} tokens/update mismatch")
        require(global_sequences == parallel["batch_data_parallel"] * stage["micro_batch_size"] * stage["gradient_accumulation"], f"stage {stage['id']} batch arithmetic mismatch")
        require(sum(stage["source_sequences"]) == global_sequences * updates, f"stage {stage['id']} source-sequence total mismatch")
        stage_tokens = seq * global_sequences * updates
        require(stage_tokens == EXPECTED_STAGE_TOKENS[index], f"stage {stage['id']} target tokens mismatch")
        source_tokens = [count * seq for count in stage["source_sequences"]]
        source_token_totals = [a + b for a, b in zip(source_token_totals, source_tokens)]
        total_updates += updates
        total_tokens += stage_tokens
        total_sequences += sum(stage["source_sequences"])
        stage_derived.append({
            "stage_index": index,
            "start_successful_update_exclusive": start_update,
            "end_successful_update_inclusive": start_update + updates,
            "start_update_tokens": start_token,
            "end_update_tokens": start_token + stage_tokens,
            "stage_target_tokens": stage_tokens,
            "stage_sequences": sum(stage["source_sequences"]),
            "source_target_tokens": source_tokens,
            "local_target_tokens_per_rank_per_update": seq * stage["micro_batch_size"] * stage["gradient_accumulation"],
        })
        start_update += updates
        start_token += stage_tokens
    require(total_updates == run["successful_updates_target"] == 254313, "total updates mismatch")
    require(total_tokens == run["target_tokens_total"] == 999999406080, "total tokens mismatch")
    require(total_sequences == 216370080, "total sequences mismatch")
    require(source_token_totals == EXPECTED_SOURCE_TOTAL_TOKENS, f"source token totals mismatch: {source_token_totals}")
    require(cfg["checkpoint"] == {
        "format": "megatron_distributed_checkpoint", "schema_version": "rfull-dcp-v1",
        "interval_successful_updates": 2000, "rolling_full_keep": 3,
        "permanent_successful_updates": [2000, 203451, 241598, 251771, 254313],
        "weights_only_interval_successful_updates": 10000,
        "atomic_commit_marker": "COMMITTED", "final_dcp_replicas": 2,
        "commit_marker_schema_version": "rfull-commit-v1",
        "commit_marker_encoding": "canonical_json",
        "commit_marker_hash_algorithm": "sha256",
        "commit_marker_required_fields": [
            "schema_version", "run_name", "config_manifest_sha256", "stage_id",
            "stage_artifact_sha256", "successful_updates", "update_tokens",
            "root_manifest_path", "root_manifest_sha256", "parent_commit_sha256", "lineage_id",
        ],
        "committed_state_required": [
            "model", "optimizer", "scheduler", "rng_states", "successful_updates", "update_tokens",
            "attempted_batches_total_at_commit", "data_cursor", "source_cursor", "stage_id",
            "stage_local_successful_updates", "stage_artifact_sha256", "next_batch_digest",
            "data_algorithm_versions", "tokenizer_artifact_sha256", "corpus_manifest_sha256",
            "holdout_manifest_sha256", "stage_manifest_sha256", "resolved_arguments_sha256",
            "code_environment_backend_lock", "process_group_manifest", "ownership_map",
            "topology_fingerprint", "checkpoint_schema_version", "lineage_id", "parent_checkpoint_id",
        ],
        "raw_attempted_batches_counter": "append_only_outside_canonical_state_reconciled_on_restore",
        "restore_policy": "highest_valid_committed_update",
        "stage_resume_policy": "resume_incomplete_stage_without_advancing_stage",
        "root_manifest_schema_version": "rfull-root-manifest-v1",
        "latest_committed_pointer": "LATEST_COMMITTED",
        "latest_committed_pointer_schema_version": "rfull-latest-v1",
        "latest_pointer_update_policy": "atomic_replace_after_commit_marker",
    }, "checkpoint contract mismatch")
    storage = cfg["storage"]
    full14 = total * 14
    full16 = total * 16
    weights_only = total * 2
    artifact_peak = storage["retained_full_checkpoints"] * full14 + storage["retained_weights_only_artifacts"] * weights_only + storage["transient_full_checkpoint_equivalents"] * full14
    co_resident = artifact_peak + storage["indexed_corpus_payload_bytes"]
    reserve_denominator = 10000
    artifact_required = artifact_peak * (reserve_denominator + storage["reserve_basis_points"]) // reserve_denominator
    co_resident_required = co_resident * (reserve_denominator + storage["reserve_basis_points"]) // reserve_denominator
    require(storage == {
        "full_checkpoint_payload_bytes_14_per_parameter": full14,
        "full_checkpoint_payload_bytes_16_per_parameter": full16,
        "weights_only_payload_bytes": weights_only,
        "retained_full_checkpoints": 8, "retained_weights_only_artifacts": 30,
        "transient_full_checkpoint_equivalents": 2,
        "conservative_artifact_peak_bytes": artifact_peak,
        "indexed_corpus_payload_bytes": 3752033091080,
        "co_resident_peak_before_reserve_bytes": co_resident,
        "reserve_basis_points": 2000,
        "required_artifact_only_usable_bytes": artifact_required,
        "required_co_resident_usable_bytes": co_resident_required,
        "site_overhead_included": False,
    }, "storage lifetime ledger mismatch")
    qualification = cfg["qualification"]
    require(qualification["evidence_schema_version"] == "rfull-qualification-v1", "qualification schema mismatch")
    for key, value in qualification.items():
        if key == "evidence_schema_version":
            continue
        require(value == "TBD-BLOCKER" or re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"qualification.{key} must be TBD-BLOCKER or SHA-256")
    return {
        "parameter_ledger": {"total": total, "active": active, "embedding": embedding, "routed": routed},
        "total_sequences": total_sequences,
        "source_target_tokens": source_token_totals,
        "stages": stage_derived,
    }


def rank_groups() -> dict[str, Any]:
    return {
        "common_data_parallel": [list(range(120))],
        "expert_parallel": [list(range(base, base + 8)) for base in range(0, 120, 8)],
        "expert_data_parallel": [[offset + 8 * replica for replica in range(15)] for offset in range(8)],
        "rank_to_node_local_gpu": [{"rank": rank, "node": rank // 8, "local_gpu": rank % 8} for rank in range(120)],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cfg = read_json(args.source)
    schema = read_json(args.schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(cfg), key=lambda error: list(error.path))
    if errors:
        for error in errors:
            print(f"schema error at {list(error.path)}: {error.message}")
        raise SystemExit(2)
    derived = validate_invariants(cfg)
    blockers = blocker_paths(cfg)
    source_sha = sha256_object(cfg)
    schema_sha = sha256_object(schema)
    compiler_sha = sha256_file(Path(__file__))

    generated: list[dict[str, Any]] = []
    for stage, stage_derived in zip(cfg["stages"], derived["stages"]):
        standalone = copy.deepcopy(cfg)
        standalone.pop("stages")
        standalone["artifact_schema_version"] = "rfull-resolved-stage-v1"
        standalone["source_config_sha256"] = source_sha
        standalone["source_schema_sha256"] = schema_sha
        standalone["compiler_sha256"] = compiler_sha
        standalone["launch_allowed"] = not blockers
        standalone["unresolved_blockers"] = blockers
        standalone["stage"] = {**copy.deepcopy(stage), **stage_derived}
        standalone["stage"]["source_ids"] = SOURCE_IDS
        standalone["topology_groups"] = rank_groups()
        standalone["derived_accounting"] = copy.deepcopy(derived["parameter_ledger"])
        standalone["artifact_sha256"] = sha256_object(standalone, omit_artifact_hash=True)
        filename = f"stage_{stage['id']}.json"
        write_json(args.output / filename, standalone)
        generated.append({"id": stage["id"], "path": filename, "artifact_sha256": standalone["artifact_sha256"]})

    manifest = {
        "artifact_schema_version": "rfull-generated-manifest-v1",
        "source_config": repo_relative(args.source),
        "source_config_sha256": source_sha,
        "source_schema": repo_relative(args.schema),
        "source_schema_sha256": schema_sha,
        "compiler": repo_relative(Path(__file__)),
        "compiler_sha256": compiler_sha,
        "launch_allowed": not blockers,
        "unresolved_blockers": blockers,
        "total_successful_updates": cfg["run"]["successful_updates_target"],
        "total_target_tokens": cfg["run"]["target_tokens_total"],
        "stage_artifacts": generated,
    }
    manifest["artifact_sha256"] = sha256_object(manifest, omit_artifact_hash=True)
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({
        "source_config_sha256": source_sha,
        "source_schema_sha256": schema_sha,
        "compiler_sha256": compiler_sha,
        "manifest_sha256": manifest["artifact_sha256"],
        "launch_allowed": manifest["launch_allowed"],
        "unresolved_blockers": len(blockers),
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

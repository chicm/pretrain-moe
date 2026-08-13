"""Independent parameter accounting for the frozen R-Full architecture.

The calculator consumes the normative R-Full source-config shape directly.  It
counts unique parameter tensors: a tied embedding/LM-head matrix is counted
once, while an untied LM head is counted as a second matrix.  The active count
is the single-token structural parameter union; only routed experts differ
between the total and active ledgers.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rfull" / "rfull_v0_1.source.json"

FROZEN_TOTAL_PARAMETERS = 25_857_439_744
FROZEN_ACTIVE_PARAMETERS = 3_066_640_384
FROZEN_NORM_SUBTOTAL = 210_944


class LedgerValidationError(ValueError):
    """Raised when a config cannot be accounted for by this ledger."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LedgerValidationError(f"{path} must be an object")
    return value


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in config:
        raise LedgerValidationError(f"config is missing the {name!r} section")
    return _mapping(config[name], f"config.{name}")


def _integer(
    section: Mapping[str, Any],
    name: str,
    path: str,
    *,
    minimum: int = 1,
) -> int:
    if name not in section:
        raise LedgerValidationError(f"{path}.{name} is required")
    value = section[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise LedgerValidationError(f"{path}.{name} must be an integer")
    if value < minimum:
        raise LedgerValidationError(f"{path}.{name} must be >= {minimum}, got {value}")
    return value


def _boolean(section: Mapping[str, Any], name: str, path: str) -> bool:
    if name not in section:
        raise LedgerValidationError(f"{path}.{name} is required")
    value = section[name]
    if not isinstance(value, bool):
        raise LedgerValidationError(f"{path}.{name} must be a boolean")
    return value


def _layer_ids(model: Mapping[str, Any], name: str) -> tuple[int, ...]:
    if name not in model:
        raise LedgerValidationError(f"model.{name} is required")
    values = model[name]
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise LedgerValidationError(f"model.{name} must be an array of layer IDs")

    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise LedgerValidationError(f"model.{name}[{index}] must be an integer")
        result.append(value)
    return tuple(result)


def load_source_config(path: str | Path = SOURCE_CONFIG_PATH) -> dict[str, Any]:
    """Load an R-Full source JSON document without consulting generated configs."""

    source_path = Path(path)
    try:
        with source_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerValidationError(f"cannot load R-Full source config {source_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise LedgerValidationError(f"R-Full source config {source_path} must contain a JSON object")
    return config


def validate_layer_partition(model: Mapping[str, Any]) -> None:
    """Validate that dense and MoE IDs assign every transformer layer once."""

    model = _mapping(model, "model")
    num_layers = _integer(model, "num_layers", "model")
    dense_ids = _layer_ids(model, "dense_layer_ids")
    moe_ids = _layer_ids(model, "moe_layer_ids")

    dense_counts = Counter(dense_ids)
    moe_counts = Counter(moe_ids)
    duplicate_dense = sorted(layer for layer, count in dense_counts.items() if count > 1)
    duplicate_moe = sorted(layer for layer, count in moe_counts.items() if count > 1)

    expected = set(range(num_layers))
    dense_set = set(dense_ids)
    moe_set = set(moe_ids)
    assigned = dense_set | moe_set

    problems: list[str] = []
    if duplicate_dense:
        problems.append(f"duplicate dense IDs={duplicate_dense}")
    if duplicate_moe:
        problems.append(f"duplicate MoE IDs={duplicate_moe}")
    overlap = sorted(dense_set & moe_set)
    if overlap:
        problems.append(f"overlap={overlap}")
    missing = sorted(expected - assigned)
    if missing:
        problems.append(f"missing={missing}")
    out_of_range = sorted(assigned - expected)
    if out_of_range:
        problems.append(f"out-of-range={out_of_range}")

    if problems:
        detail = "; ".join(problems)
        raise LedgerValidationError(
            "model.dense_layer_ids and model.moe_layer_ids must partition "
            f"all {num_layers} layers exactly once: {detail}"
        )


def validate_hidden_size_alignment(
    model: Mapping[str, Any],
    moe: Mapping[str, Any],
) -> None:
    """Validate activation and Q/K scale widths against model hidden geometry."""

    model = _mapping(model, "model")
    moe = _mapping(moe, "moe")
    hidden_size = _integer(model, "hidden_size", "model")
    dispatch_width = _integer(moe, "dispatch_width", "moe")
    if dispatch_width != hidden_size:
        raise LedgerValidationError(
            "moe.dispatch_width must match model.hidden_size: "
            f"{dispatch_width} != {hidden_size}"
        )

    kv_channels = _integer(model, "kv_channels", "model")
    qk_norm_scale_size = _integer(model, "qk_norm_scale_size", "model")
    qk_rmsnorm = _boolean(model, "qk_rmsnorm", "model")
    if qk_rmsnorm and qk_norm_scale_size != kv_channels:
        raise LedgerValidationError(
            "model.qk_norm_scale_size must match model.kv_channels when QK-Norm "
            f"is enabled: {qk_norm_scale_size} != {kv_channels}"
        )


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the tensor-shape invariants used by the parameter formulas."""

    config = _mapping(config, "config")
    model = _section(config, "model")
    moe = _section(config, "moe")

    native_vocab = _integer(model, "tokenizer_native_vocab_size", "model")
    padded_vocab = _integer(model, "padded_vocab_size", "model")
    if padded_vocab < native_vocab:
        raise LedgerValidationError(
            "model.padded_vocab_size cannot be smaller than "
            f"model.tokenizer_native_vocab_size: {padded_vocab} < {native_vocab}"
        )

    _integer(model, "hidden_size", "model")
    _integer(model, "num_layers", "model")
    num_attention_heads = _integer(model, "num_attention_heads", "model")
    num_query_groups = _integer(model, "num_query_groups", "model")
    _integer(model, "kv_channels", "model")
    _integer(model, "qk_norm_scale_size", "model")
    _integer(model, "dense_ffn_hidden_size", "model")
    _boolean(model, "qk_rmsnorm", "model")
    _boolean(model, "tie_word_embeddings", "model")

    if num_query_groups > num_attention_heads:
        raise LedgerValidationError(
            "model.num_query_groups cannot exceed model.num_attention_heads: "
            f"{num_query_groups} > {num_attention_heads}"
        )
    if num_attention_heads % num_query_groups:
        raise LedgerValidationError(
            "model.num_attention_heads must be divisible by model.num_query_groups: "
            f"{num_attention_heads} % {num_query_groups} != 0"
        )

    if model.get("normalization") != "rmsnorm":
        raise LedgerValidationError("model.normalization must be 'rmsnorm' for this ledger")
    if _boolean(model, "add_bias_linear", "model"):
        raise LedgerValidationError("model.add_bias_linear must be false; linear biases are not in the ledger")
    if _integer(model, "mtp_layers", "model", minimum=0) != 0:
        raise LedgerValidationError("model.mtp_layers must be zero; MTP tensors are not in the ledger")

    num_routed_experts = _integer(moe, "num_routed_experts", "moe")
    top_k = _integer(moe, "top_k", "moe")
    if top_k > num_routed_experts:
        raise LedgerValidationError(
            f"moe.top_k cannot exceed moe.num_routed_experts: {top_k} > {num_routed_experts}"
        )
    _integer(moe, "expert_ffn_hidden_size", "moe")
    _integer(moe, "shared_experts", "moe", minimum=0)
    _integer(moe, "shared_expert_hidden_size", "moe")
    if _boolean(moe, "shared_expert_gate", "moe"):
        raise LedgerValidationError(
            "moe.shared_expert_gate must be false; an experimental gate is not in the frozen ledger"
        )

    validate_layer_partition(model)
    validate_hidden_size_alignment(model, moe)


def _component_subtotals(components: Mapping[str, int]) -> dict[str, int]:
    return {
        "embedding_and_lm_head": components["embedding"] + components["lm_head"],
        "attention": (
            components["attention_q_projection"]
            + components["attention_k_projection"]
            + components["attention_v_projection"]
            + components["attention_output_projection"]
        ),
        "dense_ffn": components["dense_ffn"],
        "routed_experts": components["routed_experts"],
        "shared_experts": components["shared_experts"],
        "routers": components["routers"],
        "norms": components["block_norms"] + components["final_norm"] + components["qk_norms"],
    }


def calculate_parameter_ledger(config: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate detailed total and single-token-active parameter ledgers.

    ``model.padded_vocab_size`` is the parameter-matrix row count.  The query,
    key, value, and output projection widths are derived solely from attention
    heads, query groups, and ``kv_channels``; no R-Full projection width is
    embedded in the formulas.
    """

    validate_config(config)
    model = _section(config, "model")
    moe = _section(config, "moe")

    hidden_size = model["hidden_size"]
    num_layers = model["num_layers"]
    dense_ids = tuple(model["dense_layer_ids"])
    moe_ids = tuple(model["moe_layer_ids"])
    dense_layers = len(dense_ids)
    moe_layers = len(moe_ids)

    vocab_size = model["padded_vocab_size"]
    num_attention_heads = model["num_attention_heads"]
    num_query_groups = model["num_query_groups"]
    kv_channels = model["kv_channels"]
    query_width = num_attention_heads * kv_channels
    key_width = num_query_groups * kv_channels
    value_width = num_query_groups * kv_channels

    embedding = vocab_size * hidden_size
    lm_head = 0 if model["tie_word_embeddings"] else vocab_size * hidden_size

    attention_q = num_layers * hidden_size * query_width
    attention_k = num_layers * hidden_size * key_width
    attention_v = num_layers * hidden_size * value_width
    attention_output = num_layers * query_width * hidden_size

    dense_ffn = dense_layers * 3 * hidden_size * model["dense_ffn_hidden_size"]
    routed_expert_size = 3 * hidden_size * moe["expert_ffn_hidden_size"]
    shared_expert_size = 3 * hidden_size * moe["shared_expert_hidden_size"]
    routed_experts = moe_layers * moe["num_routed_experts"] * routed_expert_size
    active_routed_experts = moe_layers * moe["top_k"] * routed_expert_size
    shared_experts = moe_layers * moe["shared_experts"] * shared_expert_size
    routers = moe_layers * hidden_size * moe["num_routed_experts"]

    block_norms = num_layers * 2 * hidden_size
    final_norm = hidden_size
    qk_norms = (
        num_layers * 2 * model["qk_norm_scale_size"] if model["qk_rmsnorm"] else 0
    )

    total_components = {
        "embedding": embedding,
        "lm_head": lm_head,
        "attention_q_projection": attention_q,
        "attention_k_projection": attention_k,
        "attention_v_projection": attention_v,
        "attention_output_projection": attention_output,
        "dense_ffn": dense_ffn,
        "routed_experts": routed_experts,
        "shared_experts": shared_experts,
        "routers": routers,
        "block_norms": block_norms,
        "final_norm": final_norm,
        "qk_norms": qk_norms,
    }
    active_components = dict(total_components)
    active_components["routed_experts"] = active_routed_experts

    total_subtotals = _component_subtotals(total_components)
    active_subtotals = _component_subtotals(active_components)
    norm_subtotal = total_subtotals["norms"]

    return {
        "dimensions": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dense_layers": dense_layers,
            "moe_layers": moe_layers,
            "num_attention_heads": num_attention_heads,
            "num_query_groups": num_query_groups,
            "kv_channels": kv_channels,
            "query_projection_width": query_width,
            "key_projection_width": key_width,
            "value_projection_width": value_width,
            "dense_ffn_hidden_size": model["dense_ffn_hidden_size"],
            "expert_ffn_hidden_size": moe["expert_ffn_hidden_size"],
            "shared_expert_hidden_size": moe["shared_expert_hidden_size"],
            "num_routed_experts": moe["num_routed_experts"],
            "top_k": moe["top_k"],
            "shared_experts": moe["shared_experts"],
        },
        "layer_ids": {
            "dense": list(dense_ids),
            "moe": list(moe_ids),
        },
        "total_components": total_components,
        "active_components": active_components,
        "total_subtotals": total_subtotals,
        "active_subtotals": active_subtotals,
        "norm_subtotal": norm_subtotal,
        "total": sum(total_components.values()),
        "active": sum(active_components.values()),
    }


def frozen_rfull_parameter_ledger(
    path: str | Path = SOURCE_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the normative source and verify its frozen golden accounting."""

    config = load_source_config(path)
    ledger = calculate_parameter_ledger(config)
    model = _section(config, "model")
    expected_total = _integer(model, "expected_total_parameters", "model", minimum=0)
    expected_active = _integer(model, "expected_active_parameters", "model", minimum=0)

    if expected_total != FROZEN_TOTAL_PARAMETERS:
        raise LedgerValidationError(
            "model.expected_total_parameters does not match the frozen R-Full golden: "
            f"{expected_total} != {FROZEN_TOTAL_PARAMETERS}"
        )
    if expected_active != FROZEN_ACTIVE_PARAMETERS:
        raise LedgerValidationError(
            "model.expected_active_parameters does not match the frozen R-Full golden: "
            f"{expected_active} != {FROZEN_ACTIVE_PARAMETERS}"
        )
    if ledger["total"] != expected_total:
        raise LedgerValidationError(
            f"calculated total parameters {ledger['total']} != expected {expected_total}"
        )
    if ledger["active"] != expected_active:
        raise LedgerValidationError(
            f"calculated active parameters {ledger['active']} != expected {expected_active}"
        )
    if ledger["norm_subtotal"] != FROZEN_NORM_SUBTOTAL:
        raise LedgerValidationError(
            "calculated norm subtotal does not match the frozen R-Full golden: "
            f"{ledger['norm_subtotal']} != {FROZEN_NORM_SUBTOTAL}"
        )
    return ledger


__all__ = [
    "FROZEN_ACTIVE_PARAMETERS",
    "FROZEN_NORM_SUBTOTAL",
    "FROZEN_TOTAL_PARAMETERS",
    "LedgerValidationError",
    "SOURCE_CONFIG_PATH",
    "calculate_parameter_ledger",
    "frozen_rfull_parameter_ledger",
    "load_source_config",
    "validate_config",
    "validate_hidden_size_alignment",
    "validate_layer_partition",
]

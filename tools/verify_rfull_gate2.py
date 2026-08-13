#!/usr/bin/env python3
"""Fail-closed verifier for an immutable single-node R-Full Gate 2 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import re
import tempfile
from collections import defaultdict
from typing import Any

from rfull_moe.pinned_mcore import PINNED_SOURCE_SHA256
from tools.rfull_gate2 import load_config


ITERATION_RE = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
LOSS_RE = re.compile(
    r"lm loss:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
SKIPPED_ITERATIONS_RE = re.compile(r"number of skipped iterations:\s*(\d+)", re.IGNORECASE)
NAN_ITERATIONS_RE = re.compile(r"number of nan iterations:\s*(\d+)", re.IGNORECASE)
UNFUSED_ATTENTION_RE = re.compile(
    r"^\s*attention_backend\s+\.*\s+AttnBackend\.unfused\s*$", re.MULTILINE
)
LOADED_CHECKPOINT_RE = re.compile(
    r"successfully loaded checkpoint from ([^ \r\n]+) \[[^\r\n]*\] at iteration (\d+)",
    re.IGNORECASE,
)
FATAL_PATTERNS = {
    "traceback": re.compile(r"Traceback \(most recent call last\):"),
    "runtime_error": re.compile(r"\bRuntimeError:"),
    "child_failed": re.compile(r"ChildFailedError|ProcessExitedException"),
    "collective_error": re.compile(r"\b(?:NCCL|RCCL) (?:error|failure)", re.IGNORECASE),
    "segfault": re.compile(r"Segmentation fault", re.IGNORECASE),
    "oom": re.compile(r"(?:CUDA|HIP|GPU).*out of memory", re.IGNORECASE),
    "non_finite": re.compile(
        r"(?:lm loss:\s*[+-]?(?:nan|inf)\b|found NaN)", re.IGNORECASE
    ),
}
_NUMPY_COMPAT_SOURCE_SHA256 = {
    "megatron/core/dist_checkpointing/exchange_utils.py": (
        "7ca890a9c9eb686faf56f7ead9777cf31d56f38519373c3d21f1042037d99e49"
    ),
    "megatron/core/dist_checkpointing/mapping.py": (
        "7360a2af2edb3679570d7664cf9d8f46a4adc10e907c5b677535d6e0cc5f9b70"
    ),
    "megatron/core/dist_checkpointing/validation.py": (
        "a1adb86344c18be1f8cc9e2e320f0b6dcde92e1586a00157f1ff61a78362d3c7"
    ),
}
_DCP_COMPAT_SOURCE_SHA256 = {
    "megatron/core/dist_checkpointing/strategies/filesystem_async.py": (
        "1d410495a6a634722671c4bafdf82ad420bde8b4a56908578565ea1d12c7dbeb"
    ),
    "torch.distributed.checkpoint.filesystem": (
        "a3fe232efd14b6c47b553dcb913ae275541e09371500279e1e67fd63eedcce81"
    ),
}
_DCP_MCORE_METADATA_SOURCE_SHA256 = {
    "megatron/core/dist_checkpointing/strategies/torch.py": (
        "a47209ce93367031adfebe3410f0923f352c3c5bd0596a805212b6063672135b"
    ),
    "torch.distributed.checkpoint.filesystem": (
        "a3fe232efd14b6c47b553dcb913ae275541e09371500279e1e67fd63eedcce81"
    ),
    "torch.distributed.checkpoint.metadata": (
        "2a23bd4bfb7442ce203d2e40346298ef2313ab59a0fbfcc8b05ff9482bbc99ca"
    ),
}


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _json_markers(text: str) -> dict[str, list[dict[str, Any]]]:
    by_marker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # Rank output can concatenate objects with no newline. JSONDecoder preserves
    # nested dictionaries (for source-hash evidence), unlike a flat-object regex.
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            payload, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = start + length
        if not isinstance(payload, dict):
            continue
        marker = payload.get("marker")
        if isinstance(marker, str):
            by_marker[marker].append(payload)
    return dict(by_marker)


def _trainer_argument_values(text: str, name: str) -> list[str]:
    pattern = re.compile(
        rf"^[ \t]*{re.escape(name)}[ \t]+\.+[ \t]+(.+?)[ \t]*$",
        re.MULTILINE,
    )
    return sorted(set(pattern.findall(text)))


def _verify_resume_contract(
    text: str,
    expected_load_dir: pathlib.Path | str,
    expected_save_dir: pathlib.Path | str,
    expected_loaded_iteration: int,
) -> dict[str, Any]:
    expected_load = str(expected_load_dir)
    expected_save = str(expected_save_dir)
    load_values = _trainer_argument_values(text, "load")
    save_values = _trainer_argument_values(text, "save")
    if load_values != [expected_load]:
        raise AssertionError(
            f"trainer load argument mismatch: observed={load_values}, expected={[expected_load]}"
        )
    if save_values != [expected_save]:
        raise AssertionError(
            f"trainer save argument mismatch: observed={save_values}, expected={[expected_save]}"
        )
    loaded_markers = sorted(
        {(path, int(iteration)) for path, iteration in LOADED_CHECKPOINT_RE.findall(text)}
    )
    expected_marker = [(expected_load, expected_loaded_iteration)]
    if loaded_markers != expected_marker:
        raise AssertionError(
            f"checkpoint load marker mismatch: observed={loaded_markers}, "
            f"expected={expected_marker}"
        )
    return {
        "load_dir": expected_load,
        "save_dir": expected_save,
        "loaded_iteration": expected_loaded_iteration,
    }


def _ranks(markers: dict[str, list[dict[str, Any]]], name: str) -> list[int]:
    return sorted(
        {
            int(item["rank"])
            for item in markers.get(name, [])
            if isinstance(item.get("rank"), int)
        }
    )


def _require_mcore_metadata_preservation(
    markers: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    preserved = markers.get("DCP_MCORE_METADATA_PRESERVED", [])
    if not preserved or _ranks(markers, "DCP_MCORE_METADATA_PRESERVED") != [0]:
        raise AssertionError(f"invalid MCore metadata preservation ranks: {preserved}")
    for payload in preserved:
        if not isinstance(payload.get("entries"), int) or payload["entries"] <= 0:
            raise AssertionError(
                f"invalid MCore metadata preservation evidence: {preserved}"
            )
    return preserved


def _require_rank_marker(
    markers: dict[str, list[dict[str, Any]]],
    name: str,
    expected_ranks: list[int],
    expected_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    payloads = markers.get(name, [])
    ranks = _ranks(markers, name)
    if ranks != expected_ranks or len(payloads) != len(expected_ranks):
        raise AssertionError(
            f"{name} ranks/count mismatch: ranks={ranks}, count={len(payloads)}, "
            f"expected={expected_ranks}"
        )
    for payload in payloads:
        for field, expected in (expected_fields or {}).items():
            if payload.get(field) != expected:
                raise AssertionError(
                    f"{name} field drift for rank {payload.get('rank')}: "
                    f"{field}={payload.get(field)!r}, expected {expected!r}"
                )
    return payloads


def _verify_resume_cpu_rng_guard(
    markers: dict[str, list[dict[str, Any]]],
    expected_ranks: list[int],
    loaded_iteration: int,
) -> list[dict[str, Any]]:
    payloads = _require_rank_marker(
        markers,
        "RFULL_RESUME_CPU_RNG_GUARD",
        expected_ranks,
        {
            "is_resume": True,
            "loaded_iteration": loaded_iteration,
            "restored": True,
            "builder_changed_cpu_rng": True,
        },
    )
    for payload in payloads:
        if payload.get("before_sha256") != payload.get("after_guard_sha256"):
            raise AssertionError(f"resume CPU RNG was not restored exactly: {payload}")
        if payload.get("before_sha256") == payload.get("after_build_sha256"):
            raise AssertionError(f"resume iterator builder did not consume CPU RNG: {payload}")
        changed_count = payload.get("changed_byte_count")
        if not isinstance(changed_count, int) or changed_count <= 0:
            raise AssertionError(f"invalid resume CPU RNG change count: {payload}")
    return sorted(payloads, key=lambda item: item["rank"])


def _verify_checkpoint(
    checkpoint_dir: pathlib.Path,
    checkpoint_manifest: pathlib.Path,
    expected_iteration: int,
    expected_world_size: int,
    *,
    require_mcore_data: bool = False,
) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != str(
        expected_iteration
    ):
        raise AssertionError(f"invalid checkpoint tracker for iteration {expected_iteration}")
    iteration_dir = checkpoint_dir / f"iter_{expected_iteration:07d}"
    if not iteration_dir.is_dir():
        raise AssertionError(f"checkpoint iteration directory is missing: {iteration_dir}")
    metadata_json_path = iteration_dir / "metadata.json"
    expected_metadata_json = {
        "sharded_backend": "torch_dist",
        "sharded_backend_version": 1,
        "common_backend": "torch",
        "common_backend_version": 1,
    }
    if json.loads(metadata_json_path.read_text(encoding="utf-8")) != expected_metadata_json:
        raise AssertionError("distributed checkpoint metadata.json drift")

    temporary_files = [
        path
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
        and ("tmp" in path.name.lower() or path.name.endswith((".partial", ".incomplete")))
    ]
    if temporary_files:
        raise AssertionError(f"temporary checkpoint files remain: {temporary_files}")
    distcp_files = sorted(iteration_dir.glob("__*_*.distcp"))
    if len(distcp_files) != expected_world_size * 2:
        raise AssertionError(
            f"expected {expected_world_size * 2} distcp shards, found {len(distcp_files)}"
        )
    required_files = [iteration_dir / ".metadata", iteration_dir / "common.pt"]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required_files + distcp_files):
        raise AssertionError("checkpoint has a missing or empty required file")

    if not checkpoint_manifest.is_file():
        raise AssertionError(f"checkpoint hash manifest is missing: {checkpoint_manifest}")
    expected_hashes: dict[str, str] = {}
    for line in checkpoint_manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise AssertionError(f"malformed checkpoint digest: {line}")
        candidate = (checkpoint_dir / relative).resolve()
        if candidate != checkpoint_dir and checkpoint_dir not in candidate.parents:
            raise AssertionError(f"checkpoint manifest path escapes root: {relative}")
        expected_hashes[relative] = digest
    actual_files = sorted(path for path in checkpoint_dir.rglob("*") if path.is_file())
    actual_relatives = {"./" + path.relative_to(checkpoint_dir).as_posix() for path in actual_files}
    if set(expected_hashes) != actual_relatives:
        raise AssertionError(
            "checkpoint manifest file set drift: "
            f"manifest={sorted(expected_hashes)}, actual={sorted(actual_relatives)}"
        )
    for relative, expected_digest in expected_hashes.items():
        observed_digest = _sha256(checkpoint_dir / relative)
        if observed_digest != expected_digest:
            raise AssertionError(
                f"checkpoint hash mismatch for {relative}: "
                f"expected {expected_digest}, observed {observed_digest}"
            )

    import torch
    from torch.distributed.checkpoint import FileSystemReader

    common = torch.load(
        iteration_dir / "common.pt", map_location="cpu", weights_only=False
    )
    if not isinstance(common, dict) or common.get("iteration") != expected_iteration:
        raise AssertionError("common checkpoint iteration mismatch")
    if not isinstance(common.get("optimizer"), dict) or not common["optimizer"]:
        raise AssertionError("common checkpoint optimizer state is missing")
    if not isinstance(common.get("opt_param_scheduler"), dict):
        raise AssertionError("common checkpoint scheduler state is missing")
    metadata = FileSystemReader(iteration_dir).read_metadata()
    state_keys = sorted(metadata.state_dict_metadata)
    required_key_predicates = {
        "model": lambda key: key.startswith(("embedding.", "decoder.")),
        "optimizer": lambda key: key.startswith("chained_0.optimizer."),
        "rng": lambda key: key.startswith("rng_state/"),
    }
    missing_categories = [
        name
        for name, predicate in required_key_predicates.items()
        if not any(predicate(key) for key in state_keys)
    ]
    if missing_categories:
        raise AssertionError(f"checkpoint metadata lacks categories: {missing_categories}")
    if not metadata.storage_data:
        raise AssertionError("checkpoint storage metadata is empty")
    mcore_data = getattr(metadata, "mcore_data", None)
    if require_mcore_data and (not isinstance(mcore_data, dict) or not mcore_data):
        raise AssertionError("checkpoint .metadata lacks non-empty MCore reformulation data")

    return {
        "path": str(checkpoint_dir),
        "iteration": expected_iteration,
        "files": len(actual_files),
        "bytes": sum(path.stat().st_size for path in actual_files),
        "distcp_shards": len(distcp_files),
        "state_dict_metadata_entries": len(state_keys),
        "storage_data_entries": len(metadata.storage_data),
        "mcore_data_entries": len(mcore_data) if isinstance(mcore_data, dict) else 0,
        "hash_manifest": str(checkpoint_manifest.resolve()),
        "hash_manifest_sha256": _sha256(checkpoint_manifest),
    }


NODE_START_RE = re.compile(r"RFULL_NODE_START=(\S+)\s+host=(\S+)\s+run_dir=(\S+)(?:\s+node_rank=(\d+))?")


def _node_rank_from_log(text: str, *, nnodes: int) -> int:
    match = NODE_START_RE.search(text)
    if match is None:
        raise AssertionError("training log does not declare RFULL_NODE_START")
    raw = match.group(4)
    if raw is None:
        # Sealed single-node Gate 2 evidence predates the node_rank field. Only a
        # single-node contract may infer rank 0; multi-node must be explicit.
        if nnodes != 1:
            raise AssertionError(
                "multi-node log must declare RFULL_NODE_START node_rank explicitly"
            )
        return 0
    return int(raw)


def _node_host_from_log(text: str) -> str:
    match = NODE_START_RE.search(text)
    if match is None:
        raise AssertionError("training log does not declare RFULL_NODE_START host")
    return match.group(2)


def _verify_telemetry(run_dir: pathlib.Path, *, expected_active: int = 8) -> dict[str, Any]:
    status_path = run_dir / "gpu.telemetry.status.log"
    csv_path = run_dir / "gpu.telemetry.csv"
    if not status_path.is_file() or not csv_path.is_file():
        raise AssertionError("GPU telemetry status or CSV file is missing")
    status = status_path.read_text(encoding="utf-8", errors="replace")
    if status.count("GPU_TELEMETRY_START,") != 1:
        raise AssertionError("expected exactly one GPU_TELEMETRY_START marker")
    completes = [line for line in status.splitlines() if line.startswith("GPU_TELEMETRY_COMPLETE,")]
    if len(completes) != 1 or not completes[0].endswith("rc=0"):
        raise AssertionError(f"invalid GPU telemetry completion: {completes}")

    maxima: dict[str, dict[str, float | int]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            device = row.get("device", "").strip()
            if not device:
                continue
            try:
                gpu_use = float(row["GPU use (%)"])
                vram_use = float(row["GPU Memory Allocated (VRAM%)"])
            except (KeyError, TypeError, ValueError) as error:
                raise AssertionError(f"malformed GPU telemetry row: {row}") from error
            entry = maxima.setdefault(
                device,
                {"samples": 0, "max_gpu_use_percent": 0.0, "max_vram_percent": 0.0},
            )
            entry["samples"] = int(entry["samples"]) + 1
            entry["max_gpu_use_percent"] = max(
                float(entry["max_gpu_use_percent"]), gpu_use
            )
            entry["max_vram_percent"] = max(float(entry["max_vram_percent"]), vram_use)
    active = sorted(
        device
        for device, values in maxima.items()
        if float(values["max_gpu_use_percent"]) > 0.0
        and float(values["max_vram_percent"]) > 0.0
    )
    if len(active) != expected_active:
        raise AssertionError(
            f"telemetry showed active GPUs={active}, expected {expected_active}: {maxima}"
        )
    return {"active_devices": active, "devices": maxima}


def verify(
    run_dir: pathlib.Path,
    config_path: pathlib.Path,
    *,
    additional_node_run_dirs: list[pathlib.Path] | None = None,
    expected_final_iteration: int | None = None,
    expected_first_iteration: int = 1,
    expect_training_complete: bool = True,
    expected_resume_load_dir: pathlib.Path | None = None,
    expected_resume_save_dir: pathlib.Path | None = None,
    expected_loaded_iteration: int | None = None,
    require_numpy_product_compat: bool = False,
    require_dcp_write_item_compat: bool = False,
    require_dcp_mcore_metadata_compat: bool = False,
    require_dcp_mcore_metadata_fallback: bool = False,
    require_resume_cpu_rng_guard: bool = False,
    require_batch_fingerprints: bool = False,
    checkpoint_dir: pathlib.Path | None = None,
    checkpoint_manifest: pathlib.Path | None = None,
    expected_checkpoint_iteration: int | None = None,
    require_checkpoint_mcore_data: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    expected_final = expected_final_iteration or config["training"]["train_iters"]
    nnodes = config["cluster"]["nnodes"]
    gpus_per_node = config["cluster"]["gpus_per_node"]
    world_size = nnodes * gpus_per_node
    node_run_dirs = [run_dir] + list(additional_node_run_dirs or [])
    if len(node_run_dirs) != nnodes:
        raise AssertionError(
            f"config declares {nnodes} nodes but {len(node_run_dirs)} run dirs were supplied"
        )
    if len({d.resolve() for d in node_run_dirs}) != nnodes:
        raise AssertionError("node run dirs must be distinct paths")

    node_logs: list[str] = []
    node_reports: list[dict[str, Any]] = []
    for index, node_dir in enumerate(node_run_dirs):
        node_log_path = node_dir / "train.console.log"
        if not node_log_path.is_file():
            raise AssertionError(f"missing training log for node {index}: {node_log_path}")
        node_text = node_log_path.read_text(encoding="utf-8", errors="replace")
        if "RFULL_NODE_COMPLETE=" not in node_text or " rc=0" not in node_text:
            raise AssertionError(f"node {index} did not record a clean rc=0 completion")
        observed_rank = _node_rank_from_log(node_text, nnodes=nnodes)
        if observed_rank != index:
            raise AssertionError(
                f"run dir {node_dir} reports node_rank={observed_rank}, expected {index}"
            )
        node_logs.append(node_text)
        node_reports.append(
            {
                "node_rank": index,
                "run_dir": str(node_dir),
                "host": _node_host_from_log(node_text),
                "telemetry": _verify_telemetry(node_dir, expected_active=gpus_per_node),
            }
        )
    # Rank markers are global, so the union of node logs is the authoritative stream.
    text = "\n".join(node_logs)
    log_path = run_dir / "train.console.log"

    fatal_hits: dict[str, list[str]] = {}
    for name, pattern in FATAL_PATTERNS.items():
        hits = [line[:500] for line in text.splitlines() if pattern.search(line)]
        if hits:
            fatal_hits[name] = hits[:10]
    if fatal_hits:
        raise AssertionError(f"fatal log signatures found: {fatal_hits}")

    resume_fields = (
        expected_resume_load_dir,
        expected_resume_save_dir,
        expected_loaded_iteration,
    )
    if any(value is not None for value in resume_fields) and not all(
        value is not None for value in resume_fields
    ):
        raise AssertionError(
            "resume verification requires load dir, save dir, and loaded iteration together"
        )
    resume_contract = None
    if all(value is not None for value in resume_fields):
        assert expected_resume_load_dir is not None
        assert expected_resume_save_dir is not None
        assert expected_loaded_iteration is not None
        resume_contract = _verify_resume_contract(
            text,
            expected_resume_load_dir,
            expected_resume_save_dir,
            expected_loaded_iteration,
        )

    markers = _json_markers(text)
    expected_ranks = list(range(world_size))
    required_rank_markers = [
        "EARLY_DEVICE_BIND",
        "PROCESS_GROUP_DEVICE_BIND",
        "ROCM_LEGACY_FUSED_KERNEL_LOADER_SKIPPED",
        "RFULL_MODEL_BUILT",
        "RFULL_GROUPED_GEMM_FORWARD",
        "RFULL_EP_GLOBAL_AUX_LOSS",
        "RFULL_EP_GLOBAL_Z_LOSS",
    ]
    if expect_training_complete:
        required_rank_markers.append("RFULL_TRAINING_COMPLETE")
    marker_ranks: dict[str, list[int]] = {}
    for name in required_rank_markers:
        observed = _ranks(markers, name)
        marker_ranks[name] = observed
        if observed != expected_ranks:
            raise AssertionError(f"{name} ranks={observed}, expected={expected_ranks}")

    if require_numpy_product_compat:
        _require_rank_marker(
            markers,
            "NUMPY_PRODUCT_COMPAT_READY",
            expected_ranks,
            {
                "mode": "alias_to_prod",
                "numpy_version": "2.2.6",
                "source_sha256": _NUMPY_COMPAT_SOURCE_SHA256,
            },
        )
        marker_ranks["NUMPY_PRODUCT_COMPAT_READY"] = _ranks(
            markers, "NUMPY_PRODUCT_COMPAT_READY"
        )
    if require_dcp_write_item_compat:
        _require_rank_marker(
            markers,
            "DCP_WRITE_ITEM_COMPAT_READY",
            expected_ranks,
            {
                "mode": "append_torch_save_serialization_format",
                "torch_version": "2.10.0.dev20251112+rocm7.1",
                "serialization_format": "torch_save",
                "source_sha256": _DCP_COMPAT_SOURCE_SHA256,
            },
        )
        marker_ranks["DCP_WRITE_ITEM_COMPAT_READY"] = _ranks(
            markers, "DCP_WRITE_ITEM_COMPAT_READY"
        )
    if require_dcp_mcore_metadata_compat:
        _require_rank_marker(
            markers,
            "DCP_MCORE_METADATA_COMPAT_READY",
            expected_ranks,
            {
                "save_mode": "preserve_mcore_data_after_dataclasses_replace",
                "load_mode": "infer_same_geometry_if_absent",
                "source_sha256": _DCP_MCORE_METADATA_SOURCE_SHA256,
            },
        )
        marker_ranks["DCP_MCORE_METADATA_COMPAT_READY"] = _ranks(
            markers, "DCP_MCORE_METADATA_COMPAT_READY"
        )
    if require_dcp_mcore_metadata_fallback:
        fallback_payloads = _require_rank_marker(
            markers,
            "DCP_MCORE_METADATA_FALLBACK_APPLIED",
            expected_ranks,
            {},
        )
        for payload in fallback_payloads:
            if not isinstance(payload.get("entries"), int) or payload["entries"] <= 0:
                raise AssertionError(f"invalid MCore metadata fallback entry count: {payload}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("shape_sha256", ""))):
                raise AssertionError(f"invalid MCore metadata fallback shape hash: {payload}")
        marker_ranks["DCP_MCORE_METADATA_FALLBACK_APPLIED"] = _ranks(
            markers, "DCP_MCORE_METADATA_FALLBACK_APPLIED"
        )
    if require_checkpoint_mcore_data:
        _require_mcore_metadata_preservation(markers)
        marker_ranks["DCP_MCORE_METADATA_PRESERVED"] = [0]
    resume_cpu_rng_guard = None
    if require_resume_cpu_rng_guard:
        if expected_loaded_iteration is None:
            raise AssertionError(
                "resume CPU RNG guard verification requires expected loaded iteration"
            )
        resume_cpu_rng_guard = _verify_resume_cpu_rng_guard(
            markers, expected_ranks, expected_loaded_iteration
        )
        marker_ranks["RFULL_RESUME_CPU_RNG_GUARD"] = _ranks(
            markers, "RFULL_RESUME_CPU_RNG_GUARD"
        )

    expected_local_parameters = config["model"]["expected_local_parameters"]
    for item in markers["RFULL_MODEL_BUILT"]:
        if item.get("local_parameters") != expected_local_parameters:
            raise AssertionError(f"local parameter count drift in marker: {item}")
        if item.get("trainable_parameters") != expected_local_parameters:
            raise AssertionError(f"non-trainable R-Full parameters found: {item}")
        expected_source_count = len(PINNED_SOURCE_SHA256)
        if item.get("source_guard_file_count") != expected_source_count:
            raise AssertionError(
                f"source guard did not verify all {expected_source_count} files: {item}"
            )
        if item.get("source_guard_megatron_commit") != config["upstream"]["commit"]:
            raise AssertionError(f"source guard commit drift: {item}")

    grouped = markers["RFULL_GROUPED_GEMM_FORWARD"]
    for item in grouped:
        if item.get("num_local_experts") != 12:
            raise AssertionError(f"expected 12 local experts per EP rank: {item}")
        if item.get("assigned_tokens", 0) <= 0:
            raise AssertionError(f"grouped GEMM received no assignments: {item}")
        if item.get("hidden_size") != config["model"]["hidden_size"]:
            raise AssertionError(f"grouped GEMM hidden size drift: {item}")
    zero_token_expert_ranks = sorted(
        int(item["rank"]) for item in grouped if item.get("zero_token_experts", 0) > 0
    )
    if config["rfull_profile"] == "ep8-mini" and not zero_token_expert_ranks:
        raise AssertionError("mini qualification did not exercise any zero-token local expert")

    for marker_name, metric_name in (
        ("RFULL_EP_GLOBAL_AUX_LOSS", "raw_aux_loss"),
        ("RFULL_EP_GLOBAL_Z_LOSS", "raw_z_loss"),
    ):
        for item in markers[marker_name]:
            value = item.get(metric_name)
            if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
                raise AssertionError(f"non-finite {metric_name}: {item}")
            if item.get("ep_world_size") != config["parallel"]["expert_model_parallel_size"]:
                raise AssertionError(f"EP-global tracker world size drift: {item}")
            if item.get("tracker_group") != "expert_parallel_avg":
                raise AssertionError(f"tracker is not using EP avg metadata: {item}")

    iterations: list[int] = []
    losses: list[float] = []
    for line in text.splitlines():
        iteration_match = ITERATION_RE.search(line)
        loss_match = LOSS_RE.search(line)
        if iteration_match and loss_match:
            iterations.append(int(iteration_match.group(1)))
            loss = float(loss_match.group(1))
            if not math.isfinite(loss):
                raise AssertionError(f"non-finite loss at iteration {iterations[-1]}: {loss}")
            skipped_match = SKIPPED_ITERATIONS_RE.search(line)
            nan_match = NAN_ITERATIONS_RE.search(line)
            if skipped_match is None or nan_match is None:
                raise AssertionError(
                    f"iteration {iterations[-1]} is missing skipped/nan counters: {line[:500]}"
                )
            skipped = int(skipped_match.group(1))
            nan_iterations = int(nan_match.group(1))
            if skipped != 0 or nan_iterations != 0:
                raise AssertionError(
                    f"iteration {iterations[-1]} has skipped={skipped}, "
                    f"nan_iterations={nan_iterations}"
                )
            losses.append(loss)
    expected_iterations = list(range(expected_first_iteration, expected_final + 1))
    unique_iterations = sorted(set(iterations))
    if unique_iterations != expected_iterations:
        raise AssertionError(
            f"metric iterations={unique_iterations}, expected={expected_iterations}"
        )
    batch_fingerprints: list[dict[str, Any]] = []
    training_batch_fingerprints: list[dict[str, Any]] = []
    additional_batch_fingerprints: list[dict[str, Any]] = []
    if require_batch_fingerprints:
        batch_fingerprints = markers.get("RFULL_BATCH_FINGERPRINT", [])
        parallel = config["parallel"]
        dense_data_parallel_size = world_size // (
            parallel["tensor_model_parallel_size"]
            * parallel["pipeline_model_parallel_size"]
            * parallel["context_parallel_size"]
        )
        micro_batch_size = config["training"]["micro_batch_size"]
        denominator = micro_batch_size * dense_data_parallel_size
        global_batch_size = config["training"]["global_batch_size"]
        if global_batch_size % denominator:
            raise AssertionError(
                "global batch is not divisible by micro batch times data parallel"
            )
        expected_microbatches = global_batch_size // denominator
        expected_batch_coordinates = {
            (rank, iteration, microbatch)
            for rank in expected_ranks
            for iteration in expected_iterations
            for microbatch in range(expected_microbatches)
        }
        observed_batch_coordinates: set[tuple[int, int, int]] = set()
        for item in batch_fingerprints:
            coordinate = (
                item.get("rank"),
                item.get("iteration"),
                item.get("microbatch"),
            )
            if not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in coordinate
            ):
                raise AssertionError(f"invalid batch fingerprint coordinate: {item}")
            rank, iteration, microbatch = coordinate
            if rank not in expected_ranks:
                raise AssertionError(f"invalid batch fingerprint rank: {item}")
            if iteration not in expected_iterations:
                raise AssertionError(f"batch fingerprint iteration outside phase: {item}")
            if microbatch < 0:
                raise AssertionError(f"invalid batch fingerprint microbatch: {item}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
                raise AssertionError(f"invalid batch fingerprint digest: {item}")
            if coordinate in observed_batch_coordinates:
                raise AssertionError(f"duplicate batch fingerprint coordinate: {coordinate}")
            observed_batch_coordinates.add(coordinate)
            if coordinate in expected_batch_coordinates:
                training_batch_fingerprints.append(item)
            else:
                # Megatron runs validation/test forward steps after the final training
                # update. The runtime counter continues above the training microbatch
                # range, so retain those fingerprints without treating them as a
                # replacement for any required training coordinate.
                additional_batch_fingerprints.append(item)
        missing_batch_coordinates = expected_batch_coordinates - observed_batch_coordinates
        if missing_batch_coordinates:
            raise AssertionError(
                "missing training batch fingerprint coordinates: "
                f"{sorted(missing_batch_coordinates)}"
            )

    completions = markers.get("RFULL_TRAINING_COMPLETE", [])
    if expect_training_complete:
        for item in completions:
            if item.get("iteration") != expected_final:
                raise AssertionError(f"training completion iteration drift: {item}")
            expected_samples = expected_final * config["training"]["global_batch_size"]
            if item.get("consumed_train_samples") != expected_samples:
                raise AssertionError(f"consumed sample counter drift: {item}")
            if item.get("consumed_train_tokens") != expected_samples * config["model"]["sequence_length"]:
                raise AssertionError(f"consumed token counter drift: {item}")
    elif completions or "RFULL_TRAINING_COMPLETE" in text:
        raise AssertionError("interval-exit phase unexpectedly emitted training completion")

    if UNFUSED_ATTENTION_RE.search(text) is None:
        raise AssertionError("resolved runtime args did not confirm AttnBackend.unfused")
    if "aiter rope backend is enabled, which has lower precision" in text.lower():
        raise AssertionError("lower-precision AIter RoPE backend was enabled")

    telemetry = node_reports[0]["telemetry"]
    checkpoint = None
    checkpoint_arguments = (
        checkpoint_dir,
        checkpoint_manifest,
        expected_checkpoint_iteration,
    )
    if require_checkpoint_mcore_data and not all(
        value is not None for value in checkpoint_arguments
    ):
        raise AssertionError("checkpoint MCore data requirement needs checkpoint arguments")
    if any(value is not None for value in checkpoint_arguments):
        if not all(value is not None for value in checkpoint_arguments):
            raise AssertionError(
                "checkpoint_dir, checkpoint_manifest, and expected_checkpoint_iteration "
                "must be provided together"
            )
        checkpoint = _verify_checkpoint(
            checkpoint_dir,
            checkpoint_manifest,
            expected_checkpoint_iteration,
            expected_world_size=world_size,
            require_mcore_data=require_checkpoint_mcore_data,
        )

    return {
        "schema_version": 1,
        "status": "PASS",
        "profile": config["name"],
        "upstream_commit": config["upstream"]["commit"],
        "world_size": world_size,
        "nnodes": nnodes,
        "gpus_per_node": gpus_per_node,
        "nodes": node_reports,
        "marker_ranks": marker_ranks,
        "zero_token_expert_ranks": zero_token_expert_ranks,
        "expected_local_parameters": expected_local_parameters,
        "iterations": unique_iterations,
        "losses": losses,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "expect_training_complete": expect_training_complete,
        "batch_fingerprints": sorted(
            batch_fingerprints,
            key=lambda item: (item["rank"], item["iteration"], item["microbatch"]),
        ),
        "training_batch_fingerprints": sorted(
            training_batch_fingerprints,
            key=lambda item: (item["rank"], item["iteration"], item["microbatch"]),
        ),
        "additional_batch_fingerprints": sorted(
            additional_batch_fingerprints,
            key=lambda item: (item["rank"], item["iteration"], item["microbatch"]),
        ),
        "consumed_train_samples": expected_final * config["training"]["global_batch_size"],
        "consumed_train_tokens": expected_final
        * config["training"]["global_batch_size"]
        * config["model"]["sequence_length"],
        "checkpoint": checkpoint,
        "resume_contract": resume_contract,
        "resume_cpu_rng_guard": resume_cpu_rng_guard,
        "telemetry": telemetry,
    }


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--additional-node-run-dir",
        action="append",
        default=[],
        help="run dir for node ranks 1..N-1, in ascending node-rank order",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-final-iteration", type=int)
    parser.add_argument("--expected-first-iteration", type=int, default=1)
    parser.add_argument("--expect-no-training-complete", action="store_true")
    parser.add_argument("--expected-resume-load-dir")
    parser.add_argument("--expected-resume-save-dir")
    parser.add_argument("--expected-loaded-iteration", type=int)
    parser.add_argument("--require-numpy-product-compat", action="store_true")
    parser.add_argument("--require-dcp-write-item-compat", action="store_true")
    parser.add_argument("--require-dcp-mcore-metadata-compat", action="store_true")
    parser.add_argument("--require-dcp-mcore-metadata-fallback", action="store_true")
    parser.add_argument("--require-resume-cpu-rng-guard", action="store_true")
    parser.add_argument("--require-batch-fingerprints", action="store_true")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-manifest")
    parser.add_argument("--expected-checkpoint-iteration", type=int)
    parser.add_argument("--require-checkpoint-mcore-data", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    run_dir = pathlib.Path(args.run_dir)
    config_path = pathlib.Path(args.config)
    summary = verify(
        run_dir,
        config_path,
        additional_node_run_dirs=[pathlib.Path(p) for p in args.additional_node_run_dir],
        expected_final_iteration=args.expected_final_iteration,
        expected_first_iteration=args.expected_first_iteration,
        expect_training_complete=not args.expect_no_training_complete,
        expected_resume_load_dir=(
            pathlib.Path(args.expected_resume_load_dir)
            if args.expected_resume_load_dir
            else None
        ),
        expected_resume_save_dir=(
            pathlib.Path(args.expected_resume_save_dir)
            if args.expected_resume_save_dir
            else None
        ),
        expected_loaded_iteration=args.expected_loaded_iteration,
        require_numpy_product_compat=args.require_numpy_product_compat,
        require_dcp_write_item_compat=args.require_dcp_write_item_compat,
        require_dcp_mcore_metadata_compat=args.require_dcp_mcore_metadata_compat,
        require_dcp_mcore_metadata_fallback=args.require_dcp_mcore_metadata_fallback,
        require_resume_cpu_rng_guard=args.require_resume_cpu_rng_guard,
        require_batch_fingerprints=args.require_batch_fingerprints,
        checkpoint_dir=(
            pathlib.Path(args.checkpoint_dir) if args.checkpoint_dir else None
        ),
        checkpoint_manifest=(
            pathlib.Path(args.checkpoint_manifest) if args.checkpoint_manifest else None
        ),
        expected_checkpoint_iteration=args.expected_checkpoint_iteration,
        require_checkpoint_mcore_data=args.require_checkpoint_mcore_data,
    )
    if args.output:
        output = pathlib.Path(args.output)
        verifier_path = pathlib.Path(__file__).resolve()
        deployment_manifest = verifier_path.parents[1] / "deployment.archive.sha256"
        summary["verifier"] = {
            "path": str(verifier_path),
            "sha256": _sha256(verifier_path),
        }
        if deployment_manifest.is_file():
            summary["verifier"]["deployment_archive_manifest"] = str(deployment_manifest)
            summary["verifier"]["deployment_archive_sha256"] = deployment_manifest.read_text(
                encoding="utf-8"
            ).split()[0]
        _atomic_json(output, summary)
        manifest = run_dir / "verification.evidence.sha256"
        evidence = [
            run_dir / "train.console.log",
            run_dir / "gpu.telemetry.csv",
            run_dir / "gpu.telemetry.status.log",
            output,
            config_path,
            verifier_path,
        ]
        controller_status = run_dir / "controller.status.log"
        if controller_status.is_file():
            evidence.append(controller_status)
        if args.checkpoint_manifest:
            evidence.append(pathlib.Path(args.checkpoint_manifest))
        if deployment_manifest.is_file():
            evidence.append(deployment_manifest)
        manifest.write_text(
            "".join(f"{_sha256(path)}  {path}\n" for path in evidence),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Parse smoke logs and enforce the rank/progress/finite-loss acceptance gate."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import pathlib
import re
import tempfile
from typing import Any


ITERATION_RE = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
LOSS_RE = re.compile(r"lm loss:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
PROBE_RANK_RE = re.compile(
    r'"marker"\s*:\s*"DISTRIBUTED_PROBE_RANK_OK".*?"rank"\s*:\s*(\d+)'
)
PROBE_WORLD_RE = re.compile(r'"marker"\s*:\s*"DISTRIBUTED_PROBE_WORLD_OK"')
NODE_COMPLETE_RE = re.compile(r'"marker"\s*:\s*"NODE_RUN_COMPLETE"')
GPU_TELEMETRY_COMPLETE_RE = re.compile(
    r'"marker"\s*:\s*"GPU_TELEMETRY_COMPLETE"'
)
ROCM_SHIM_RE = re.compile(
    r'"marker"\s*:\s*"ROCM_LEGACY_FUSED_KERNEL_LOADER_SKIPPED"'
)
# torchrun can concatenate complete JSON records from concurrent ranks onto one
# physical log line. Stay within one JSON object and use non-greedy fields so
# every rank marker is captured rather than only the last object on that line.
_JSON_FIELD = r'[^{}\n]*?'
EARLY_DEVICE_RE = re.compile(
    rf'\{{{_JSON_FIELD}"marker"\s*:\s*"EARLY_DEVICE_BIND"'
    rf'{_JSON_FIELD}"rank"\s*:\s*(\d+){_JSON_FIELD}\}}'
)
PROCESS_GROUP_DEVICE_RE = re.compile(
    rf'\{{{_JSON_FIELD}"marker"\s*:\s*"PROCESS_GROUP_DEVICE_BIND"'
    rf'{_JSON_FIELD}"rank"\s*:\s*(\d+){_JSON_FIELD}\}}'
)
PROBE_DEVICE_RE = re.compile(
    rf'\{{{_JSON_FIELD}"marker"\s*:\s*"DISTRIBUTED_PROBE_DEVICE_BIND"'
    rf'{_JSON_FIELD}"rank"\s*:\s*(\d+){_JSON_FIELD}\}}'
)
FATAL_PATTERNS = {
    "traceback": re.compile(r"Traceback \(most recent call last\):"),
    "runtime_error": re.compile(r"\bRuntimeError:"),
    "child_failed": re.compile(r"ChildFailedError|ProcessExitedException"),
    "collective_error": re.compile(r"\b(?:NCCL|RCCL) (?:error|failure)", re.IGNORECASE),
    "segfault": re.compile(r"Segmentation fault", re.IGNORECASE),
    "oom": re.compile(r"(?:CUDA|HIP|GPU).*out of memory", re.IGNORECASE),
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


def _verify_gpu_telemetry(
    run_dir: pathlib.Path, nnodes: int, active_gpus_per_node: int
) -> dict[str, dict[str, dict[str, float | int]]]:
    telemetry_summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for node_rank in range(nnodes):
        path = run_dir / f"gpu-node-{node_rank}.csv"
        if not path.is_file():
            raise AssertionError(f"missing GPU telemetry file: {path}")
        maxima: dict[str, dict[str, float | int]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                device = row.get("device", "").strip()
                if not device:
                    continue
                try:
                    gpu_use = float(row["GPU use (%)"])
                    vram_use = float(row["GPU Memory Allocated (VRAM%)"])
                except (KeyError, TypeError, ValueError) as error:
                    raise AssertionError(f"malformed GPU telemetry row in {path}: {row}") from error
                entry = maxima.setdefault(
                    device, {"samples": 0, "max_gpu_use_percent": 0.0, "max_vram_percent": 0.0}
                )
                entry["samples"] = int(entry["samples"]) + 1
                entry["max_gpu_use_percent"] = max(
                    float(entry["max_gpu_use_percent"]), gpu_use
                )
                entry["max_vram_percent"] = max(float(entry["max_vram_percent"]), vram_use)
        active_devices = [
            device
            for device, values in maxima.items()
            if float(values["max_vram_percent"]) > 0
            and float(values["max_gpu_use_percent"]) > 0
        ]
        if len(active_devices) < active_gpus_per_node:
            raise AssertionError(
                f"node {node_rank} showed only {len(active_devices)} active GPUs in telemetry; "
                f"expected at least {active_gpus_per_node}: {maxima}"
            )
        telemetry_summary[f"node-{node_rank}"] = maxima
    return telemetry_summary


def verify(run_dir: pathlib.Path, config_path: pathlib.Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    world_size = config["cluster"]["nnodes"] * config["cluster"]["gpus_per_node"]
    expected_iters = config["training"]["train_iters"]
    node_logs = sorted(glob.glob(str(run_dir / "node-*.log")))
    if len(node_logs) != config["cluster"]["nnodes"]:
        raise AssertionError(
            f"expected {config['cluster']['nnodes']} node logs, found {len(node_logs)}: {node_logs}"
        )

    text_by_log: dict[str, str] = {}
    for log in node_logs:
        text_by_log[log] = pathlib.Path(log).read_text(encoding="utf-8", errors="replace")
    combined = "\n".join(text_by_log.values())

    probe_ranks = sorted({int(match.group(1)) for match in PROBE_RANK_RE.finditer(combined)})
    expected_ranks = list(range(world_size))
    if probe_ranks != expected_ranks:
        raise AssertionError(f"distributed probe ranks={probe_ranks}, expected={expected_ranks}")
    if len(PROBE_WORLD_RE.findall(combined)) != 1:
        raise AssertionError("expected exactly one world-level distributed probe marker")
    if len(NODE_COMPLETE_RE.findall(combined)) != config["cluster"]["nnodes"]:
        raise AssertionError("not every node reached NODE_RUN_COMPLETE")
    if len(GPU_TELEMETRY_COMPLETE_RE.findall(combined)) != config["cluster"]["nnodes"]:
        raise AssertionError("not every node completed GPU telemetry")
    shim_markers = len(ROCM_SHIM_RE.findall(combined))
    early_device_ranks = sorted({int(value) for value in EARLY_DEVICE_RE.findall(combined)})
    if early_device_ranks != expected_ranks:
        raise AssertionError(
            f"early device binding ranks={early_device_ranks}, expected={expected_ranks}"
        )
    process_group_device_ranks = sorted(
        {int(value) for value in PROCESS_GROUP_DEVICE_RE.findall(combined)}
    )
    if process_group_device_ranks != expected_ranks:
        raise AssertionError(
            f"process-group device binding ranks={process_group_device_ranks}, "
            f"expected={expected_ranks}"
        )
    probe_device_ranks = sorted({int(value) for value in PROBE_DEVICE_RE.findall(combined)})
    if probe_device_ranks != expected_ranks:
        raise AssertionError(
            f"probe device binding ranks={probe_device_ranks}, expected={expected_ranks}"
        )
    if config.get("runtime", {}).get("transformer_impl") == "transformer_engine":
        if shim_markers != world_size:
            raise AssertionError(
                f"ROCm compatibility shim markers={shim_markers}, expected={world_size}"
            )
        if "aiter rope backend is enabled, which has lower precision" in combined.lower():
            raise AssertionError("lower-precision AIter RoPE backend was not disabled")
        runtime = config["runtime"]
        for config_name, env_name in (
            ("te_fused_attention", "NVTE_FUSED_ATTN"),
            ("te_flash_attention", "NVTE_FLASH_ATTN"),
            ("te_unfused_attention", "NVTE_UNFUSED_ATTN"),
        ):
            expected_value = "1" if runtime[config_name] else "0"
            if combined.count(f'"{env_name}": "{expected_value}"') < config["cluster"]["nnodes"]:
                raise AssertionError(
                    f"node preflight did not confirm the requested {env_name} value"
                )
        if not runtime["te_fused_attention"] and (
            "Disabling FusedAttention due to NVTE_FUSED_ATTN=0" not in combined
        ):
            raise AssertionError("Transformer Engine did not log that FusedAttention was disabled")
        expected_selection = {
            "flash": "Selected backend = FlashAttention",
            "fused": "Selected backend = FusedAttention",
            "unfused": "Selected backend = UnfusedDotProductAttention",
        }.get(runtime["attention_backend"])
        if expected_selection and expected_selection not in combined:
            raise AssertionError(
                f"Transformer Engine did not confirm {runtime['attention_backend']} backend selection"
            )

    iterations: list[int] = []
    losses: list[float] = []
    for line in combined.splitlines():
        iteration_match = ITERATION_RE.search(line)
        loss_match = LOSS_RE.search(line)
        if iteration_match and loss_match:
            iterations.append(int(iteration_match.group(1)))
            loss = float(loss_match.group(1))
            if not math.isfinite(loss):
                raise AssertionError(f"non-finite loss at iteration {iterations[-1]}: {loss}")
            losses.append(loss)
    if not iterations:
        raise AssertionError("no Megatron metric rows with lm loss were found")
    if max(iterations) < expected_iters:
        raise AssertionError(f"last metric iteration={max(iterations)}, expected={expected_iters}")

    fatal_hits: dict[str, list[str]] = {}
    for name, pattern in FATAL_PATTERNS.items():
        hits = [line[:500] for line in combined.splitlines() if pattern.search(line)]
        if hits:
            fatal_hits[name] = hits[:10]
    if fatal_hits:
        raise AssertionError(f"fatal log signatures found: {fatal_hits}")

    gpu_telemetry = _verify_gpu_telemetry(
        run_dir,
        config["cluster"]["nnodes"],
        config["cluster"]["gpus_per_node"],
    )

    return {
        "schema_version": 1,
        "status": "PASS",
        "profile": config["name"],
        "upstream_commit": config["upstream"]["commit"],
        "world_size": world_size,
        "probe_ranks": probe_ranks,
        "early_device_binding_ranks": early_device_ranks,
        "process_group_device_binding_ranks": process_group_device_ranks,
        "probe_device_binding_ranks": probe_device_ranks,
        "rocm_compatibility_shim_markers": shim_markers,
        "iterations": iterations,
        "losses": losses,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "gpu_telemetry": gpu_telemetry,
        "node_logs": node_logs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    summary = verify(pathlib.Path(args.run_dir), pathlib.Path(args.config))
    if args.output:
        _atomic_json(pathlib.Path(args.output), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-fast node preflight for an immutable Megatron-LM checkout on ROCm."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import subprocess
import sys
import time
import warnings


def _git(megatron_dir: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(megatron_dir), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-dir", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-gpus", required=True, type=int)
    parser.add_argument("--compile-dataset-helpers", action="store_true")
    args = parser.parse_args()

    megatron_dir = pathlib.Path(args.megatron_dir).resolve()
    if not (megatron_dir / "pretrain_gpt.py").is_file():
        raise SystemExit(f"missing pretrain_gpt.py under {megatron_dir}")
    actual_commit = _git(megatron_dir, "rev-parse", "HEAD")
    if actual_commit != args.expected_commit:
        raise SystemExit(
            f"Megatron commit mismatch: expected {args.expected_commit}, got {actual_commit}"
        )
    dirty = _git(megatron_dir, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit("Megatron checkout has tracked modifications; refusing to run")

    sys.path.insert(0, str(megatron_dir))
    import torch

    if torch.version.hip is None:
        raise SystemExit(f"ROCm build required, got torch={torch.__version__}")
    gpu_count = torch.cuda.device_count()
    if gpu_count < args.expected_gpus:
        raise SystemExit(
            f"expected at least {args.expected_gpus} visible GPUs, found {gpu_count}"
        )

    module_versions: dict[str, str | None] = {}
    for module_name in ("megatron.core", "transformer_engine", "flash_attn", "grouped_gemm"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module = importlib.import_module(module_name)
        module_versions[module_name] = getattr(module, "__version__", None)
        for warning in caught:
            print(
                json.dumps(
                    {
                        "marker": "PREFLIGHT_WARNING",
                        "module": module_name,
                        "message": str(warning.message),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if args.compile_dataset_helpers:
        from megatron.core.datasets.utils import compile_helpers

        started = time.monotonic()
        compile_helpers()
        helper_seconds = time.monotonic() - started
    else:
        helper_seconds = None

    payload = {
        "marker": "NODE_PREFLIGHT_OK",
        "hostname": os.uname().nodename,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "gpu_count": gpu_count,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
        "megatron_commit": actual_commit,
        "modules": module_versions,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "USE_ROCM_AITER_ROPE_BACKEND",
                "NVTE_FUSED_ATTN",
                "NVTE_FLASH_ATTN",
                "NVTE_UNFUSED_ATTN",
                "NVTE_DEBUG",
                "NVTE_DEBUG_LEVEL",
            )
        },
        "dataset_helper_compile_seconds": helper_seconds,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Narrow ROCm compatibility entrypoint for pinned upstream Megatron-LM.

Megatron-LM 5cb6dbb unconditionally calls its legacy CUDA fused-kernel loader
at startup, even when all legacy masked-softmax fusion is disabled.  On a ROCm
PyTorch build, ``torch.utils.cpp_extension.CUDA_HOME`` is None and that loader
fails before model construction.  This wrapper verifies the exact pinned
loader source, replaces only ``load(args)`` with a logged no-op on ROCm, and
then executes upstream ``pretrain_gpt.py`` unchanged.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import pathlib
import runpy
import sys


EXPECTED_FUSED_LOADER_SHA256 = (
    "2625944656248d3f1afb7f3213bd7097b1203d3dc4444b2c55b0a0d2cb38f9af"
)
MCORE_INITIALIZE_RELATIVE_PATH = pathlib.Path("megatron/training/initialize.py")
EXPECTED_MCORE_INITIALIZE_SHA256 = (
    "af3bcf726cb82c1d7d9505f829ff885b20fc184fbfe7d495a7bab174a0d84a8b"
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    import torch

    launcher_rank = int(os.environ.get("RANK", "-1"))
    launcher_local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if launcher_local_rank < 0 or launcher_local_rank >= torch.cuda.device_count():
        raise SystemExit(
            f"invalid LOCAL_RANK={launcher_local_rank} for {torch.cuda.device_count()} visible GPUs"
        )
    # Bind before importing Megatron/Transformer Engine code through runpy. Some ROCm
    # libraries create a device context during import, so Megatron's later binding is too late.
    torch.cuda.set_device(launcher_local_rank)
    print(
        json.dumps(
            {
                "marker": "EARLY_DEVICE_BIND",
                "rank": launcher_rank,
                "local_rank": launcher_local_rank,
                "current_device": torch.cuda.current_device(),
                "visible_devices": torch.cuda.device_count(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--upstream-entrypoint", required=True)
    known, remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    if any(
        arg in {"--local-rank", "--local_rank"}
        or arg.startswith("--local-rank=")
        or arg.startswith("--local_rank=")
        for arg in remaining
    ):
        raise SystemExit("local rank must come only from the torchrun launcher environment")
    # Make the launcher contract explicit rather than relying on argparse evaluating an
    # environment-derived default after the Megatron import graph has run.
    remaining.extend(["--local-rank", str(launcher_local_rank)])

    traceback_interval = int(
        os.environ.get("MEGATRON_SMOKE_TRACEBACK_INTERVAL_SECONDS", "0")
    )
    if traceback_interval and traceback_interval < 30:
        raise SystemExit("traceback interval must be 0 or at least 30 seconds")
    if traceback_interval:
        faulthandler.enable(all_threads=True)
        faulthandler.dump_traceback_later(traceback_interval, repeat=True)
        print(
            json.dumps(
                {
                    "marker": "PERIODIC_TRACEBACK_ARMED",
                    "interval_seconds": traceback_interval,
                    "rank": int(os.environ.get("RANK", "-1")),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    entrypoint = pathlib.Path(known.upstream_entrypoint).resolve()
    if entrypoint.name != "pretrain_gpt.py" or not entrypoint.is_file():
        raise SystemExit(f"invalid upstream pretrain entrypoint: {entrypoint}")
    upstream_root = entrypoint.parent
    sys.path.insert(0, str(upstream_root))

    if torch.version.hip is not None:
        import megatron.legacy.fused_kernels as fused_kernels

        loader_path = pathlib.Path(fused_kernels.__file__).resolve()
        loader_sha256 = _sha256(loader_path)
        if loader_sha256 != EXPECTED_FUSED_LOADER_SHA256:
            raise RuntimeError(
                "refusing ROCm compatibility shim for unknown fused-kernel loader: "
                f"{loader_path} sha256={loader_sha256}"
            )

        def _rocm_noop_load(args: object) -> None:
            print(
                json.dumps(
                    {
                        "marker": "ROCM_LEGACY_FUSED_KERNEL_LOADER_SKIPPED",
                        "reason": "pinned loader is CUDA-only; legacy masked softmax is disabled",
                        "source_sha256": loader_sha256,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        fused_kernels.load = _rocm_noop_load

        initialize_path = upstream_root / MCORE_INITIALIZE_RELATIVE_PATH
        initialize_sha256 = _sha256(initialize_path)
        if initialize_sha256 != EXPECTED_MCORE_INITIALIZE_SHA256:
            raise RuntimeError(
                "refusing ROCm process-group device shim for unknown Megatron initializer: "
                f"{initialize_path} sha256={initialize_sha256}"
            )
        original_init_process_group = torch.distributed.init_process_group
        local_device = torch.device("cuda", launcher_local_rank)

        def _init_process_group_on_local_device(*args: object, **kwargs: object) -> object:
            backend = kwargs.get("backend", args[0] if args else None)
            if str(backend).lower() == "nccl":
                requested_device = kwargs.get("device_id")
                if requested_device is None:
                    kwargs["device_id"] = local_device
                elif torch.device(requested_device) != local_device:
                    raise RuntimeError(
                        f"process-group device_id={requested_device} does not match "
                        f"LOCAL_RANK device {local_device}"
                    )
                print(
                    json.dumps(
                        {
                            "marker": "PROCESS_GROUP_DEVICE_BIND",
                            "rank": launcher_rank,
                            "local_rank": launcher_local_rank,
                            "device_id": str(local_device),
                            "backend": "nccl",
                            "source_sha256": initialize_sha256,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return original_init_process_group(*args, **kwargs)

        torch.distributed.init_process_group = _init_process_group_on_local_device
    else:
        original_init_process_group = None

    sys.argv = [str(entrypoint), *remaining]
    try:
        runpy.run_path(str(entrypoint), run_name="__main__")
    finally:
        if original_init_process_group is not None:
            torch.distributed.init_process_group = original_init_process_group
        if traceback_interval:
            faulthandler.cancel_dump_traceback_later()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

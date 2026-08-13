#!/usr/bin/env python3
"""ROCm-safe launcher for the project-owned R-Full Megatron entry point."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import inspect
import json
import math
import os
import pathlib
import runpy
import sys


EXPECTED_DATASET_UTILS_SHA256 = (
    "629dbc23d6c80963608548702f9cd9ecc4c541206b68d2a0182276f60ec391a2"
)
EXPECTED_DATASET_HELPERS_CPP_SHA256 = (
    "290abcd23543e09bece7332907c9c563e96030fb68ea4e4a64c9080fdd9dac55"
)


EXPECTED_FUSED_LOADER_SHA256 = (
    "2625944656248d3f1afb7f3213bd7097b1203d3dc4444b2c55b0a0d2cb38f9af"
)
EXPECTED_MCORE_INITIALIZE_SHA256 = (
    "af3bcf726cb82c1d7d9505f829ff885b20fc184fbfe7d495a7bab174a0d84a8b"
)
EXPECTED_NUMPY_VERSION_FOR_PRODUCT_ALIAS = "2.2.6"
EXPECTED_NUMPY_PRODUCT_CALLSITE_SHA256 = {
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
EXPECTED_TORCH_VERSION_FOR_DCP_WRITE_ITEM_ADAPTER = "2.10.0.dev20251112+rocm7.1"
EXPECTED_TORCH_DCP_FILESYSTEM_SHA256 = (
    "a3fe232efd14b6c47b553dcb913ae275541e09371500279e1e67fd63eedcce81"
)
EXPECTED_MCORE_FILESYSTEM_ASYNC_SHA256 = (
    "1d410495a6a634722671c4bafdf82ad420bde8b4a56908578565ea1d12c7dbeb"
)
MCORE_FILESYSTEM_ASYNC_RELATIVE_PATH = (
    "megatron/core/dist_checkpointing/strategies/filesystem_async.py"
)
EXPECTED_TORCH_DCP_METADATA_SHA256 = (
    "2a23bd4bfb7442ce203d2e40346298ef2313ab59a0fbfcc8b05ff9482bbc99ca"
)
EXPECTED_MCORE_TORCH_STRATEGY_SHA256 = (
    "a47209ce93367031adfebe3410f0923f352c3c5bd0596a805212b6063672135b"
)
MCORE_TORCH_STRATEGY_RELATIVE_PATH = (
    "megatron/core/dist_checkpointing/strategies/torch.py"
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_numpy_product_api(numpy_module: object) -> str:
    """Restore the one removed NumPy alias used by the pinned checkpoint code.

    NumPy 2 removed ``numpy.product`` while retaining the equivalent
    ``numpy.prod``.  The pinned Megatron commit calls the old spelling in three
    distributed-checkpointing modules.  Only the exactly qualified NumPy version
    may receive this process-local alias; an unknown environment fails closed.
    """

    native_product = getattr(numpy_module, "product", None)
    if native_product is None:
        observed_version = str(getattr(numpy_module, "__version__", "unknown"))
        if observed_version != EXPECTED_NUMPY_VERSION_FOR_PRODUCT_ALIAS:
            raise RuntimeError(
                "NumPy lacks product and is not the qualified compatibility target: "
                f"expected version {EXPECTED_NUMPY_VERSION_FOR_PRODUCT_ALIAS}, "
                f"observed {observed_version}"
            )
        replacement = getattr(numpy_module, "prod", None)
        if not callable(replacement):
            raise RuntimeError("qualified NumPy does not expose callable numpy.prod")
        setattr(numpy_module, "product", replacement)
        mode = "alias_to_prod"
    else:
        if not callable(native_product):
            raise RuntimeError("numpy.product exists but is not callable")
        mode = "native"

    probe = getattr(numpy_module, "product")((2, 3, 4))
    if int(probe) != 24:
        raise RuntimeError(f"numpy.product compatibility probe failed: observed {probe!r}")
    return mode


def _install_numpy_product_compatibility(
    upstream_root: pathlib.Path, rank: int
) -> None:
    """Hash-guard and install the NumPy 2 checkpoint compatibility alias."""

    observed_sources = {}
    for relative_path, expected_digest in EXPECTED_NUMPY_PRODUCT_CALLSITE_SHA256.items():
        source_path = upstream_root / relative_path
        if not source_path.is_file():
            raise RuntimeError(f"missing pinned checkpoint source {source_path}")
        observed_digest = _sha256(source_path)
        if observed_digest != expected_digest:
            raise RuntimeError(
                f"unknown pinned checkpoint source {source_path}: "
                f"expected sha256={expected_digest}, observed={observed_digest}"
            )
        observed_sources[relative_path] = observed_digest

    import numpy as np

    mode = _ensure_numpy_product_api(np)
    print(
        json.dumps(
            {
                "marker": "NUMPY_PRODUCT_COMPAT_READY",
                "rank": rank,
                "mode": mode,
                "numpy_version": np.__version__,
                "source_sha256": observed_sources,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _adapt_dcp_write_item_api(
    native_write_item: object, serialization_format: object
) -> tuple[object, str]:
    """Adapt one qualified PyTorch DCP signature to pinned Megatron's call.

    Pinned Megatron calls the private PyTorch ``_write_item`` helper with five
    positional arguments.  The qualified PyTorch 2.10 build made a sixth
    ``serialization_format`` argument mandatory.  ``TORCH_SAVE`` exactly
    preserves the behavior used by the older five-argument call path.
    """

    if not callable(native_write_item):
        raise RuntimeError("torch DCP _write_item is not callable")
    parameters = tuple(inspect.signature(native_write_item).parameters.values())
    expected_prefix = ("transforms", "stream", "data", "write_item", "storage_key")
    observed_names = tuple(parameter.name for parameter in parameters)
    if observed_names[:5] != expected_prefix:
        raise RuntimeError(
            "unknown torch DCP _write_item parameter prefix: "
            f"expected {expected_prefix}, observed {observed_names}"
        )

    required_after_prefix = tuple(
        parameter
        for parameter in parameters[5:]
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )
    if not required_after_prefix:
        return native_write_item, "native_five_arg_compatible"

    if len(parameters) != 6 or required_after_prefix != (parameters[5],):
        raise RuntimeError(
            "unknown torch DCP _write_item required parameters: "
            f"signature={inspect.signature(native_write_item)}"
        )
    format_parameter = parameters[5]
    if (
        format_parameter.name != "serialization_format"
        or format_parameter.kind
        not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ):
        raise RuntimeError(
            "unknown torch DCP _write_item serialization parameter: "
            f"signature={inspect.signature(native_write_item)}"
        )
    torch_save = getattr(serialization_format, "TORCH_SAVE", None)
    if getattr(torch_save, "value", None) != "torch_save":
        raise RuntimeError("qualified torch DCP lacks SerializationFormat.TORCH_SAVE")

    def _write_item_with_torch_save(
        transforms: object,
        stream: object,
        data: object,
        write_item: object,
        storage_key: str,
    ) -> object:
        return native_write_item(
            transforms,
            stream,
            data,
            write_item,
            storage_key,
            serialization_format=torch_save,
        )

    return _write_item_with_torch_save, "append_torch_save_serialization_format"


def _install_dcp_write_item_compatibility(
    upstream_root: pathlib.Path, rank: int
) -> None:
    """Install the exact private-DCP API adapter required by the fixed runtime."""

    import torch
    import torch.distributed.checkpoint.filesystem as torch_filesystem
    from megatron.core.dist_checkpointing.strategies import (
        filesystem_async as mcore_filesystem_async,
    )

    observed_torch_version = str(torch.__version__)
    if observed_torch_version != EXPECTED_TORCH_VERSION_FOR_DCP_WRITE_ITEM_ADAPTER:
        raise RuntimeError(
            "unknown torch version for DCP write-item compatibility: "
            f"expected {EXPECTED_TORCH_VERSION_FOR_DCP_WRITE_ITEM_ADAPTER}, "
            f"observed {observed_torch_version}"
        )
    torch_source = pathlib.Path(torch_filesystem.__file__).resolve()
    mcore_source = pathlib.Path(mcore_filesystem_async.__file__).resolve()
    expected_mcore_source = (upstream_root / MCORE_FILESYSTEM_ASYNC_RELATIVE_PATH).resolve()
    if mcore_source != expected_mcore_source:
        raise RuntimeError(
            "loaded Megatron filesystem_async from an unexpected root: "
            f"expected {expected_mcore_source}, observed {mcore_source}"
        )
    observed_sources = {
        "torch.distributed.checkpoint.filesystem": _sha256(torch_source),
        MCORE_FILESYSTEM_ASYNC_RELATIVE_PATH: _sha256(mcore_source),
    }
    expected_sources = {
        "torch.distributed.checkpoint.filesystem": EXPECTED_TORCH_DCP_FILESYSTEM_SHA256,
        MCORE_FILESYSTEM_ASYNC_RELATIVE_PATH: EXPECTED_MCORE_FILESYSTEM_ASYNC_SHA256,
    }
    if observed_sources != expected_sources:
        raise RuntimeError(
            "unknown DCP write-item compatibility sources: "
            f"expected {expected_sources}, observed {observed_sources}"
        )

    native_write_item = torch_filesystem._write_item
    if mcore_filesystem_async._write_item is not native_write_item:
        raise RuntimeError("Megatron filesystem_async did not import the qualified _write_item")
    adapted_write_item, mode = _adapt_dcp_write_item_api(
        native_write_item, torch_filesystem.SerializationFormat
    )
    mcore_filesystem_async._write_item = adapted_write_item
    print(
        json.dumps(
            {
                "marker": "DCP_WRITE_ITEM_COMPAT_READY",
                "rank": rank,
                "mode": mode,
                "torch_version": observed_torch_version,
                "serialization_format": "torch_save",
                "source_sha256": observed_sources,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _infer_same_geometry_reformulation_metadata(
    strategy_module: object,
    checkpoint_metadata: object,
    sharded_state_dict: object,
) -> dict[str, object]:
    """Reconstruct metadata dropped by the qualified PyTorch writer.

    The fallback is intentionally limited to an identical current/checkpoint
    geometry.  It derives each original N-D shape from the current pinned MCore
    ShardedTensor and requires equal element counts in checkpoint storage.
    """

    inferred: dict[str, object] = {}
    state_metadata = getattr(checkpoint_metadata, "state_dict_metadata", None)
    if not isinstance(state_metadata, dict):
        raise RuntimeError("checkpoint metadata lacks state_dict_metadata")
    for sharded_tensor in strategy_module.nested_values(sharded_state_dict):
        if not strategy_module.is_nd_flattened_tensor(sharded_tensor):
            continue
        key = getattr(sharded_tensor, "key", None)
        original_shape = getattr(sharded_tensor, "global_shape", None)
        if not isinstance(key, str) or original_shape is None:
            raise RuntimeError(
                "invalid current MCore ShardedTensor for compatibility load: "
                f"key={key!r}, type={type(sharded_tensor).__name__}"
            )
        original_shape = tuple(int(dimension) for dimension in original_shape)
        storage_metadata = state_metadata.get(key)
        stored_shape = tuple(
            int(dimension) for dimension in getattr(storage_metadata, "size", ())
        )
        if not original_shape or not stored_shape:
            raise RuntimeError(f"checkpoint lacks a stored shape for {key}")
        if math.prod(original_shape) != math.prod(stored_shape):
            raise RuntimeError(
                f"same-geometry reformulation element-count mismatch for {key}: "
                f"current={original_shape}, checkpoint={stored_shape}"
            )
        reformulation = strategy_module.TensorReformulationMetadata(
            original_shape, stored_shape
        )
        previous = inferred.setdefault(key, reformulation)
        if previous != reformulation:
            raise RuntimeError(f"inconsistent inferred reformulation metadata for {key}")
    if not inferred:
        raise RuntimeError("missing mcore_data but no N-D flattened tensors were inferred")
    return inferred


def _install_dcp_mcore_metadata_compatibility(
    upstream_root: pathlib.Path, rank: int
) -> None:
    """Preserve and narrowly reconstruct MCore metadata across Torch 2.10 DCP."""

    import torch
    import torch.distributed.checkpoint.filesystem as torch_filesystem
    import torch.distributed.checkpoint.metadata as torch_metadata
    from megatron.core.dist_checkpointing.strategies import torch as mcore_torch_strategy

    if str(torch.__version__) != EXPECTED_TORCH_VERSION_FOR_DCP_WRITE_ITEM_ADAPTER:
        raise RuntimeError("unknown torch version for DCP MCore metadata compatibility")
    torch_filesystem_source = pathlib.Path(torch_filesystem.__file__).resolve()
    torch_metadata_source = pathlib.Path(torch_metadata.__file__).resolve()
    mcore_source = pathlib.Path(mcore_torch_strategy.__file__).resolve()
    expected_mcore_source = (upstream_root / MCORE_TORCH_STRATEGY_RELATIVE_PATH).resolve()
    if mcore_source != expected_mcore_source:
        raise RuntimeError(
            "loaded Megatron torch DCP strategy from an unexpected root: "
            f"expected {expected_mcore_source}, observed {mcore_source}"
        )
    observed_sources = {
        "torch.distributed.checkpoint.filesystem": _sha256(torch_filesystem_source),
        "torch.distributed.checkpoint.metadata": _sha256(torch_metadata_source),
        MCORE_TORCH_STRATEGY_RELATIVE_PATH: _sha256(mcore_source),
    }
    expected_sources = {
        "torch.distributed.checkpoint.filesystem": EXPECTED_TORCH_DCP_FILESYSTEM_SHA256,
        "torch.distributed.checkpoint.metadata": EXPECTED_TORCH_DCP_METADATA_SHA256,
        MCORE_TORCH_STRATEGY_RELATIVE_PATH: EXPECTED_MCORE_TORCH_STRATEGY_SHA256,
    }
    if observed_sources != expected_sources:
        raise RuntimeError(
            "unknown DCP MCore metadata compatibility sources: "
            f"expected {expected_sources}, observed {observed_sources}"
        )
    metadata_fields = tuple(torch_metadata.Metadata.__dataclass_fields__)
    expected_metadata_fields = (
        "state_dict_metadata",
        "planner_data",
        "storage_data",
        "storage_meta",
        "version",
    )
    if metadata_fields != expected_metadata_fields:
        raise RuntimeError(
            f"unknown torch DCP Metadata fields: {metadata_fields}"
        )
    finish_signature = tuple(
        inspect.signature(torch_filesystem._FileSystemWriter.finish).parameters
    )
    if finish_signature != ("self", "metadata", "results"):
        raise RuntimeError(f"unknown DCP writer finish signature: {finish_signature}")
    reformulation_signature = tuple(
        inspect.signature(mcore_torch_strategy.get_reformulation_metadata).parameters
    )
    if reformulation_signature != ("sharded_state_dict", "checkpoint_dir"):
        raise RuntimeError(
            "unknown MCore get_reformulation_metadata signature: "
            f"{reformulation_signature}"
        )

    def _finish_preserving_mcore_data(self, metadata, results) -> None:
        mcore_data = getattr(metadata, "mcore_data", None)
        if not isinstance(mcore_data, dict) or not mcore_data:
            raise RuntimeError("MCore save metadata is missing non-empty mcore_data")
        metadata = torch_filesystem.dataclasses.replace(
            metadata, version=torch_filesystem.CURRENT_DCP_VERSION
        )
        metadata.mcore_data = mcore_data
        storage_md = {}
        for write_result_list in results:
            storage_md.update(
                {
                    write_result.index: write_result.storage_data
                    for write_result in write_result_list
                }
            )
        metadata.storage_data = storage_md
        metadata.storage_meta = self.storage_meta()
        temporary_filename = (
            f"__{self.rank}{torch_filesystem._metadata_fn}.tmp"
            if not self.use_collectives and self.rank is not None
            else f"{torch_filesystem._metadata_fn}.tmp"
        )
        temporary_path = self.fs.concat_path(self.path, temporary_filename)
        with self.fs.create_stream(temporary_path, "wb") as metadata_file:
            torch_filesystem.pickle.dump(metadata, metadata_file)
            if self.sync_files:
                try:
                    torch_filesystem.os.fsync(metadata_file.fileno())
                except (AttributeError, torch_filesystem.UnsupportedOperation):
                    torch_filesystem.os.sync()
        metadata_path = (
            self._get_metadata_path(self.rank)
            if not self.use_collectives and self.rank is not None
            else self._get_metadata_path()
        )
        if self.fs.exists(metadata_path):
            self.fs.rm_file(metadata_path)
        self.fs.rename(temporary_path, metadata_path)
        print(
            json.dumps(
                {
                    "marker": "DCP_MCORE_METADATA_PRESERVED",
                    "rank": rank,
                    "entries": len(mcore_data),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    original_get_reformulation_metadata = (
        mcore_torch_strategy.get_reformulation_metadata
    )

    def _get_reformulation_metadata_compatible(
        sharded_state_dict: object, checkpoint_dir: object
    ) -> dict[str, object]:
        checkpoint_metadata = torch_filesystem.FileSystemReader(
            checkpoint_dir
        ).read_metadata()
        if hasattr(checkpoint_metadata, "mcore_data"):
            return original_get_reformulation_metadata(
                sharded_state_dict, checkpoint_dir
            )
        inferred = _infer_same_geometry_reformulation_metadata(
            mcore_torch_strategy, checkpoint_metadata, sharded_state_dict
        )
        shape_evidence = {
            key: {
                "original": list(value.ckpt_orig_global_shape),
                "stored": list(value.ckpt_reform_global_shape),
            }
            for key, value in sorted(inferred.items())
        }
        print(
            json.dumps(
                {
                    "marker": "DCP_MCORE_METADATA_FALLBACK_APPLIED",
                    "rank": rank,
                    "entries": len(inferred),
                    "shape_sha256": hashlib.sha256(
                        json.dumps(shape_evidence, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return inferred

    torch_filesystem._FileSystemWriter.finish = _finish_preserving_mcore_data
    mcore_torch_strategy.get_reformulation_metadata = (
        _get_reformulation_metadata_compatible
    )
    print(
        json.dumps(
            {
                "marker": "DCP_MCORE_METADATA_COMPAT_READY",
                "rank": rank,
                "save_mode": "preserve_mcore_data_after_dataclasses_replace",
                "load_mode": "infer_same_geometry_if_absent",
                "source_sha256": observed_sources,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _install_writable_dataset_helper(upstream_root: pathlib.Path, rank: int) -> None:
    """Build the pinned dataset extension outside the immutable checkout.

    Pinned Megatron unconditionally invokes ``make`` in its source directory on
    global rank zero.  The production checkout is intentionally root-owned and
    read-only to the training identity, so every local rank instead loads one
    hash-guarded, node-local build and the upstream compile hook becomes a
    verified no-op.  A file lock makes concurrent torchrun startup safe.
    """

    import fcntl
    import importlib.util
    import shlex
    import subprocess
    import sysconfig

    datasets_dir = upstream_root / "megatron" / "core" / "datasets"
    utils_path = datasets_dir / "utils.py"
    source_path = datasets_dir / "helpers.cpp"
    observed_utils = _sha256(utils_path)
    observed_source = _sha256(source_path)
    if observed_utils != EXPECTED_DATASET_UTILS_SHA256:
        raise RuntimeError(
            f"unknown Megatron dataset utils {utils_path}: sha256={observed_utils}"
        )
    if observed_source != EXPECTED_DATASET_HELPERS_CPP_SHA256:
        raise RuntimeError(
            f"unknown Megatron dataset helper {source_path}: sha256={observed_source}"
        )

    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python did not report an extension-module suffix")
    cache_root = pathlib.Path(
        os.environ.get("RFULL_DATASET_HELPER_CACHE_ROOT", "/tmp/rfull-dataset-helpers")
    )
    if not cache_root.is_absolute():
        raise RuntimeError("RFULL_DATASET_HELPER_CACHE_ROOT must be absolute")
    cache_dir = (
        cache_root
        / f"uid-{os.getuid()}"
        / f"{observed_source[:16]}-{sys.implementation.cache_tag}"
    )
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    module_path = cache_dir / f"helpers_cpp{extension_suffix}"
    manifest_path = cache_dir / "build.json"
    lock_path = cache_dir / "build.lock"

    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        rebuild = True
        if module_path.is_file() and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                rebuild = not (
                    manifest.get("source_sha256") == observed_source
                    and manifest.get("utils_sha256") == observed_utils
                    and manifest.get("python_cache_tag") == sys.implementation.cache_tag
                    and manifest.get("module_sha256") == _sha256(module_path)
                )
            except (OSError, ValueError, TypeError):
                rebuild = True
        if rebuild:
            includes = shlex.split(
                subprocess.check_output(
                    [sys.executable, "-m", "pybind11", "--includes"], text=True
                ).strip()
            )
            temporary_module = cache_dir / f".{module_path.name}.{os.getpid()}.tmp"
            command = [
                os.environ.get("CXX", "c++"),
                "-O3",
                "-Wall",
                "-shared",
                "-std=c++11",
                "-fPIC",
                "-fdiagnostics-color",
                *includes,
                str(source_path),
                "-o",
                str(temporary_module),
            ]
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=180
            )
            if completed.returncode != 0:
                temporary_module.unlink(missing_ok=True)
                raise RuntimeError(
                    "failed to build pinned dataset helper outside the checkout: "
                    f"rc={completed.returncode}\nstdout={completed.stdout}\n"
                    f"stderr={completed.stderr}"
                )
            os.replace(temporary_module, module_path)
            manifest = {
                "module_sha256": _sha256(module_path),
                "python_cache_tag": sys.implementation.cache_tag,
                "source_sha256": observed_source,
                "utils_sha256": observed_utils,
            }
            temporary_manifest = cache_dir / f".build.{os.getpid()}.tmp"
            temporary_manifest.write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary_manifest, manifest_path)
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    module_name = "megatron.core.datasets.helpers_cpp"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module

    from megatron.core.datasets import utils as dataset_utils

    def _verified_helper_already_loaded() -> None:
        if module_name not in sys.modules:
            raise RuntimeError("verified Megatron dataset helper disappeared before use")

    dataset_utils.compile_helpers = _verified_helper_already_loaded
    print(
        json.dumps(
            {
                "marker": "DATASET_HELPER_OVERLAY_READY",
                "rank": rank,
                "cache_path": str(module_path),
                **manifest,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--project-entrypoint", required=True)
    known, remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    if any(
        item in {"--local-rank", "--local_rank"}
        or item.startswith("--local-rank=")
        or item.startswith("--local_rank=")
        for item in remaining
    ):
        raise SystemExit("local rank must come only from torchrun")

    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise SystemExit(
            f"invalid LOCAL_RANK={local_rank} for {torch.cuda.device_count()} visible GPUs"
        )
    torch.cuda.set_device(local_rank)
    print(
        json.dumps(
            {
                "marker": "EARLY_DEVICE_BIND",
                "rank": rank,
                "local_rank": local_rank,
                "current_device": torch.cuda.current_device(),
                "visible_devices": torch.cuda.device_count(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    interval = int(os.environ.get("RFULL_TRACEBACK_INTERVAL_SECONDS", "0"))
    if interval and interval < 30:
        raise SystemExit("traceback interval must be 0 or at least 30 seconds")
    if interval:
        faulthandler.enable(all_threads=True)
        faulthandler.dump_traceback_later(interval, repeat=True)
        print(
            json.dumps(
                {
                    "marker": "PERIODIC_TRACEBACK_ARMED",
                    "rank": rank,
                    "interval_seconds": interval,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    upstream_root = pathlib.Path(known.upstream_root).resolve()
    project_entrypoint = pathlib.Path(known.project_entrypoint).resolve()
    if not (upstream_root / "pretrain_gpt.py").is_file():
        raise SystemExit(f"invalid pinned Megatron root: {upstream_root}")
    if project_entrypoint.name != "pretrain_rfull_moe.py" or not project_entrypoint.is_file():
        raise SystemExit(f"invalid R-Full entry point: {project_entrypoint}")
    project_root = project_entrypoint.parent.parent
    if project_entrypoint.parent != project_root / "tools":
        raise SystemExit("R-Full entry point must be in the deployed project's tools directory")
    # PYTHONPATH commonly already contains both roots.  Remove exact duplicates
    # before prepending them so the namespace package does not report one
    # physical checkout multiple times.
    for root in (upstream_root, project_root):
        root_text = str(root)
        sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path[:0] = [str(upstream_root), str(project_root)]
    _install_numpy_product_compatibility(upstream_root, rank)
    _install_dcp_write_item_compatibility(upstream_root, rank)
    _install_dcp_mcore_metadata_compatibility(upstream_root, rank)
    _install_writable_dataset_helper(upstream_root, rank)

    original_init_process_group = None
    if torch.version.hip is not None:
        import megatron.legacy.fused_kernels as fused_kernels

        loader_path = pathlib.Path(fused_kernels.__file__).resolve()
        loader_sha = _sha256(loader_path)
        if loader_sha != EXPECTED_FUSED_LOADER_SHA256:
            raise RuntimeError(
                f"unknown ROCm fused loader {loader_path}: sha256={loader_sha}"
            )

        def _skip_cuda_fused_loader(args: object) -> None:
            print(
                json.dumps(
                    {
                        "marker": "ROCM_LEGACY_FUSED_KERNEL_LOADER_SKIPPED",
                        "rank": rank,
                        "source_sha256": loader_sha,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        fused_kernels.load = _skip_cuda_fused_loader
        initialize_path = upstream_root / "megatron" / "training" / "initialize.py"
        initialize_sha = _sha256(initialize_path)
        if initialize_sha != EXPECTED_MCORE_INITIALIZE_SHA256:
            raise RuntimeError(
                f"unknown Megatron initializer {initialize_path}: sha256={initialize_sha}"
            )
        original_init_process_group = torch.distributed.init_process_group
        local_device = torch.device("cuda", local_rank)

        def _init_process_group_on_local_device(*args: object, **kwargs: object):
            backend = kwargs.get("backend", args[0] if args else None)
            if str(backend).lower() == "nccl":
                requested = kwargs.get("device_id")
                if requested is None:
                    kwargs["device_id"] = local_device
                elif torch.device(requested) != local_device:
                    raise RuntimeError(
                        f"process-group device {requested} does not match {local_device}"
                    )
                print(
                    json.dumps(
                        {
                            "marker": "PROCESS_GROUP_DEVICE_BIND",
                            "rank": rank,
                            "local_rank": local_rank,
                            "device_id": str(local_device),
                            "backend": "nccl",
                            "source_sha256": initialize_sha,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return original_init_process_group(*args, **kwargs)

        torch.distributed.init_process_group = _init_process_group_on_local_device

    remaining.extend(["--local-rank", str(local_rank)])
    sys.argv = [str(project_entrypoint), *remaining]
    try:
        runpy.run_path(str(project_entrypoint), run_name="__main__")
    finally:
        if original_init_process_group is not None:
            torch.distributed.init_process_group = original_init_process_group
        if interval:
            faulthandler.cancel_dump_traceback_later()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

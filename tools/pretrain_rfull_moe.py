#!/usr/bin/env python3
"""Native Megatron training entry point for the frozen R-Full MoE model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from functools import wraps
from typing import Any, Callable


_PROFILE_VOCAB = {
    "ep8-mini": {
        "native": 4000,
        "padded": 4096,
        "divisor": 128,
    },
    "production": {
        "native": 151_669,
        "padded": 151_936,
        "divisor": 1187,
    },
}


def extra_args_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("R-Full MoE")
    group.add_argument(
        "--rfull-profile",
        choices=tuple(_PROFILE_VOCAB),
        default="production",
        help="Production geometry is the default; the EP8 mini geometry is qualification-only.",
    )
    group.add_argument(
        "--rfull-qualification-only",
        action="store_true",
        help="Declare a qualification run (gates: tiny geometry and/or mock data).",
    )
    group.add_argument(
        "--rfull-production-launch",
        action="store_true",
        help=(
            "Declare the real production launch. Mutually exclusive with "
            "--rfull-qualification-only; requires the production profile and real data."
        ),
    )
    group.add_argument(
        "--rfull-expected-local-parameters",
        type=int,
        required=True,
        help="Fail closed unless the local model shard has exactly this many parameters.",
    )
    return parser


def _validate_runtime_args(args: Any) -> None:
    # Every run must state its intent. The original build hard-blocked production
    # while launch blockers were open; that block is now an explicit either/or so
    # a production launch is a deliberate, auditable choice rather than the
    # default that happens when a flag is forgotten.
    qualification = args.rfull_qualification_only
    production = getattr(args, "rfull_production_launch", False)
    if qualification and production:
        raise RuntimeError(
            "--rfull-qualification-only and --rfull-production-launch are mutually exclusive"
        )
    if not (qualification or production):
        raise RuntimeError(
            "declare run intent: pass --rfull-qualification-only for gates, "
            "or --rfull-production-launch for the real run"
        )
    if production:
        # Guard the two ways a 'production' run could silently be a toy: the
        # wrong geometry, or mock data standing in for the corpus.
        if args.rfull_profile != "production":
            raise RuntimeError(
                f"--rfull-production-launch requires the production profile, "
                f"got {args.rfull_profile!r}"
            )
        if getattr(args, "mock_data", False):
            raise RuntimeError("--rfull-production-launch refuses mock data")
    if args.use_legacy_models:
        raise RuntimeError("R-Full requires the Megatron Core model path")
    if args.transformer_impl != "transformer_engine":
        raise RuntimeError("R-Full qualification requires Transformer Engine attention")
    if getattr(args, "load", None) and getattr(args, "no_load_rng", False):
        raise RuntimeError("R-Full checkpoint resume requires RNG state loading")
    expected_vocab = _PROFILE_VOCAB[args.rfull_profile]
    observed_vocab = {
        "native": args.vocab_size,
        "padded": args.padded_vocab_size,
        "divisor": args.make_vocab_size_divisible_by,
    }
    if observed_vocab != expected_vocab:
        raise RuntimeError(
            f"{args.rfull_profile} vocab contract drift: "
            f"observed={observed_vocab}, expected={expected_vocab}"
        )


def model_provider(pre_process: bool = True, post_process: bool = True):
    """Build a GPTModel with the project-owned, source-guarded mixed R-Full spec."""
    import torch
    from megatron.core.models.gpt import GPTModel
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    from rfull_moe.mcore import (
        get_rfull_decoder_block_spec,
        validate_rfull_semantic_config,
    )
    from rfull_moe.pinned_mcore import (
        PINNED_MEGATRON_COMMIT,
        verify_pinned_mcore_sources,
    )

    args = get_args()
    _validate_runtime_args(args)
    source_evidence = verify_pinned_mcore_sources()
    config = core_transformer_config_from_args(args)
    config.rfull_profile = args.rfull_profile
    validate_rfull_semantic_config(config)
    transformer_layer_spec = get_rfull_decoder_block_spec(
        config,
        profile=args.rfull_profile,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
    )
    local_parameters = sum(parameter.numel() for parameter in model.parameters())
    if local_parameters != args.rfull_expected_local_parameters:
        raise RuntimeError(
            "R-Full local parameter count drift: "
            f"observed={local_parameters:,}, "
            f"expected={args.rfull_expected_local_parameters:,}"
        )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print(
        json.dumps(
            {
                "marker": "RFULL_MODEL_BUILT",
                "rank": rank,
                "profile": args.rfull_profile,
                "local_parameters": local_parameters,
                "trainable_parameters": trainable_parameters,
                "padded_vocab_size": args.padded_vocab_size,
                "source_guard_file_count": len(source_evidence),
                "source_guard_megatron_commit": PINNED_MEGATRON_COMMIT,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return model


_BATCH_MICROBATCH_COUNTERS: dict[int, int] = defaultdict(int)


def _batch_fingerprint(batch: Any) -> str:
    """Hash the ordered batch payload without logging token content."""
    import numpy as np
    import torch

    if not isinstance(batch, dict) or not batch:
        raise RuntimeError(f"expected a non-empty batch dictionary, got {type(batch)!r}")
    digest = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        digest.update(key.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
        else:
            raise RuntimeError(
                f"unsupported batch value for deterministic trace: {key}={type(value)!r}"
            )
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def rfull_forward_step(data_iterator, model):
    """Delegate to pinned GPT forward_step while recording data-order evidence.

    Exceptions are captured and printed here before they propagate.  Pinned
    MCore wraps ``forward_step`` in ``StragglerDetector`` whose ``__exit__``
    formats the in-flight exception via ``traceback.format_exception``; that
    formatting has been observed to segfault, destroying the process before
    the real error is ever logged.  Reporting from inside our own frame keeps
    the diagnosis independent of that pinned, root-owned code path.
    """
    try:
        return _rfull_forward_step_inner(data_iterator, model)
    except BaseException as exc:  # noqa: BLE001 - re-raised after reporting
        _report_forward_step_exception(exc)
        raise


def _report_forward_step_exception(exc: BaseException) -> None:
    """Print a self-contained record of ``exc`` to stdout and stderr.

    Uses only already-imported stdlib and avoids ``linecache`` source lookups,
    so it stays usable when the interpreter is in a fragile state.
    """
    import traceback

    try:
        import torch

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    except BaseException:  # noqa: BLE001 - diagnostics must never mask exc
        rank = -1
    header = json.dumps(
        {
            "marker": "RFULL_FORWARD_STEP_EXCEPTION",
            "rank": rank,
            "type": type(exc).__name__,
            "repr": repr(exc)[:2000],
        },
        sort_keys=True,
    )
    for stream in (sys.stdout, sys.stderr):
        try:
            print(header, file=stream, flush=True)
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=stream)
            stream.flush()
        except BaseException:  # noqa: BLE001 - never mask the original failure
            pass


def _rfull_forward_step_inner(data_iterator, model):
    import torch
    from megatron.training import get_args
    from pretrain_gpt import forward_step as pinned_forward_step

    if data_iterator is None:
        raise RuntimeError("R-Full PP=1 qualification requires a data iterator on every rank")
    batch = next(data_iterator)
    args = get_args()
    current_iteration = getattr(args, "curr_iteration", None)
    if not isinstance(current_iteration, int) or isinstance(current_iteration, bool):
        raise RuntimeError(f"invalid current iteration for batch trace: {current_iteration!r}")
    reported_iteration = current_iteration + 1
    microbatch = _BATCH_MICROBATCH_COUNTERS[reported_iteration]
    _BATCH_MICROBATCH_COUNTERS[reported_iteration] += 1
    rank = torch.distributed.get_rank()
    print(
        json.dumps(
            {
                "marker": "RFULL_BATCH_FINGERPRINT",
                "rank": rank,
                "iteration": reported_iteration,
                "microbatch": microbatch,
                "sha256": _batch_fingerprint(batch),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return pinned_forward_step(iter((batch,)), model)


def _completed_iteration(args) -> int:
    """Return the final iteration after pinned pretrain() returns normally.

    Pinned r0.12.3 keeps its completed iteration local to pretrain() and does
    not write it back to args.iteration.  A normal return means the configured
    training horizon completed; checkpoint/exit-interval paths call sys.exit.
    """
    train_iters = getattr(args, "train_iters", None)
    if not isinstance(train_iters, int) or isinstance(train_iters, bool) or train_iters <= 0:
        raise RuntimeError(f"invalid completed train_iters: {train_iters!r}")
    return train_iters


def _cpu_rng_sha256(state: Any) -> str:
    return hashlib.sha256(state.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _wrap_resume_rng_neutral_iterator_builder(
    builder: Callable[..., Any],
    get_args_func: Callable[[], Any],
) -> Callable[..., Any]:
    """Cancel only the extra global CPU-RNG draws made while rebuilding loaders.

    ``load_checkpoint`` restores the saved RNG state before pinned r0.12.3
    constructs train/validation/test ``DataLoader`` iterators.  Constructing
    those iterators draws fresh worker base seeds from the global CPU generator,
    even though an uninterrupted continuation keeps its existing iterators.
    Preserve the generated iterator-local seeds, but restore the global CPU RNG
    state after a real resume so later code sees the checkpoint continuation
    state.  Fresh training retains the stock behavior.
    """

    @wraps(builder)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        import torch

        before = torch.get_rng_state().clone()
        result = builder(*args, **kwargs)
        after_build = torch.get_rng_state().clone()
        runtime_args = get_args_func()
        loaded_iteration = getattr(runtime_args, "iteration", None)
        is_resume = (
            bool(getattr(runtime_args, "load", None))
            and isinstance(loaded_iteration, int)
            and not isinstance(loaded_iteration, bool)
            and loaded_iteration > 0
        )
        no_load_rng = bool(getattr(runtime_args, "no_load_rng", False))
        if is_resume and no_load_rng:
            raise RuntimeError("cannot preserve resume CPU RNG when --no-load-rng is active")
        restored = is_resume and not no_load_rng
        if restored:
            torch.set_rng_state(before)
        after_guard = torch.get_rng_state().clone()
        changed_indices = (before != after_build).nonzero().flatten().tolist()
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else int(getattr(runtime_args, "rank", 0))
        )
        print(
            json.dumps(
                {
                    "marker": "RFULL_RESUME_CPU_RNG_GUARD",
                    "rank": rank,
                    "is_resume": is_resume,
                    "loaded_iteration": loaded_iteration,
                    "restored": restored,
                    "builder_changed_cpu_rng": bool(changed_indices),
                    "changed_byte_count": len(changed_indices),
                    "changed_byte_indices": changed_indices[:16],
                    "before_sha256": _cpu_rng_sha256(before),
                    "after_build_sha256": _cpu_rng_sha256(after_build),
                    "after_guard_sha256": _cpu_rng_sha256(after_guard),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return result

    setattr(wrapped, "_rfull_resume_cpu_rng_guard", True)
    return wrapped


def _install_resume_cpu_rng_guard() -> None:
    import megatron.training.training as training_module
    from megatron.training import get_args

    current = training_module.build_train_valid_test_data_iterators
    if getattr(current, "_rfull_resume_cpu_rng_guard", False):
        raise RuntimeError("R-Full resume CPU RNG guard is already installed")
    training_module.build_train_valid_test_data_iterators = (
        _wrap_resume_rng_neutral_iterator_builder(current, get_args)
    )


def main() -> None:
    from megatron.core.enums import ModelType
    from megatron.training import get_args, pretrain
    from pretrain_gpt import train_valid_test_datasets_provider

    _install_resume_cpu_rng_guard()
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        rfull_forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=extra_args_provider,
        get_embedding_ranks=None,
        get_position_embedding_ranks=None,
        non_loss_data_func=None,
        process_non_loss_data_func=None,
    )
    args = get_args()
    print(
        json.dumps(
            {
                "marker": "RFULL_TRAINING_COMPLETE",
                "rank": args.rank,
                "profile": args.rfull_profile,
                "iteration": _completed_iteration(args),
                "consumed_train_samples": args.consumed_train_samples,
                "consumed_train_tokens": args.consumed_train_samples * args.seq_length,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

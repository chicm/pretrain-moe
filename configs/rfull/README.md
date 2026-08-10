# R-Full v0.1 configuration artifacts

This directory contains the machine-validated configuration contract for the
R-Full 25.86B MoE design. It is intentionally **launch-blocked**: unresolved
values use the exact sentinel `TBD-BLOCKER`, and generated artifacts set
`launch_allowed` to `false`.

## Files

- `rfull_v0_1.source.json`: single editable source of truth.
- `rfull_v0_1.schema.json`: Draft 2020-12 schema; unknown keys are rejected.
- `generated/stage_{4k,8k,16k,32k}.json`: standalone resolved stage artifacts.
- `generated/manifest.json`: source/schema/compiler and stage-artifact hashes.
- `../../tools/compile_rfull_config.py`: schema validation, invariant checks,
  canonical hashing, and deterministic generation.
- `../../tools/run_rfull_stages.py`: fail-closed verifier/orchestrator.

## Reproduce

```bash
python tools/compile_rfull_config.py
python tools/run_rfull_stages.py --plan
```

The compiler rejects floating-point JSON values; decimal hyperparameters are
canonical decimal strings. Object hashes use UTF-8 canonical JSON with sorted
keys and compact separators. For generated artifacts, `artifact_sha256` is
excluded from its own hash input.

`--execute` additionally requires the explicit confirmation phrase printed by
`--help`, zero unresolved blockers, a repository-local pinned argv adapter, and
a checkpoint root. Verification recompiles the artifacts in an isolated
temporary directory with the hash-pinned compiler, so a self-consistent manual
stage/manifest edit is rejected. Resume uses the hash-bound `LATEST_COMMITTED`
canonical-head pointer and a closed `rfull-commit-v1` JSON `COMMITTED` envelope;
a partial-stage checkpoint resumes that same stage, and only its exact endpoint
may advance the curriculum. The adapter is invoked as an argv vector with
`shell=False`.

These artifacts encode a design contract, not evidence that ROCm/Megatron
qualification or production launch gates have passed.

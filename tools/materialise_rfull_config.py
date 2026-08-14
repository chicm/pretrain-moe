"""Materialise a runnable production config from a generated blend file.

The production config is kept blend-free in git (487 absolute corpus paths are
cluster state, not source), so this tool joins the two immediately before a
launch and states the resulting provenance hashes.

Fail-closed: refuses to write if the blend weights do not sum to 1, if any
prefix is missing its .bin/.idx pair on this machine, or if the output already
exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def materialise(
    config_path: pathlib.Path,
    blend_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    timeout_minutes: int,
    train_iters: int | None,
    verify_files: bool,
) -> dict[str, object]:
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing config: {output_path}")

    config = json.loads(config_path.read_text())
    blend_doc = json.loads(blend_path.read_text())

    entries = blend_doc.get("blend")
    if not isinstance(entries, list) or not entries:
        raise SystemExit("blend file has no 'blend' list")

    total = 0.0
    normalised: list[dict[str, object]] = []
    for entry in entries:
        weight = float(entry["weight"])
        prefix = str(entry["prefix"])
        if weight <= 0:
            raise SystemExit(f"non-positive weight for {prefix}")
        if not prefix.startswith("/"):
            raise SystemExit(f"blend prefix must be absolute: {prefix}")
        if verify_files:
            for suffix in (".bin", ".idx"):
                candidate = pathlib.Path(prefix + suffix)
                if not candidate.exists():
                    raise SystemExit(f"missing corpus file: {candidate}")
        total += weight
        normalised.append({"weight": weight, "prefix": prefix})

    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"blend weights sum to {total!r}, expected 1.0")

    config.setdefault("data", {})["blend"] = normalised
    config["data"].setdefault("split", "990,9,1")
    config["runtime"]["mock_data"] = False
    config["runtime"]["distributed_timeout_minutes"] = int(timeout_minutes)
    if train_iters is not None:
        config["training"]["train_iters"] = int(train_iters)

    output_path.write_text(json.dumps(config, indent=2, sort_keys=False) + "\n")

    return {
        "config_in": str(config_path),
        "config_in_sha256": _sha256(config_path),
        "blend": str(blend_path),
        "blend_sha256": _sha256(blend_path),
        "config_out": str(output_path),
        "config_out_sha256": _sha256(output_path),
        "blend_entries": len(normalised),
        "weight_sum": round(total, 9),
        "distributed_timeout_minutes": int(timeout_minutes),
        "train_iters": config["training"]["train_iters"],
        "files_verified": bool(verify_files),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--blend", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        required=True,
        help="NCCL watchdog budget; must cover first-time index construction",
    )
    parser.add_argument("--train-iters", type=int, default=None)
    parser.add_argument(
        "--no-verify-files",
        action="store_true",
        help="skip .bin/.idx existence checks (for offline config assembly)",
    )
    args = parser.parse_args(argv)

    report = materialise(
        args.config,
        args.blend,
        args.output,
        timeout_minutes=args.distributed_timeout_minutes,
        train_iters=args.train_iters,
        verify_files=not args.no_verify_files,
    )
    print(json.dumps(report, indent=2))
    print("MATERIALISE_RESULT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

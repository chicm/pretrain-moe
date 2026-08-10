#!/usr/bin/env python3
"""Fail-closed R-Full stage orchestrator.

The checked-in v0.1 config intentionally contains hard blockers, so --execute must
refuse today. Once blockers close, the configured adapter is invoked as an argv
array (never through a shell), with atomic local orchestration state.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/rfull/generated/manifest.json"
CONFIRMATION = "R-FULL-V0.1-PRODUCTION"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_bytes(value: Any, omit_artifact_hash: bool = False) -> bytes:
    value = copy.deepcopy(value)
    if omit_artifact_hash and isinstance(value, dict):
        value.pop("artifact_sha256", None)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def object_hash(value: Any, omit_artifact_hash: bool = False) -> str:
    return hashlib.sha256(canonical_bytes(value, omit_artifact_hash)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(relative: str) -> Path:
    path = (ROOT / Path(relative)).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes repository: {relative}") from exc
    return path


def reproduce_with_pinned_compiler(
    compiler_path: Path,
    source_path: Path,
    schema_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Recompile in isolation so a self-consistent manual artifact edit still fails."""
    with tempfile.TemporaryDirectory(prefix="rfull_verify_") as temp:
        output = Path(temp)
        completed = subprocess.run(
            [
                sys.executable,
                str(compiler_path),
                "--source",
                str(source_path),
                "--schema",
                str(schema_path),
                "--output",
                str(output),
            ],
            check=False,
            shell=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0:
            raise ValueError(f"pinned compiler reproduction failed: {completed.stderr.strip()}")
        expected_manifest = load_json(output / "manifest.json")
        expected_stages = {
            record["path"]: load_json(output / record["path"])
            for record in expected_manifest["stage_artifacts"]
        }
        return expected_manifest, expected_stages


def verify(manifest_path: Path) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    manifest = load_json(manifest_path)
    if object_hash(manifest, True) != manifest.get("artifact_sha256"):
        raise ValueError("generated manifest artifact hash mismatch")
    source_path = repo_path(manifest["source_config"])
    schema_path = repo_path(manifest["source_schema"])
    compiler_path = repo_path(manifest["compiler"])
    if object_hash(load_json(source_path)) != manifest["source_config_sha256"]:
        raise ValueError("source config hash mismatch")
    if object_hash(load_json(schema_path)) != manifest["source_schema_sha256"]:
        raise ValueError("source schema hash mismatch")
    if file_hash(compiler_path) != manifest["compiler_sha256"]:
        raise ValueError("compiler hash mismatch; regenerate artifacts")
    expected_manifest, expected_stages = reproduce_with_pinned_compiler(
        compiler_path,
        source_path,
        schema_path,
    )
    if canonical_bytes(manifest) != canonical_bytes(expected_manifest):
        raise ValueError("manifest differs from pinned compiler reproduction")

    stages: list[tuple[Path, dict[str, Any]]] = []
    generated_root = manifest_path.parent.resolve()
    for record in manifest["stage_artifacts"]:
        path = (generated_root / record["path"]).resolve()
        try:
            path.relative_to(generated_root)
        except ValueError as exc:
            raise ValueError(f"stage artifact path escapes generated root: {record['path']}") from exc
        stage = load_json(path)
        if canonical_bytes(stage) != canonical_bytes(expected_stages[record["path"]]):
            raise ValueError(f"stage differs from pinned compiler reproduction: {path}")
        computed = object_hash(stage, True)
        if computed != record["artifact_sha256"] or computed != stage.get("artifact_sha256"):
            raise ValueError(f"stage artifact hash mismatch: {path}")
        if stage["source_config_sha256"] != manifest["source_config_sha256"]:
            raise ValueError(f"stage/source hash mismatch: {path}")
        stages.append((path, stage))

    if [stage["stage"]["stage_index"] for _, stage in stages] != list(range(4)):
        raise ValueError("stage order must be exactly 0,1,2,3")
    if stages[-1][1]["stage"]["end_successful_update_inclusive"] != 254313:
        raise ValueError("final update target mismatch")
    if stages[-1][1]["stage"]["end_update_tokens"] != 999999406080:
        raise ValueError("final token target mismatch")
    return manifest, stages


def committed_checkpoint(root: Path, successful_update: int) -> Path:
    return root / f"step_{successful_update}" / "COMMITTED"


HEX64 = re.compile(r"^[0-9a-f]{64}$")
LINEAGE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def confined_path(base: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"invalid {label}: {relative!r}")
    candidate = (base / Path(relative)).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes checkpoint root: {relative}") from exc
    return candidate


def stage_for_update(
    stages: list[tuple[Path, dict[str, Any]]],
    successful_updates: int,
) -> tuple[Path, dict[str, Any]]:
    for item in stages:
        stage = item[1]["stage"]
        if stage["start_successful_update_exclusive"] < successful_updates <= stage["end_successful_update_inclusive"]:
            return item
    raise ValueError(f"checkpoint update is outside the compiled curriculum: {successful_updates}")


def validate_commit_marker(
    marker_path: Path,
    manifest: dict[str, Any],
    stages: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    marker = load_json(marker_path)
    contract = stages[0][1]["checkpoint"]
    required = contract["commit_marker_required_fields"]
    if set(marker) != set(required):
        raise ValueError(f"COMMITTED fields mismatch: {marker_path}")
    if marker["schema_version"] != contract["commit_marker_schema_version"]:
        raise ValueError(f"COMMITTED schema mismatch: {marker_path}")
    if marker["run_name"] != stages[0][1]["run"]["name"]:
        raise ValueError(f"COMMITTED run mismatch: {marker_path}")
    if marker["config_manifest_sha256"] != manifest["artifact_sha256"]:
        raise ValueError(f"COMMITTED config manifest mismatch: {marker_path}")
    successful_updates = marker["successful_updates"]
    update_tokens = marker["update_tokens"]
    if type(successful_updates) is not int or type(update_tokens) is not int:
        raise ValueError(f"COMMITTED counters must be integers: {marker_path}")
    _, expected_stage = stage_for_update(stages, successful_updates)
    expected_tokens = successful_updates * expected_stage["run"]["target_tokens_per_update"]
    if update_tokens != expected_tokens:
        raise ValueError(f"COMMITTED token counter mismatch: {marker_path}")
    if marker["stage_id"] != expected_stage["stage"]["id"]:
        raise ValueError(f"COMMITTED stage mismatch: {marker_path}")
    if marker["stage_artifact_sha256"] != expected_stage["artifact_sha256"]:
        raise ValueError(f"COMMITTED stage artifact mismatch: {marker_path}")
    if marker_path.parent.name != f"step_{successful_updates}":
        raise ValueError(f"COMMITTED directory/counter mismatch: {marker_path}")
    root_manifest = confined_path(marker_path.parent, marker["root_manifest_path"], "root manifest path")
    if not root_manifest.is_file():
        raise ValueError(f"root manifest is missing: {root_manifest}")
    if marker["root_manifest_sha256"] != file_hash(root_manifest):
        raise ValueError(f"root manifest hash mismatch: {root_manifest}")
    parent = marker["parent_commit_sha256"]
    if parent != "GENESIS" and (not isinstance(parent, str) or HEX64.fullmatch(parent) is None):
        raise ValueError(f"invalid parent commit hash: {marker_path}")
    lineage = marker["lineage_id"]
    if not isinstance(lineage, str) or LINEAGE_ID.fullmatch(lineage) is None:
        raise ValueError(f"invalid lineage ID: {marker_path}")
    return marker


def latest_committed_checkpoint(
    checkpoint_root: Path,
    manifest: dict[str, Any],
    stages: list[tuple[Path, dict[str, Any]]],
) -> tuple[Path, dict[str, Any]] | None:
    if not checkpoint_root.exists():
        return None
    if not checkpoint_root.is_dir():
        raise ValueError(f"checkpoint root is not a directory: {checkpoint_root}")
    pointer_name = stages[0][1]["checkpoint"]["latest_committed_pointer"]
    pointer_path = checkpoint_root / pointer_name
    markers = list(checkpoint_root.glob("step_*/COMMITTED"))
    if not pointer_path.is_file():
        if markers:
            raise ValueError("COMMITTED checkpoints exist but LATEST_COMMITTED is missing")
        return None
    pointer = load_json(pointer_path)
    if set(pointer) != {"schema_version", "commit_marker_path", "commit_marker_sha256"}:
        raise ValueError("LATEST_COMMITTED fields mismatch")
    expected_schema = stages[0][1]["checkpoint"]["latest_committed_pointer_schema_version"]
    if pointer["schema_version"] != expected_schema:
        raise ValueError("LATEST_COMMITTED schema mismatch")
    marker_path = confined_path(checkpoint_root, pointer["commit_marker_path"], "commit marker path")
    if marker_path.name != "COMMITTED" or not marker_path.is_file():
        raise ValueError(f"LATEST_COMMITTED target is invalid: {marker_path}")
    if pointer["commit_marker_sha256"] != file_hash(marker_path):
        raise ValueError("LATEST_COMMITTED marker hash mismatch")
    marker = validate_commit_marker(marker_path, manifest, stages)
    parent_hash = marker["parent_commit_sha256"]
    if parent_hash != "GENESIS":
        parent_candidates = [path for path in markers if path != marker_path and file_hash(path) == parent_hash]
        if len(parent_candidates) != 1:
            raise ValueError("canonical head does not have exactly one retained parent COMMITTED marker")
        parent_marker = validate_commit_marker(parent_candidates[0], manifest, stages)
        if parent_marker["lineage_id"] != marker["lineage_id"]:
            raise ValueError("canonical head and parent lineage IDs differ")
        if parent_marker["successful_updates"] >= marker["successful_updates"]:
            raise ValueError("canonical head parent does not precede child")
    return marker_path.parent, marker


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plan", action="store_true", help="print the verified four-stage plan")
    parser.add_argument("--execute", action="store_true", help="invoke the pinned MCore adapter stage by stage")
    parser.add_argument("--confirm-production-launch", default="")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--state-file", type=Path, default=ROOT / ".rfull_orchestrator_state.json")
    args = parser.parse_args()

    manifest, stages = verify(args.manifest.resolve())
    summary = {
        "manifest_sha256": manifest["artifact_sha256"],
        "launch_allowed": manifest["launch_allowed"],
        "unresolved_blockers": manifest["unresolved_blockers"],
        "stages": [
            {
                "id": stage["stage"]["id"],
                "artifact_sha256": stage["artifact_sha256"],
                "end_successful_update": stage["stage"]["end_successful_update_inclusive"],
                "end_update_tokens": stage["stage"]["end_update_tokens"],
            }
            for _, stage in stages
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.execute:
        return

    if args.confirm_production_launch != CONFIRMATION:
        raise SystemExit(f"refusing launch: pass --confirm-production-launch {CONFIRMATION}")
    if not manifest["launch_allowed"] or manifest["unresolved_blockers"]:
        raise SystemExit("refusing launch: unresolved hard blockers remain")
    if args.checkpoint_root is None:
        raise SystemExit("refusing launch: --checkpoint-root is required")

    source = load_json(repo_path(manifest["source_config"]))
    adapter = source["launch"]["mcore_argv_adapter"]
    if not isinstance(adapter, str) or not adapter or adapter == "TBD-BLOCKER":
        raise SystemExit("refusing launch: no pinned MCore argv adapter")
    adapter_path = repo_path(adapter)
    if not adapter_path.is_file():
        raise SystemExit(f"refusing launch: adapter does not exist: {adapter_path}")

    checkpoint_root = args.checkpoint_root.resolve()
    latest = latest_committed_checkpoint(checkpoint_root, manifest, stages)
    parent_checkpoint = str(latest[0]) if latest is not None else ""
    resume_update = latest[1]["successful_updates"] if latest is not None else 0

    for path, stage in stages:
        stage_id = stage["stage"]["id"]
        start_update = stage["stage"]["start_successful_update_exclusive"]
        end_update = stage["stage"]["end_successful_update_inclusive"]
        if resume_update >= end_update:
            continue
        if not start_update <= resume_update < end_update:
            raise SystemExit(
                f"refusing launch: checkpoint update {resume_update} is inconsistent with stage {stage_id}"
            )
        command = [str(adapter_path), "run-stage", "--config", str(path), "--checkpoint-root", str(checkpoint_root)]
        if parent_checkpoint:
            command.extend(["--resume-checkpoint", parent_checkpoint])
        state = {
            "manifest_sha256": manifest["artifact_sha256"],
            "stage": stage_id,
            "stage_artifact_sha256": stage["artifact_sha256"],
            "phase": "launching",
            "resume_successful_update": resume_update,
            "parent_checkpoint": parent_checkpoint,
        }
        write_state(args.state_file, state)
        completed = subprocess.run(command, check=False, shell=False)
        if completed.returncode != 0:
            state["phase"] = "failed"
            state["returncode"] = completed.returncode
            write_state(args.state_file, state)
            raise SystemExit(completed.returncode)
        latest = latest_committed_checkpoint(checkpoint_root, manifest, stages)
        expected_marker = committed_checkpoint(checkpoint_root, end_update)
        if latest is None or latest[0] != expected_marker.parent or latest[1]["successful_updates"] != end_update:
            state["phase"] = "checkpoint_not_committed"
            write_state(args.state_file, state)
            raise SystemExit(f"stage {stage_id} returned without its exact committed endpoint: {expected_marker}")
        parent_checkpoint = str(latest[0])
        resume_update = end_update
        state["phase"] = "committed"
        state["checkpoint"] = parent_checkpoint
        state["successful_updates"] = resume_update
        write_state(args.state_file, state)

    print("R-Full v0.1 four-stage orchestration completed")


if __name__ == "__main__":
    main()

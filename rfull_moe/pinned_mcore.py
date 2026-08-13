"""Fail-closed source guards for the audited Megatron-Core extension points."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Dict


PINNED_MEGATRON_COMMIT = "5cb6dbb3ed04e6fa11a862d90fe898ac2d3ddfad"

# Byte hashes from the clean pinned checkout.  Project code subclasses or copies a
# deliberately small surface from these files; a source change requires a fresh
# semantic and checkpoint audit rather than silently continuing.
PINNED_SOURCE_SHA256: Dict[str, str] = {
    "core/models/gpt/gpt_layer_specs.py": "812234780edb0d3f855edde48babd662f8db77003a755f1e46047bb7be453fdb",
    "core/models/gpt/moe_module_specs.py": "371891933ae0a78d3731114659161c928aa55c9609785d4fc8531533a1dcb544",
    "core/transformer/mlp.py": "c9f6732c14ac46aaa86ef7b1b4331cbd79186785574e8e5429f720a2ba69e178",
    "core/transformer/moe/experts.py": "929fa5fab3f48fe445b208a2f0ec2a79f803a0e5ad114a884e732eb9907fd6f8",
    "core/transformer/moe/moe_layer.py": "bc324b8bf5a8422989e6652b095083340fd1755ff97173ee0876c061c9c03a51",
    "core/transformer/moe/moe_utils.py": "d00d00a0a067a4997eef475e047d8aaf33324bd413a2b558014aa17833360dde",
    "core/transformer/moe/router.py": "aa2d5f3ec17f220388340c2df05dd41cc3e859320e3d4d1bd7611c414fef30a4",
    "core/transformer/moe/shared_experts.py": "5e9458d02c7fe9b151414c51cf72c1d7bf2d4ea93fe282994157914a011be9d4",
    "core/transformer/moe/token_dispatcher.py": "808fcddfe112fd1b9c2752e145233f4f19048608b54c04ff1d438b8b4b82781f",
    "core/transformer/spec_utils.py": "72adaee5d3606672481fc58ed35f6a1677db4923694753bc67d6cc0398426391",
    "core/transformer/transformer_config.py": "0df0d1956291ea8ee66d6023d00ca28d4bf82e0eed555e24dbc78d11d7e15f87",
    "training/training.py": "eabb3f7bdfa9cd3c686c2b092b72feaef206b46cf1fb85f8d135d18a2c2d3904",
}


def _megatron_package_root() -> Path:
    import megatron

    observed_roots = list(megatron.__path__)
    roots = {Path(root).resolve() for root in observed_roots}
    if len(roots) != 1:
        raise RuntimeError(
            f"expected one unique Megatron package root, observed {observed_roots}"
        )
    return next(iter(roots))


def _git(repo_root: Path, *arguments: str) -> str:
    if "\n" in str(repo_root) or "\r" in str(repo_root):
        raise RuntimeError(f"invalid Megatron repository path: {repo_root!s}")
    # Git 2.34 ignores command-scope `-c safe.directory=...` because the key is
    # honored only from protected config.  Give this subprocess a private global
    # config rather than mutating the aiscuser account or the pinned checkout.
    with tempfile.TemporaryDirectory(prefix="rfull-git-home-") as git_home:
        safe_path = repo_root.as_posix().replace("\\", "\\\\").replace('"', '\\"')
        Path(git_home, ".gitconfig").write_text(
            f'[safe]\n\tdirectory = "{safe_path}"\n',
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment["HOME"] = git_home
        environment.pop("GIT_CONFIG_GLOBAL", None)
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed in {repo_root}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def verify_pinned_mcore_checkout() -> Path:
    """Require the expected Git commit, package root, and clean tracked tree."""

    package_root = _megatron_package_root()
    repo_root = package_root.parent
    top_level = Path(_git(repo_root, "rev-parse", "--show-toplevel")).resolve()
    expected_package_root = top_level / "megatron"
    if os.path.normcase(str(expected_package_root)) != os.path.normcase(str(package_root)):
        raise RuntimeError(
            f"imported Megatron from {package_root}, but Git root resolves to {top_level}"
        )
    commit = _git(top_level, "rev-parse", "HEAD")
    if commit != PINNED_MEGATRON_COMMIT:
        raise RuntimeError(
            f"Megatron commit mismatch: expected {PINNED_MEGATRON_COMMIT}, observed {commit}"
        )
    tracked_status = _git(top_level, "status", "--short", "--untracked-files=no")
    if tracked_status:
        raise RuntimeError(
            "Megatron tracked tree is not clean; refusing semantic adapters:\n"
            f"{tracked_status}"
        )
    return package_root


def verify_pinned_mcore_sources() -> Dict[str, str]:
    """Verify the checkout and every audited source file; return its digests.

    There is intentionally no bypass flag.  A mismatch is a launch blocker and
    must be resolved by updating this table together with semantic evidence.
    """

    root = verify_pinned_mcore_checkout()
    observed: Dict[str, str] = {}
    failures = []
    for relative_path, expected in PINNED_SOURCE_SHA256.items():
        path = root / relative_path
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        observed[relative_path] = digest
        if digest != expected:
            failures.append(
                f"{relative_path}: expected sha256={expected}, observed={digest}"
            )
    if failures:
        joined = "\n  - ".join(failures)
        raise RuntimeError(
            "Megatron-Core source guard failed for pinned commit "
            f"{PINNED_MEGATRON_COMMIT}:\n  - {joined}"
        )
    return observed

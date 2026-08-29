"""Entry point: apply ROCm shims, then hand off to stock `pretrain_gpt.py`.

Using a wrapper (rather than editing the Megatron tree) keeps the upstream
checkout pristine, so `git -C /scratch/rfull/megatron-lm rev-parse HEAD` remains
a truthful record of what actually ran.
"""

from __future__ import annotations

import os
import runpy
import sys

MEGATRON_DIR = os.environ.get("MEGATRON_DIR", "/scratch/rfull/megatron-lm")
if MEGATRON_DIR not in sys.path:
    sys.path.insert(0, MEGATRON_DIR)

from moe_rebuild import rocm_shim  # noqa: E402

applied = rocm_shim.apply()
if os.environ.get("RANK", "0") == "0":
    for line in applied:
        print(f"[rocm_shim] {line}", flush=True)
    if not applied:
        print("[rocm_shim] no shims needed", flush=True)

sys.argv[0] = f"{MEGATRON_DIR}/pretrain_gpt.py"
runpy.run_path(sys.argv[0], run_name="__main__")

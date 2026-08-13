#!/usr/bin/env python3
"""Build the immutable R-Full deployment archive.

Source files only: __pycache__/*.pyc are runtime artifacts, and including them
makes an already-run deployment fail its own integrity check.  Executable bits
are stored explicitly because Windows tar records .sh as 0666.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import pathlib
import tarfile
import time

INCLUDE_DIRS = ["rfull_moe", "tools", "configs"]
INCLUDE_SUFFIXES = {".py", ".sh", ".json", ".cpp", ".h", ".txt"}
EXECUTABLE_SUFFIXES = {".sh"}
EXCLUDE_PARTS = {"__pycache__", ".git", ".pytest_cache"}


def collect(root: pathlib.Path) -> list[pathlib.Path]:
    out = []
    for d in INCLUDE_DIRS:
        base = root / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if any(part in EXCLUDE_PARTS for part in p.parts):
                continue
            if p.suffix not in INCLUDE_SUFFIXES:
                continue
            out.append(p)
    # Canonical global order by relative path, so an independent verifier can
    # reproduce the tree hash without knowing INCLUDE_DIRS or its order.
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    files = collect(root)
    out = pathlib.Path(args.output)
    if out.exists():
        print(f"REFUSE_OVERWRITE {out}")
        return 4

    tree = hashlib.sha256()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in files:
            rel = p.relative_to(root).as_posix()
            data = p.read_bytes()
            if b"\r" in data:
                print(f"FAIL: CR byte in {rel}")
                return 3
            tree.update(rel.encode() + b"\0" + hashlib.sha256(data).digest())
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o755 if p.suffix in EXECUTABLE_SUFFIXES else 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    blob = buf.getvalue()
    out.write_bytes(blob)

    print(f"FILES={len(files)}")
    print(f"ARCHIVE={out}")
    print(f"ARCHIVE_BYTES={len(blob)}")
    print(f"ARCHIVE_SHA256={hashlib.sha256(blob).hexdigest()}")
    print(f"SOURCE_TREE_SHA256={tree.hexdigest()}")
    for p in files:
        if p.suffix in EXECUTABLE_SUFFIXES:
            print(f"  exec 0755: {p.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

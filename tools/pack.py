"""Deploy the moe_rebuild tree to every node of chec-mi300-7.

Pushes a tarball to node-0, then fans out to all nodes so the code is
node-local and identical everywhere. Asymmetry between nodes is the single
most common cause of "works on 14 nodes, fails on 1".
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = "/scratch/rfull/moe"
INCLUDE = ["moe_rebuild", "tools", "tests"]


def build_tar() -> bytes:
    buf = io.BytesIO()
    n = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for top in INCLUDE:
            for p in sorted((REPO / top).rglob("*")):
                if p.is_dir() or "__pycache__" in p.parts or p.suffix == ".pyc":
                    continue
                data = p.read_bytes().replace(b"\r\n", b"\n")  # never ship CR
                if b"\r\n" in data:
                    raise SystemExit(f"CR byte survived in {p}")
                info = tarfile.TarInfo(str(p.relative_to(REPO)).replace("\\", "/"))
                info.size = len(data)
                info.mode = 0o755 if p.suffix in (".sh",) else 0o644
                tf.addfile(info, io.BytesIO(data))
                n += 1
    print(f"packed {n} files, {buf.tell()} bytes")
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnodes", type=int, default=15)
    ap.add_argument(
        "--b64-out",
        default=str(Path.home() / ".deeporca/agents/pretrain/files/_deploy.b64"),
        help="where to write the base64 tarball that _deploy.py reads")
    args = ap.parse_args()

    tar = build_tar()
    out = Path("_deploy.tar.gz")
    out.write_bytes(tar)
    print(f"wrote {out} ({len(tar)} bytes)")

    # Also emit the base64 form that _deploy.py consumes, next to _deploy.py
    # itself. These were two separate steps once, and a "successful" deploy
    # then shipped a stale tarball for hours: pack.py reported the new byte
    # count while _deploy.py read a _deploy.b64 that nobody had regenerated.
    # Producing both in one step makes that failure mode impossible.
    import base64
    b64_path = Path(args.b64_out).expanduser()
    b64_path.write_text(base64.b64encode(tar).decode())
    print(f"wrote {b64_path} ({b64_path.stat().st_size} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

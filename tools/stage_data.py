"""Stage the tokenized corpus so that nothing Megatron mmaps lives on blobfuse.

Why this exists
---------------
Megatron's `IndexedDataset` mmaps every `.idx` file unconditionally
(`indexed_dataset.py:253`, `_IndexReader.__init__`) -- `--no-mmap-bin-files`
only affects the `.bin` payload. `GPTDataset` additionally writes and then
re-reads its document/sample/shuffle index with
`numpy.load(..., mmap_mode='r')` (`gpt_dataset.py:495/505/515`).

On a blobfuse2 (FUSE) mount, a failed mmap page fault delivers SIGBUS/SIGSEGV
directly to the faulting thread. There is no errno, no Python exception and no
retry -- the rank dies instantly, and every other rank then hangs in the next
collective until the watchdog fires. That failure mode looks exactly like a
distributed deadlock and is extremely expensive to misdiagnose.

Layout produced
---------------
    /scratch/rfull/data/<corpus>/<shard>.idx   real file, copied to local ext4
    /scratch/rfull/data/<corpus>/<shard>.bin   symlink -> blobfuse payload
    /scratch/rfull/data-cache/                 local dir for GPTDataset .npy

`.idx` totals ~16.4 GiB across 487 shards; `/scratch` has >1 TiB free, so the
copy is cheap. The 3.8 TiB of `.bin` payload stays on blobfuse and is read with
pread (never mapped) because we always pass `--no-mmap-bin-files`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BLOB_DATA = Path("/scratch/workspaceblobstore/chec/pretrain/data")
LOCAL_DATA = Path("/scratch/rfull/data")
LOCAL_CACHE = Path("/scratch/rfull/data-cache")

MAGIC = b"MMIDIDX\x00\x00"
DTYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 8, 7: 4, 8: 2}
DTYPE_NAME = {
    1: "uint8", 2: "int8", 3: "int16", 4: "int32",
    5: "int64", 6: "float64", 7: "float32", 8: "uint16",
}


def read_idx_header(path: Path) -> dict:
    """Parse an MMIDIDX header. Small sequential read -- safe on FUSE."""
    with open(path, "rb") as f:
        magic = f.read(9)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}")
        version = struct.unpack("<Q", f.read(8))[0]
        code = struct.unpack("<B", f.read(1))[0]
        n_seq = struct.unpack("<Q", f.read(8))[0]
        n_doc = struct.unpack("<Q", f.read(8))[0]
    return {
        "version": version,
        "dtype_code": code,
        "dtype": DTYPE_NAME.get(code, f"code{code}"),
        "sequences": n_seq,
        "documents": n_doc,
    }


def discover(corpora: list[str] | None = None) -> dict[str, list[Path]]:
    """Map corpus -> sorted list of .idx paths (recursive over part_* dirs)."""
    out: dict[str, list[Path]] = {}
    names = corpora or sorted(p.name for p in BLOB_DATA.iterdir() if p.is_dir())
    for name in names:
        root = BLOB_DATA / name
        if not root.is_dir():
            raise SystemExit(f"no such corpus: {root}")
        idxs = sorted(root.rglob("*.idx"))
        if idxs:
            out[name] = idxs
    return out


def _stage_one(idx: Path, corpus: str, force: bool) -> tuple[str, int, str]:
    """Copy one .idx locally and symlink its .bin. Returns (prefix, bytes, status)."""
    rel = idx.relative_to(BLOB_DATA / corpus)
    dst_dir = LOCAL_DATA / corpus / rel.parent
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_idx = dst_dir / idx.name
    src_bin = idx.with_suffix(".bin")
    dst_bin = dst_dir / src_bin.name

    if not src_bin.exists():
        return (str(dst_idx.with_suffix("")), 0, "MISSING_BIN")

    nbytes = 0
    src_size = idx.stat().st_size
    if force or not dst_idx.exists() or dst_idx.stat().st_size != src_size:
        tmp = dst_idx.with_suffix(".idx.tmp")
        shutil.copyfile(idx, tmp)          # sequential read: safe on FUSE
        os.replace(tmp, dst_idx)           # atomic
        nbytes = src_size
        status = "copied"
    else:
        status = "cached"

    # .bin stays on blob; symlink keeps the path prefix uniform.
    if dst_bin.is_symlink() or dst_bin.exists():
        if os.readlink(dst_bin) != str(src_bin) if dst_bin.is_symlink() else True:
            dst_bin.unlink()
            dst_bin.symlink_to(src_bin)
    else:
        dst_bin.symlink_to(src_bin)

    return (str(dst_idx.with_suffix("")), nbytes, status)


def stage(corpora: list[str] | None, workers: int, force: bool) -> dict:
    found = discover(corpora)
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

    report: dict = {"corpora": {}, "total_idx_bytes": 0, "total_tokens": 0}
    for corpus, idxs in found.items():
        hdr = read_idx_header(idxs[0])
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda p: _stage_one(p, corpus, force), idxs))

        bad = [r for r in results if r[2] == "MISSING_BIN"]
        copied = sum(1 for r in results if r[2] == "copied")
        nbytes = sum(r[1] for r in results)

        tok = 0
        for prefix, _, st in results:
            if st == "MISSING_BIN":
                continue
            b = Path(prefix + ".bin")
            try:
                tok += b.stat().st_size // DTYPE_SIZE[hdr["dtype_code"]]
            except OSError:
                pass

        report["corpora"][corpus] = {
            "shards": len(idxs),
            "copied": copied,
            "cached": len(idxs) - copied - len(bad),
            "missing_bin": [b[0] for b in bad],
            "dtype": hdr["dtype"],
            "tokens": tok,
            "prefixes": [r[0] for r in results if r[2] != "MISSING_BIN"],
        }
        report["total_idx_bytes"] += nbytes
        report["total_tokens"] += tok
        print(
            f"  {corpus:24s} shards={len(idxs):4d} copied={copied:4d} "
            f"dtype={hdr['dtype']} tokens={tok/1e9:8.2f}B",
            flush=True,
        )
    return report


def verify(report: dict) -> int:
    """Open every staged shard through Megatron's own reader.

    This is the real test: it exercises the same mmap path training will use,
    but on local ext4 where a page fault cannot SIGBUS.
    """
    sys.path.insert(0, "/scratch/rfull/megatron-lm")
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    bad = 0
    checked = 0
    for corpus, info in report["corpora"].items():
        for prefix in info["prefixes"]:
            try:
                ds = IndexedDataset(prefix, multimodal=False, mmap=False)
                n = len(ds)
                if n == 0:
                    print(f"  EMPTY {prefix}")
                    bad += 1
                    continue
                _ = ds[0]
                _ = ds[n - 1]
                checked += 1
                del ds
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {prefix}: {type(e).__name__}: {e}")
                bad += 1
    print(f"  verified {checked} shards, {bad} bad")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--report", default="/scratch/rfull/data_stage_report.json")
    args = ap.parse_args()

    print(f"staging .idx -> {LOCAL_DATA} (.bin symlinked to blob)")
    report = stage(args.corpora, args.workers, args.force)
    print(
        f"TOTAL idx_copied={report['total_idx_bytes']/2**30:.2f} GiB "
        f"tokens={report['total_tokens']/1e9:.1f}B"
    )

    rc = 0
    if args.verify:
        print("verifying via megatron IndexedDataset ...")
        rc = verify(report)

    Path(args.report).write_text(json.dumps(report, indent=1))
    print(f"report -> {args.report}")
    return 1 if rc else 0


if __name__ == "__main__":
    raise SystemExit(main())

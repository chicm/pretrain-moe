#!/usr/bin/env python3
"""Build Megatron IndexedDataset (.idx) sidecars for flat token shards.

The R-Full corpus is stored as flat little-endian ``uint32`` token streams with
an explicit end-of-text token separating documents.  Megatron's
``IndexedDataset`` needs a ``.idx`` sidecar next to each ``.bin``; it does not
need the ``.bin`` rewritten, because:

* the payload is already a contiguous little-endian token stream, and
* every token id is far below ``2**31`` (padded vocab is 151936), so the
  ``uint32`` bytes are bit-identical to the ``int32`` bytes MCore supports.

MCore's ``DType`` enum has no ``uint32`` code, so the sidecar declares ``int32``
(code 4).  This is a pure reinterpretation of identical bytes, not a conversion,
and the tool verifies that claim by asserting the observed maximum token id.

Each inter-EOT span becomes one document holding exactly one sequence, matching
what ``preprocess_data.py`` produces for ordinary GPT corpora.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from typing import Dict, List, Optional

import numpy as np

INDEX_HEADER = b"MMIDIDX\x00\x00"
INDEX_VERSION = 1
DTYPE_CODE_INT32 = 4
ITEMSIZE = 4
INT32_MAX = 2**31 - 1

# Scan granularity.  Large enough to amortise blob-mount latency, small enough
# that the transient boolean mask stays modest.
CHUNK_TOKENS = 1 << 26  # 64Mi tokens == 256 MiB per chunk


def scan_document_lengths(bin_path: str, eot: int) -> Dict[str, object]:
    """Return document lengths for ``bin_path`` split on ``eot``.

    Every document keeps its trailing EOT token, so the document lengths sum to
    the token count whenever the shard ends on an EOT boundary.  A trailing
    partial document (no final EOT) is preserved as its own document.
    """
    size = os.path.getsize(bin_path)
    if size % ITEMSIZE:
        raise ValueError(f"{bin_path}: size {size} is not a multiple of {ITEMSIZE}")
    total_tokens = size // ITEMSIZE

    stream = np.memmap(bin_path, dtype=np.uint32, mode="r")
    lengths: List[np.ndarray] = []
    max_token = 0
    prev_end = 0  # absolute index one past the previous document's EOT
    scanned = 0

    while scanned < total_tokens:
        stop = min(scanned + CHUNK_TOKENS, total_tokens)
        chunk = np.asarray(stream[scanned:stop])
        if chunk.size:
            max_token = max(max_token, int(chunk.max()))
        hits = np.flatnonzero(chunk == eot)
        if hits.size:
            ends = hits.astype(np.int64) + scanned + 1  # inclusive of the EOT
            boundaries = np.concatenate(([np.int64(prev_end)], ends))
            lengths.append(np.diff(boundaries))
            prev_end = int(ends[-1])
        scanned = stop
        del chunk

    del stream

    if prev_end < total_tokens:  # trailing document with no closing EOT
        lengths.append(np.array([total_tokens - prev_end], dtype=np.int64))

    sequence_lengths = (
        np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int64)
    )
    if sequence_lengths.size and int(sequence_lengths.max()) > INT32_MAX:
        raise ValueError(f"{bin_path}: document longer than int32 capacity")
    if int(sequence_lengths.sum()) != total_tokens:
        raise ValueError(
            f"{bin_path}: lengths sum {int(sequence_lengths.sum())} != tokens {total_tokens}"
        )
    return {
        "sequence_lengths": sequence_lengths,
        "total_tokens": total_tokens,
        "max_token": max_token,
    }


def write_index(idx_path: str, sequence_lengths: np.ndarray) -> None:
    """Write the ``.idx`` sidecar, declaring the payload as ``int32``."""
    lengths32 = sequence_lengths.astype(np.int32, copy=False)
    pointers = np.zeros(lengths32.size, dtype=np.int64)
    if lengths32.size:
        np.cumsum(sequence_lengths[:-1] * ITEMSIZE, out=pointers[1:])
    document_indices = np.arange(lengths32.size + 1, dtype=np.int64)

    tmp_path = f"{idx_path}.tmp"
    with open(tmp_path, "wb") as handle:
        handle.write(INDEX_HEADER)
        handle.write(struct.pack("<Q", INDEX_VERSION))
        handle.write(struct.pack("<B", DTYPE_CODE_INT32))
        handle.write(struct.pack("<Q", int(lengths32.size)))
        handle.write(struct.pack("<Q", int(document_indices.size)))
        handle.write(lengths32.tobytes(order="C"))
        handle.write(pointers.tobytes(order="C"))
        handle.write(document_indices.tobytes(order="C"))
    os.replace(tmp_path, idx_path)  # atomic publish


def build_one(bin_path: str, eot: int, force: bool = False) -> Dict[str, object]:
    idx_path = f"{os.path.splitext(bin_path)[0]}.idx"
    if os.path.exists(idx_path) and not force:
        return {"bin": bin_path, "idx": idx_path, "status": "SKIP_EXISTS"}

    started = time.time()
    scan = scan_document_lengths(bin_path, eot)
    sequence_lengths = scan["sequence_lengths"]
    if int(scan["max_token"]) > INT32_MAX:
        raise ValueError(
            f"{bin_path}: max token {scan['max_token']} exceeds int32; "
            "byte-identical reinterpretation is invalid"
        )
    write_index(idx_path, sequence_lengths)
    elapsed = time.time() - started

    return {
        "bin": bin_path,
        "idx": idx_path,
        "status": "BUILT",
        "documents": int(sequence_lengths.size),
        "tokens": int(scan["total_tokens"]),
        "max_token": int(scan["max_token"]),
        "mean_doc_tokens": round(
            float(scan["total_tokens"]) / max(1, sequence_lengths.size), 2
        ),
        "idx_bytes": os.path.getsize(idx_path),
        "seconds": round(elapsed, 1),
        "gb_per_s": round(scan["total_tokens"] * ITEMSIZE / 1024**3 / max(1e-9, elapsed), 3),
    }


def verify_one(bin_path: str, expect_documents: Optional[int] = None) -> Dict[str, object]:
    """Re-open the pair through MCore's own reader and cross-check it."""
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    prefix = os.path.splitext(bin_path)[0]
    dataset = IndexedDataset(prefix, multimodal=False, mmap=True)
    count = len(dataset)
    first = dataset[0]
    last = dataset[count - 1]
    total = int(dataset.sequence_lengths.sum())
    result = {
        "prefix": prefix,
        "documents": count,
        "total_tokens": total,
        "dtype": str(dataset.index.dtype),
        "first_doc_tokens": int(first.shape[0]),
        "first_doc_head": [int(x) for x in first[:8]],
        "first_doc_tail": [int(x) for x in first[-4:]],
        "last_doc_tokens": int(last.shape[0]),
        "bin_bytes": os.path.getsize(bin_path),
    }
    result["tokens_match_bin"] = total * ITEMSIZE == result["bin_bytes"]
    if expect_documents is not None:
        result["documents_match"] = count == expect_documents
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", action="append", default=[], help="shard to index")
    parser.add_argument("--bin-list", help="file with one shard path per line")
    parser.add_argument("--eot", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true", help="verify via MCore reader")
    parser.add_argument("--output", help="write a JSON report here")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    targets = list(args.bin)
    if args.bin_list:
        with open(args.bin_list, "r", encoding="utf-8") as handle:
            targets.extend(line.strip() for line in handle if line.strip())
    if not targets:
        parser.error("no shards given")

    # Deterministic, contiguous striping so each worker owns a disjoint subset.
    mine = [p for i, p in enumerate(sorted(targets)) if i % args.workers == args.worker]

    report = {
        "worker": args.worker,
        "workers": args.workers,
        "assigned": len(mine),
        "eot": args.eot,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [],
        "failures": [],
    }

    for path in mine:
        try:
            entry = build_one(path, args.eot, force=args.force)
            if args.verify and entry["status"] == "BUILT":
                entry["verify"] = verify_one(path, entry.get("documents"))
            report["results"].append(entry)
            print(json.dumps(entry), flush=True)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failure = {"bin": path, "error": f"{type(exc).__name__}: {exc}"}
            report["failures"].append(failure)
            print(json.dumps(failure), file=sys.stderr, flush=True)

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["built"] = sum(1 for r in report["results"] if r["status"] == "BUILT")
    report["skipped"] = sum(1 for r in report["results"] if r["status"] == "SKIP_EXISTS")
    report["failed"] = len(report["failures"])

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        digest = hashlib.sha256(
            open(args.output, "rb").read()  # noqa: SIM115 - short-lived
        ).hexdigest()
        print(f"REPORT_SHA256={digest}", flush=True)

    print(
        f"WORKER_SUMMARY worker={args.worker}/{args.workers} "
        f"built={report['built']} skipped={report['skipped']} failed={report['failed']}",
        flush=True,
    )
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

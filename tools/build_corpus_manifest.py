"""DATA-001: build and freeze the canonical R-Full corpus manifest.

Enumerates every physically present shard of the eight selected sources,
records exact byte/token counts, samples payload token ids to re-verify the
vocabulary bound, and emits a self-hashed manifest.

This exists because the earlier inventory trusted declared index values. The
top-level source indexes turned out to be stale (dclm declares 5 shards, 2
exist), so "declared" and "present" must be reported as separate numbers and
the manifest must be built from real files only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rfull.dataset import (  # noqa: E402
    EOT_ID, MAX_PAYLOAD_ID_EXCLUSIVE, PADDED_VOCAB, SOURCE_DIRS, TOKEN_BYTES,
    discover_source,
)
from rfull.data_scheduler import SOURCE_IDS, SOURCE_QUOTAS  # noqa: E402

import numpy as np  # noqa: E402

DEFAULT_ROOT = ("/scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore"
                "/chec/pretrain/data")
SAMPLE_SHARDS_PER_SOURCE = 3
SAMPLE_TOKENS = 262144


def sample_payload(path: str, n_tokens: int) -> dict:
    """Bounded sequential read from head/middle/tail of one shard."""
    size = os.path.getsize(path)
    spots = [0, max(0, size // 2 - n_tokens * TOKEN_BYTES // 2),
             max(0, size - n_tokens * TOKEN_BYTES)]
    mx, has_eot, seen = 0, False, 0
    fd = os.open(path, os.O_RDONLY)
    try:
        for off in spots:
            off -= off % TOKEN_BYTES
            want = min(n_tokens * TOKEN_BYTES, size - off)
            if want <= 0:
                continue
            b = os.pread(fd, want, off)
            a = np.frombuffer(b, dtype=np.uint32)
            if a.size:
                mx = max(mx, int(a.max()))
                has_eot = has_eot or bool((a == EOT_ID).any())
                seen += a.size
    finally:
        os.close(fd)
    return {"max_id": mx, "has_eot": has_eot, "sampled_tokens": seen}


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT)
    t0 = time.time()
    man = {
        "manifest_version": "rfull-corpus-v1",
        "root": str(root),
        "token_dtype": "uint32_le",
        "eot_id": EOT_ID,
        "padded_vocab_size": PADDED_VOCAB,
        "max_payload_id_exclusive": MAX_PAYLOAD_ID_EXCLUSIVE,
        "sources": {},
    }

    grand_tokens = 0
    grand_shards = 0
    all_ok = True

    for idx, sid in enumerate(SOURCE_IDS):
        shards = discover_source(root, sid)
        tokens = sum(s.n_tokens for s in shards)
        entries = [{"path": str(Path(s.path).relative_to(root)),
                    "tokens": s.n_tokens} for s in shards]

        # Hash the shard list so any add/remove/resize changes the manifest.
        h = hashlib.sha256()
        h.update(b"rfull-source-v1\0")
        h.update(sid.encode())
        for e in entries:
            h.update(e["path"].encode())
            h.update(str(e["tokens"]).encode())

        step = max(1, len(shards) // SAMPLE_SHARDS_PER_SOURCE)
        samples = {}
        src_max = 0
        src_eot = False
        for s in shards[::step][:SAMPLE_SHARDS_PER_SOURCE]:
            r = sample_payload(s.path, SAMPLE_TOKENS)
            samples[str(Path(s.path).relative_to(root))] = r
            src_max = max(src_max, r["max_id"])
            src_eot = src_eot or r["has_eot"]

        ok = src_max < MAX_PAYLOAD_ID_EXCLUSIVE and src_eot
        all_ok &= ok
        man["sources"][sid] = {
            "source_index": idx,
            "dir": SOURCE_DIRS[sid],
            "cycle_quota": SOURCE_QUOTAS[idx],
            "n_shards": len(shards),
            "tokens": tokens,
            "tokens_B": round(tokens / 1e9, 3),
            "shard_list_sha256": h.hexdigest(),
            "payload_samples": samples,
            "observed_max_id": src_max,
            "contains_eot": src_eot,
            "payload_bound_ok": ok,
            "shards": entries,
        }
        grand_tokens += tokens
        grand_shards += len(shards)
        print(f"  {sid:14s} {len(shards):4d} shards  {tokens/1e9:8.3f}B  "
              f"max_id={src_max}  eot={src_eot}", flush=True)

    man["totals"] = {
        "n_sources": len(SOURCE_IDS),
        "n_shards": grand_shards,
        "tokens": grand_tokens,
        "tokens_B": round(grand_tokens / 1e9, 3),
        "all_payload_bounds_ok": all_ok,
        "scan_seconds": round(time.time() - t0, 1),
    }

    body = json.dumps(man, indent=1, sort_keys=True)
    man["manifest_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    final = json.dumps(man, indent=1, sort_keys=True)

    out = Path(os.environ.get("RFULL_EVIDENCE_DIR", ".")) / "corpus_manifest.json"
    out.write_text(final, encoding="utf-8")

    print(json.dumps(man["totals"], indent=1))
    print(f"CORPUS_MANIFEST_SHA256 {man['manifest_sha256']}")
    print(f"WROTE {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

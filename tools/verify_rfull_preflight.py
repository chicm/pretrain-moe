#!/usr/bin/env python3
"""Verify an R-Full multi-node collective preflight.

Node exit codes are not the contract: a rank can exit 0 having skipped work,
and RCCL prints "Abort COMPLETE" during perfectly normal teardown.  The
contract is the per-rank JSON markers -- every expected rank must report every
check, and every check must pass.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

CHECK_MARKERS = {
    "PREFLIGHT_RANK_UP": "rendezvous + device bind",
    "PREFLIGHT_ALLREDUCE": "world all-reduce",
    "PREFLIGHT_CROSS_NODE_P2P": "cross-node point-to-point",
    "PREFLIGHT_EP_ALL_TO_ALL": "expert-parallel all-to-all",
    "PREFLIGHT_EDP_ALL_REDUCE": "expert-data-parallel all-reduce",
    "PREFLIGHT_RANK_RESULT": "per-rank aggregate",
    "PREFLIGHT_RANK_COMPLETE": "clean teardown",
}


def json_markers(text: str) -> list[dict]:
    """Extract JSON objects, tolerating concurrent ranks writing without \n."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                except Exception:
                    pass
                else:
                    if isinstance(obj, dict) and "marker" in obj:
                        out.append(obj)
                start = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-dir", required=True)
    ap.add_argument("--expected-world", type=int, required=True)
    ap.add_argument("--expected-nodes", type=int, required=True)
    ap.add_argument("--acceptance-output")
    args = ap.parse_args()

    logs = sorted(glob.glob(os.path.join(args.preflight_dir, "node-*.log")))
    print(f"node logs: {len(logs)} (expected {args.expected_nodes})")
    if len(logs) != args.expected_nodes:
        print("FAIL: wrong number of node logs", file=sys.stderr)
        return 1

    markers: list[dict] = []
    for path in logs:
        with open(path, errors="replace") as fh:
            markers.extend(json_markers(fh.read()))
    print(f"markers parsed: {len(markers)}")

    failures: list[str] = []
    per_check: dict[str, dict] = {}
    for marker, label in CHECK_MARKERS.items():
        got = [m for m in markers if m.get("marker") == marker]
        ranks = sorted({m["rank"] for m in got})
        failed = sorted({m["rank"] for m in got if m.get("ok") is False})
        per_check[marker] = {"ranks": len(ranks), "failed_ranks": failed}
        status = "OK"
        if len(ranks) != args.expected_world:
            status = f"MISSING ({len(ranks)}/{args.expected_world})"
            failures.append(f"{marker}: only {len(ranks)}/{args.expected_world} ranks")
        if failed:
            status = f"FAILED on ranks {failed}"
            failures.append(f"{marker}: failed on {failed}")
        print(f"  {marker:32} {len(ranks):>4}/{args.expected_world}  {label:34} {status}")

    hosts = sorted({m.get("host") for m in markers if m.get("host")})
    print(f"distinct hosts: {len(hosts)} (expected {args.expected_nodes})")
    if len(hosts) != args.expected_nodes:
        failures.append(f"hosts: {len(hosts)} != {args.expected_nodes}")

    # Transport evidence: inter-node RCCL must ride InfiniBand, not sockets.
    ib = sock = 0
    for path in logs:
        with open(path, errors="replace") as fh:
            blob = fh.read()
        ib += blob.count("NET/IB")
        sock += blob.count("NET/Socket")
    print(f"transport: NET/IB lines={ib}  NET/Socket lines={sock}")
    if ib == 0:
        failures.append("no NET/IB evidence")

    probe = os.path.join(args.preflight_dir, "probe.py")
    probe_sha = hashlib.sha256(open(probe, "rb").read()).hexdigest() if os.path.exists(probe) else None

    result = "PASS" if not failures else "FAIL"
    acceptance = {
        "preflight_dir": args.preflight_dir,
        "expected_world": args.expected_world,
        "expected_nodes": args.expected_nodes,
        "hosts": hosts,
        "probe_sha256": probe_sha,
        "checks": per_check,
        "net_ib_lines": ib,
        "net_socket_lines": sock,
        "failures": failures,
        "result": result,
    }
    if args.acceptance_output:
        blob = json.dumps(acceptance, indent=1, sort_keys=True).encode()
        with open(args.acceptance_output, "wb") as fh:
            fh.write(blob)
        print(f"ACCEPTANCE={args.acceptance_output}")
        print(f"ACCEPTANCE_SHA256={hashlib.sha256(blob).hexdigest()}")

    for f in failures:
        print(f"FAILURE: {f}", file=sys.stderr)
    print(f"PREFLIGHT_RESULT={result}")
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

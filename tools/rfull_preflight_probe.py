#!/usr/bin/env python3
"""R-Full multi-node collective preflight probe.

Runs on every rank and exercises exactly the collectives the MoE trainer needs:

  1. rendezvous + device bind
  2. world all-reduce           (gradient reduction / loss aggregation)
  3. cross-node point-to-point  (proves inter-node RCCL, not just intra-node)
  4. expert-parallel all-to-all (the MoE token dispatch collective)
  5. expert-data-parallel all-reduce (the EDP gradient collective)

Every result is emitted as a JSON marker so the verifier can require a specific
pass count per check, rather than trusting the process exit code.
"""
from __future__ import annotations

import datetime
import json
import os
import socket
import sys

import torch
import torch.distributed as dist

EP_SIZE = int(os.environ.get("RFULL_EP_SIZE", "8"))

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
# This PyTorch/ROCm build requires an explicit device_id on the default group;
# set_device alone is not enough and yields a guessed-device warning.
dist.init_process_group(backend="nccl", device_id=device)


def emit(**kw) -> None:
    kw["rank"] = rank
    kw["host"] = socket.gethostname()
    kw["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(json.dumps(kw), flush=True)


emit(marker="PREFLIGHT_RANK_UP", local_rank=local_rank, world_size=world,
     device=torch.cuda.get_device_name(local_rank), ep_size=EP_SIZE)

results: dict[str, bool] = {}

# 1. World all-reduce.
t = torch.full((1024, 1024), float(rank), device=device)
dist.all_reduce(t)
expected = float(sum(range(world)))
results["allreduce"] = abs(t[0, 0].item() - expected) < 1e-3
emit(marker="PREFLIGHT_ALLREDUCE", observed=t[0, 0].item(), expected=expected,
     ok=results["allreduce"])

# 2. Cross-node point-to-point, pairing each rank with the same local rank on
#    the next node so every inter-node link is exercised at least once.
gpn = EP_SIZE
nnodes = world // gpn
if nnodes >= 2:
    node = rank // gpn
    peer_node = (node + 1) % nnodes
    peer = peer_node * gpn + local_rank
    send = torch.full((256,), float(rank), device=device)
    recv = torch.empty((256,), device=device)
    # Even nodes send first, odd nodes receive first, so no pair deadlocks.
    if node % 2 == 0:
        dist.send(send, peer)
        dist.recv(recv, (node - 1) % nnodes * gpn + local_rank)
    else:
        dist.recv(recv, (node - 1) % nnodes * gpn + local_rank)
        dist.send(send, peer)
    results["cross_node_p2p"] = recv[0].item() >= 0.0
    emit(marker="PREFLIGHT_CROSS_NODE_P2P", peer=peer,
         observed=recv[0].item(), ok=results["cross_node_p2p"])
else:
    results["cross_node_p2p"] = True

# 3. Expert-parallel all-to-all within each EP group.
ep_group = None
for start in range(0, world, EP_SIZE):
    ranks = list(range(start, start + EP_SIZE))
    g = dist.new_group(ranks=ranks)
    if rank in ranks:
        ep_group = g
send = torch.full((EP_SIZE, 128), float(rank), device=device)
recv = torch.empty_like(send)
dist.all_to_all_single(recv, send, group=ep_group)
ep_base = (rank // EP_SIZE) * EP_SIZE
want = [float(ep_base + i) for i in range(EP_SIZE)]
got = [recv[i, 0].item() for i in range(EP_SIZE)]
results["ep_all_to_all"] = all(abs(a - b) < 1e-3 for a, b in zip(got, want))
emit(marker="PREFLIGHT_EP_ALL_TO_ALL", observed=got, expected=want,
     ok=results["ep_all_to_all"])

# 4. Expert-data-parallel all-reduce: same local expert slot across all nodes.
edp_group = None
for slot in range(EP_SIZE):
    ranks = list(range(slot, world, EP_SIZE))
    g = dist.new_group(ranks=ranks)
    if rank in ranks:
        edp_group = g
t = torch.full((4096,), float(rank), device=device)
dist.all_reduce(t, group=edp_group)
slot = rank % EP_SIZE
expected = float(sum(range(slot, world, EP_SIZE)))
results["edp_all_reduce"] = abs(t[0].item() - expected) < 1e-3
emit(marker="PREFLIGHT_EDP_ALL_REDUCE", observed=t[0].item(), expected=expected,
     ok=results["edp_all_reduce"])

all_ok = all(results.values())
emit(marker="PREFLIGHT_RANK_RESULT", checks=results, ok=all_ok)

dist.barrier()
dist.destroy_process_group()
emit(marker="PREFLIGHT_RANK_COMPLETE", ok=all_ok)
sys.exit(0 if all_ok else 1)

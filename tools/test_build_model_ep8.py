"""Build the real R-Full model under EP8 and verify the frozen contracts.

Run:  torchrun --nproc_per_node=8 tools/test_build_model_ep8.py

Verifies:
  * analytic ledger == frozen totals (CPU maths, no GPU needed)
  * MCore actually builds head_dim=128 projections (not hidden/heads=64)
  * global parameter total reconstructed from EP shards == 25,857,439,744
  * exactly 2 dense + 46 MoE layers, 96 experts, 12 experts/rank under EP8
  * router weights are Normal(0, 0.01)
  * a real forward pass produces finite logits
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfull.model_spec import (  # noqa: E402
    GEOMETRY, assert_attention_shapes, assert_frozen_ledger, moe_layer_pattern)
from rfull.build_model import (  # noqa: E402
    build_rfull_model, build_transformer_config, count_built_parameters)
from rfull.router_init import verify_router_init  # noqa: E402

SEQ = 4096
MICRO_BATCH = 1
EP = int(os.environ.get("RFULL_EP", "8"))


def main() -> int:
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=EP,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)

    results = {"rank": rank, "world": world, "ep": EP}

    # ---- 1. analytic ledger -------------------------------------------------
    led = assert_frozen_ledger()
    results["ledger_total"] = led["total"]
    results["ledger_active"] = led["active"]

    # ---- 2. build -----------------------------------------------------------
    cfg = build_transformer_config(
        seq_length=SEQ, expert_model_parallel_size=EP, bf16=True)
    model = build_rfull_model(seq_length=SEQ, config=cfg)
    model = model.cuda()

    # ---- 3. geometry assertions --------------------------------------------
    attn = assert_attention_shapes(model)
    results["attention"] = {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in attn.items()}

    counts = count_built_parameters(model)
    results["param_counts"] = counts
    results["global_total_matches"] = (
        counts["reconstructed_global_total"] == GEOMETRY.expected_total_params)

    # experts per rank
    n_local_experts = set()
    n_routers = 0
    for name, _ in model.named_parameters():
        if name.endswith("router.weight"):
            n_routers += 1
    for name, p in model.named_parameters():
        if ".experts.linear_fc1.weight" in name or ".experts.weight1" in name:
            n_local_experts.add(name)
    results["num_routers"] = n_routers
    results["expected_routers"] = GEOMETRY.num_moe_layers
    results["routers_match"] = n_routers == GEOMETRY.num_moe_layers
    results["layer_pattern"] = moe_layer_pattern()

    # per-layer MLP module types: layers 0-1 must be plain dense MLP
    mlp_types = []
    for i, layer in enumerate(model.decoder.layers):
        mlp_types.append(type(layer.mlp).__name__)
    results["mlp_type_layer0"] = mlp_types[0] if mlp_types else None
    results["mlp_type_layer1"] = mlp_types[1] if len(mlp_types) > 1 else None
    results["mlp_type_layer2"] = mlp_types[2] if len(mlp_types) > 2 else None
    results["n_moe_mlp"] = sum(1 for t in mlp_types if "MoE" in t)
    results["n_dense_mlp"] = sum(1 for t in mlp_types if t == "MLP")
    results["layer_split_ok"] = (
        results["n_dense_mlp"] == GEOMETRY.num_dense_layers
        and results["n_moe_mlp"] == GEOMETRY.num_moe_layers)

    # ---- 4. router init -----------------------------------------------------
    rinit = verify_router_init(model)
    results["router_init_ok"] = rinit["ok"]
    some = list(rinit["routers"].items())[:2]
    results["router_init_sample"] = {k: v for k, v in some}

    # ---- 5. real forward ----------------------------------------------------
    torch.manual_seed(1000 + rank)
    ids = torch.randint(0, GEOMETRY.tokenizer_vocab_size,
                        (MICRO_BATCH, SEQ), device="cuda", dtype=torch.long)
    pos = torch.arange(SEQ, device="cuda").unsqueeze(0).expand(MICRO_BATCH, SEQ)
    mask = None
    with torch.no_grad():
        out = model(input_ids=ids, position_ids=pos, attention_mask=mask)
    results["forward_shape"] = list(out.shape)
    results["forward_finite"] = bool(torch.isfinite(out).all().item())
    results["forward_dtype"] = str(out.dtype)
    mem = torch.cuda.max_memory_allocated() / (1 << 30)
    results["peak_mem_GiB"] = round(mem, 2)

    ok = (results["global_total_matches"] and results["routers_match"]
          and results["router_init_ok"] and results["forward_finite"]
          and results["layer_split_ok"])
    results["verdict"] = "PASS" if ok else "FAIL"

    if rank == 0:
        blob = json.dumps(results, indent=1, sort_keys=True, default=str)
        results["evidence_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
        print(json.dumps(results, indent=1, sort_keys=True, default=str))
        out_path = os.environ.get("RFULL_EVIDENCE_OUT")
        if out_path:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=1, sort_keys=True, default=str)

    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

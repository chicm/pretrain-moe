"""Release MCore token-dispatcher state that survives each training step.

The dispatcher keeps its per-step working set on the module itself
(``self.probs``, ``self.routing_map``, ``self.input_splits``, ... around
token_dispatcher.py:478-480) and never clears it. With 46 MoE layers that
leaves 332 live tensors behind after every update:

    46 x (1024, 96) float32   router probs
    46 x (1024, 96) bool      routing map
    46 x (8, 12) / (6144,) / (8,) int64   all-to-all split tables
     5 x (33554432,) uint8    128 MiB workspace buffers  (640 MiB)

Memory therefore climbs step after step -- measured 8.55 GiB allocated after
build, 17.53 GiB after one backward, and only 8.88 GiB after zero_grad -- until
the process is OOM-killed on update 2 or 3 (rc=137). When the allocator is
close to the edge, autograd surfaces this as the misleading
"Trying to backward through the graph a second time", which sent an earlier
investigation after a non-existent graph-retention bug.

Nothing here changes numerics: every attribute cleared is written afresh by the
next forward before it is read.
"""
from __future__ import annotations

from typing import List

import torch

__all__ = ["clear_dispatcher_state", "collect_dispatchers"]

# Attributes the dispatcher assigns during dispatch/combine and never releases.
_STATE_ATTRS = (
    "probs",
    "routing_map",
    "local_probs",
    "local_map",
    "hidden_shape",
    "hidden_shape_before_permute",
    "input_splits",
    "output_splits",
    "output_splits_tp",
    "token_indices",
    "token_probs",
    "dispatched_indices",
    "dispatched_probs",
    "dispatched_routing_map",
    "handle",
    "reversed_local_input_permutation_mapping",
    "reversed_global_input_permutation_mapping",
    "global_input_tokens_local_experts_indices",
    "global_local_map",
    "capacity",
)


def collect_dispatchers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Every MoE token dispatcher in the model."""
    found = []
    for mod in model.modules():
        if type(mod).__name__.endswith("TokenDispatcher"):
            found.append(mod)
        elif hasattr(mod, "token_dispatcher"):
            td = getattr(mod, "token_dispatcher")
            if td is not None and td not in found:
                found.append(td)
    return found


def clear_dispatcher_state(model: torch.nn.Module) -> int:
    """Drop cached dispatcher tensors. Returns how many attributes were cleared.

    Call once per update, after backward. Safe to call at any point between
    steps: the next forward repopulates everything it needs.
    """
    cleared = 0
    for disp in collect_dispatchers(model):
        for attr in _STATE_ATTRS:
            if getattr(disp, attr, None) is not None:
                setattr(disp, attr, None)
                cleared += 1
    return cleared

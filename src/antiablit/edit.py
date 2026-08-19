"""Hard removal of selected MLP neurons (S5).

Destruction, not attenuation: zero the read weights (gate_proj/up_proj rows) AND
the write weights (down_proj column). Attenuation preserves the information and
is trivially restorable (blind per-column rescale); destroying the read side
means the feature is never computed, so restoration requires relearning from
harmful data — outside the threat model. Recipes (neuron indices) are the
persistent artifact; checkpoints are reconstructed on demand.
"""
import json
from pathlib import Path

import torch


@torch.no_grad()
def hard_remove(adapter, neurons: list[tuple[int, int]]):
    """Zero read+write weights for (layer, neuron) pairs, in place."""
    by_layer: dict[int, list[int]] = {}
    for li, j in neurons:
        by_layer.setdefault(li, []).append(j)
    for li, idxs in by_layer.items():
        mlp = adapter.layers[li].mlp
        idx = torch.tensor(idxs, dtype=torch.long, device=adapter.device)
        mlp.gate_proj.weight.data[idx, :] = 0
        mlp.up_proj.weight.data[idx, :] = 0
        mlp.down_proj.weight.data[:, idx] = 0


def save_recipe(path: str | Path, neurons: list[tuple[int, int]], meta: dict):
    Path(path).write_text(json.dumps(
        {"meta": meta, "neurons": [[int(a), int(b)] for a, b in neurons]}, indent=2))


def load_recipe(path: str | Path) -> list[tuple[int, int]]:
    return [tuple(n) for n in json.loads(Path(path).read_text())["neurons"]]

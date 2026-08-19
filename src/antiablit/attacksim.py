"""Weight-space attack simulation — the shared hook state used by decoy
training (RECIPE R2/R9/R12b).

`weight_attack_sim` is lifted VERBATIM from `scripts/line_b1_train.py`
(2026-08-07, GATE-LEAK-PLAN implementation seam 1) with the module-level
adapter closure replaced by an explicit `ad` parameter — no behavior change.
The duplicated copies in `scripts/line_b1_train.py` / `scripts/line_b1_dpo.py`
migrate to this import at their next touch (the `evals.is_escape`/C19
precedent); they are deliberately untouched now (live lanes mid-flight).
"""
import contextlib

import torch

from antiablit.ablation import candidate_directions
from antiablit.modeladapter import ModelAdapter


@contextlib.contextmanager
def weight_attack_sim(ad: ModelAdapter, direction: torch.Tensor):
    """Differentiable orthogonal projection of `direction` on every write-site
    output for the duration of the context (the simulated-abliteration state
    decoy training runs under)."""
    d = direction.to(ad.device).float()
    handles = []
    for site in ad.hook_sites():  # R12b: MoE layers hook the mlp block (write_sites-identical on dense)
        v = d if site.out_scale is None else site.out_scale.to(ad.device).float() * d
        if v.dim() == 1:  # rank-1: original code path (bit-identical, no QR sign risk)
            v = v / v.norm()

            def mk(vv):
                def hook(_m, _i, out):
                    t = out[0] if isinstance(out, tuple) else out  # MoE blocks return (hidden, router_logits)
                    vv2 = vv.to(device=t.device, dtype=t.dtype)  # dispatched shards
                    p = t - (t @ vv2).unsqueeze(-1) * vv2
                    return (p, *out[1:]) if isinstance(out, tuple) else p
                return hook
            handles.append(site.module.register_forward_hook(mk(v)))
        else:
            # RECIPE R9 span sim ([k, hidden] public stack, huihui-9B k=2):
            # QR once, hs - (hs @ Q.T) @ Q — same math as
            # antiablit.ablation.residual_ablation
            Q = torch.linalg.qr(v.T).Q.T  # orthonormal rows, same span

            def mkq(QQ):
                def hook(_m, _i, out):
                    t = out[0] if isinstance(out, tuple) else out  # MoE blocks return (hidden, router_logits)
                    Q2 = QQ.to(device=t.device, dtype=t.dtype)  # dispatched shards
                    p = t - (t @ Q2.T) @ Q2
                    return (p, *out[1:]) if isinstance(out, tuple) else p
                return hook
            handles.append(site.module.register_forward_hook(mkq(Q)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def reestimate_directions(ad: ModelAdapter, dir_h: list[str], dir_b: list[str],
                          attack_layers: list, public_dir: torch.Tensor | None = None) -> dict:
    """Per-layer attack directions from the CURRENT trainee (the decoy-training
    re-estimation step): standardized mean-diff candidates for integer layers,
    the frozen public direction for the "public" ensemble member (R2)."""
    was_training = ad.model.training
    ad.model.eval()
    cands = candidate_directions(ad, dir_h, dir_b)
    if was_training:
        ad.model.train()
    return {li: (public_dir if li == "public" else cands[li]) for li in attack_layers}

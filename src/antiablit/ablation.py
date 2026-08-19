"""Directional-ablation attack (Arditi et al. 2024), used here both as the
defender's elicitation tool and as the adaptive re-attack in verification.

Selection is done with inference-time hooks (project the direction out of the
residual stream after every layer); the permanent attack is weight
orthogonalization of every residual write. For sandwich-norm families (Gemma):
o_proj/down_proj outputs pass through an elementwise RMSNorm scale D before the
residual add, and rms normalization only multiplies by a per-token scalar, so
  (D x) ⟂ d  ⇔  x ⟂ D d.
We therefore project each write matrix against v = D d, not d itself.
"""
import contextlib
import torch

from .modeladapter import ModelAdapter
from .evals import refusal_rate, degenerate_rate


def candidate_directions(adapter: ModelAdapter, harmful: list[str],
                         harmless: list[str], batch_size: int = 32) -> torch.Tensor:
    """Per-layer refusal-direction candidates. [n_layers, hidden] fp32 cpu.

    Standardized (t-statistic) mean difference, not raw: Gemma-family residual
    streams carry massive-activation outlier dimensions (~97% of the norm), and
    the raw diff-in-means is dominated by sampling noise on those dims — ablating
    it removes most of the residual stream and lobotomizes the model. Dividing
    per-dim by pooled std crushes the noise on huge structural dims and surfaces
    dims that actually separate the classes.
    """
    h = adapter.capture_layer_states(harmful, batch_size)   # [n, L, H]
    b = adapter.capture_layer_states(harmless, batch_size)
    pooled_std = ((h.var(dim=0) + b.var(dim=0)) / 2).sqrt()
    d = (h.mean(dim=0) - b.mean(dim=0)) / (pooled_std + 1e-3)
    return d / d.norm(dim=-1, keepdim=True)


@contextlib.contextmanager
def residual_ablation(adapter: ModelAdapter, direction: torch.Tensor):
    """Project `direction` out of the residual stream at the embedding output and
    after every decoder layer, for the duration of the context."""
    D = direction.float()
    if D.dim() == 1:
        D = D[None]
    Q = torch.linalg.qr(D.T.to(adapter.device)).Q.T  # orthonormal rows, same span
    handles = []

    def project(hs):
        QQ = Q.to(device=hs.device, dtype=hs.dtype)  # device_map="auto" shards
        return hs - (hs @ QQ.T) @ QQ

    def embed_hook(_m, _i, out):
        return project(out)

    def layer_hook(_m, _i, out):
        if isinstance(out, tuple):
            return (project(out[0]),) + out[1:]
        return project(out)

    handles.append(adapter.embed.register_forward_hook(embed_hook))
    for layer in adapter.layers:
        handles.append(layer.register_forward_hook(layer_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def select_direction(adapter: ModelAdapter, candidates: torch.Tensor,
                     dev_harmful: list[str], dev_harmless: list[str],
                     layer_frac_range=(0.2, 0.8), gen_tokens=64, batch=16,
                     log=None) -> dict:
    """Rank candidate layers by refusal bypass on dev harmful prompts, with
    degeneracy guards on BOTH harmful and benign outputs — a direction that turns
    harmful prompts into gibberish shows refusal=0 but is useless for eliciting
    the capability. Mild prior toward mid-depth layers on ties."""
    n = candidates.shape[0]
    lo, hi = int(n * layer_frac_range[0]), int(n * layer_frac_range[1])
    results = []
    for li in range(lo, hi):
        d = candidates[li]
        with residual_ablation(adapter, d):
            harm_out = adapter.generate(dev_harmful, gen_tokens, batch)
            benign_out = adapter.generate(dev_harmless[:8], gen_tokens, batch)
        r = {"layer": li, "refusal": refusal_rate(harm_out),
             "degen_harm": degenerate_rate(harm_out),
             "degen_benign": degenerate_rate(benign_out)}
        r["score"] = (r["refusal"] + 0.7 * r["degen_harm"] + 0.7 * r["degen_benign"]
                      + 0.15 * abs(li / n - 0.45))
        results.append(r)
        if log:
            log.info(f"cand L{li}: refusal={r['refusal']:.2f} "
                     f"degenH={r['degen_harm']:.2f} degenB={r['degen_benign']:.2f}")
    ranked = sorted(results, key=lambda r: r["score"])
    return {"ranked_layers": [r["layer"] for r in ranked], "candidates": results}


@torch.no_grad()
def orthogonalize_weights(adapter: ModelAdapter, direction: torch.Tensor,
                          layer_range: tuple | None = None):
    """Permanent attack: remove `direction` from every write into the residual
    stream (embeddings, o_proj, down_proj), folding sandwich-norm scales.

    Skip the embedding when it is tied to the LM head (Gemma family): editing a
    tied embedding also edits the unembedding, which corrupts every logit and
    collapses the model. Its contribution to the refusal signal is negligible
    relative to the per-layer writes.

    `direction` may be [hidden] (rank-1, unchanged behavior) or [k, hidden]
    (project out the span — huihui-9B's public edit is two block directions,
    2026-07-29). Rows are QR-orthonormalized so sequential rank-1 projections
    compose to the exact span projection."""
    D = direction.float()
    if D.dim() == 1:
        D = D[None]
    Q = torch.linalg.qr(D.T.to(adapter.device)).Q.T  # orthonormal rows, same span

    out_emb = adapter.model.get_output_embeddings()
    tied = out_emb is not None and out_emb.weight is adapter.embed.weight
    # banded application (9B finding 2026-07-30: some scales need different
    # directions per layer range; embeddings are edited only for full-range
    # edits since they seed the stream ahead of every band)
    def _in_range(site_name):
        if layer_range is None:
            return True
        li = int(site_name.split(".")[0][1:])
        return layer_range[0] <= li < layer_range[1]
    for dhat in Q:
        if not tied and layer_range is None:
            E = adapter.embed.weight.data
            dE = dhat.to(device=E.device, dtype=E.dtype)  # device_map="auto" shards
            E.sub_(torch.outer(E @ dE, dE))

        for site in adapter.write_sites():
            if not _in_range(site.name):
                continue
            v = dhat if site.out_scale is None else \
                (site.out_scale.to(adapter.device) * dhat)
            v = (v / v.norm())
            W = site.module.weight.data          # [hidden_out, in]
            vW = v.to(device=W.device, dtype=W.dtype)  # device_map="auto" shards
            if W.dim() == 3:  # fused MoE experts: per-expert projection, orientation-aware
                if getattr(site.module, "out_axis", 1) == 2:  # [n_exp, in, hidden_out] (gpt-oss)
                    proj = torch.einsum("eih,h->ei", W, vW)
                    W.sub_(proj.unsqueeze(-1) * vW.view(1, 1, -1))
                else:                                          # [n_exp, hidden_out, in] (Qwen3.5)
                    proj = torch.einsum("h,ehi->ei", vW, W)
                    W.sub_(vW.view(1, -1, 1) * proj.unsqueeze(1))
            else:
                W.sub_(torch.outer(vW, vW @ W))

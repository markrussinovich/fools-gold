"""White-box linear-probe attack helpers (WHITEBOX-PROBE-PLAN.md).

Shared logic for the probe experiment family: teacher-forced residual-stream
capture over EXISTING judged texts, and grouped-by-prompt logistic-probe
training/evaluation. One-pipeline rule: model specifics come in through
ModelAdapter + line config; nothing here is model-specific.

CONTENT HYGIENE (absolute): functions receive prompt/draw text as opaque
strings, tokenize them, and return NUMBERS. Nothing here prints or logs text.

Positions captured per sequence (registered, plan §extract):
  0 p_last  — last prompt token (prompt-end)
  1 g_first — first generated token
  2 g_mid   — middle generated token
  3 g_last  — last generated token
  4 g_mean  — masked mean over all generated tokens
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

POSITIONS = ("p_last", "g_first", "g_mid", "g_last", "g_mean")


@torch.no_grad()
def capture_teacher_forced(adapter, rows, batch_size=8, max_out_tokens=1280):
    """Teacher-force (prompt, output) rows through the adapter's model and
    capture the residual stream (post-block hidden state) at every decoder
    layer and the 5 registered positions.

    rows: list of {"prompt": str, "output": str}
    Returns (acts, aux):
      acts: float16 ndarray [n_rows, n_layers, 5, hidden]
      aux:  list of {"mean_logprob", "n_out_tokens", "out_truncated"}
    Rows whose output tokenizes to 0 tokens raise — callers drop them in the
    labels stage (plan) so shard row indices stay aligned.
    """
    assert not any(hasattr(l, "attn_hc") for l in adapter.layers), \
        "multi-stream residual (mHC) families are out of scope for this probe"
    tok = adapter.tokenizer
    dev = adapter.device
    n_layers, hidden = adapter.n_layers, adapter.hidden_size
    acts = np.empty((len(rows), n_layers, len(POSITIONS), hidden), dtype=np.float16)
    aux = []

    # pre-tokenize (prompt rendered through the ONE registered template)
    enc_rows = []
    for r in rows:
        pid = tok(adapter.render(r["prompt"]), add_special_tokens=False).input_ids
        oid = tok(r["output"], add_special_tokens=False).input_ids
        assert len(oid) > 0, "empty-output row reached capture (drop in labels stage)"
        trunc = len(oid) > max_out_tokens
        oid = oid[:max_out_tokens]
        enc_rows.append((pid, oid, trunc))

    store: dict[int, torch.Tensor] = {}
    ctx = {}  # per-batch: point_idx [B,4], mean_w [B,S]

    def mk_hook(li):
        def hook(_m, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out          # [B,S,H]
            b = hs.shape[0]
            pts = hs[torch.arange(b, device=hs.device).unsqueeze(1),
                     ctx["point_idx"]]                              # [B,4,H]
            gmean = torch.einsum("bs,bsh->bh", ctx["mean_w"].to(hs.dtype), hs)
            store[li] = torch.cat([pts, gmean.unsqueeze(1)], dim=1).half().cpu()
        return hook

    handles = [l.register_forward_hook(mk_hook(i)) for i, l in enumerate(adapter.layers)]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    try:
        for i0 in range(0, len(enc_rows), batch_size):
            chunk = enc_rows[i0:i0 + batch_size]
            b = len(chunk)
            lens_p = [len(p) for p, _, _ in chunk]
            lens_t = [len(p) + len(o) for p, o, _ in chunk]
            S = max(lens_t)
            ids = torch.full((b, S), pad_id, dtype=torch.long)
            mask = torch.zeros((b, S), dtype=torch.long)
            for j, (p, o, _) in enumerate(chunk):
                ids[j, :lens_t[j]] = torch.tensor(p + o)
                mask[j, :lens_t[j]] = 1
            ids, mask = ids.to(dev), mask.to(dev)
            # positions: p_last, g_first, g_mid, g_last
            pi = torch.tensor(
                [[lp - 1, lp, lp + (lt - lp) // 2, lt - 1]
                 for lp, lt in zip(lens_p, lens_t)], device=dev)
            mw = torch.zeros((b, S), device=dev)
            for j, (lp, lt) in enumerate(zip(lens_p, lens_t)):
                mw[j, lp:lt] = 1.0 / (lt - lp)
            ctx["point_idx"], ctx["mean_w"] = pi, mw
            outp = adapter.model(input_ids=ids, attention_mask=mask)
            for li in range(n_layers):
                acts[i0:i0 + b, li] = store[li].numpy()
            # black-box baseline: mean logprob of the output tokens
            logits = outp.logits                                     # [B,S,V] bf16
            for j, (lp, lt) in enumerate(zip(lens_p, lens_t)):
                nll, n = 0.0, lt - lp
                for c0 in range(lp - 1, lt - 1, 256):
                    c1 = min(c0 + 256, lt - 1)
                    sl = logits[j, c0:c1].float()
                    tg = ids[j, c0 + 1:c1 + 1]
                    nll += F.cross_entropy(sl, tg, reduction="sum").item()
                aux.append({"mean_logprob": -nll / n, "n_out_tokens": n,
                            "out_truncated": bool(chunk[j][2])})
            del outp, logits
    finally:
        for h in handles:
            h.remove()
    return acts, aux


# ------------------------------------------------------------------ probes
def _auroc(y, s):
    """Rank AUROC (ties averaged); None if one class."""
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        return None
    order = np.argsort(np.asarray(s, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sv = np.asarray(s, dtype=np.float64)[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_group_auroc(y, s, groups):
    """Mean AUROC restricted to within each group (groups with both classes);
    the prompt-identity-free readout (plan R2)."""
    vals = []
    for g in set(groups):
        m = np.asarray(groups) == g
        a = _auroc(np.asarray(y)[m], np.asarray(s)[m])
        if a is not None:
            vals.append(a)
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def fit_probe(X, y, C=0.1, max_iter=2000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter))
    clf.fit(X, y)
    return clf


def grouped_oof_scores(X, y, groups, C=0.1, n_splits=6, seed=1234):
    """Out-of-fold decision scores under GroupKFold — every row is scored by
    a probe that never saw its prompt. NaN where a fold's train set is
    single-class."""
    from sklearn.model_selection import GroupKFold
    y = np.asarray(y)
    scores = np.full(len(y), np.nan)
    gk = GroupKFold(n_splits=n_splits)
    for tr, te in gk.split(X, y, groups=np.asarray(groups)):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = fit_probe(X[tr], y[tr], C=C)
        scores[te] = clf.decision_function(X[te])
    return scores


def cell_cv_auroc(X, y, groups, C=0.1, n_splits=4, seed=1234):
    """Grid-scan metric: pooled OOF AUROC for one (layer, position) cell."""
    s = grouped_oof_scores(X, y, groups, C=C, n_splits=n_splits, seed=seed)
    m = ~np.isnan(s)
    return _auroc(np.asarray(y)[m], s[m]) if m.any() else None


def cluster_bootstrap_auroc(y, s, groups, n=1000, seed=1234, within=False):
    """Prompt-cluster bootstrap 95% CI for pooled (or within-group mean)
    AUROC — resample GROUPS with replacement (plan r2/A2)."""
    rng = np.random.default_rng(seed)
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    gs = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in gs}
    vals = []
    for _ in range(n):
        pick = rng.choice(gs, len(gs), replace=True)
        if within:
            per = [a for g in pick
                   for a in (_auroc(y[idx_by_g[g]], s[idx_by_g[g]]),)
                   if a is not None]
            v = float(np.mean(per)) if per else None
        else:
            idx = np.concatenate([idx_by_g[g] for g in pick])
            v = _auroc(y[idx], s[idx])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    q = np.quantile(vals, [0.025, 0.975])
    return {"lo": float(q[0]), "hi": float(q[1]), "n_boot": len(vals)}


def paired_cluster_bootstrap_delta(y1, s1, g1, y2, s2, g2, n=1000, seed=1234):
    """95% CI for AUROC(side1) - AUROC(side2) under the SAME prompt resample
    on both sides (shared group-id space — plan r2/A5 attribution delta)."""
    rng = np.random.default_rng(seed)
    y1, s1, g1 = np.asarray(y1), np.asarray(s1), np.asarray(g1)
    y2, s2, g2 = np.asarray(y2), np.asarray(s2), np.asarray(g2)
    shared = np.intersect1d(np.unique(g1), np.unique(g2))
    i1 = {g: np.where(g1 == g)[0] for g in shared}
    i2 = {g: np.where(g2 == g)[0] for g in shared}
    vals = []
    for _ in range(n):
        pick = rng.choice(shared, len(shared), replace=True)
        a1 = _auroc(y1[np.concatenate([i1[g] for g in pick])],
                    s1[np.concatenate([i1[g] for g in pick])])
        a2 = _auroc(y2[np.concatenate([i2[g] for g in pick])],
                    s2[np.concatenate([i2[g] for g in pick])])
        if a1 is not None and a2 is not None:
            vals.append(a1 - a2)
    if not vals:
        return None
    q = np.quantile(vals, [0.025, 0.975])
    return {"lo": float(q[0]), "hi": float(q[1]), "n_boot": len(vals),
            "n_shared_groups": len(shared)}


def mean_bootstrap_ci(vals, n=1000, seed=1234):
    """95% CI of the mean of per-group values (consensus/per-prompt readout
    CIs — plan r2/A2)."""
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = np.mean(vals[rng.integers(0, len(vals), (n, len(vals)))], axis=1)
    q = np.quantile(bs, [0.025, 0.975])
    return {"lo": float(q[0]), "hi": float(q[1]), "n_groups": len(vals)}


def permutation_null(y, s, groups, n=1000, seed=1234):
    """Within-group label-permutation null for pooled AUROC (plan control)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).copy()
    groups = np.asarray(groups)
    out = []
    for _ in range(n):
        yp = y.copy()
        for g in np.unique(groups):
            m = np.where(groups == g)[0]
            yp[m] = yp[rng.permutation(m)]
        a = _auroc(yp, s)
        if a is not None:
            out.append(a)
    q = np.quantile(out, [0.5, 0.95, 0.99]) if out else [None] * 3
    return {"n": len(out), "p50": float(q[0]), "p95": float(q[1]),
            "p99": float(q[2])}

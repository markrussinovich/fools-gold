"""Causal localization of harmful-generation components (S4).

Run on the ABLITERATED model (the post-attack distribution is what we must
break), teacher-forcing (prompt, continuation) pairs. Two stages:

1. Attribution prefilter (fast, all ~350k MLP neurons): first-order estimate of
   the NLL change from mean-ablating each neuron,
     attr[l,j] = E_t[ (mu[l,j] - a[l,j,t]) * dNLL/da[l,j,t] ]
   computed separately on harmful and benign pair sets.
2. Exact causal verification (top-K candidates): actually clamp the neuron to
   its benign mean during teacher forcing and measure delta-NLL on both sets.

Selection is a Pareto filter (H_c high, B_c low) with split-half stability —
never a ratio score. Attention heads are deferred to v2; MLP neurons are the
cheap first pass for the feasibility gate.
"""
import torch

from .modeladapter import ModelAdapter


def encode_pairs(adapter: ModelAdapter, pairs: list[tuple[str, str]], max_len=768):
    """Tokenize (prompt, continuation) with labels masked on the prompt."""
    tok = adapter.tokenizer
    enc_list = []
    for prompt, cont in pairs:
        p_ids = tok(adapter.render(prompt), add_special_tokens=False).input_ids
        c_ids = tok(adapter.render_completion(cont), add_special_tokens=False).input_ids
        ids = (p_ids + c_ids)[:max_len]
        labels = ([-100] * len(p_ids) + c_ids)[:max_len]
        enc_list.append((ids, labels))
    return enc_list


def _batches(enc_list, batch_size, device, pad_id):
    for i in range(0, len(enc_list), batch_size):
        chunk = enc_list[i:i + batch_size]
        maxlen = max(len(ids) for ids, _ in chunk)
        input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        labels = torch.full((len(chunk), maxlen), -100, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for r, (ids, labs) in enumerate(chunk):
            input_ids[r, :len(ids)] = torch.tensor(ids)
            labels[r, :len(labs)] = torch.tensor(labs)
            attn[r, :len(ids)] = 1
        yield (input_ids.to(device), attn.to(device), labels.to(device))


class MLPHooks:
    """Capture (and optionally patch) post-gating MLP activations = down_proj inputs."""

    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter
        self.acts: dict[int, torch.Tensor] = {}
        self.handles = []
        self.patch: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}  # layer -> (idx, values)

    def __enter__(self):
        for li, layer in enumerate(self.adapter.layers):
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(
                self._mk(li), with_kwargs=False))
        return self

    def _mk(self, li):
        def hook(_m, inputs):
            x = inputs[0]
            if li in self.patch:
                idx, val = self.patch[li]
                x = x.clone()
                x[..., idx] = val.to(x.dtype)
            if self.capture:
                x.retain_grad() if x.requires_grad else None
                self.acts[li] = x
            return (x,)
        return hook

    capture = False

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def neuron_means(adapter: ModelAdapter, pairs, batch_size=8) -> torch.Tensor:
    """Mean post-gating activation per neuron over continuation tokens. [L, I] fp32."""
    enc = encode_pairs(adapter, pairs)
    L, I = adapter.n_layers, adapter.layers[0].mlp.down_proj.weight.shape[1]
    total = torch.zeros(L, I, dtype=torch.float64)
    count = 0
    with MLPHooks(adapter) as h:
        h.capture = True
        for ids, attn, labels in _batches(enc, batch_size, adapter.device,
                                          adapter.tokenizer.pad_token_id or 0):
            adapter.model(input_ids=ids, attention_mask=attn)
            mask = (labels != -100)
            n = int(mask.sum())
            for li in range(L):
                total[li] += h.acts[li][mask].float().sum(dim=0).double().cpu()
            count += n
    return (total / max(count, 1)).float()


def attribution(adapter: ModelAdapter, pairs, means: torch.Tensor,
                batch_size=4) -> torch.Tensor:
    """First-order estimate of delta-NLL from mean-ablating each neuron. [L, I] fp32.
    Positive value ~ mean-ablation INCREASES the NLL of these continuations."""
    enc = encode_pairs(adapter, pairs)
    L, I = means.shape
    attr = torch.zeros(L, I, dtype=torch.float64)
    n_ex = 0
    mu = means.to(adapter.device)
    with MLPHooks(adapter) as h:
        h.capture = True
        for ids, attn, labels in _batches(enc, batch_size, adapter.device,
                                          adapter.tokenizer.pad_token_id or 0):
            for p in adapter.model.parameters():
                p.requires_grad_(False)
            with torch.enable_grad():
                emb = adapter.model.get_input_embeddings()(ids)
                emb.requires_grad_(True)   # makes activations require grad
                out = adapter.model(inputs_embeds=emb, attention_mask=attn)
                logits = out.logits[:, :-1]
                tgt = labels[:, 1:]
                mask = tgt != -100
                nll = torch.nn.functional.cross_entropy(
                    logits[mask].float(), tgt[mask], reduction="mean")
                grads = torch.autograd.grad(nll, [h.acts[li] for li in range(L)],
                                            allow_unused=True)
            amask = (labels != -100)
            for li in range(L):
                g = grads[li]
                if g is None:
                    continue
                a = h.acts[li].detach()
                contrib = ((mu[li] - a[amask].float()) * g[amask].float()).sum(dim=0)
                attr[li] += contrib.double().cpu()
            n_ex += ids.shape[0]
    return (attr / max(n_ex, 1)).float()


@torch.no_grad()
def nll(adapter: ModelAdapter, pairs, batch_size=8,
        patch: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None) -> float:
    """Mean per-token NLL of continuations, optionally with neurons clamped to means."""
    enc = encode_pairs(adapter, pairs)
    tot, cnt = 0.0, 0
    with MLPHooks(adapter) as h:
        h.capture = False
        if patch:
            h.patch = patch
        for ids, attn, labels in _batches(enc, batch_size, adapter.device,
                                          adapter.tokenizer.pad_token_id or 0):
            logits = adapter.model(input_ids=ids, attention_mask=attn).logits[:, :-1]
            tgt = labels[:, 1:]
            mask = tgt != -100
            tot += torch.nn.functional.cross_entropy(
                logits[mask].float(), tgt[mask], reduction="sum").item()
            cnt += int(mask.sum())
    return tot / max(cnt, 1)


def causal_verify(adapter: ModelAdapter, candidates: list[tuple[int, int]],
                  pairs_h, pairs_b, means: torch.Tensor, batch_size=8,
                  group: int = 1, log=None) -> list[dict]:
    """Exact mean-ablation effect of each candidate neuron (or small group) on
    harmful vs benign continuation NLL."""
    base_h = nll(adapter, pairs_h, batch_size)
    base_b = nll(adapter, pairs_b, batch_size)
    if log:
        log.info(f"base NLL harmful={base_h:.4f} benign={base_b:.4f}")
    out = []
    for i in range(0, len(candidates), group):
        chunk = candidates[i:i + group]
        patch = {}
        for li, j in chunk:
            idx, val = patch.get(li, (torch.tensor([], dtype=torch.long),
                                      torch.tensor([])))
            patch[li] = (torch.cat([idx, torch.tensor([j])]),
                         torch.cat([val, means[li, j:j + 1]]))
        patch = {li: (idx.to(adapter.device), val.to(adapter.device))
                 for li, (idx, val) in patch.items()}
        h_c = nll(adapter, pairs_h, batch_size, patch) - base_h
        b_c = nll(adapter, pairs_b, batch_size, patch) - base_b
        out.append({"neurons": chunk, "H": h_c, "B": b_c})
        if log and (i // group) % 50 == 0:
            log.info(f"verified {i + len(chunk)}/{len(candidates)}: "
                     f"last H={h_c:+.4f} B={b_c:+.4f}")
    return out

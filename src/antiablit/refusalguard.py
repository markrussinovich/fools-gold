"""Benign-refusal guard (T1 benign-collateral repair; registration:
docs/experiments/MUSE-T1-REVIVAL-REGISTRATION.md, PENDING USER RATIFICATION).

Mechanism (9B-TRACE-FORENSICS.md §3/§5 T1; same failure as Muse r4): the
refusal pin's gradient generalizes onto benign prompts, and the benign KL
leash cannot see it — its anchors are teacher-forced along the reference's
own helpful completions, so a model that would refuse FREE-RUN still scores
near-identical forced-path probabilities (training benignKL <= .02 through
the 9B gate break at .146). Two seams close the gap:

1. TWO-SIDED PIN (`benign_refusal_pin`): a hinge penalty on the probability
   of the model's OWN refusal openers appearing after benign near-boundary
   prompts — relu(logp_trainee(opener|p) - logp_ref(opener|p) - margin).
   Zero at adapter init (trainee == reference), activates exactly when the
   "refuse everything" descent direction starts moving benign mass, and
   saturates instead of pushing refusal-prob below the reference.
2. FREE-RUN MONITOR (`benign_refusal_monitor`): periodic in-training probes
   — teacher-forced opener delta-logp grid (cheap, every ~25 steps) and
   greedy free-run refusal rate on held-out topic-matched benign prompts
   (every ~100 steps) — with a registered stop rule. Today the benign gate
   is post-hoc per rung: an overshoot consolidates for a full round before
   anyone measures it (Muse r4 went to .688 unseen). The monitor converts
   that into a mid-rung loud stop (trainer exit code 7 + marker artifact).

Openers are derived from the round's own clean-src refusal-pin targets, so
the guard needs no new data and no per-model configuration (one-pipeline
rule: arming is config keys only).

CONTENT HYGIENE: opener strings are generation text — they never enter
logs or artifacts; sha256 stubs + counts only.
"""
import hashlib
import json
import os


def _sha12(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def derive_refusal_openers(refusal_texts: list[str], top_m: int = 8,
                           max_words: int = 10, min_count: int = 2):
    """Most-common word-boundary openers of the src model's own refusal
    targets. Returns (openers, stats); stats carries hashes/counts only."""
    counts: dict[str, int] = {}
    n_nonempty = 0
    for t in refusal_texts:
        words = (t or "").split()
        if not words:
            continue
        n_nonempty += 1
        op = " ".join(words[:max_words])
        counts[op] = counts.get(op, 0) + 1
    assert counts, "derive_refusal_openers: no non-empty refusal targets"
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = [(op, c) for op, c in ranked if c >= min_count][:top_m]
    relaxed = False
    if not kept:  # diverse refusals: fall back to the most common regardless
        kept = ranked[:top_m]
        relaxed = True
    openers = [op for op, _ in kept]
    stats = {"n_texts": len(refusal_texts), "n_nonempty": n_nonempty,
             "n_unique_openers": len(counts), "min_count_relaxed": relaxed,
             "kept": [{"sha256_12": _sha12(op), "count": c,
                       "n_words": len(op.split())} for op, c in kept]}
    return openers, stats


def hinge_pair_loss(pairs, logp_train, logp_ref, margin: float = 0.0):
    """Mean over (benign_prompt, opener) pairs of
    relu(logp_train - logp_ref - margin). logp_train(pc) returns a scalar
    tensor WITH grad; logp_ref(pc) a detached scalar tensor."""
    import torch
    assert pairs, "hinge_pair_loss: empty pair list"
    tot = None
    for pc in pairs:
        h = torch.relu(logp_train(pc) - logp_ref(pc).detach() - margin)
        tot = h if tot is None else tot + h
    return tot / len(pairs)


class BenignRefusalMonitor:
    """In-training benign-refusal drift monitor with a registered stop rule.

    probe_prompts: FIXED held-out benign prompts (registered default: the
    first `probes` prompts of data/train/benign_matched_v1.jsonl — topic-
    matched twins of the harmful pool that are leash-anchored but never
    CE-pinned, i.e. exactly where free-run drift shows first).
    Stop rule: `patience` CONSECUTIVE free-run probes with
    (refusal_rate - baseline) > stop_delta.
    """

    def __init__(self, probe_prompts, openers, artifact_path, *,
                 every: int = 25, gen_every: int = 100,
                 stop_delta: float = 0.125, patience: int = 2,
                 gen_tokens: int = 48, trigger_dlogp: float = 2.0,
                 meta: dict | None = None):
        assert probe_prompts and openers
        assert gen_every % every == 0, \
            "gen_every must be a multiple of every (single probe cadence)"
        self.probe_prompts = list(probe_prompts)
        self.openers = list(openers)[:3]  # dlogp grid: top-3 openers
        self.artifact_path = str(artifact_path)
        self.every, self.gen_every = int(every), int(gen_every)
        self.stop_delta, self.patience = float(stop_delta), int(patience)
        self.gen_tokens = int(gen_tokens)
        # efficiency-review F5: a hot dlogp reading (mean opener-logp drift
        # above trigger_dlogp nats) fires an immediate out-of-cadence gen
        # probe — worst-case detection drops ~200 -> ~50 steps at ~zero cost
        # (extra gen probes fire only while drifting)
        self.trigger_dlogp = float(trigger_dlogp)
        self.baseline_refusal = None
        self.dlogp_series, self.gen_series = [], []
        self._consecutive = 0
        self._meta = dict(meta or {})

    # -- internals ---------------------------------------------------------
    def _grid(self):
        return [(p, o) for p in self.probe_prompts for o in self.openers]

    def _write(self, stop=None):
        art = {"marker": "benign_refusal_monitor", **self._meta,
               "n_probes": len(self.probe_prompts),
               "openers_sha256_12": [_sha12(o) for o in self.openers],
               "every": self.every, "gen_every": self.gen_every,
               "stop_delta": self.stop_delta, "patience": self.patience,
               "baseline_refusal": self.baseline_refusal,
               "dlogp_series": self.dlogp_series,
               "gen_series": self.gen_series}
        if stop is not None:
            art["stop"] = stop
        tmp = self.artifact_path + ".tmp"
        json.dump(art, open(tmp, "w"), indent=1)
        os.replace(tmp, self.artifact_path)

    def _gen_probe(self, step, gen_fn, refusal_rate_fn, triggered=False):
        rr = refusal_rate_fn(gen_fn(self.probe_prompts, self.gen_tokens))
        rec = {"step": step, "refusal_rate": rr}
        if triggered:
            rec["triggered"] = True
        self.gen_series.append(rec)
        return rr

    def _dlogp_probe(self, step, dlogp_fn):
        ds = dlogp_fn(self._grid())
        mean = sum(ds) / len(ds)
        mx = max(ds)
        self.dlogp_series.append({"step": step, "dlogp_mean": round(mean, 4),
                                  "dlogp_max": round(mx, 4)})
        return mean

    # -- API ---------------------------------------------------------------
    def baseline(self, gen_fn, refusal_rate_fn, dlogp_fn):
        """Capture the pre-training state (adapter zero-init == reference)."""
        self.baseline_refusal = self._gen_probe(-1, gen_fn, refusal_rate_fn)
        self._dlogp_probe(-1, dlogp_fn)
        self._write()
        return self.baseline_refusal

    def maybe_probe(self, step, steps_total, gen_fn, refusal_rate_fn,
                    dlogp_fn) -> dict | None:
        """Call after each optimizer step with the 0-based `step`. Probes on
        the registered cadence (and at the final step); returns a stop dict
        (drift evidence) when the stop rule fires, else None."""
        assert self.baseline_refusal is not None, "baseline() not captured"
        s1 = step + 1
        final = s1 == steps_total
        if s1 % self.every != 0 and not final:
            return None
        dmean = self._dlogp_probe(step, dlogp_fn)
        stop = None
        due = s1 % self.gen_every == 0 or final
        triggered = (not due) and dmean > self.trigger_dlogp
        if due or triggered:
            rr = self._gen_probe(step, gen_fn, refusal_rate_fn,
                                 triggered=triggered)
            drift = rr - self.baseline_refusal
            self._consecutive = self._consecutive + 1 if drift > self.stop_delta else 0
            if self._consecutive >= self.patience:
                stop = {"step": step, "refusal_rate": rr,
                        "baseline_refusal": self.baseline_refusal,
                        "drift": round(drift, 4),
                        "stop_delta": self.stop_delta,
                        "patience": self.patience}
        self._write(stop=stop)
        return stop

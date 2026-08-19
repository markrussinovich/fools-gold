"""Shared statistical CI machinery (paper tracker item E8).

Two methods, applied by scripts/bootstrap_cis.py to every booked proportion:

- PRIMARY: prompt-level cluster bootstrap. The prompt is the sampling unit
  (draws within a prompt are correlated); resample prompts with replacement
  keeping all K draws per prompt, recompute the metric, `reps` resamples,
  95% percentile interval. Metrics are expressed as per-cluster
  (numerator, denominator) pairs so both pooled ratios (per-draw rates:
  sum num / sum den) and mean-of-prompt-means metrics (e.g. C18 element
  recovery) share one implementation.
- SECONDARY: Wilson 95% score intervals for simple counts (unit = draw /
  item / accept; anti-conservative when units are correlated within
  prompts — callers must label the unit).

Deltas between arms that share the prompt set resample prompts JOINTLY
(paired bootstrap); arms with disjoint units would need independent
resampling per arm (not currently used — every booked delta is paired).

All RNG is seeded; per-cell seeds derive deterministically from a base seed
plus the cell id (crc32) so cells are independent but reproducible.

Content hygiene: this module only ever sees numbers, never text.
"""

import math
import zlib

import numpy as np

Z95 = 1.959963984540054  # two-sided 95% normal quantile


def cell_seed(base_seed: int, cell_id: str) -> int:
    """Deterministic per-cell seed (recorded in the output)."""
    return (int(base_seed) * 1000003 + zlib.crc32(cell_id.encode())) % 2**32


def wilson_ci(k: int, n: int, z: float = Z95):
    """Wilson 95% score interval for k successes out of n Bernoulli units."""
    if n <= 0:
        return [None, None]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def wilson(k: int, n: int, unit: str):
    return {"method": "wilson", "k": int(k), "n": int(n), "unit": unit,
            "ci95": wilson_ci(k, n)}


def _ratio(num, den, idx, agg):
    """Bootstrap statistic matrix -> vector of resampled metric values."""
    if agg == "pooled":
        d = den[idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = num[idx].sum(axis=1) / d
        return s
    r = num / den                      # per-cluster means (den>0 asserted)
    return r[idx].mean(axis=1)


def _point(num, den, agg):
    if agg == "pooled":
        return float(num.sum() / den.sum())
    return float((num / den).mean())


def cluster_boot(num, den, reps, seed, agg="pooled", unit="prompt"):
    """Prompt-level cluster bootstrap percentile CI.

    num/den: per-cluster numerator/denominator (cluster = prompt, keeping
    all its draws). agg='pooled' -> sum(num)/sum(den); agg='mean' ->
    mean(num/den) (unweighted mean of per-prompt means).
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    assert num.shape == den.shape and num.ndim == 1 and len(num) > 0
    dropped = 0
    if agg == "mean":
        keep = den > 0
        dropped = int((~keep).sum())
        num, den = num[keep], den[keep]
    n = len(num)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(reps), n))
    stats = _ratio(num, den, idx, agg)
    stats = stats[~np.isnan(stats)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    out = {"method": "cluster_bootstrap", "point": _point(num, den, agg),
           "ci95": [float(lo), float(hi)], "reps": int(reps),
           "n_effective_reps": int(len(stats)),
           "seed": int(seed), "n_units": n, "agg": agg, "unit": unit}
    if lo == hi:
        out["degenerate"] = True  # zero-variance resamples — prefer Wilson
    if dropped:
        out["n_units_dropped_zero_denom"] = dropped
    return out


def paired_delta_boot(num_a, den_a, num_b, den_b, reps, seed,
                      agg="pooled", unit="prompt"):
    """Bootstrap CI for metric(A) - metric(B) with arms PAIRED on the same
    clusters (shared prompt set): prompts are resampled jointly."""
    num_a = np.asarray(num_a, dtype=float)
    den_a = np.asarray(den_a, dtype=float)
    num_b = np.asarray(num_b, dtype=float)
    den_b = np.asarray(den_b, dtype=float)
    assert len(num_a) == len(num_b), "paired arms must align on clusters"
    n = len(num_a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(reps), n))
    stats = _ratio(num_a, den_a, idx, agg) - _ratio(num_b, den_b, idx, agg)
    stats = stats[~np.isnan(stats)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    out = {"method": "paired_cluster_bootstrap",
           "point": _point(num_a, den_a, agg) - _point(num_b, den_b, agg),
           "ci95": [float(lo), float(hi)], "reps": int(reps),
           "n_effective_reps": int(len(stats)),
           "seed": int(seed), "n_units": n, "agg": agg, "unit": unit,
           "pairing": "joint prompt resampling (arms share the prompt set)"}
    if lo == hi:
        out["degenerate"] = True
    return out


def category_boot(counts, reps, seed, unit="prompt"):
    """Bootstrap CIs for multinomial category COUNTS (e.g. the K-draw
    consensus triple c/w/n) by resampling per-prompt category labels, plus
    the precision c/(c+w) conditional on >=1 accept in the resample."""
    counts = [int(c) for c in counts]
    n = sum(counts)
    cats = np.repeat(np.arange(len(counts)), counts)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(reps), n))
    lab = cats[idx]
    boot = np.stack([(lab == j).sum(axis=1) for j in range(len(counts))], axis=1)
    cis = [[int(math.floor(lo)), int(math.ceil(hi))] for lo, hi in
           (np.percentile(boot[:, j], [2.5, 97.5]) for j in range(len(counts)))]
    acc = boot[:, 0] + boot[:, 1]
    ok = acc > 0
    prec = boot[ok, 0] / acc[ok]
    out = {"method": "category_bootstrap", "counts": counts, "n_units": n,
           "reps": int(reps), "seed": int(seed), "unit": unit,
           "count_ci95": cis}
    if counts[0] + counts[1] > 0 and len(prec):
        plo, phi = np.percentile(prec, [2.5, 97.5])
        out["precision"] = {
            "point": counts[0] / (counts[0] + counts[1]),
            "ci95": [float(plo), float(phi)],
            "n_resamples_zero_accepts": int((~ok).sum()),
            "note": "conditional on >=1 accept in the resample"}
        if plo == phi:
            out["precision"]["degenerate"] = True  # zero-variance resamples — prefer Wilson
    elif counts[0] + counts[1] > 0:  # unreachable in practice: every
        out["precision"] = {         # resample missed every accept
            "point": counts[0] / (counts[0] + counts[1]),
            "ci95": [None, None],
            "note": "no resample contained an accept — CI unavailable"}
    else:
        out["precision"] = {"point": None, "ci95": [None, None],
                            "note": "0 accepts — precision undefined"}
    return out

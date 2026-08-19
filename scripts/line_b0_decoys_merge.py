"""Two-pass B0 corpus merge/verify — formation-time completion of an UNBOOKED
decoy corpus (reviewer remediation prescription, 2026-08-07, glm45_air B0).

A cluster/base pass that shipped short of full pool coverage (no_candidate
ids) is completed by a TARGETED box pass (line_b0_decoys.py under
B0_ONLY_IDS, fresh checkpoint namespace, same gates verbatim). This script
produces the ONE final <data_dir>/decoys_B0.jsonl (pool order — byte-copied
rows, exactly what a single full-pool run would have assembled) and the ONE
combined <results_prefix>decoys.json whose two_pass_formation block documents
both passes (counts, ids, per-id retries — content hygiene: never text).

No mixed vintage: refuses unless the fill ids are a subset of the base pass's
no_candidate set (restriction proof), both passes ran the canonical-fatal
mode under the relaxed-amendment-1 gate, and every gate re-runs here on the
COMBINED rows: canonical contract fields, generator identity, TELL_UNION +
MARKERS (single-sourced from antiablit.tells), rule-6 formatting, id
uniqueness/disjointness, full containment in the signed train pool.

    python3 scripts/line_b0_decoys_merge.py --line <line> \
        --base-corpus <jsonl> --base-summary <json> \
        [--fill-corpus <path>] [--fill-summary <path>] [--out <path>] \
        [--fleet-set configs/fleet/fleet_set_v1.json] \
        [--pass1-label <str>] [--pass2-label <str>]

Defaults: fill-corpus/out = <data_dir>/decoys_B0.jsonl (read-then-atomic-
replace), fill-summary = <results_prefix>decoys.json (the targeted pass's own
summary — preserved verbatim inside the combined summary before overwrite).
Downstream contract kept: gate_tell_hits/gate_marker_hits/
frac_fully_falsified/n_generated etc. are recomputed over the COMBINED
formation (line_b1.sh preflight reads them).

Exit 0 = merged + verified; exit 1 = any refusal (nothing written).
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from antiablit.line import load_line
from antiablit.tells import TELL_UNION, marker_hits


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def die(msg):
    print(f"[b0-merge] REFUSED: {msg}", flush=True)
    sys.exit(1)


ap = argparse.ArgumentParser()
ap.add_argument("--line", required=True)
ap.add_argument("--base-corpus", required=True)
ap.add_argument("--base-summary", required=True)
ap.add_argument("--fill-corpus")
ap.add_argument("--fill-summary")
ap.add_argument("--out")
ap.add_argument("--fleet-set")
ap.add_argument("--pass1-label", default="pass1")
ap.add_argument("--pass2-label", default="pass2 (targeted completion)")
args = ap.parse_args()

L = load_line(args.line)
DQ = Path(L["data_dir_path"])
fill_corpus = Path(args.fill_corpus) if args.fill_corpus else DQ / "decoys_B0.jsonl"
fill_summary_p = Path(args.fill_summary) if args.fill_summary \
    else ROOT / (L["results_prefix"] + "decoys.json")
out_p = Path(args.out) if args.out else DQ / "decoys_B0.jsonl"
sum_out_p = ROOT / (L["results_prefix"] + "decoys.json")

for p, what in ((Path(args.base_corpus), "base corpus"),
                (Path(args.base_summary), "base summary"),
                (fill_corpus, "fill corpus"), (fill_summary_p, "fill summary")):
    if not p.is_file():
        die(f"{what} missing: {p}")

# ---- signed pool: train order + id set (merge order == pool order) ----
train_rows = [json.loads(l) for l in open(DQ / "associations_gated.jsonl")
              if json.loads(l).get("split") == "train"]
train_order = [r["id"] for r in train_rows]
train_set = set(train_order)
if len(train_order) != len(train_set):
    die("duplicate ids in the gated train split")

base_sha = sha256(args.base_corpus)
fill_sha = sha256(fill_corpus)
base = [json.loads(l) for l in open(args.base_corpus)]
fill = [json.loads(l) for l in open(fill_corpus)]
base_ids = [r["id"] for r in base]
fill_ids = [r["id"] for r in fill]
bsum = json.load(open(args.base_summary))
fsum = json.load(open(fill_summary_p))

# ---- structural refusals (all before any write) ----
if len(set(base_ids)) != len(base_ids):
    die("duplicate ids in base corpus")
if len(set(fill_ids)) != len(fill_ids):
    die("duplicate ids in fill corpus")
overlap = set(base_ids) & set(fill_ids)
if overlap:
    die(f"fill corpus overlaps base on {len(overlap)} ids (already merged? "
        f"e.g. {sorted(overlap)[:3]}) — refusing")
stray = (set(base_ids) | set(fill_ids)) - train_set
if stray:
    die(f"{len(stray)} corpus ids outside the signed train pool")
no_cand_base = set(bsum.get("no_candidate", []))
not_allowed = set(fill_ids) - no_cand_base
if not_allowed:
    die(f"restriction violated: {len(not_allowed)} fill ids are NOT in the "
        f"base pass no_candidate set: {sorted(not_allowed)}")
for s, tag in ((bsum, "base"), (fsum, "fill")):
    if s.get("mode") != "canonical-fatal" or s.get("gate") != "relaxed-amendment-1":
        die(f"{tag} summary mode/gate mismatch: {s.get('mode')}/{s.get('gate')}")

# ---- gate re-run on the COMBINED rows (identical single-sourced patterns) --
merged_by_id = {r["id"]: r for r in base + fill}
gen_id = f"self:{L['line']}-M0a"
bad_contract, bad_tell, bad_marker, bad_fmt = [], [], [], []
for i, r in merged_by_id.items():
    if not (r.get("canonical_element") and r.get("canonical_false_value")
            and r.get("canonical_fatal") is True and r.get("generator") == gen_id
            and r.get("split") == "train" and isinstance(r.get("corrupted"), str)
            and r["corrupted"].strip() and r.get("prompt") and r.get("continuation")):
        bad_contract.append(i)
        continue
    if TELL_UNION.search(r["corrupted"]):
        bad_tell.append(i)
    if marker_hits(r["corrupted"]):
        bad_marker.append(i)
    if "\n\n" not in r["corrupted"] and len(r["corrupted"]) > 500:
        bad_fmt.append(i)
if bad_contract:
    die(f"{len(bad_contract)} rows fail the canonical contract/identity: {sorted(bad_contract)[:5]}")
if bad_tell or bad_marker or bad_fmt:
    die(f"combined ship gate failed: tells={sorted(bad_tell)} "
        f"markers={sorted(bad_marker)} rule6={sorted(bad_fmt)}")

merged_ids = [i for i in train_order if i in merged_by_id]     # pool order
remaining = [i for i in train_order if i not in merged_by_id]
n, n_train = len(merged_ids), len(train_order)
frac = n / max(n_train, 1)

fleet_set_cov = None
if args.fleet_set:
    fs = set(json.load(open(args.fleet_set))["ids"])
    fleet_set_cov = {"covered": len(set(merged_ids) & fs), "set_size": len(fs),
                     "missing": sorted(fs - set(merged_ids))}

# ---- combined summary (downstream keys recomputed over the formation) ----
acc_ret = [merged_by_id[i].get("retries", 0) for i in merged_ids]
rounds = ([{**x, "pass": 1} for x in bsum.get("rounds", [])]
          + [{**x, "pass": 2} for x in fsum.get("rounds", [])])
summary = {
    "n_train": n_train, "extract_failed": [], "n_generated": n,
    "no_candidate": remaining, "rounds": rounds,
    "fully_falsified": n, "frac_fully_falsified": frac,
    "marker_scrubbed": bsum.get("marker_scrubbed", 0) + fsum.get("marker_scrubbed", 0),
    "reflowed": bsum.get("reflowed", 0) + fsum.get("reflowed", 0),
    "gate_tell_hits": [], "gate_marker_hits": [],
    "retries_per_accepted": dict(Counter(acc_ret)),
    "mean_retries_accepted": sum(acc_ret) / max(len(acc_ret), 1),
    "mode": "canonical-fatal", "no_canonical": [],
    "canonical_clean": n, "frac_canonical_clean": frac,
    "gate": "relaxed-amendment-1",
    "strict_clean": bsum.get("strict_clean", 0) + fsum.get("strict_clean", 0),
    "nll_sanity": bsum.get("nll_sanity"),
    "fleet_pool": {"n_fleet_train": n_train, "shipped": n, "coverage": frac},
    "two_pass_formation": {
        "doctrine": ("formation-time completion of an UNBOOKED corpus — one "
                     "final corpus, one gate regime (canonical-falsification "
                     "audit, relaxed-amendment-1, tell/marker union scrub), "
                     "gates re-run on the combined rows by this merge"),
        "merged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pass1": {"label": args.pass1_label, "corpus_sha256": base_sha,
                  "shipped": len(base_ids),
                  "no_candidate": sorted(no_cand_base),
                  "rounds": bsum.get("rounds", []),
                  "summary_verbatim": bsum},
        "pass2": {"label": args.pass2_label, "corpus_sha256": fill_sha,
                  "targeted_ids": sorted(no_cand_base),
                  "filled_ids": sorted(fill_ids),
                  "unfilled_ids": sorted(no_cand_base - set(fill_ids)),
                  "per_id_retries": {i: merged_by_id[i].get("retries")
                                     for i in sorted(fill_ids)},
                  "rounds": fsum.get("rounds", []),
                  "summary_verbatim": fsum},
    },
}
if fleet_set_cov is not None:
    summary["fleet_set_coverage"] = fleet_set_cov

# ---- atomic writes (corpus first, then summary) ----
out_p.parent.mkdir(parents=True, exist_ok=True)
fd, tmp = tempfile.mkstemp(dir=out_p.parent, prefix="._b0merge_")
with os.fdopen(fd, "w") as f:
    for i in merged_ids:
        f.write(json.dumps(merged_by_id[i]) + "\n")
os.replace(tmp, out_p)
json.dump(summary, open(sum_out_p, "w"), indent=1)

fs_txt = (f" fleet_set={fleet_set_cov['covered']}/{fleet_set_cov['set_size']}"
          if fleet_set_cov else "")
print(f"[b0-merge] OK {L['line']}: merged {n}/{n_train} "
      f"(base {len(base_ids)} + fills {len(fill_ids)}){fs_txt} "
      f"remaining_no_candidate={len(remaining)} {sorted(remaining)} "
      f"out_sha={sha256(out_p)[:12]} -> {out_p}", flush=True)

"""B0.5b — corpus-expansion split promotion, line-generalized (v2 rule).

General pipeline (configs/lines/<line>.json). Promotes every UNASSIGNED pool
id (spare + the below-floor category leftovers the C1c rule excluded) into
train/holdout under the registered v2 expansion rule (CBRN-V2-PLAN.md §2;
reference implementation cbrn_v2_prep.py; 122B splits `_expansion` precedent
2026-08-01):

  * existing assignments preserved VERBATIM — train stays train, holdout
    stays holdout, the direction/dev attack reserve is never touched (the
    trainer re-estimates attack directions from it; putting those prompts in
    the corpus would contaminate the attack estimation);
  * new ids: per axis (sorted axis order), sort by id, shuffle with
    random.Random(1234), first ceil(0.25*n_axis) -> holdout, rest -> train
    (0.25 = the smoke elicited holdout share, 32/128).

CPU-only, deterministic, no GPU or judge calls; prints counts/ids only.
NOT idempotent by design: refuses to run if splits.json already carries an
`_expansion` marker (delete splits.json and re-run line_b0_splits.py first
to rebuild from scratch). splits.json is backed up before writing.

    python3 scripts/line_b0_expand_splits.py --line <line>

Outputs: <data_dir>/splits.json           (expanded + _expansion provenance)
         <data_dir>/splits.json.pre_expansion   (backup of the smoke splits)
         <data_dir>/associations.jsonl    (informational train file, rewritten)
"""
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antiablit.line import load_line

L = load_line()
DQ = L["data_dir_path"]
SEED, HOLDOUT_FRAC = 1234, 0.25   # registered constants (CBRN-V2-PLAN §2)

pool = [json.loads(l) for l in open(DQ / "pool_refused.jsonl")]
by_id = {r["id"]: r for r in pool}
splits = json.load(open(DQ / "splits.json"))
assert "_expansion" not in splits, \
    "splits.json already expanded — refusing to re-expand (see module docstring)"

reserved = set(splits["direction"]) | set(splits["dev"])
assigned = set(splits["train"]) | set(splits["holdout"]) | reserved
new_ids = [r["id"] for r in pool if r["id"] not in assigned]
n_spare, n_leftover = len(splits.get("spare", [])), len(new_ids) - len(splits.get("spare", []))
print(f"pool={len(pool)} assigned={len(assigned)} -> promoting {len(new_ids)} "
      f"unassigned ids ({n_spare} spare + {n_leftover} below-floor leftovers)")
print("  by axis: " + " ".join(
    f"{a}:{n}" for a, n in Counter(by_id[i]["axis"] for i in new_ids).most_common()))

# registered rule: per axis (sorted order), shuffle sorted-by-id ids with one
# rng, first ceil(0.25*n) -> holdout, rest -> train
rng = random.Random(SEED)
by_ax = defaultdict(list)
for i in new_ids:
    by_ax[by_id[i]["axis"]].append(i)
new_hold, new_train = [], []
for ax in sorted(by_ax):
    ids = sorted(by_ax[ax])
    rng.shuffle(ids)
    k = math.ceil(HOLDOUT_FRAC * len(ids))
    new_hold += ids[:k]
    new_train += ids[k:]

out = {
    "direction": splits["direction"],
    "dev": splits["dev"],
    "holdout": splits["holdout"] + new_hold,
    "train": splits["train"] + new_train,
    "spare": [],
    "_expansion": {
        "date": date.today().isoformat(),
        "rule": ("v2 corpus expansion (CBRN-V2-PLAN §2 semantics): existing "
                 "assignments preserved; ALL unassigned ids (spare + below-floor "
                 "leftovers) assigned 75/25 per axis, Random(1234)"),
        "n_added_train": len(new_train),
        "n_added_holdout": len(new_hold),
    },
}
flat = [i for k, v in out.items() if not k.startswith("_") for i in v]
assert len(flat) == len(set(flat)) == len(pool), \
    "expanded splits must cover the pool exactly once"

backup = DQ / "splits.json.pre_expansion"
assert not backup.exists(), f"{backup} already exists — refusing to clobber the backup"
shutil.copy(DQ / "splits.json", backup)
json.dump(out, open(DQ / "splits.json", "w"), indent=1)
with open(DQ / "associations.jsonl", "w") as f:   # informational train file (splits-stage parity)
    for i in out["train"]:
        f.write(json.dumps(by_id[i]) + "\n")

print(f"expanded splits: train={len(out['train'])} (+{len(new_train)}) "
      f"holdout={len(out['holdout'])} (+{len(new_hold)}) "
      f"direction={len(out['direction'])} dev={len(out['dev'])} spare=0")
for name in ("train", "holdout"):
    print(f"  {name:8s} axes: " + " ".join(
        f"{a}:{n}" for a, n in Counter(by_id[i]["axis"] for i in out[name]).most_common()))
print(f"wrote {L['data_dir']}/splits.json (backup: splits.json.pre_expansion) "
      f"and rewrote associations.jsonl ({len(out['train'])})")

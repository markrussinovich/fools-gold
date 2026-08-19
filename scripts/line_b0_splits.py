"""B0.5 — category-balanced smoke splits from the screen survivors, line-generalized.

General pipeline (configs/lines/<line>.json); conversion of the debugged
q35_b0_splits.py — see REFACTOR-GENERAL-PIPELINE.md.

Port of cbrn_c1c_splits.py at smoke scale (plan §2 B0.5): same
category-selection and disjoint-assignment logic, budgets capped at
96 train / 32 holdout / 40 direction+dev (split 32 direction / 8 dev — the C1c
80:20 ratio), fixed seed 1234, frozen to <data_dir>/splits.json.

Category rule (as C1c): a category is included only if it can fill its
per-category quota; categories below the floor are recorded and excluded rather
than carried as strata too small to say anything about. Because the smoke pool
is smaller, the include-set additionally sheds its smallest category until
every remaining one can fill the per-category need (C1c asserted instead;
unattended-chain robustness).

The gemma-line attacker files data/train/cbrn_harmful_{direction,dev}.jsonl are
NOT touched — the line-scoped copies are written under <data_dir>/ (the B0 attack
itself uses the gemma-line direction sets per plan; these frozen splits are the
line's direction/dev reserve for B1 re-estimation).

    python3 scripts/line_b0_splits.py --line <line>

Outputs: <data_dir>/splits.json, <data_dir>/associations.jsonl,
         <data_dir>/cbrn_harmful_{direction,dev}.jsonl
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antiablit.line import load_line
L = load_line()
DQ = L["data_dir_path"]
DQ.mkdir(parents=True, exist_ok=True)

MIN_CATEGORY = 20                     # floor scaled from C1c's 40 (343 -> 168 budget)
N_DIRECTION, N_DEV = 32, 8            # 40 direction+dev total (plan cap)
N_HOLDOUT, N_TRAIN = 32, 96           # plan caps
SEED = 1234

rows = [json.loads(l) for l in open(DQ / "pool_refused.jsonl")]
by_ax = defaultdict(list)
for r in rows:
    by_ax[r["axis"]].append(r)

included = sorted([a for a, v in by_ax.items() if len(v) >= MIN_CATEGORY],
                  key=lambda a: -len(by_ax[a]))
# shed the smallest category until every included one can fill its quota
while included:
    k = len(included)
    per = {"direction": N_DIRECTION // k, "dev": N_DEV // k,
           "holdout": N_HOLDOUT // k, "train": N_TRAIN // k}
    need = sum(per.values())
    if all(len(by_ax[a]) >= need for a in included):
        break
    included = included[:-1]
assert included, "no category can support the smoke quotas"
excluded = {a: len(v) for a, v in by_ax.items() if a not in included}
print(f"categories included (>={MIN_CATEGORY}, fillable): "
      + ", ".join(f"{a}={len(by_ax[a])}" for a in included))
print(f"categories excluded: {excluded}")
print(f"\nper category: {per}  (x{len(included)} categories)")

rng = random.Random(SEED)
splits = defaultdict(list)
for ax in included:
    pool = sorted(by_ax[ax], key=lambda r: r["id"])
    rng.shuffle(pool)
    i = 0
    for name in ("direction", "dev", "holdout", "train"):
        splits[name] += pool[i:i + per[name]]
        i += per[name]
    splits["spare"] += pool[i:]

# distribute integer-division remainders from spare, up to each split's cap
rng.shuffle(splits["spare"])
for name, cap in (("train", N_TRAIN), ("holdout", N_HOLDOUT),
                  ("direction", N_DIRECTION), ("dev", N_DEV)):
    short = cap - len(splits[name])
    if short > 0:
        splits[name] += splits["spare"][:short]
        splits["spare"] = splits["spare"][short:]

ids = [r["id"] for name in splits for r in splits[name]]
assert len(ids) == len(set(ids)), "splits overlap"
assert len(splits["train"]) <= N_TRAIN and len(splits["holdout"]) <= N_HOLDOUT \
    and len(splits["direction"]) + len(splits["dev"]) <= N_DIRECTION + N_DEV, "cap exceeded"
print()
for name in ("direction", "dev", "holdout", "train", "spare"):
    print(f"  {name:10s} {len(splits[name]):4d}  "
          + " ".join(f"{a}:{n}" for a, n in
                     Counter(r["axis"] for r in splits[name]).most_common()))

json.dump({k: [r["id"] for r in v] for k, v in splits.items()},
          open(DQ / "splits.json", "w"), indent=1)
with open(DQ / "associations.jsonl", "w") as f:
    for r in splits["train"]:
        f.write(json.dumps(r) + "\n")

# attacker-facing splits, in the format the ablation pipeline expects (line-scoped
# copies under the line data_dir — do NOT overwrite the gemma-line data/train files)
for name, rel in (("direction", "cbrn_harmful_direction"), ("dev", "cbrn_harmful_dev")):
    with open(DQ / f"{rel}.jsonl", "w") as f:
        for r in splits[name]:
            f.write(json.dumps({"prompt": r["prompt"], "category": r["axis"]}) + "\n")
print(f"\nwrote {L['data_dir']}/splits.json, associations.jsonl ({len(splits['train'])}), "
      f"and {L['data_dir']}/cbrn_harmful_{{direction,dev}}.jsonl")

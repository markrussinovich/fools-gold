"""Trim a fleet line's self-decoy corpus to the SIGNED fleet set (Option-A
phase 2; fleet-SET formation ruling 2026-08-03 ~20:20 + GO ruling 2026-08-04
~13:50: the three fleet lines train on the IDENTICAL association subset).

Reads the line config's fleet_pool.fleet_set (ids file, sha256-pinned by
fleet_pool.fleet_set_sha256 — both committed) and filters
<data_dir>/decoys_B0.jsonl to exactly those ids:
  - refuses if the pin is absent/mismatched (substitution defense);
  - refuses if any fleet-set id is MISSING from the corpus (a trim can only
    narrow — a missing id means this line's gen never covered it and the
    set was formed wrong);
  - pre-trim corpus preserved once at decoys_B0_prefleetset.jsonl;
  - atomic replace; idempotent (already-exact -> no write, exit 0).
Counts/ids only in logs (content hygiene).

    python3 scripts/line_trim_fleet_set.py --line <line> [--dry-run]
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antiablit.line import load_line  # noqa: E402

L = load_line()
DRY = "--dry-run" in sys.argv
fp = L.get("fleet_pool") or {}
set_path = fp.get("fleet_set")
pin = fp.get("fleet_set_sha256")
assert set_path and pin, (
    f"line {L['line']}: fleet_pool.fleet_set / fleet_set_sha256 not configured")
p = ROOT / set_path
raw = p.read_bytes()
got = hashlib.sha256(raw).hexdigest()
assert got == pin.lower(), (
    f"fleet_set sha256 mismatch: config pin {pin[:16]}.. != {got[:16]}.. — refusing")
fleet = json.loads(raw)
ids = set(fleet["ids"])
assert len(ids) == fleet["n"], "fleet_set ids/n mismatch"

dq = Path(L["data_dir_path"])
src = dq / "decoys_B0.jsonl"
rows = [json.loads(l) for l in open(src)]
have = {r["id"] for r in rows}
missing = ids - have
assert not missing, (f"{len(missing)} fleet-set ids missing from {src} "
                     f"({sorted(missing)[:5]}..) — formation error, refusing")
keep = [r for r in rows if r["id"] in ids]
drop = sorted(have - ids)
if len(keep) == len(rows) and not drop:
    print(f"[trim] {L['line']}: corpus already exactly the fleet set "
          f"({len(keep)}) — nothing to do")
    sys.exit(0)
print(f"[trim] {L['line']}: {len(rows)} -> {len(keep)} decoys "
      f"(fleet set {fleet['version']} n={fleet['n']}); dropping {drop}")
if DRY:
    print("[trim] dry-run: no writes")
    sys.exit(0)
bak = dq / "decoys_B0_prefleetset.jsonl"
if not bak.exists():
    bak.write_bytes(src.read_bytes())
    print(f"[trim] pre-trim corpus preserved at {bak}")
fd, tmp = tempfile.mkstemp(dir=dq, prefix="._trim_")
with os.fdopen(fd, "w") as f:
    for r in keep:
        f.write(json.dumps(r) + "\n")
os.replace(tmp, src)
print(f"[trim] {L['line']} OK: {src} now {len(keep)} decoys == fleet set "
      f"{fleet['version']} (sha {got[:12]}..)")

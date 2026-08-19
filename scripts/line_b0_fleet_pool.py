"""B0 fleet-pool stage — verify + materialize the SIGNED shared association
pool (corpus/recipe integrity directive 2026-08-03; seam logic in
src/antiablit/fleetpool.py, spec in configs/lines/deepseek_v4_flash.json
_fleet_pool_note + tracker 2026-08-03 ~20:20/21:30).

Replaces b0_screen/b0_splits/b0_elicit for lines with a "fleet_pool" config
block: verifies the sign-off marker + frozen sha256 manifest, cross-checks
registry wiring, then materializes byte-exact copies of the signed pool files
into the line's data dir with fleet_pool_provenance.json. Refuses to run (exit
1, nothing written) on any mismatch. CPU-only.

    python3 scripts/line_b0_fleet_pool.py --line <line> [--verify-only]

--verify-only writes nothing and additionally requires the materialized
copies + provenance to already exist and match (B1 preflight re-check).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from antiablit.line import load_line
from antiablit.fleetpool import FleetPoolError, verify_and_materialize

L = load_line()
VERIFY_ONLY = "--verify-only" in sys.argv
try:
    prov = verify_and_materialize(L, verify_only=VERIFY_ONLY)
except FleetPoolError as e:
    print(f"[b0-fleet-pool] REFUSED ({L['line']}): {e}", flush=True)
    sys.exit(1)
print(f"[b0-fleet-pool] OK ({L['line']}, "
      f"{'verify-only' if VERIFY_ONLY else 'materialized'}): "
      f"{len(prov['files'])} files sha-verified vs manifest "
      f"{prov['manifest_sha256'][:12]}.., fleet train set "
      f"{prov['n_fleet_train']} (marker {prov['signoff_marker']})")

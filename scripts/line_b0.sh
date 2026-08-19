#!/bin/bash
# General-pipeline B0 foundation chain (REFACTOR-GENERAL-PIPELINE.md; recipe
# from the debugged q35_b0.sh). Reads configs/lines/$LINE.json, waits for ALL
# GPUs to go idle (other jobs may own them), then runs the five B0 stages
# sequentially:
#   b0_screen -> b0_attack -> b0_splits -> b0_elicit -> b0_decoys
# The attack stage script is the config seam "attack_script" (default
# line_b0_attack4.py = in-house compliance-ranked derivation). Lines with a
# public abliterated build register line_b0_attack3.py — the PUBLIC-anchored
# attack (extracts the public-recipe direction from the (M0, public) pair;
# attack-realism directive 2026-08-03). 31b/122B produced their booked M0-a
# via attack3 run as a pre-stage (b0_attack3_{extract,accept}.log); the stage
# guard below then skips on the accepted artifact.
# Any stage exiting 3 is a pre-registered STOP gate (STOP-B1 screen coverage,
# STOP-B2 decoy falsification) — the chain prints the marker and aborts. Any
# other non-zero exit aborts without a STOP marker (infra failure).
# B1 training is deliberately NOT chained here: it is gated on human review of
# the B0 outputs (standing rule). line_auto_next.sh chains B1 for
# auto-continued lines per the explicit 2026-07-29 user directive.
#
#   LINE=<line> nohup bash scripts/line_b0.sh > <run_dir>/logs/line_b0_launcher.log 2>&1 &
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
LINE=${LINE:?set LINE=<name> (reads configs/lines/<name>.json)}
export LINE
C=configs/lines/$LINE.json
PY=$(jq -r .python "$C")
RUN=$(jq -r .run_dir "$C")
DQ=$(jq -r .data_dir "$C")
HFID=$(jq -r .hf_id "$C")
PUBID=$(jq -r .public_abliterated_id "$C")
RES=$(jq -r .results_prefix "$C")
DECOY_FLOOR=$(jq -r .decoy_floor "$C")
FLEET_POOL=$(jq -r '.fleet_pool != null' "$C")
LOG=$RUN/logs
mkdir -p "$LOG" "$DQ" "$RUN/artifacts" "$(dirname "${RES}x")"

ts () { date +%H:%M:%S; }

run_stage () {  # $1 = stage name, rest = command
  local name=$1; shift
  echo "[$(ts)] [$name] start"
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  if [ "$rc" -eq 3 ]; then
    local stop
    stop=$(grep -o 'STOP-B[0-9]*' "$LOG/$name.log" | tail -1)
    echo "[$(ts)] [$name] ${stop:-STOP} — pre-registered stop gate tripped (exit 3), chain aborted"
    exit 3
  elif [ "$rc" -ne 0 ]; then
    echo "[$(ts)] [$name] FAILED (exit $rc) — chain aborted"
    exit "$rc"
  fi
  echo "[$(ts)] [$name] done"
}

# fleet-pool seam (2026-08-04, corpus/recipe integrity directive 2026-08-03):
# lines with a fleet_pool config block consume the SIGNED shared association
# pool — verify the sign-off marker + frozen sha256 manifest, then materialize
# byte-exact copies into $DQ (fleet_pool_provenance.json records the set).
# b0_screen/b0_splits/b0_elicit are skipped below: they would re-derive a
# per-line pool and clobber the shared one. CPU-only + fail-fast, so it runs
# before the model download and the GPU wait.
if [ "$FLEET_POOL" = "true" ]; then
  run_stage b0_fleet_pool $PY scripts/line_b0_fleet_pool.py --line $LINE
fi

# preflight (network/CPU only — the GPUs may still belong to other jobs): make
# sure BOTH checkpoints are in the HF cache (HF_HOME) before we
# queue — attack3 extracts the public direction from the abliterated pair, so
# it needs the public checkpoint alongside M0. Cache-first (2026-08-09 seam,
# trio-B0): resolve local_files_only before touching the hub — an online
# snapshot_download on a curated cache (ms4: consolidated dupes/images
# deliberately removed) would re-pull excluded files; PUBID may be absent
# (fleet lines whose accepted attack artifact already exists — null-safe).
echo "[$(ts)] preflight: ensuring $HFID${PUBID:+ + $PUBID} are downloaded"
$PY - "$HFID" "$PUBID" > "$LOG/b0_download.log" 2>&1 <<'PYEOF' || { echo "[$(ts)] model download FAILED"; exit 1; }
import sys
from huggingface_hub import snapshot_download
for rid in sys.argv[1:]:
    if not rid or rid == "null":
        continue
    try:
        snapshot_download(rid, local_files_only=True)
        print(f"cache-hit {rid}")
    except Exception:
        print(f"cache-miss {rid} — downloading")
        snapshot_download(rid)
PYEOF

# cache is complete after the preflight — go hub-offline so 8 concurrent
# worker loads can't race hub file resolution (worker crashed on a malformed
# "model.safetensors-00004-of-00011" lookup under concurrent from_pretrained)
export HF_HUB_OFFLINE=1

# served-backend seam (2026-08-04, Option-A 122B fleet gen): lines with
# backend=="served" run b0_elicit/b0_decoys as HTTP shards against the
# PRE-MATERIALIZED M0-a checkpoint (m0a_model_dir served under
# served_models.m0a at served_defaults.served_url) — serve it here, PID-file
# managed (archived q122_b0_chain.sh pattern; never pkill -f, self-match
# doctrine). All values are config seams; absent backend key = old behavior.
# Scope: m0a stages only — b0_screen on a served NON-fleet line would need an
# m0 serve and is not wired (no such line exists; 397B B0 ran as a cluster job).
BACKEND=$(jq -r '.backend // "hf"' "$C")
SERVE_PIDF=/tmp/${LINE}_b0_vllm.pid
stop_serve () {
  [ -f "$SERVE_PIDF" ] || return 0
  kill "$(cat "$SERVE_PIDF")" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$(cat "$SERVE_PIDF")" 2>/dev/null || break; sleep 2; done
  kill -9 "$(cat "$SERVE_PIDF")" 2>/dev/null || true
  rm -f "$SERVE_PIDF"; sleep 5
}
serve_m0a () {
  local m0a_dir served_name port tp kv vllm_bin
  m0a_dir=$(jq -r .m0a_model_dir "$C")
  served_name=$(jq -r .served_models.m0a "$C")
  port=$(jq -r .served_defaults.served_url "$C" | sed 's|.*:||; s|/.*||')
  tp=$(jq -r '.vllm_tp // 4' "$C")
  # vllm_env config seam (2026-08-09, trio-B0; anchor_gate_driver/m0screen
  # parity): lines that need serve-side env (ms4 VLLM_MLA_DISABLE=1 marlin/MLA
  # incident) register it in config — never a per-model script fork
  while IFS= read -r kv; do
    [ -n "$kv" ] && export "${kv?}" && echo "[$(ts)] vllm_env: $kv"
  done < <(jq -r '(.vllm_env // {}) | to_entries[] | "\(.key)=\(.value)"' "$C")
  local serve_args=()
  while IFS= read -r a; do serve_args+=("$a"); done \
    < <(jq -r '.vllm_serve_args // [] | .[]' "$C")
  [ -d "$m0a_dir" ] || { echo "[$(ts)] served backend: m0a_model_dir missing: $m0a_dir — chain aborted"; exit 1; }
  echo "[$(ts)] serving $served_name from $m0a_dir (TP=$tp, port $port${CUDA_VISIBLE_DEVICES:+, GPUs $CUDA_VISIBLE_DEVICES})"
  # vllm resolution (public-repo correctness review B2; same chain as the
  # lm_eval seam): config key "vllm_bin" overrides; else dirname(python)/vllm
  # when it exists (venv layout — unchanged behavior); else PATH vllm.
  vllm_bin=$(jq -r '.vllm_bin // empty' "$C")
  [ -n "$vllm_bin" ] || vllm_bin=$(dirname "$PY")/vllm
  [ -x "$vllm_bin" ] || vllm_bin=$(command -v vllm || echo "$vllm_bin")
  "$vllm_bin" serve "$m0a_dir" --served-model-name "$served_name" \
      --port "$port" --tensor-parallel-size "$tp" ${serve_args[@]+"${serve_args[@]}"} \
      > "$LOG/vllm_${served_name}_b0.log" 2>&1 &
  echo $! > "$SERVE_PIDF"
  for _ in $(seq 1 180); do
    curl -s "localhost:$port/health" > /dev/null 2>&1 \
        && { echo "[$(ts)] $served_name healthy"; return 0; }
    kill -0 "$(cat "$SERVE_PIDF")" 2>/dev/null \
        || { echo "[$(ts)] vLLM $served_name died on startup (see $LOG/vllm_${served_name}_b0.log) — chain aborted"; rm -f "$SERVE_PIDF"; exit 1; }
    sleep 10
  done
  echo "[$(ts)] vLLM $served_name not healthy in 1800s — chain aborted"; stop_serve; exit 1
}
trap stop_serve EXIT

echo "[$(ts)] waiting for all GPUs idle (<2GB)${CUDA_VISIBLE_DEVICES:+ [restricted to $CUDA_VISIBLE_DEVICES]}"
GPU_SEL=${CUDA_VISIBLE_DEVICES:+-i $CUDA_VISIBLE_DEVICES}
until [ "$(nvidia-smi $GPU_SEL --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)" -eq 0 ]; do
  sleep 60
done
echo "[$(ts)] GPUs free — B0 chain starts"

if [ "$FLEET_POOL" = "true" ]; then
  echo "[$(ts)] [b0_screen] SKIP: fleet_pool set — signed shared pool materialized by b0_fleet_pool"
else
  run_stage b0_screen $PY scripts/line_b0_screen.py --line $LINE
fi
if [ -f "$RUN/artifacts/cbrn_attack_M0a.json" ] && $PY -c "import json,sys; sys.exit(0 if json.load(open('$RUN/artifacts/cbrn_attack_M0a.json')).get('attack_clean') else 1)" 2>/dev/null; then
  echo "[$(ts)] [b0_attack] SKIP: accepted attack artifact already present"
else
  ATK=$(jq -r '.attack_script // "line_b0_attack4.py"' "$C")
  run_stage b0_attack $PY scripts/$ATK --line $LINE
fi
if [ "$BACKEND" = "served" ]; then
  serve_m0a   # m0a stages below (b0_elicit + b0_decoys) generate via HTTP
fi
if [ "$FLEET_POOL" = "true" ]; then
  echo "[$(ts)] [b0_splits/b0_elicit] SKIP: fleet_pool set — splits + gated associations come byte-exact from the signed pool"
else
  run_stage b0_splits $PY scripts/line_b0_splits.py --line $LINE
  run_stage b0_elicit $PY scripts/line_b0_elicit.py --line $LINE
fi
# fleet-SET seam (Option-A phase 2, review finding F4): once a line's corpus
# has been trimmed to the signed common subset (fleet_pool.fleet_set +
# line_trim_fleet_set.py), a b0 re-entry must NOT regenerate the excluded
# ids or trip the full-pool coverage gate — corpus == signed set is DONE.
FLEET_SET=$(jq -r '.fleet_pool.fleet_set // empty' "$C")
FLEET_SET_SHA=$(jq -r '.fleet_pool.fleet_set_sha256 // empty' "$C")
fleet_set_done () {
  [ -n "$FLEET_SET" ] && [ -f "$DQ/decoys_B0.jsonl" ] || return 1
  $PY - "$DQ/decoys_B0.jsonl" "$FLEET_SET" "$FLEET_SET_SHA" <<'EOF'
import hashlib, json, sys
raw = open(sys.argv[2], "rb").read()
assert hashlib.sha256(raw).hexdigest() == sys.argv[3].lower(), "fleet_set pin mismatch"
want = set(json.loads(raw)["ids"])
have = [json.loads(l)["id"] for l in open(sys.argv[1])]
sys.exit(0 if (set(have) == want and len(have) == len(want)) else 1)
EOF
}
if fleet_set_done; then
  echo "[$(ts)] [b0_decoys] SKIP: corpus already == the signed fleet set ($FLEET_SET) — no regeneration"
else
  run_stage b0_decoys $PY scripts/line_b0_decoys.py --line $LINE
fi
if [ "$BACKEND" = "served" ]; then
  stop_serve
fi

# decoy-count floor (gate yield is line-dependent — RECIPE R6 floor form; the
# B1 preflight enforces the same floor, catch it here so the failure is
# attributed to B0). Fleet-pool lines gate on FLEET-SET COVERAGE instead:
# floor = |fleet train set| (integrity directive — same prompts, same size),
# never the per-line decoy_floor.
N_DECOYS=$(wc -l < "$DQ/decoys_B0.jsonl")
if [ "$FLEET_POOL" = "true" ]; then
  # floor = signed fleet-set size when configured (post-trim lines), else
  # the full fleet train count (generation-time lines)
  DECOY_FLOOR=$($PY - "$DQ/associations_gated.jsonl" "$FLEET_SET" <<'EOF'
import json, sys
if sys.argv[2]:
    print(json.load(open(sys.argv[2]))["n"])
else:
    print(sum(1 for l in open(sys.argv[1]) if json.loads(l)["split"] == "train"))
EOF
)
  if [ "$N_DECOYS" -lt "$DECOY_FLOOR" ]; then
    echo "[$(ts)] [b0_decoys] FAILED: fleet-set coverage $N_DECOYS/$DECOY_FLOOR incomplete ($C) — chain aborted (fleet-set formation ruling 2026-08-03 ~20:20: common gate-passing subset needs reviewer sign-off before any narrowing)"
    exit 1
  fi
elif [ "$N_DECOYS" -lt "$DECOY_FLOOR" ]; then
  echo "[$(ts)] [b0_decoys] FAILED: $N_DECOYS decoys < floor $DECOY_FLOOR ($C) — chain aborted"
  exit 1
fi

echo "[$(ts)] B0 CHAIN DONE — $N_DECOYS decoys; review $DQ/ + ${RES}decoys.json before launching B1 (human sign-off required)"

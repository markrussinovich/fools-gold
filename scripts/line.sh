#!/bin/bash
# THE single entry point for a model line (user directive 2026-08-02):
# takes any configured line from raw base model to certified champion —
# B0 foundation -> B1 seed -> ceiling-search DPO ladder (smoke + full eval
# per rung) -> GSM8K gate -> verdict -> capability battery.
#
#   LINE=<name> bash scripts/line.sh                 # full run / exact resume
#   LINE=<name> STAGES=b1,post bash scripts/line.sh  # subset
#   LINE=<name> DRYRUN=1 bash scripts/line.sh        # branch trace, read-only
#
# Every stage is resume-aware (existence + freshness guards inside the stage
# scripts), so relaunching after any crash continues where it stopped.
#
# Policy gates preserved:
#   * B0 -> B1 requires human sign-off on the B0 decoy corpus (standing
#     rule). If B0 artifacts are missing, this script runs B0 and STOPS with
#     exit 5 so a human can review; set B0_SIGNED_OFF=1 to proceed into B1
#     in the same invocation (only for corpora already reviewed).
#   * The battery honors /tmp/antiablit_defer_battery (campaign mode:
#     ladders first, batteries at the end).
#   * NEVER promote any checkpoint without human sign-off.
#
# Exit codes: 0 done (verdict PASS), 2 verdict FAIL, 3 STOP-B1SEED,
#             4 STOP-B1D0A, 5 B0-awaiting-sign-off, 1 infra.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LINE=${LINE:?set LINE=<name> (reads configs/lines/<name>.json)}
export LINE
C=configs/lines/$LINE.json
[ -f "$C" ] || { echo "[line] no such line config: $C"; exit 1; }
PY=$(jq -r .python "$C")
RUN=$(jq -r .run_dir "$C")
DQ=$(jq -r .data_dir "$C")
RES=$(jq -r .results_prefix "$C")
STAGES=${STAGES:-b0,b1,post}
ts () { date +%H:%M:%S; }
mkdir -p "$RUN/logs"

want () { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }

# ---- B0 foundation (skips itself when the corpus is complete) ----
b0_done () {
    [ -f "$DQ/decoys_B0.jsonl" ] && [ -f "${RES}decoys.json" ] \
        && [ -f "$RUN/artifacts/cbrn_attack_M0a.json" ]
}
if want b0; then
    # sign-off marker (adversarial-review finding 7: the exit-5 gate must
    # survive resume — record sign-off durably, never infer it). Grandfather:
    # lines whose B1 already ran were signed off in-session.
    SIGNOFF="$RUN/artifacts/b0_signoff"
    if [ "${B0_SIGNED_OFF:-0}" = 1 ] && [ ! -f "$SIGNOFF" ]; then
        date -u +%FT%TZ > "$SIGNOFF"
    fi
    if [ ! -f "$SIGNOFF" ] && ls "$RUN"/evals/cbrn_smoke_B1*.json >/dev/null 2>&1; then
        echo "grandfathered: B1 evals predate the marker ($(date -u +%FT%TZ))" > "$SIGNOFF"
    fi
    if b0_done; then
        echo "[$(ts)] [line] B0 complete ($DQ/decoys_B0.jsonl + gates) — skipping"
        if [ ! -f "$SIGNOFF" ]; then
            echo "[$(ts)] [line] B0 complete but NOT signed off — human review of the decoy" \
                 "corpus required. Re-run with B0_SIGNED_OFF=1 to proceed."
            exit 5
        fi
    elif [ "${DRYRUN:-0}" = 1 ]; then
        echo "[line DRYRUN] B0 would run (line_b0.sh)"
    else
        echo "[$(ts)] [line] B0 foundation -> $RUN/logs/line_b0_launcher.log"
        LINE=$LINE bash scripts/line_b0.sh > "$RUN/logs/line_b0_launcher.log" 2>&1
        b0_done || { echo "[$(ts)] [line] B0 did not produce a complete foundation"; exit 1; }
        if [ "${B0_SIGNED_OFF:-0}" != 1 ]; then
            echo "[$(ts)] [line] B0 DONE — human review of the decoy corpus required" \
                 "(standing rule). Re-run with B0_SIGNED_OFF=1 to continue into B1."
            exit 5
        fi
    fi
fi

# ---- B1: seed -> ladder (smoke+full per rung) -> GSM8K -> verdict ----
rc=0
if want b1; then
    if [ "${DRYRUN:-0}" = 1 ]; then
        DRYRUN=1 LINE=$LINE bash scripts/line_b1.sh
    else
        echo "[$(ts)] [line] B1 chain -> $RUN/logs/line_b1_launcher.log"
        LINE=$LINE bash scripts/line_b1.sh > "$RUN/logs/line_b1_launcher.log" 2>&1 || rc=$?
        grep -hE "B1 (PASS|FAIL)|STOP-B1" "$RUN/logs/line_b1_launcher.log" | tail -1 || true
        case "$rc" in
            0|2) ;;                          # verdict in — battery may proceed
            3|4) echo "[$(ts)] [line] pre-registered stop (rc=$rc)"; exit "$rc" ;;
            *)   echo "[$(ts)] [line] B1 infra failure (rc=$rc) — see launcher log"; exit 1 ;;
        esac
    fi
fi

# ---- capability battery (defer-flag aware; verdict must exist) ----
if want post && [ "${DRYRUN:-0}" != 1 ]; then
    echo "[$(ts)] [line] battery -> $RUN/logs/line_b1_post.log"
    LINE=$LINE bash scripts/line_b1_post.sh > "$RUN/logs/line_b1_post.log" 2>&1 \
        || echo "[$(ts)] [line] battery rc=$? (non-fatal)"
fi
echo "[$(ts)] [line] DONE (b1 rc=$rc)"
exit "$rc"

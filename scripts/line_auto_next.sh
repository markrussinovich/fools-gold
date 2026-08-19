#!/bin/bash
# Auto-continuation between pipeline lines (user directive 2026-07-29:
# the next line starts automatically once the previous line's B1 verdict is
# in AND its post battery has finished; full evals per model are encoded in
# line_b1_post.sh). Replaces the one-off q9_auto.sh idea with a parameterized
# handoff: waits on <prev_line>'s logs, then runs <next_line>'s full chain
# (b0 -> b1 + post). Any stage exiting non-zero prints a FAILED line and
# exits non-zero — a monitor watches this log.
#
# The *b1_launcher.log / *b1_post.log globs match both the frozen q35_* names
# (q35_b1_launcher.log, q35_b1_post.log) and the general line_* names.
#
#   nohup bash scripts/line_auto_next.sh <prev_line> <next_line> \
#       > runs/auto_<prev>_to_<next>.log 2>&1 &
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PREV=${1:?usage: line_auto_next.sh <prev_line> <next_line>}
NEXT=${2:?usage: line_auto_next.sh <prev_line> <next_line>}
PREV_RUN=$(jq -r .run_dir "configs/lines/$PREV.json")
NEXT_RUN=$(jq -r .run_dir "configs/lines/$NEXT.json")

ts () { date +%H:%M:%S; }

fail () {  # $1 = stage, $2 = rc
    echo "[$(ts)] [AUTO] FAILED: $1 exited $2 (line $NEXT) — auto-continuation aborted"
    exit "$2"
}

echo "[$(ts)] [AUTO] waiting on $PREV: B1 verdict in $PREV_RUN/logs/*b1_launcher.log + battery done in $PREV_RUN/logs/*b1_post.log"
until grep -qE "B1 (PASS|FAIL)" "$PREV_RUN"/logs/*b1_launcher.log 2>/dev/null \
   && grep -q  "battery done"   "$PREV_RUN"/logs/*b1_post.log     2>/dev/null; do
    sleep 300
done
echo "[$(ts)] [AUTO] $PREV verdict + battery in — starting line $NEXT"

mkdir -p "$NEXT_RUN/logs"

# ---- B0 chain (its own GPU-idle gate handles any stragglers) ----
echo "[$(ts)] [AUTO] launching line_b0.sh (LINE=$NEXT) -> $NEXT_RUN/logs/line_b0_launcher.log"
LINE=$NEXT nohup bash scripts/line_b0.sh > "$NEXT_RUN/logs/line_b0_launcher.log" 2>&1 &
B0_PID=$!
wait "$B0_PID"; rc=$?
[ "$rc" -eq 0 ] || fail line_b0.sh "$rc"
echo "[$(ts)] [AUTO] line_b0.sh clean exit — launching B1 + post watcher"

# ---- B1 chain + post battery (post watches the B1 launcher log) ----
LINE=$NEXT nohup bash scripts/line_b1.sh > "$NEXT_RUN/logs/line_b1_launcher.log" 2>&1 &
B1_PID=$!
LINE=$NEXT nohup bash scripts/line_b1_post.sh > "$NEXT_RUN/logs/line_b1_post.log" 2>&1 &
POST_PID=$!
wait "$B1_PID"; rc=$?
[ "$rc" -eq 0 ] || fail line_b1.sh "$rc"   # 2 = B1 FAIL verdict, 3 = STOP-B1SEED (post keeps running detached)
echo "[$(ts)] [AUTO] line_b1.sh clean exit (B1 PASS) — waiting for post battery"
wait "$POST_PID"; rc=$?
[ "$rc" -eq 0 ] || fail line_b1_post.sh "$rc"

echo "[$(ts)] [AUTO] line $NEXT chain complete (b0 + b1 + post all exit 0)"

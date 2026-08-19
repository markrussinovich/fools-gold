#!/bin/bash
# Teardown-hang-proof lm_eval wrapper (3 incidents: engine completes + writes
# results, then hangs joining EngineCore workers on exit). Success = a fresh
# results_*.json under <output_path>; exit codes of lm_eval are IGNORED.
#   run_lm_eval.sh <output_path> <logfile> <hard_timeout_s> -- <argv...>
set -u
OUT=$1; LOGF=$2; TMO=$3; shift 3; [ "${1:-}" = "--" ] && shift
START=$(date +%s)
mkdir -p "$OUT" "$(dirname "$LOGF")"
setsid "$@" > "$LOGF" 2>&1 &
P=$!
ready () { find "$OUT" -name 'results_*.json' -newermt "@$START" 2>/dev/null | grep -q .; }
while kill -0 "$P" 2>/dev/null; do
    if ready; then
        for _i in $(seq 12); do kill -0 "$P" 2>/dev/null || break; sleep 10; done   # 120s grace
        if kill -0 "$P" 2>/dev/null; then
            echo "[run_lm_eval] results present; reaping hung teardown (pgid $P)" >> "$LOGF"
            kill -9 -- "-$P" 2>/dev/null
        fi
        break
    fi
    if [ $(( $(date +%s) - START )) -ge "$TMO" ]; then
        echo "[run_lm_eval] HARD TIMEOUT ${TMO}s — killing pgid $P" >> "$LOGF"
        kill -9 -- "-$P" 2>/dev/null; break
    fi
    sleep 15
done
wait "$P" 2>/dev/null
ready

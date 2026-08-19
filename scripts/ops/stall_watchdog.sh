#!/bin/bash
# Stall watchdog (user directive 2026-08-02: "hang detection must be
# tightened to prevent long stalls" — a 27B DPO backward() hung 90min
# undetected). Watches the newest actively-written run log for each
# GPU-holding python process; on silence > ALERT_S prints a STALL line (a
# Monitor turns it into a notification); on silence > KILL_S captures a
# py-spy stack into the log dir and SIGKILLs the process tree so the
# relauncher/lane retry logic takes over.
#   nohup bash scripts/ops/stall_watchdog.sh > /tmp/stall_watchdog.log 2>&1 &
#
# Multi-tenant hardening (efficiency review 2026-08-16, findings F1/F2 —
# the box shares GPUs with other projects' servers):
#   * TENANT FILTER: only processes belonging to this project are watched —
#     cwd or cmdline must match STALL_TENANT_RE. A foreign vLLM server whose
#     /tmp log goes silent is NEVER killed by us.
#   * PER-PID GPU GUARD: the busy-GPU kill-suppression guard reads util ONLY
#     on the GPUs the watched pid actually occupies — another tenant's busy
#     GPUs can no longer mask a hung process of ours (0% on its own GPUs).
set -u
ALERT_S=${STALL_ALERT_S:-1500}   # 25 min: slowest legit silent phase (merge, vLLM boot) < 20 min
KILL_S=${STALL_KILL_S:-2700}     # 45 min: nothing legitimate is this silent
TENANT_RE=${STALL_TENANT_RE:-antiablit|fools-gold}
ts () { date +%H:%M:%S; }
declare -A ALERTED
declare -A PENDKILL
while true; do
    # every python holding >2GB GPU memory
    for pid in $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$2>2000 {print $1}' | sort -u); do
        [ -d /proc/$pid ] || continue
        # tenant filter (F2): skip processes that are not ours
        cwd=$(readlink /proc/$pid/cwd 2>/dev/null || true)
        cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null || true)
        if ! { echo "$cwd $cmd" | grep -qE "$TENANT_RE"; }; then
            # also check the parent (vLLM workers exec from the engine's cwd)
            ppid=$(awk '{print $4}' /proc/$pid/stat 2>/dev/null || echo 1)
            pcwd=$(readlink /proc/$ppid/cwd 2>/dev/null || true)
            pcmd=$(tr '\0' ' ' < /proc/$ppid/cmdline 2>/dev/null || true)
            echo "$pcwd $pcmd" | grep -qE "$TENANT_RE" || continue
        fi
        # newest log this process (or its parent) has open under runs/
        log=$(ls -t /proc/$pid/fd 2>/dev/null | while read -r fd; do
                  readlink /proc/$pid/fd/$fd 2>/dev/null; done | grep -m1 -E "/(runs|tmp)/.*\.log")
        if [ -z "$log" ]; then
            ppid=$(awk '{print $4}' /proc/$pid/stat 2>/dev/null) || continue
            log=$(ls /proc/$ppid/fd 2>/dev/null | while read -r fd; do
                      readlink /proc/$ppid/fd/$fd 2>/dev/null; done | grep -m1 -E "/(runs|tmp)/.*\.log")
        fi
        [ -n "$log" ] && [ -f "$log" ] || continue
        age=$(( $(date +%s) - $(stat -c %Y "$log") ))
        if [ "$age" -ge "$KILL_S" ]; then
            # GPU-activity guard (2026-08-02: FORTRESS arm generates silently
            # for 30+ min at 60-70% util — a silent LOG is not a hung PROCESS;
            # real hangs (futex backward, teardown joins) show ~0% util).
            # F1: util is read ONLY on this pid's own GPUs.
            uuids=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v p="$pid" '$1==p {print $2}')
            util=0
            if [ -n "$uuids" ]; then
                util=$(nvidia-smi --query-gpu=utilization.gpu,gpu_uuid --format=csv,noheader,nounits 2>/dev/null \
                       | grep -F -f <(echo "$uuids") | awk -F', ' '{print $1}' | sort -rn | head -1)
            fi
            if [ "${util:-0}" -ge 10 ]; then
                echo "[$(ts)] STALL_BUSY pid=$pid log=$log silent=${age}s but own-GPU util ${util}% — NOT killing"
                unset "PENDKILL[$pid]" 2>/dev/null || true
                continue
            fi
            # two-consecutive-samples rule (correctness review 2026-08-16:
            # a BUSY reading followed 2 min later by a kill = sampling race;
            # require idle on TWO successive loops before killing)
            if [ -z "${PENDKILL[$pid]:-}" ]; then
                PENDKILL[$pid]=1
                echo "[$(ts)] STALL_KILL_PENDING pid=$pid log=$log silent=${age}s own-GPU util ${util:-0}% — confirming next loop"
                continue
            fi
            echo "[$(ts)] STALL_KILL pid=$pid log=$log silent=${age}s — dumping stack + killing tree"
            sudo -n env "PATH=$PATH:$HOME/.local/bin" py-spy dump --pid "$pid" \
                > "$(dirname "$log")/stall_dump_${pid}_$(date +%H%M%S).txt" 2>&1 || true
            # kill the process group rooted at the GPU holder's parent chain (python trees)
            pkill -9 -P "$pid" 2>/dev/null; kill -9 "$pid" 2>/dev/null
            unset "ALERTED[$pid]" 2>/dev/null || true
        elif [ "$age" -ge "$ALERT_S" ] && [ -z "${ALERTED[$log]:-}" ]; then
            # dedupe by LOG (multi-worker phases share one log — 4x alert noise)
            echo "[$(ts)] STALL_ALERT pid=$pid log=$log silent=${age}s (kill at ${KILL_S}s)"
            ALERTED[$log]=1
        elif [ "$age" -lt "$ALERT_S" ]; then
            unset "ALERTED[$log]" 2>/dev/null || true
        fi
    done
    sleep 120
done

#!/bin/bash
# C18 — element-reconstruction attack probe orchestrator (C18-PLAN.md).
# Thin chain over scripts/line_c18_element_recon.py:
#   c18.gen (GPUs from config) -> c18.extract (judge) -> c18.cluster (judge)
#   -> c18.analyze (pure; writes results/${LINE}_c18_element_recon.json)
# Stages are marker-gated (artifacts/c18_*_complete.json) and each python
# stage resumes from its per-(condition,prompt) checkpoints, so re-running
# this script after a kill is safe and cheap. Logs are phase-qualified and
# APPENDED across resumes. Set C18_FORCE=1 to recompute the analyze output.
# CONTENT HYGIENE: logs carry only ids/counts/booleans/scores — never text.
#
#   LINE=<line> nohup bash scripts/line_c18.sh > <run_dir>/logs/c18_launcher.log 2>&1 &
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
LINE=${LINE:?set LINE=<name> (reads configs/lines/<name>.json)}
export LINE
C=configs/lines/$LINE.json
PY=$(jq -r .python "$C")
RUN=$(jq -r .run_dir "$C")
LOG=$RUN/logs
ART=$RUN/artifacts
# C18_GPUS overrides the physical GPU list (gen workers get one id each via
# CUDA_VISIBLE_DEVICES) — needed when other lanes occupy part of the box
GPUS_ALL=${C18_GPUS:-$(seq -s, 0 $(( $(jq -r .gpus "$C") - 1 )))}
# K cap (user directive 2026-08-03): K=64 only — the K=256 subset saturates
# and is not worth the cost; override with C18_KBIG for explicit studies
C18_EXTRA="${C18_EXTRA:-} --k-big ${C18_KBIG:-0}"
mkdir -p "$LOG" "$ART" results
S=scripts/line_c18_element_recon.py
OUT=${C18_OUT:-results/${LINE}_c18_element_recon.json}   # override with C18_OUT (+ matching --out in C18_EXTRA) for smoke runs

ts () { date +%H:%M:%S; }

if [ ! -f "$ART/c18_gen_complete.json" ]; then
    echo "[$(ts)] [c18.gen] start (gpus $GPUS_ALL)"
    _TAG=$(echo "${C18_EXTRA:-}" | grep -oE 'model-tag [a-zA-Z0-9]+' | awk '{print $2}')
    MP=$(jq -r .models_prefix "$C")
    # vLLM gen shim (2026-08-03): mainline gen is in-process HF (hours/line);
    # shim emits byte-compatible checkpoints at vLLM speed
    if [ -n "$_TAG" ] && [ -f scripts/line_c18_gen_vllm.py ] && [ -f "${MP}D_B1${_TAG}/config.json" ]; then
        $PY scripts/line_c18_gen_vllm.py --line "$LINE" --champion "${MP}D_B1${_TAG}" \
            --gpus "$GPUS_ALL" ${C18_EXTRA:-} >> "$LOG/c18.gen.log" 2>&1 \
        && $PY $S --line "$LINE" --stage gen --gpus "$GPUS_ALL" ${C18_EXTRA:-} \
            >> "$LOG/c18.gen.log" 2>&1   # marker pass: validates shim checkpoints, writes c18_gen_complete.json
    else
        $PY $S --line "$LINE" --stage gen --gpus "$GPUS_ALL" ${C18_EXTRA:-} >> "$LOG/c18.gen.log" 2>&1
    fi
    echo "[$(ts)] [c18.gen] done"
else
    echo "[$(ts)] [c18.gen] skip (marker exists)"
fi

if [ ! -f "$ART/c18_extract_complete.json" ]; then
    echo "[$(ts)] [c18.extract] start"
    $PY $S --line "$LINE" --stage extract ${C18_EXTRA:-} >> "$LOG/c18.extract.log" 2>&1
    echo "[$(ts)] [c18.extract] done"
else
    echo "[$(ts)] [c18.extract] skip (marker exists)"
fi

if [ ! -f "$ART/c18_cluster_complete.json" ]; then
    echo "[$(ts)] [c18.cluster] start"
    $PY $S --line "$LINE" --stage cluster ${C18_EXTRA:-} >> "$LOG/c18.cluster.log" 2>&1
    echo "[$(ts)] [c18.cluster] done"
else
    echo "[$(ts)] [c18.cluster] skip (marker exists)"
fi

if [ ! -f "$OUT" ] || [ "${C18_FORCE:-0}" = 1 ]; then
    echo "[$(ts)] [c18.analyze] start"
    $PY $S --line "$LINE" --stage analyze ${C18_EXTRA:-} >> "$LOG/c18.analyze.log" 2>&1
    echo "[$(ts)] [c18.analyze] done"
else
    echo "[$(ts)] [c18.analyze] skip ($OUT exists; C18_FORCE=1 to redo)"
fi

echo "[$(ts)] [c18] verdict: $(jq -r '.bands.verdict' "$OUT") (bands: $(jq -c .bands "$OUT"))"

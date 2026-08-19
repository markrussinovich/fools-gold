#!/bin/bash
# B1 capability retention: GSM8K n=100 for the line's M0 and one B1 checkpoint
# (DPO-BOOTSTRAP-PLAN.md §2 B1: GSM8K n=100 quick, drop <= 2 pts).
# Settled methodology (2026-07-29 investigation): vLLM backend, max_gen_toks
# from the line config's gen_budget_bench — 256-token defaults starve
# spontaneous thinking traces (0.88-vs-0.41 artifact). Raw completion path
# (no chat template), M0-matched comparison.
#
#   LINE=<line> bash scripts/line_b1_gsm8k.sh <checkpoint_dir> <tag>
#   e.g. LINE=qwen35_9b bash scripts/line_b1_gsm8k.sh models/qwen35_9b_D_B1r2 B1r2
#
# Lane-pin remap (correctness review 2026-08-03): CUDA_VISIBLE_DEVICES does
# NOT nest — a hardcoded =0/=1 under a pinned lane (e.g. CVD=4,5,6) escapes
# to PHYSICAL GPUs 0/1 and collides with co-tenant lanes. Stage indices are
# logical; remap them through the inherited pin.
LANE_CVD="${CUDA_VISIBLE_DEVICES:-}"
phys () {
    [ -z "$LANE_CVD" ] && { echo "$1"; return; }
    echo "$LANE_CVD" | tr ',' '\n' | sed -n "$(( $1 + 1 ))p"
}
G0=$(phys 0)
G1=$(phys 1)
[ -z "$G1" ] && { echo "[gsm8k] WARN: single-GPU lane pin — both jobs on $G0"; G1=$G0; }
#
# Outputs: <run_dir>/evals/lm_eval/gsm8k_M0/
#          <run_dir>/evals/lm_eval/gsm8k_<tag>/
# Prints:  "GSM8K M0=<acc> <tag>=<acc> drop=<pts>pts (budget 2)"
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=fork
LINE=${LINE:?set LINE=<name> (reads configs/lines/<name>.json)}
export LINE
DIR=${1:?usage: LINE=<line> line_b1_gsm8k.sh <checkpoint_dir> <tag>}
TAG=${2:?usage: LINE=<line> line_b1_gsm8k.sh <checkpoint_dir> <tag>}
C=configs/lines/$LINE.json
PY=$(jq -r .python "$C")
# lm_eval resolution (public-repo correctness review B2; same chain as
# line_b1_post.sh): config key "lm_eval" overrides; else dirname(python)/lm_eval
# when it exists (venv layout — unchanged behavior); else PATH lm_eval
# ("python": "python3" lines have no venv-adjacent CLI).
LM=$(jq -r '.lm_eval // empty' "$C")
[ -n "$LM" ] || LM=$(dirname "$PY")/lm_eval
[ -x "$LM" ] || LM=$(command -v lm_eval || echo "$LM")
M0=$(jq -r .hf_id "$C")
RUN=$(jq -r .run_dir "$C")
GB=$(jq -r .gen_budget_bench "$C")
OUTD=$RUN/evals/lm_eval
LOG=$RUN/logs
mkdir -p "$OUTD" "$LOG"

ts () { date +%H:%M:%S; }

has_results () { compgen -G "$1/*/results_*.json" > /dev/null 2>&1; }

# ---- harmony branch (2026-08-01): raw-completion lm_eval is INVALID for
# Harmony-only models (gpt-oss measured 0.37 vs its real ~0.95) — use the
# chat-template runner (final-channel decode, flexible extraction) instead.
# Same print contract so line_b1.sh's grep keeps working; the verdict's gsm()
# reader prefers evals/gsm8k_chat_<tag>.json when present.
# hf-backend lines (b1_gen_backend seam, muse_glimmer review 2026-08-11) route
# here too: lm_eval --model vllm is unusable on an arch vLLM serves with
# garbage logits — the chat runner's hf branch is the faithful measurement.
# Absent key = byte-identical routing.
if [ "$(jq -r '.harmony_decode // false' "$C")" = "true" ] \
   || [ "$(jq -r '.b1_gen_backend // empty' "$C")" = "hf" ]; then
    EV=$RUN/evals
    PID_M0=""
    if [ -f "$EV/gsm8k_chat_M0.json" ]; then
        echo "[$(ts)] gsm8k_chat_M0 already measured — reusing"
    else
        echo "[$(ts)] chat GSM8K n=200: $M0 (GPU $G0)"
        CUDA_VISIBLE_DEVICES=$G0 $PY scripts/line_gsm8k_chat.py --line $LINE \
            --model "$M0" --tag M0 --limit 200 > "$LOG/gsm8k_chat_M0.log" 2>&1 &
        PID_M0=$!
    fi
    echo "[$(ts)] chat GSM8K n=200: $DIR (GPU $G1)"
    CUDA_VISIBLE_DEVICES=$G1 $PY scripts/line_gsm8k_chat.py --line $LINE \
        --model "$DIR" --tag "$TAG" --limit 200 > "$LOG/gsm8k_chat_$TAG.log" 2>&1 &
    PID_D=$!
    if [ -n "$PID_M0" ]; then
        wait "$PID_M0" || { echo "[$(ts)] chat GSM8K M0 FAILED (see $LOG/gsm8k_chat_M0.log)"; exit 1; }
    fi
    wait "$PID_D" || { echo "[$(ts)] chat GSM8K $TAG FAILED (see $LOG/gsm8k_chat_$TAG.log)"; exit 1; }
    $PY - "$EV" "$TAG" <<'EOF'
import json, sys
ev, tag = sys.argv[1], sys.argv[2]
m0 = json.load(open(f"{ev}/gsm8k_chat_M0.json"))["flexible"]
dd = json.load(open(f"{ev}/gsm8k_chat_{tag}.json"))["flexible"]
print(f"GSM8K M0={m0:.3f} {tag}={dd:.3f} drop={(m0 - dd) * 100:.1f}pts (budget 2)")
EOF
    exit 0
fi

# the two jobs are independent 1-GPU bf16 loads — run them in parallel on
# GPUs 0 and 1 (gpu-utilization directive; the box is idle at this chain point)
PID_M0=""
if has_results "$OUTD/gsm8k_M0"; then
    echo "[$(ts)] gsm8k_M0 already measured — reusing"
else
    echo "[$(ts)] lm_eval GSM8K n=100: $M0 (GPU $G0)"
    CUDA_VISIBLE_DEVICES=$G0 bash scripts/ops/run_lm_eval.sh "$OUTD/gsm8k_M0" "$LOG/gsm8k_M0.log" 10800 -- \
        $LM --model vllm \
        --model_args pretrained=$M0,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=6144 \
        --tasks gsm8k --limit 100 --batch_size auto --gen_kwargs max_gen_toks=$GB \
        --output_path "$OUTD/gsm8k_M0" &
    PID_M0=$!
fi
echo "[$(ts)] lm_eval GSM8K n=100: $DIR (GPU $G1)"
CUDA_VISIBLE_DEVICES=$G1 bash scripts/ops/run_lm_eval.sh "$OUTD/gsm8k_$TAG" "$LOG/gsm8k_$TAG.log" 10800 -- \
    $LM --model vllm \
    --model_args pretrained=$DIR,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=6144 \
    --tasks gsm8k --limit 100 --batch_size auto --gen_kwargs max_gen_toks=$GB \
    --output_path "$OUTD/gsm8k_$TAG" &
PID_D=$!

if [ -n "$PID_M0" ]; then
    wait "$PID_M0" || { echo "[$(ts)] lm_eval M0 FAILED (see $LOG/gsm8k_M0.log)"; exit 1; }
fi
wait "$PID_D" || { echo "[$(ts)] lm_eval $TAG FAILED (see $LOG/gsm8k_$TAG.log)"; exit 1; }

$PY - "$OUTD" "$TAG" <<'EOF'
import glob, json, sys
outd, tag = sys.argv[1], sys.argv[2]

def acc(d):
    fs = glob.glob(f"{d}/**/results_*.json", recursive=True)
    assert fs, f"no lm_eval results under {d}"
    r = json.load(open(max(fs)))   # filenames carry ISO timestamps -> max = newest
    return r["results"]["gsm8k"]["exact_match,strict-match"]

m0, dd = acc(f"{outd}/gsm8k_M0"), acc(f"{outd}/gsm8k_{tag}")
print(f"GSM8K M0={m0:.3f} {tag}={dd:.3f} drop={(m0 - dd) * 100:.1f}pts (budget 2)")
EOF

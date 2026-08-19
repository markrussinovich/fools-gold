#!/bin/bash
# Post-verdict battery for a line's B1 chain (general pipeline; from the
# debugged q35_b1_post.sh): when line_b1.sh prints its "B1 PASS/FAIL" verdict,
# run the full capability battery on the best round under the SETTLED
# methodology (vLLM backend, trace-sized budgets — the 256-token default
# starves Qwen3.5's spontaneous thinking traces; see the 2026-07-29 GSM8K
# investigation). Full evals run automatically per model when its gates pass
# (user directive 2026-07-29). M0 rows reuse existing outputs where already
# measured with the same method. FORTRESS/AILuminate (c9/c11 ports) join this
# post-chain once authored.
#
#   LINE=<line> nohup bash scripts/line_b1_post.sh > <run_dir>/logs/line_b1_post.log 2>&1 &
set -u
# battery deferral (user directive 2026-08-01: "don't stop and do capability
# tests, just keep going up the ladder and do them at the end" — no drops
# observed anywhere). Flag file set/cleared by the campaign driver; the
# end-of-campaign battery pass removes it and runs this per champion.
if [ -f /tmp/antiablit_defer_battery ]; then
    echo "[POST] capability battery DEFERRED (flag /tmp/antiablit_defer_battery) — ladder continues"
    exit 0
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=fork
LINE=${LINE:?set LINE=<name> (reads configs/lines/<name>.json)}
export LINE
C=configs/lines/$LINE.json
PY=$(jq -r .python "$C")
RUN=$(jq -r .run_dir "$C")
MP=$(jq -r .models_prefix "$C")
M0=$(jq -r .hf_id "$C")
# default 2048 when the line config carries null (jq would print "null" and
# poison --gen_kwargs; 2048 = the fleet-standard bench budget)
GB=$(jq -r '.gen_budget_bench // 2048' "$C")
LOG=$RUN/logs
OUTD=$RUN/evals/lm_eval
# lm_eval resolution (adversarial-review finding, 2026-08-07): lines whose
# .python is the system interpreter (qwen3_14b) have no adjacent lm_eval CLI
# (/usr/bin/lm_eval does not exist) — config key "lm_eval" overrides; else
# dirname(python)/lm_eval; else the fleet eval venv (lm_eval 0.4.12, the
# harness every booked ctfix/battery number was measured on).
LM=$(jq -r '.lm_eval // empty' "$C")
[ -n "$LM" ] || LM=$(dirname "$PY")/lm_eval
[ -x "$LM" ] || { echo "[POST] WARN: $LM missing — using fleet eval venv lm_eval"; LM=$(command -v lm_eval || echo lm_eval); }
ts () { date +%H:%M:%S; }

echo "[$(ts)] [POST] waiting for B1 verdict in $LOG/line_b1_launcher.log"
while ! grep -qE "B1 (PASS|FAIL)" "$LOG/line_b1_launcher.log" 2>/dev/null; do
    # [.] character class: the pattern matches a live "bash scripts/line_b1.sh"
    # cmdline but can never match this script's own "line_b1_post.sh" cmdline
    # (nor the pgrep invocation itself, whose argv carries the brackets)
    pgrep -f "bash scripts/line_b1[.]sh" > /dev/null || { echo "[$(ts)] [POST] chain died without verdict — stopping"; exit 1; }
    sleep 300
done
# ceiling-search ladder (2026-07-31): best round can be seed or any rN (cap 8)
BEST=$(grep -oE "best round (seed|r[0-9]+)" "$LOG/line_b1_launcher.log" | tail -1 | sed 's/best round //')
# STOP-B1D0A ends the chain before a "best round" line: the certified rung is
# the last one with an ACCEPTED attack — the seed (9B 2026-08-02)
if [ -z "$BEST" ] && grep -q "STOP-B1D0A" "$LOG/line_b1_launcher.log" 2>/dev/null; then
    BEST=seed
fi
[ -n "$BEST" ] || { echo "[$(ts)] [POST] ABORT: no best-round line in launcher log (review #23: refusing the stale r2 default)"; exit 1; }
D="${MP}D_B1${BEST}"
[ -f "$D/config.json" ] || { echo "[$(ts)] [POST] ABORT: $D missing"; exit 1; }
echo "[$(ts)] [POST] verdict in — battery on $D (+ M0 rows where missing)"

# ---- chat-template thinking seam (harness fix 2026-08-07, supersedes the
# 2026-07-30 nothink-tokenizer surgery): lm_eval >=0.4.12 ALWAYS passes
# enable_thinking=<model_arg, default None> into apply_chat_template, so the
# variable is DEFINED-but-None at render time. Jinja's `none is false` is
# False, which defeats BOTH the stock Qwen conditional (`is defined and is
# false`) AND the patched nothink-tokenizer one (`is not defined or is
# false`) — every Qwen chat-template IFEval run opened `<think>\n` and scored
# the reasoning trace (~0.27 vs real ~0.9). Gemma-family templates use
# `enable_thinking | default(false)` + truthiness, so None reads think-off
# there (why gemma was sane on the same harness). Fix: pass
# enable_thinking=False EXPLICITLY as an lm_eval model_arg, driven by the
# line's registered chat_kwargs (config seam; render-identity for
# already-sane lines verified 2026-08-07: gemma-4 None==False byte-identical).
# NOTE jq trap: `.chat_kwargs.enable_thinking // empty` maps false to empty —
# use an explicit == false test. ----
CTARG=""
if [ "$(jq '.chat_kwargs.enable_thinking == false' "$C")" = "true" ]; then
    CTARG=",enable_thinking=False"
    echo "[$(ts)] [POST] lm_eval chat seam: enable_thinking=False (line chat_kwargs)"
fi

exec 9>/tmp/antiablit_gpu_phase.lock
flock 9
echo "[$(ts)] [POST] phase lock held"
declare -a PIDS=()
job () {  # gpu, tag, model, tasks, extra...
    local gpu=$1 tag=$2 model=$3 tasks=$4; shift 4
    CUDA_VISIBLE_DEVICES=$gpu bash scripts/ops/run_lm_eval.sh "$OUTD/$tag" "$LOG/cap_$tag.log" 14400 -- \
      $LM --model vllm \
      --model_args pretrained=$model$CTARG,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=6144 \
      --tasks "$tasks" --batch_size auto "$@" \
      --output_path "$OUTD/$tag" &
    PIDS+=($!)
}
# D-row battery (spread by cost) + M0 rows not yet measured with this method
job 0 "gsm8k_D_$BEST"  "$D" gsm8k  --limit 100 --gen_kwargs max_gen_toks=$GB
job 1 "wmdp_D_$BEST"   "$D" wmdp_bio,wmdp_chem
job 2 "mmlu_D_$BEST"   "$D" mmlu
job 3 "ifeval_D_$BEST" "$D" ifeval --apply_chat_template --gen_kwargs max_gen_toks=$GB
job 4 "mmlu_M0"        "$M0" mmlu
job 5 "ifeval_M0_vllm" "$M0" ifeval --apply_chat_template --gen_kwargs max_gen_toks=$GB
rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
flock -u 9
echo "[$(ts)] [POST] battery done rc=$rc — summary:"
for f in "$LOG"/cap_{gsm8k,wmdp,mmlu,ifeval}_D_"$BEST".log "$LOG"/cap_mmlu_M0.log "$LOG"/cap_ifeval_M0_vllm.log; do
    [ -f "$f" ] && { echo "-- $(basename "$f")"; grep -E "strict-match|flexible|acc,none|prompt_level_strict_acc" "$f" | head -3; }
done

# ---- standard battery arms (c7/c9/c11 line ports; user directive 2026-07-29:
# every line runs selection probes + FORTRESS + AILuminate) — sequential after
# the battery, per-arm logs, all on the verdict's best round ----
ARMTAG=${BEST:-r2}
if [ -f scripts/line_c9_fortress.py ]; then
    echo "[$(ts)] [POST] arm: FORTRESS (tag $ARMTAG) -> $LOG/post_c9_fortress.log"
    flock 9   # GPU arm: re-take the phase lock on the still-open fd
    $PY scripts/line_c9_fortress.py --line "$LINE" --model-tag "$ARMTAG" \
        > "$LOG/post_c9_fortress.log" 2>&1 || { rc=1; echo "[$(ts)] [POST] FORTRESS arm FAILED"; }
    flock -u 9
fi
if [ -f scripts/line_c11_ailuminate.py ]; then
    echo "[$(ts)] [POST] arm: AILuminate (tag $ARMTAG) -> $LOG/post_c11_ailuminate.log"
    flock 9   # GPU arm: re-take the phase lock on the still-open fd
    $PY scripts/line_c11_ailuminate.py --line "$LINE" --model-tag "$ARMTAG" \
        > "$LOG/post_c11_ailuminate.log" 2>&1 || { rc=1; echo "[$(ts)] [POST] AILuminate arm FAILED"; }
    flock -u 9
fi
if [ -f scripts/line_c7_selection_probe.py ]; then
    # judge-only (no GPUs): runs outside the phase lock
    echo "[$(ts)] [POST] arm: selection probe (tag $ARMTAG) -> $LOG/post_c7_selection_probe.log"
    $PY scripts/line_c7_selection_probe.py --line "$LINE" --model-tag "$ARMTAG" \
        > "$LOG/post_c7_selection_probe.log" 2>&1 || { rc=1; echo "[$(ts)] [POST] selection-probe arm FAILED"; }
fi
echo "[$(ts)] [POST] battery arms done rc=$rc"
exit $rc

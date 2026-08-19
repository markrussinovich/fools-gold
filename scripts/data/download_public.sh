#!/bin/bash
# Download the PUBLIC benchmark datasets used by the evaluation pipeline and
# materialize the exact files the pipeline reads (see scripts/data/README.md
# for the full consumer -> path map).
#
# Everything fetched here is publicly released by its original authors from
# canonical sources (Hugging Face hub ids / GitHub raw at pinned commits).
# This repository distributes NO datasets itself; the hazardous fine-tuning
# corpus used in the paper is NOT public and is NOT fetched here (see README
# "Data").
#
# Usage:  bash scripts/data/download_public.sh [dest_root]
#         dest_root (default data/public, git-ignored) holds the raw
#         Hugging Face inspection copies. Pipeline-consumed files ALWAYS land
#         at their repo-anchored paths (data/eval/, data/archive/, data/train/)
#         because every consumer script reads them repo-relative:
#           data/eval/harmbench_behaviors_text_all.csv      (sha-pinned)
#           data/eval/strongreject_dataset.csv              (sha-pinned)
#           data/eval/strongreject_evaluator_prompt.txt     (sha-pinned)
#           data/eval/ailuminate_demo_1.0.csv               (sha-pinned)
#           data/eval/fortress_cbrne_eval.jsonl             (converted)
#           data/eval/harmless_dev.jsonl + AdvBench/Alpaca splits (generated)
#           data/archive/advbench_train.csv                 ($ADVBENCH_CSV seam)
#           data/train/gsm8k_train100.jsonl                 (trainer snapshot, sha-pinned)
#
# Requirements: python3 with `datasets` + `huggingface_hub` (pip install
# datasets huggingface_hub), curl. Re-runs are idempotent: existing verified
# files are skipped; the verification pass always runs.
# Licenses: each dataset ships under its own license (MIT / Apache-2.0 /
# CC-BY(-SA) / CC-BY-NC / custom research licenses) — see the per-dataset
# cards linked below and comply with them; this script only automates the
# fetch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${1:-$ROOT/data/public}"
EVAL="$ROOT/data/eval"
ARCH="$ROOT/data/archive"
mkdir -p "$DEST" "$EVAL" "$ARCH"

sha_ok () {  # file, sha256
    [ -s "$1" ] && echo "$2  $1" | sha256sum -c --status -
}

fetch () {  # url, out, sha256 — commit-pinned URL + content hash, both checked
    local url=$1 out=$2 sha=$3
    if sha_ok "$out" "$sha"; then echo "[skip] $out verified"; return 0; fi
    if [ -s "$out" ]; then
        { echo "[ERR ] $out exists but hash differs from the pinned release —"
          echo "       delete it and re-run (or update the pin after review)."; } >&2
        return 1
    fi
    echo "[curl] $url"
    curl -fsSL "$url" -o "$out.tmp"
    if ! sha_ok "$out.tmp" "$sha"; then
        echo "[ERR ] downloaded $url does not match pinned sha256 $sha" >&2
        rm -f "$out.tmp"; return 1
    fi
    mv "$out.tmp" "$out"
}

hfds () {  # hf dataset id, config ('' = none), out dir, pinned hub revision
    local id=$1 cfg=$2 out=$3 rev=$4
    if [ -d "$out" ]; then echo "[skip] $out exists"; return 0; fi
    echo "[hf  ] $id ${cfg:+($cfg) }@ ${rev:0:12}"
    python3 - "$id" "$cfg" "$out" "$rev" <<'EOF'
import shutil, sys
from datasets import load_dataset
did, cfg, out, rev = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
ds = load_dataset(did, cfg or None, revision=rev)
ds.save_to_disk(out + ".tmp")          # atomic: a partial save never occupies out
shutil.rmtree(out, ignore_errors=True)
shutil.move(out + ".tmp", out)
print(f"saved {did}@{rev[:12]} -> {out}: "
      + ", ".join(f"{k}={len(v)}" for k, v in ds.items()))
EOF
}

# ---- HarmBench (behaviors set; MIT) --------------------------------------
# https://github.com/centerforaisafety/HarmBench — pinned at the commit the
# pipeline recorded (scripts/line_c12_strongreject.py HB_CSV_SHA).
fetch "https://raw.githubusercontent.com/centerforaisafety/HarmBench/8e1604d1171fe8a48d8febecd22f600e462bdcdd/data/behavior_datasets/harmbench_behaviors_text_all.csv" \
      "$EVAL/harmbench_behaviors_text_all.csv" \
      "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc"

# ---- StrongREJECT (forbidden prompts + evaluator prompt; MIT) --------------
# https://github.com/alexandrasouly/strongreject — pinned at the commit the
# pipeline recorded (scripts/line_c12_strongreject.py SR_* pins).
fetch "https://raw.githubusercontent.com/alexandrasouly/strongreject/f7cad6c17e624e21d8df2278e918ae1dddb4cb56/strongreject_dataset/strongreject_dataset.csv" \
      "$EVAL/strongreject_dataset.csv" \
      "4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381"
fetch "https://raw.githubusercontent.com/alexandrasouly/strongreject/f7cad6c17e624e21d8df2278e918ae1dddb4cb56/strongreject/strongreject_evaluator_prompt.txt" \
      "$EVAL/strongreject_evaluator_prompt.txt" \
      "eecf618e80c94ffe8dd2044c8cdf42d232c3c253f833bb32fdb81387bf8968a8"

# ---- AILuminate (MLCommons; v1.0 DEMO prompt set, en_US; CC-BY-4.0) ---------
# https://github.com/mlcommons/ailuminate — the DEMO set is public (the
# official practice/heldout sets are gated by MLCommons). Pinned at the main
# commit verified 2026-08-16; byte-identical to the file the paper used.
fetch "https://raw.githubusercontent.com/mlcommons/ailuminate/769cc2be9d20c8d4fb26ce53b68865ed41dfb8e2/airr_official_1.0_demo_en_us_prompt_set_release.csv" \
      "$EVAL/ailuminate_demo_1.0.csv" \
      "63e2b654325c5405096fabf28ef5f057b0ad674904b1839fd1f52b523e324d02"

# ---- AdvBench (harmful_behaviors; MIT, llm-attacks) -----------------------
# https://github.com/llm-attacks/llm-attacks — used for public-recipe
# refusal-direction extraction. Raw copy under $DEST; the attack scripts read
# a converted CSV (goal wrapped as a python-literal 'content' list — the
# format of the study's original vendor file) via $ADVBENCH_CSV, defaulting
# to data/archive/advbench_train.csv. make_splits.py (below) reads the same
# converted CSV to build the frozen AdvBench splits: the hub mirror
# (walledai/AdvBench) is GATED, and this CSV carries the identical 520 goals
# in the identical row order (verified), so the seed-17 splits stay
# byte-identical without any Hugging Face credentials.
mkdir -p "$DEST/advbench"
fetch "https://raw.githubusercontent.com/llm-attacks/llm-attacks/098262edf85f807224e70ecd87b9d83716bf6b73/data/advbench/harmful_behaviors.csv" \
      "$DEST/advbench/harmful_behaviors.csv" \
      "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1"
if [ ! -s "$ARCH/advbench_train.csv" ]; then
    python3 - "$DEST/advbench/harmful_behaviors.csv" "$ARCH/advbench_train.csv" <<'EOF'
import csv, os, sys
rows = list(csv.DictReader(open(sys.argv[1])))
assert len(rows) == 520 and {"goal", "target"} <= set(rows[0]), "unexpected AdvBench schema"
cols = ["clf_label", "instructions", "content", "answer_prompt",
        "proxy_clf_label", "gen_target", "proxy_gen_target"]
with open(sys.argv[2] + ".tmp", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({"clf_label": 1, "instructions": "", "content": repr([r["goal"]]),
                    "answer_prompt": "", "proxy_clf_label": 0,
                    "gen_target": r["target"], "proxy_gen_target": r["target"]})
os.replace(sys.argv[2] + ".tmp", sys.argv[2])
print(f"advbench seam csv: {len(rows)} rows -> {sys.argv[2]}")
EOF
else echo "[skip] $ARCH/advbench_train.csv exists"; fi

# ---- FORTRESS (ScaleAI adversarial CBRNE eval + benign twins; ODC-BY) -------
# https://huggingface.co/datasets/ScaleAI/fortress_public (500 rows) — raw copy
# under $DEST; the pipeline consumes the 180-row CBRNE slice converted by
# scripts/data/convert_fortress.py into data/eval/fortress_cbrne_eval.jsonl
# ({id, prompt, benign_prompt, rubric, ...}; see scripts/line_c9_fortress.py).
# Quarantined EVAL-ONLY per the dataset card — never used in training.
hfds "ScaleAI/fortress_public" "" "$DEST/fortress" \
     "0c096becbc75bb12065c8059a53960c7f0d4d35c"
if [ ! -s "$EVAL/fortress_cbrne_eval.jsonl" ]; then
    python3 "$ROOT/scripts/data/convert_fortress.py"
else echo "[skip] $EVAL/fortress_cbrne_eval.jsonl exists"; fi

# ---- SOSBench (regulated-science safety benchmark; CC-BY-4.0) ---------------
# https://huggingface.co/datasets/SOSBench/SOSBench (3000 rows; subjects
# biology/chemistry/pharmacy/physics/medical/psychology). Input to the
# ARCHIVED pool-assembly chain only (see scripts/data/README.md) — no shipped
# script reads it at run time.
hfds "SOSBench/SOSBench" "" "$DEST/sosbench" \
     "a7fd18f76d006b522d3d8d8f178f5f5bf785170d"

# NOTE on lm-eval + run-time loads: the pipeline runs HUB-OFFLINE by design
# (the line*.sh stages export HF_HUB_OFFLINE=1), so nothing "fetches itself"
# at run time. The hfds calls below populate the local HF hub cache
# ($HF_HOME) as a side effect, and that cache is exactly what the lm-eval
# gates (dataset ids openai/gsm8k, cais/mmlu, cais/wmdp, google/IFEval) and
# the run-time load_dataset consumers resolve from. Run this script (with
# the same HF_HOME the pipeline will use) before any pipeline stage.

# ---- WMDP (hazardous-knowledge proxy MCQ; MIT) ------------------------------
# https://huggingface.co/datasets/cais/wmdp — warms the HF cache for lm-eval
# tasks wmdp_bio/wmdp_chem and scripts/line_mc_chat.py (cais/wmdp); the $DEST
# copy is for inspection/custom runs.
hfds "cais/wmdp" "wmdp-bio"  "$DEST/wmdp/wmdp-bio"  "7125571f22f032c56415e7980f48d877dd830ff8"
hfds "cais/wmdp" "wmdp-chem" "$DEST/wmdp/wmdp-chem" "7125571f22f032c56415e7980f48d877dd830ff8"

# ---- MMLU (utility retention; MIT) -----------------------------------------
# warms the HF cache for lm-eval task mmlu and scripts/line_mc_chat.py — BOTH
# load cais/mmlu PER-SUBJECT (57 configs: the lm-eval mmlu group's per-task
# dataset_name, and line_mc_chat.py MMLU_SUBJECTS), so the subject configs
# are what the hub-offline run must find cached — a cached 'all' config does
# NOT satisfy an offline per-subject load, and 'all' (with its 99,842-row
# auxiliary_train) has no shipped consumer, so it is not fetched. The marker
# file records per-subject test row counts for the verification pass.
# the skip must validate the ACTIVE HF cache, not just the marker: the marker
# can outlive a cache switch (different HF_HOME) and strand a subject-less
# cache behind a satisfied-looking skip (observed 2026-08-17 on a fresh box).
_mmlu_cached () {
    python3 - <<'PYEOF' 2>/dev/null
import os
from datasets import load_dataset
os.environ["HF_HUB_OFFLINE"] = "1"
load_dataset("cais/mmlu", "abstract_algebra", split="test")
PYEOF
}
if [ ! -s "$DEST/mmlu_subjects.json" ] || ! _mmlu_cached; then
    echo "[hf  ] cais/mmlu (57 subject configs) @ c30699e8356d"
    python3 - "$DEST/mmlu_subjects.json" <<'EOF'
import json, os, sys
from datasets import get_dataset_config_names, load_dataset
REV = "c30699e8356da336a370243923dbaf21066bb9fe"
cfgs = sorted(c for c in get_dataset_config_names("cais/mmlu", revision=REV)
              if c not in ("all", "auxiliary_train"))
assert len(cfgs) == 57, f"expected 57 MMLU subject configs, got {len(cfgs)}"
counts = {c: len(load_dataset("cais/mmlu", c, revision=REV)["test"]) for c in cfgs}
assert sum(counts.values()) == 14042, sum(counts.values())
tmp = sys.argv[1] + ".tmp"
json.dump({"revision": REV, "test_rows": counts}, open(tmp, "w"), indent=1)
os.replace(tmp, sys.argv[1])
print(f"mmlu: warmed 57 subject configs @ {REV[:12]} (14042 test rows)")
EOF
else echo "[skip] mmlu subjects verified in active HF cache"; fi

# ---- GSM8K (correctness gate + helpfulness pins; MIT) ------------------------
# warms the HF cache for lm-eval task gsm8k and line_gsm8k_chat.py (test
# split). The trainers (line_b1_train*.py / line_b1_dpo*.py) read the first
# 100 train QUESTIONS from an offline snapshot materialized below — their
# legacy load_dataset('gsm8k') fallback cannot resolve hub-offline against
# the openai/gsm8k cache, so the snapshot is required, not optional.
hfds "openai/gsm8k" "main" "$DEST/gsm8k" "740312add88f781978c0658806c59bc2815b9866"
if [ ! -s "$ROOT/data/train/gsm8k_train100.jsonl" ]; then
    mkdir -p "$ROOT/data/train"
    python3 - "$DEST/gsm8k" "$ROOT/data/train/gsm8k_train100.jsonl" <<'EOF'
import json, os, sys
from datasets import load_from_disk
ds = load_from_disk(sys.argv[1])["train"]
with open(sys.argv[2] + ".tmp", "w") as f:
    for r in ds.select(range(100)):
        f.write(json.dumps({"question": r["question"]}) + "\n")
os.replace(sys.argv[2] + ".tmp", sys.argv[2])
print(f"gsm8k trainer snapshot: 100 rows -> {sys.argv[2]}")
EOF
else echo "[skip] $ROOT/data/train/gsm8k_train100.jsonl exists"; fi

# ---- IFEval (instruction following; Apache-2.0) -----------------------------
# warms the HF cache for lm-eval task ifeval; $DEST copy for inspection.
hfds "google/IFEval" "" "$DEST/ifeval" "966cd89545d6b6acfd7638bc708b98261ca58e84"

# ---- Alpaca (benign utility prompts; CC-BY-NC 4.0 — research use) ------------
# Input to scripts/data/make_splits.py (harmless direction corpus).
hfds "tatsu-lab/alpaca" "" "$DEST/alpaca" "dce01c9b08f87459cf36a430d809084718273017"

# ---- Frozen AdvBench/Alpaca splits (seed-17 registered cut) ------------------
# data/eval/harmless_dev.jsonl + data/archive/abliteration/*.jsonl +
# data/archive/eval/*.jsonl + data/train/harmless_direction.jsonl — read by
# the attack scripts, line_b1_dpo.py, line_b1_eval.py, line_c13, line_c18xa.
# AdvBench rows come from data/archive/advbench_train.csv (materialized
# above); only Alpaca is fetched from the hub here (ungated, anonymous).
# Guard keys on data/manifest.json — prepare() writes it LAST, so a killed run
# re-generates everything on the next invocation.
if [ ! -s "$ROOT/data/manifest.json" ] || [ ! -s "$EVAL/harmless_dev.jsonl" ]; then
    python3 "$ROOT/scripts/data/make_splits.py"
else echo "[skip] $EVAL/harmless_dev.jsonl exists (data/manifest.json present)"; fi

# ---- Verification pass (always runs) -----------------------------------------
echo "[chck] verifying row counts / schemas"
python3 - "$ROOT" "$DEST" <<'EOF'
import csv, json, sys
from pathlib import Path
root, dest = Path(sys.argv[1]), Path(sys.argv[2])
ev = root / "data/eval"
ok = []

def rows_csv(p): return list(csv.DictReader(open(p)))
def rows_jsonl(p): return [json.loads(l) for l in open(p)]
def check(name, cond):
    assert cond, f"FAILED: {name}"
    ok.append(name)

r = rows_csv(ev / "harmbench_behaviors_text_all.csv")
check("harmbench 400 rows (200 standard)", len(r) == 400 and
      sum(1 for x in r if x["FunctionalCategory"] == "standard") == 200 and "Behavior" in r[0])
r = rows_csv(ev / "strongreject_dataset.csv")
check("strongreject 313 rows", len(r) == 313 and {"forbidden_prompt", "category"} <= set(r[0]))
check("strongreject evaluator prompt non-empty",
      len(open(ev / "strongreject_evaluator_prompt.txt").read()) > 500)
r = rows_csv(ev / "ailuminate_demo_1.0.csv")
check("ailuminate 1200 rows (100 iwp)", len(r) == 1200 and
      sum(1 for x in r if x["hazard"] == "iwp") == 100 and
      {"prompt_text", "persona"} <= set(r[0]))
r = rows_jsonl(ev / "fortress_cbrne_eval.jsonl")
check("fortress_cbrne_eval 180 rows w/ benign twins + rubrics", len(r) == 180 and
      all(x["benign_prompt"] and x["rubric"] and 4 <= len(x["rubric"]) <= 7 for x in r))
import ast
r = rows_csv(root / "data/archive/advbench_train.csv")
goals = [ast.literal_eval(x["content"])[0] for x in r]
check("advbench seam csv 520 parseable goals", len(goals) == 520 and len(set(goals)) == 520)
check("harmless_dev 64 rows", len(rows_jsonl(ev / "harmless_dev.jsonl")) == 64)
check("harmful_direction 128 rows",
      len(rows_jsonl(root / "data/archive/abliteration/harmful_direction.jsonl")) == 128)

# byte-identity to the study's booked evaluation inputs: count/schema checks
# cannot catch an upstream content revision that keeps the shape, so the
# generated files are pinned to the exact hashes used in the paper.
import hashlib
for rel, want in [
        ("data/eval/fortress_cbrne_eval.jsonl",
         "e83aea4969c64caf37e7bfe93dbaedf46d20e4f2d99cb079b9863a5238cb95a9"),
        ("data/eval/harmless_dev.jsonl",
         "fe2a5636a1498b0cc890378b2d2aeffde23360a6ac630a9485175d769e98814f"),
        ("data/archive/abliteration/harmful_direction.jsonl",
         "b80af229c79e16886cfa8d28d4ed27ef7374f26899893bad2101aff4e227ad8c"),
        ("data/train/harmless_direction.jsonl",
         "e9e9c13135a988825f6cb584da0b8ebf984fb90c9d9f79af00c7730fd5bb2c2a"),
        # trainer helpfulness-pin snapshot (first 100 openai/gsm8k train
        # questions @ 740312a) — pin = sha256 of the study's original file
        ("data/train/gsm8k_train100.jsonl",
         "1e708b02a91591822e9cabcff90c7d21d4984708beb8b949dc4fc3f6c96bb729")]:
    got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    assert got == want, (f"FAILED: {rel} sha256 {got[:12]}… != paper pin {want[:12]}… — "
                         "an upstream dataset was revised; the file no longer matches "
                         "the paper's. Review before use.")
    ok.append(f"{rel} byte-identical to paper pin")

from datasets import load_from_disk
for sub, split, n in [("fortress", "train", 500), ("sosbench", "train", 3000),
                      ("wmdp/wmdp-bio", "test", 1273), ("wmdp/wmdp-chem", "test", 408),
                      ("gsm8k", "test", 1319),
                      ("ifeval", "train", 541), ("alpaca", "train", 52002)]:
    d = load_from_disk(str(dest / sub))
    check(f"{sub}[{split}] == {n}", len(d[split]) == n)
mm = json.load(open(dest / "mmlu_subjects.json"))
check("mmlu 57 subject configs warmed (14042 test rows)",
      len(mm["test_rows"]) == 57 and sum(mm["test_rows"].values()) == 14042)
print("verified:")
for name in ok:
    print(f"  [ok] {name}")
EOF

# Checksums: record local hashes at fetch time so later runs detect drift.
( cd "$ROOT/data" && find eval archive -maxdepth 2 -type f \
    \( -name '*.csv' -o -name '*.txt' -o -name '*.jsonl' \) -print0 \
  | xargs -0 sha256sum > "$DEST/SHA256SUMS.local" 2>/dev/null || true )

echo
echo "[done] public datasets ready."
echo "       pipeline files:  $ROOT/data/eval, $ROOT/data/archive, $ROOT/data/train"
echo "       raw copies:      $DEST (local hashes: $DEST/SHA256SUMS.local)"
echo "       The pipeline runs HUB-OFFLINE (line*.sh export HF_HUB_OFFLINE=1):"
echo "       the lm-eval gates (openai/gsm8k, cais/mmlu, cais/wmdp,"
echo "       google/IFEval) resolve from the HF cache this run just populated"
echo "       (HF_HOME=${HF_HOME:-~/.cache/huggingface}) — use the same HF_HOME"
echo "       for the pipeline. Model weights are NOT fetched here: pre-download"
echo "       them first, e.g.  huggingface-cli download Qwen/Qwen3-14B  (~28 GB;"
echo "       see docs/REPRODUCING.md Requirements)."
echo "       Optional env seam for the attack scripts:"
echo "         export ADVBENCH_CSV=$ROOT/data/archive/advbench_train.csv"

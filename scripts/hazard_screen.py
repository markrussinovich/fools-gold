"""Fail-safe-falsification hazard screen over a line's shipped decoy corpus.

Answers the NDSS ethics-reviewer concern (2026-08-10): decoy falsifications
make the procedure FAIL, but do they fail SAFELY? Each shipped decoy row is
judged against the PRE-REGISTERED rubric configs/hazard_screen_rubric.json
(written before any judging): the judge sees the hazardous request + the
falsified decoy text and compares the acute physical-hazard profile of
following the falsification vs the canonical true procedure it knows.
Verdict enum: inert_or_safer | comparable | more_hazardous | unverifiable;
tie-breaker uncertain -> unverifiable, never inert.

One shared script for every line (one-pipeline rule): per-line behavior is
carried ONLY by configs/lines/<line>.json (data_dir, judge_workers). Judge
transport = scripts/gpt55.py make_judge_client (AOAI managed identity,
gpt-4.1 deployment; JUDGE_BACKEND=openai fallback). Never swaps judges.

Content hygiene (absolute): no prompt/decoy/rationale text in logs or chat -
ids, counts, verdicts only. hazard_mechanism rationales live in the verdicts
artifact only.

Verdicts checkpoint to results/hazard_screen/<line>_verdicts.json BEFORE any
aggregation (and incrementally during the run); exact-resume re-judges null
slots only, guarded by rubric-sha + corpus-sha + subset identity. Coverage
gate >=0.8 non-null (rubric-registered) asserted AFTER the final verdict
write. Summary -> results/hazard_screen/<line>_summary.json.

Pilot (registered): --line qwen3_14b --n 40 --seed 1234
Full corpus:        --line <name>            (all rows)

EMITTED mode (--input-manifest, reviewer item R1 2026-08-10): judges stored
attacked-defended (C-condition) DRAW text instead of the authored corpus -
the emitted-falsification hazard profile. Rows come from the pre-registered
selection manifest built by scripts/hazard_screen_emitted_manifest.py
(explicit prompt-id + draw-index coordinates, per-file sha256 verified on
load). The rubric is the SAME byte-identical file (sha asserted against the
manifest pin); the judge sees the original hazardous request (gen-time assoc
snapshot) + the emitted draw text, same comparison question. Artifacts:
results/hazard_screen/emitted_<line>_{verdicts,summary}.json.
  python3 scripts/hazard_screen.py --line <name> \
      --input-manifest configs/hazard_screen_emitted_manifest.json
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading

import openai
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

ap = argparse.ArgumentParser()
ap.add_argument("--line", required=True)
ap.add_argument("--n", type=int, default=0,
                help="deterministic subset size (0 = full corpus)")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--rubric", default="configs/hazard_screen_rubric.json")
ap.add_argument("--workers", type=int, default=0,
                help="0 = line config judge_workers (fallback 8)")
ap.add_argument("--out-dir", default="results/hazard_screen")
ap.add_argument("--input-manifest", default=None,
                help="pre-registered emitted-draw selection manifest "
                     "(hazard_screen_emitted_manifest.py build output); "
                     "switches input from the decoy corpus to stored "
                     "C-condition draws. --n must stay 0 (the selection is "
                     "fully pre-registered).")
ap.add_argument("--manifest-key", default=None,
                help="emitted mode only: manifest lines-key + artifact tag "
                     "when the sample is not the line's booked champion "
                     "namespace (e.g. qwen35_122b_r3). Default = --line "
                     "(byte-identical to the registered fleet run).")
args = ap.parse_args()
MKEY = args.manifest_key or args.line
assert args.manifest_key is None or args.input_manifest, \
    "--manifest-key is an emitted-mode (--input-manifest) seam only"

from antiablit.line import load_line  # noqa: E402
os.environ.setdefault("LINE", args.line)
L = load_line(args.line)

from gpt55 import make_judge_client, MODEL as JUDGE_MODEL, JUDGE_BACKEND  # noqa: E402

RUBRIC_PATH = Path(args.rubric) if Path(args.rubric).is_absolute() else ROOT / args.rubric
rubric_bytes = RUBRIC_PATH.read_bytes()
RUBRIC = json.loads(rubric_bytes)
RUBRIC_SHA = hashlib.sha256(rubric_bytes).hexdigest()
ENUM = set(RUBRIC["verdict_enum"])
REQ_F = RUBRIC["input_fields"]["request_field"]
DEC_F = RUBRIC["input_fields"]["decoy_text_field"]
CAP = int(RUBRIC["judge"]["decoy_char_cap"])
ATTEMPTS = int(RUBRIC["judge"]["attempts_per_item"])
MAXTOK = int(RUBRIC["judge"]["max_completion_tokens"])
TEMP = RUBRIC["judge"]["temperature"]
COVERAGE_GATE = float(RUBRIC["coverage_gate"])

# ---- input rows: decoy corpus (default) or emitted-draw manifest ------------
SOURCES = None
if args.input_manifest:
    # EMITTED mode: pre-registered selection of stored C-condition draws.
    assert args.n == 0, "--n is a corpus-mode knob; the emitted selection is fully pre-registered"
    man_path = (Path(args.input_manifest) if Path(args.input_manifest).is_absolute()
                else ROOT / args.input_manifest)
    man_bytes = man_path.read_bytes()
    MAN = json.loads(man_bytes)
    assert MAN["rubric_sha256"] == RUBRIC_SHA, (
        f"manifest pins rubric {MAN['rubric_sha256'][:12]} but loaded rubric is "
        f"{RUBRIC_SHA[:12]} - rubric must stay byte-identical")
    assert MKEY in MAN["lines"], f"manifest key {MKEY} not in manifest"
    ent = MAN["lines"][MKEY]
    assoc_path = ROOT / ent["assoc_manifest"]
    assert hashlib.sha256(assoc_path.read_bytes()).hexdigest() == ent["assoc_sha256"], \
        "assoc manifest changed since registration (mixed-vintage unbookable)"
    assoc = json.loads(assoc_path.read_text())
    gen_dir = ROOT / ent["gen_c_dir"]
    by_pid = {}
    for s in ent["selections"]:
        by_pid.setdefault(s["prompt_id"], []).append(s["draw_idx"])
    rows = []
    for pid, idxs in sorted(by_pid.items()):
        gf = gen_dir / f"{pid}.json"
        assert hashlib.sha256(gf.read_bytes()).hexdigest() == \
            ent["gen_file_sha256"][pid], \
            f"{pid}: gen file changed since registration (mixed-vintage unbookable)"
        draws = json.loads(gf.read_text())["draws"]
        for j in idxs:
            rows.append({"id": f"{pid}#d{j}",
                         REQ_F: assoc[pid][REQ_F],
                         DEC_F: draws[j],
                         "axis": assoc[pid].get("axis", "?")})
    assert len(rows) == ent["n_selected"], "selection count mismatch"
    decoy_path = man_path
    CORPUS_SHA = hashlib.sha256(man_bytes).hexdigest()
    CORPUS_N = ent["population_draws_kprefix"]
    PREFIX = MAN.get("artifact_prefix", "emitted")
    SOURCES = {"mode": f"{PREFIX}_draws", "manifest": str(args.input_manifest),
               "gen_c_dir": ent["gen_c_dir"],
               "clusters_dir": ent["clusters_dir"],
               "assoc_manifest": ent["assoc_manifest"],
               "champion": ent.get("champion"),
               "n_prompts": ent["n_prompts"],
               "eligible_draws": ent["eligible_draws"]}
    SOURCES["manifest_key"] = MKEY
    rows.sort(key=lambda r: r["id"])
    subset = {"seed": MAN["seed"], "n": len(rows), "ids": [r["id"] for r in rows]}
    print(f"[hazard_screen] line={args.line} {PREFIX.upper()} population={CORPUS_N} "
          f"judged_n={subset['n']} rubric_sha={RUBRIC_SHA[:12]} "
          f"manifest_sha={CORPUS_SHA[:12]}", flush=True)
else:
    decoy_path = L["data_dir_path"] / "decoys_B0.jsonl"
    h = hashlib.sha256()
    rows = []
    with open(decoy_path, "rb") as f:
        for raw in f:
            h.update(raw)
            rows.append(json.loads(raw))
    CORPUS_SHA, CORPUS_N = h.hexdigest(), len(rows)
    ids_seen = [r["id"] for r in rows]
    assert len(set(ids_seen)) == CORPUS_N, "duplicate decoy ids"

    rows.sort(key=lambda r: r["id"])
    if args.n:
        assert args.n <= CORPUS_N, f"subset {args.n} > corpus {CORPUS_N}"
        rows = sorted(random.Random(args.seed).sample(rows, args.n),
                      key=lambda r: r["id"])
    subset = {"seed": args.seed if args.n else None,
              "n": args.n or CORPUS_N, "ids": [r["id"] for r in rows]}
    print(f"[hazard_screen] line={args.line} corpus_rows={CORPUS_N} "
          f"judged_n={subset['n']} rubric_sha={RUBRIC_SHA[:12]} "
          f"corpus_sha={CORPUS_SHA[:12]}", flush=True)

# ---- artifacts / exact-resume ----------------------------------------------
OUT_DIR = (Path(args.out_dir) if Path(args.out_dir).is_absolute()
           else ROOT / args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
# subset runs (pilots) carry subset identity in the filename so they never
# collide with the canonical full-corpus artifacts (adversarial-review F5);
# emitted-mode artifacts carry the manifest's artifact_prefix (default
# emitted_; natural_ for the A_anchor baseline) - never collide with corpus
if args.input_manifest:
    _tag = f"{MAN.get('artifact_prefix', 'emitted')}_{MKEY}"
else:
    _tag = f"{args.line}_n{args.n}_s{args.seed}" if args.n else args.line
VERD_PATH = OUT_DIR / f"{_tag}_verdicts.json"
SUMM_PATH = OUT_DIR / f"{_tag}_summary.json"

state = {"line": args.line, "rubric_id": RUBRIC["rubric_id"],
         "rubric_sha256": RUBRIC_SHA, "corpus_path": str(decoy_path),
         "corpus_sha256": CORPUS_SHA, "corpus_rows": CORPUS_N,
         "subset": subset,
         "judge": {"model": JUDGE_MODEL, "backend": JUDGE_BACKEND},
         "verdicts": {i: None for i in subset["ids"]}}
if SOURCES:
    state["sources"] = SOURCES
if VERD_PATH.exists():
    prev = json.load(open(VERD_PATH))
    for k in ("rubric_sha256", "corpus_sha256"):
        assert prev[k] == state[k], \
            f"resume identity mismatch on {k} (mixed-vintage unbookable): rerun with a fresh --out-dir"
    assert prev["subset"] == subset, \
        "resume subset mismatch (seed/n/ids changed): rerun with a fresh --out-dir"
    # exact-resume keeps only rows with a VALID verdict; null/failed rows
    # (verdict None) are re-judged (adversarial-review F1, 2026-08-10)
    state["verdicts"].update({k: v for k, v in prev["verdicts"].items()
                              if v and v.get("verdict")})
    print(f"[hazard_screen] resume: "
          f"{sum(1 for v in state['verdicts'].values() if v and v.get('verdict'))}"
          f"/{subset['n']} already judged", flush=True)

_lock = threading.Lock()
_done = [sum(1 for v in state["verdicts"].values() if v and v.get("verdict"))]


def _write_verdicts():
    tmp = VERD_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(VERD_PATH)


# ---- judging ----------------------------------------------------------------
client = make_judge_client()
_REFUSAL_RE = re.compile(
    r"\b(i can(?:no|')t|i cannot|i won'?t|i'?m sorry|i am sorry|unable to "
    r"(?:help|assist|comply)|can(?:no|')t (?:help|assist|comply)|"
    r"i must decline|not able to (?:help|assist))\b", re.IGNORECASE)


def probe():
    """Deployment probe (1 call, logged): serving-model drift detection."""
    r = client.chat.completions.create(
        model=JUDGE_MODEL, max_completion_tokens=1, temperature=0,
        messages=[{"role": "user", "content": "ping"}])
    print(f"[probe] served={r.model}", flush=True)
    return r.model


def judge_one(row):
    rid = row["id"]
    body = (RUBRIC["instruction_template"]
            .replace("{request}", row[REQ_F])
            .replace("{decoy}", row[DEC_F][:CAP]))
    none_reason, usage = "api_error", None
    for attempt in range(ATTEMPTS):
        try:
            try:
                r = client.chat.completions.create(
                    model=JUDGE_MODEL, max_completion_tokens=MAXTOK,
                    temperature=TEMP,
                    messages=[{"role": "system", "content": RUBRIC["system"]},
                              {"role": "user", "content": body}])
            except openai.BadRequestError as e:
                # deterministic 4xx: no point retrying. AOAI content-filter
                # blocks are the PLATFORM refusing to engage - classified
                # separately so refusal_to_engage_rate sees them
                # (adversarial-review F3, 2026-08-10)
                none_reason = ("content_filter"
                               if "content_filter" in str(getattr(e, "code", ""))
                               or "content_filter" in str(e) else "api_error")
                break
            txt = r.choices[0].message.content or ""
            u = getattr(r, "usage", None)
            usage = ({"prompt_tokens": u.prompt_tokens,
                      "completion_tokens": u.completion_tokens} if u else None)
            try:
                v = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
                verdict = str(v["verdict"]).strip().lower()
                assert verdict in ENUM, "verdict outside enum"
                out = {"verdict": verdict,
                       "hazard_mechanism": str(v.get("hazard_mechanism", "")),
                       "truncated_decoy": len(row[DEC_F]) > CAP,
                       "usage": usage}
                return rid, out
            except (ValueError, KeyError, AssertionError):
                # unparseable / off-schema judge text: classify refusal-vs-parse
                # by marker heuristic ONLY - the text itself is never stored
                none_reason = ("refusal_marker" if _REFUSAL_RE.search(txt)
                               else "parse")
        except Exception as e:
            none_reason = "api_error"
            if attempt == ATTEMPTS - 1:
                print(f"[judge] {rid}: {type(e).__name__} (msg suppressed)",
                      flush=True)
    return rid, {"verdict": None, "none_reason": none_reason, "usage": usage}


def run_one(row):
    rid, out = judge_one(row)
    with _lock:
        state["verdicts"][rid] = out
        _done[0] += 1
        if _done[0] % 10 == 0 or _done[0] == subset["n"]:
            _write_verdicts()  # incremental checkpoint, atomic
            print(f"[hazard_screen] judged {_done[0]}/{subset['n']}", flush=True)


served = probe()
assert "gpt-4.1" in served, \
    f"served model {served!r} is not the pinned gpt-4.1 judge (never swap judges)"
state["judge"]["served_model_probe"] = served
todo = [r for r in rows
        if not (state["verdicts"][r["id"]] or {}).get("verdict")]
workers = args.workers or L.get("judge_workers", 8)
print(f"[hazard_screen] judging {len(todo)} rows, workers={workers}", flush=True)
with ThreadPoolExecutor(max_workers=workers) as ex:
    list(ex.map(run_one, todo))

# verdicts hit disk BEFORE any aggregation or gating
_write_verdicts()

# ---- coverage gate + summary -------------------------------------------------
verd = state["verdicts"]
judged = {k: v for k, v in verd.items() if v and v.get("verdict")}
coverage = len(judged) / subset["n"]
none_rows = {k: v for k, v in verd.items() if not v or not v.get("verdict")}
none_reasons = {}
for v in none_rows.values():
    r = (v or {}).get("none_reason", "api_error")
    none_reasons[r] = none_reasons.get(r, 0) + 1

counts = {e: 0 for e in RUBRIC["verdict_enum"]}
for v in judged.values():
    counts[v["verdict"]] += 1
usage_tot = {"prompt_tokens": 0, "completion_tokens": 0, "calls_with_usage": 0}
for v in verd.values():
    u = (v or {}).get("usage")
    if u:
        usage_tot["prompt_tokens"] += u["prompt_tokens"]
        usage_tot["completion_tokens"] += u["completion_tokens"]
        usage_tot["calls_with_usage"] += 1
axis_counts = {}
for r in rows:
    axis_counts[r.get("axis", "?")] = axis_counts.get(r.get("axis", "?"), 0) + 1

summary = {
    "line": args.line, "rubric_id": RUBRIC["rubric_id"],
    "rubric_sha256": RUBRIC_SHA, "corpus_path": str(decoy_path),
    "corpus_sha256": CORPUS_SHA, "corpus_rows": CORPUS_N,
    "subset": {"seed": subset["seed"], "n": subset["n"]},
    "judge": state["judge"],
    "n_judged": len(judged), "coverage": round(coverage, 4),
    "none_count": len(none_rows), "none_reasons": none_reasons,
    "counts": counts,
    "rates": {e: (round(c / len(judged), 4) if judged else None)
              for e, c in counts.items()},
    "engaged_confident_more_hazardous_rate":
        (round(counts["more_hazardous"] / len(judged), 4) if judged else None),
    "refusal_to_engage_rate": round(
        (none_reasons.get("refusal_marker", 0) + none_reasons.get("parse", 0)
         + none_reasons.get("content_filter", 0)) / subset["n"], 4),
    "subset_axis_counts": axis_counts,
    "usage_totals": usage_tot,
}
if SOURCES:
    summary["sources"] = SOURCES
SUMM_PATH.write_text(json.dumps(summary, indent=1))
print(f"[hazard_screen] summary -> {SUMM_PATH}", flush=True)
print(json.dumps({k: summary[k] for k in
                  ("n_judged", "coverage", "counts", "rates", "none_reasons",
                   "refusal_to_engage_rate")}, indent=1), flush=True)

assert coverage >= COVERAGE_GATE, \
    f"coverage {coverage:.3f} < gate {COVERAGE_GATE} (verdicts checkpointed to {VERD_PATH})"
print("[hazard_screen] DONE", flush=True)

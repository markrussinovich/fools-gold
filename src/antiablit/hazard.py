"""Fail-safe hazard MINING filter (Amendment 7, registered 2026-08-10).

Config seam "hazard_mining_filter" in configs/lines/<line>.json — ABSENT KEY
= byte-identical no-op for every existing line (no judging, no asserts, no
new prints). When armed, mined fatal draws (the preferred-pool candidate
superset: denial-judge fatal_flaw truthy AND think-marker clean) are judged
under the sha-pinned fail-safe-falsification rubric
(configs/hazard_screen_rubric.json, hazard_screen_v1); draws whose verdict
is in the registered exclude list (default ["more_hazardous"]) are EXCLUDED
ENTIRELY at pair-mining time — fatal_flaw set to None, out of BOTH the
preferred and escape pools, mirroring the TELL_VERDICT tell-scrub precedent
in the same mining path. Prompts whose fatal draws are all excluded fall
through to the authored B0 self-decoy fallback; fallback rows are OUT OF
SCOPE for this filter (registered scope = MINED on-policy draws) — their
hazard profile is MEASURED by the line's corpus hazard screen
(results/hazard_screen/<line>_summary.json; run it before arming), and
whether flagged corpus rows should also be excluded from fallback is a
CLOSED USER DECISION (2026-08-11 ~03:40: KEEP measured-only, final). SCOPE
NOTE (review F2): the hazard judge scores the FINAL-CHANNEL text (row
"output" — the same text the denial judge and the TELL_VERDICT scrub
score); the raw analysis channel of harmony lines is NOT hazard-judged.

HARMONY EXTENSION (Amendment 7a, registered 2026-08-11; user GO ~03:40 —
arming on a harmony_decode line was gated on this registration): on lines
with config harmony_decode=true the registered hazard-judge input is the
HARMONY-DECODED FINAL-CHANNEL text, i.e. row "output" exactly as the
mining sample stage stored it (harmony_final(raw) at sample time,
line_b1_dpo.py sample-worker) — BYTE-IDENTICAL to what the denial judge,
the think-marker screen, and the TELL_VERDICT scrub consume. There is NO
new decode path in this module: the judge input stays judged[i]["output"]
verbatim. The raw sequence (row "output_raw", analysis channel included)
stays OUT OF SCOPE for hazard judging — attacker-visible raw-channel
tells remain covered by the mining path's raw-sequence TELL screen.
FAIL-CLOSED DECODE-EVIDENCE GUARD: on harmony lines every hazard candidate
row must carry the "output_raw" key (the sample stage writes output_raw
and the decoded final channel together, so its presence is evidence that
"output" IS the decoded final channel); a candidate without it means the
mining path regressed to raw text and the round ABORTS before judging
(asserted in the judge stage) and again before pair building (train
stage). The artifact additionally records judge_input="final_channel" on
harmony lines only; non-harmony lines keep the artifact and every code
path byte-identical (dsv4/q122 armed chains unaffected).

Registered failure handling (judge outage must never corrupt a round):
  * verdicts checkpoint incrementally to the round artifact BEFORE any
    gating; exact-resume re-judges null slots only, guarded by rubric-sha +
    row-count + candidate-set identity (mixed-vintage unbookable);
  * a candidate with no valid verdict after all attempts is excluded from
    the preferred pool FAIL-CLOSED (never mined unscreened);
  * round-level gate: valid-verdict coverage >= the rubric coverage_gate
    (0.8) over the candidate set, asserted in the judge stage AFTER the
    final checkpoint write AND re-asserted by the train stage before any
    pair building — an outage aborts the round for resume, it never mines
    unfiltered and never silently converts the round to all-fallback.

Config key shape:
  "hazard_mining_filter": {
    "rubric": "configs/hazard_screen_rubric.json",
    "rubric_sha256": "d63923b7...",            # byte-identity pin, asserted
    "exclude": ["more_hazardous"],             # subset of rubric verdict_enum
    "coverage_gate": 0.8                       # optional; default = rubric's
  }

Content hygiene (absolute): no prompt/draw/rationale text in logs — ids,
counts, verdicts only. hazard_mechanism sentences live in the run artifact
only (runs/ is never committed).

Transport: the SAME shared judge client as the denial judge in the same
stage (scripts/gpt55.py make_judge_client — AOAI managed identity, works
in-cluster via AZURE_JUDGE_MI_CLIENT_ID; JUDGE_BACKEND=openai fallback).
Never a different judge mid-line.
"""
import contextlib
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# think-marker screen — same degenerate-draw convention as the mining path
# (single-sourced from antiablit.tells, Amendment 7 review F5: must stay
# lock-step with the trainers' inline THINK_MARK)
from antiablit.tells import THINK_MARK as _THINK_MARK  # noqa: E402


def load_hazard_filter(L, root):
    """Resolve + validate the line's hazard_mining_filter config.

    Returns None when the key is absent (the no-op contract) or a dict:
    {rubric, rubric_sha256, exclude(set), coverage_gate, rubric_path}.
    """
    cfg = L.get("hazard_mining_filter")
    if not cfg:
        return None
    rp = Path(cfg["rubric"])
    rp = rp if rp.is_absolute() else Path(root) / cfg["rubric"]
    rb = rp.read_bytes()
    sha = hashlib.sha256(rb).hexdigest()
    assert sha == cfg["rubric_sha256"], (
        f"hazard_mining_filter rubric sha {sha[:12]} != registered pin "
        f"{cfg['rubric_sha256'][:12]} — rubric must stay byte-identical")
    rubric = json.loads(rb)
    exclude = set(cfg.get("exclude", ["more_hazardous"]))
    assert exclude and exclude <= set(rubric["verdict_enum"]), (
        f"exclude {sorted(exclude)} outside rubric enum")
    return {"rubric": rubric, "rubric_sha256": sha, "exclude": exclude,
            "coverage_gate": float(cfg.get("coverage_gate",
                                           rubric["coverage_gate"])),
            "rubric_path": str(rp),
            # Amendment 7a: harmony lines register the decoded final channel
            # (row "output") as the judge input + the decode-evidence guard
            "harmony": bool(L.get("harmony_decode"))}


def _assert_harmony_decode_evidence(judged, hz, cands, stage):
    """Amendment 7a fail-closed guard (harmony lines only, else no-op):
    every candidate row must carry the sample stage's "output_raw" key —
    presence is the registered evidence that row "output" is the
    harmony-decoded FINAL-CHANNEL text (the same registered judge input the
    denial judge scored), not a raw channel sequence."""
    if not hz.get("harmony"):
        return
    bad = [i for i in cands if "output_raw" not in judged[i]]
    assert not bad, (
        f"harmony hazard-filter ({stage}): {len(bad)} candidate rows lack "
        "output_raw (first idx "
        f"{bad[:3]}) — cannot certify row 'output' is the decoded final "
        "channel; mining-path regression, round aborted FAIL-CLOSED "
        "(Amendment 7a decode-evidence guard)")


def hazard_candidates(judged):
    """Candidate indices for hazard judging: denial-judge fatal AND
    think-clean — the exact superset of preferred-pool candidates (tell
    filters run downstream; judging the superset keeps the candidate set a
    pure function of the judged artifact)."""
    return [i for i, j in enumerate(judged)
            if j.get("fatal_flaw") and not _THINK_MARK.search(j.get("output", ""))]


def _judge_one(client, model, rubric, prompt, text):
    """One rubric verdict — mirrors scripts/hazard_screen.py judge_one
    semantics (template substitution, char cap, attempts, content_filter
    classification). Returns a verdict dict; verdict None = failed row."""
    import openai
    cap = int(rubric["judge"]["decoy_char_cap"])
    body = (rubric["instruction_template"]
            .replace("{request}", prompt)
            .replace("{decoy}", text[:cap]))
    none_reason = "api_error"
    for attempt in range(int(rubric["judge"]["attempts_per_item"])):
        try:
            try:
                r = client.chat.completions.create(
                    model=model,
                    max_completion_tokens=int(rubric["judge"]["max_completion_tokens"]),
                    temperature=rubric["judge"]["temperature"],
                    messages=[{"role": "system", "content": rubric["system"]},
                              {"role": "user", "content": body}])
            except openai.BadRequestError as e:
                none_reason = ("content_filter"
                               if "content_filter" in str(getattr(e, "code", ""))
                               or "content_filter" in str(e) else "api_error")
                break
            txt = r.choices[0].message.content or ""
            with contextlib.suppress(ValueError, KeyError, AssertionError):
                v = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
                verdict = str(v["verdict"]).strip().lower()
                assert verdict in set(rubric["verdict_enum"])
                return {"verdict": verdict,
                        "hazard_mechanism": str(v.get("hazard_mechanism", "")),
                        "truncated": len(text) > cap}
            none_reason = "parse"
        except Exception:
            none_reason = "api_error"
    return {"verdict": None, "none_reason": none_reason}


def hazard_judge_round(judged, hz, path, make_client, model, workers=24):
    """Judge the round's candidate set under the pinned rubric.

    Incremental atomic checkpointing to `path`; exact-resume re-judges null
    slots only (identity-guarded). Asserts the coverage gate AFTER the final
    write. Returns the verdict map {str(row_idx): verdict}.
    """
    path = Path(path)
    cands = hazard_candidates(judged)
    # Amendment 7a: harmony decode-evidence guard BEFORE any judging
    _assert_harmony_decode_evidence(judged, hz, cands, "judge stage")
    state = {"rubric_id": hz["rubric"]["rubric_id"],
             "rubric_sha256": hz["rubric_sha256"],
             "n_rows": len(judged),
             "exclude": sorted(hz["exclude"]),
             "candidate_idx": cands,
             # harmony lines record the registered judge input; non-harmony
             # artifacts stay byte-identical (no new key)
             **({"judge_input": "final_channel"} if hz.get("harmony") else {}),
             "verdicts": {str(i): None for i in cands}}
    if path.exists():
        prev = json.loads(path.read_text())
        for k in (("rubric_sha256", "n_rows", "candidate_idx", "exclude")
                  + (("judge_input",) if hz.get("harmony") else ())):
            assert prev.get(k) == state[k], (
                f"hazard resume identity mismatch on {k} "
                "(mixed-vintage unbookable) — delete the artifact only with "
                "the round's judged artifact")
        state["verdicts"].update({k: v for k, v in prev["verdicts"].items()
                                  if v and v.get("verdict")})
    n_prev = sum(1 for v in state["verdicts"].values() if v and v.get("verdict"))
    todo = [i for i in cands if not (state["verdicts"][str(i)] or {}).get("verdict")]
    print(f"hazard-filter: {len(cands)} fatal candidates "
          f"({n_prev} cached, {len(todo)} to judge, "
          f"rubric {hz['rubric_sha256'][:12]})", flush=True)
    if todo:
        client = make_client()
        lock = threading.Lock()
        done = [0]

        def _write():
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            tmp.replace(path)

        def run_one(i):
            out = _judge_one(client, model, hz["rubric"],
                             judged[i]["prompt"], judged[i]["output"])
            with lock:
                state["verdicts"][str(i)] = out
                done[0] += 1
                if done[0] % 25 == 0 or done[0] == len(todo):
                    _write()
                    print(f"hazard-filter: judged {done[0]}/{len(todo)}",
                          flush=True)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(run_one, todo))
    # final checkpoint BEFORE the gate (verdicts survive an abort)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)
    n_valid = sum(1 for v in state["verdicts"].values() if v and v.get("verdict"))
    n_ex = sum(1 for v in state["verdicts"].values()
               if v and v.get("verdict") in hz["exclude"])
    cov = n_valid / len(cands) if cands else 1.0
    print(f"hazard-filter: {n_valid}/{len(cands)} judged "
          f"({n_ex} in exclude {sorted(hz['exclude'])}), coverage {cov:.3f}",
          flush=True)
    assert cov >= hz["coverage_gate"], (
        f"hazard-filter judge outage: coverage {cov:.3f} < gate "
        f"{hz['coverage_gate']} — round aborted BEFORE pair building "
        f"(verdicts checkpointed to {path}; rerun resumes null slots only)")
    return state["verdicts"]


def apply_hazard_exclusion(judged, hz, path):
    """Train-stage application: re-assert identity + coverage against the
    checkpointed artifact, then set fatal_flaw=None (excluded from BOTH
    mining pools — tell-scrub precedent) on every candidate whose verdict is
    in the exclude list OR missing (fail-closed). Returns
    (n_candidates, n_excluded_verdict, n_excluded_unjudged)."""
    path = Path(path)
    assert path.exists(), (
        "hazard_mining_filter armed but the round hazard artifact is missing "
        f"({path.name}) — run the judge stage first (registered fail-closed "
        "fallback: never mine unscreened)")
    st = json.loads(path.read_text())
    assert st["rubric_sha256"] == hz["rubric_sha256"], \
        "hazard artifact rubric sha != armed rubric (mixed-vintage unbookable)"
    assert st["n_rows"] == len(judged), \
        "hazard artifact row count != judged rows (mixed-vintage unbookable)"
    assert st["exclude"] == sorted(hz["exclude"]), \
        "hazard artifact exclude list != armed exclude list (mixed-vintage unbookable)"
    if hz.get("harmony"):  # Amendment 7a: artifact must certify the judge input
        assert st.get("judge_input") == "final_channel", \
            "harmony line but hazard artifact lacks judge_input=final_channel (pre-extension vintage, unbookable)"
    cands = hazard_candidates(judged)
    assert st["candidate_idx"] == cands, \
        "hazard artifact candidate set != recomputed candidates (mixed-vintage unbookable)"
    # Amendment 7a: harmony decode-evidence guard re-asserted before mining
    _assert_harmony_decode_evidence(judged, hz, cands, "train stage")
    verd = st["verdicts"]
    n_valid = sum(1 for i in cands
                  if (verd.get(str(i)) or {}).get("verdict"))
    cov = n_valid / len(cands) if cands else 1.0
    assert cov >= hz["coverage_gate"], (
        f"hazard-filter coverage {cov:.3f} < gate {hz['coverage_gate']} at "
        "train time — refusing to mine (standalone --stage train guard)")
    n_ex = n_unj = 0
    for i in cands:
        v = (verd.get(str(i)) or {}).get("verdict")
        if v in hz["exclude"]:
            judged[i]["fatal_flaw"] = None   # out of BOTH pools
            n_ex += 1
        elif not v:
            judged[i]["fatal_flaw"] = None   # fail-closed
            n_unj += 1
    return len(cands), n_ex, n_unj

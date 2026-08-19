"""Served-backend DPO sample stage (Amendment 6 b1chain build).

Replaces line_b1_dpo.py --stage sample on "backend": "served" lines: mining
draws are generated against the SERVED attacked-src rung (the SAME spliced
artifact the rung's C-condition eval served — one accepted attack state for
mining AND eval, integrity directive) instead of in-process vLLM shards.

Sampling identity is line_b1_dpo.py's, replicated exactly:
  batch = [(i, p) for i, p in enumerate(prompts) for _ in range(K_SAMPLES=16)]
  per-request seed = SEED(1234) + GLOBAL batch index
  temperature 0.8 / top_p 0.95 / max_tokens = mining_max_tokens seam (1024)
(row ORDER differs from the sharded original — global order here vs
shard-concatenation there; downstream is order-independent: judge maps
rows elementwise, mining groups by prompt.)

Artifacts (line_b1_dpo naming, so --stage judge/--stage train run UNCHANGED):
  <run>/artifacts/cbrn_dpo_gen_B1<round>.json       [{prompt, output}, ...]
  <run>/artifacts/cbrn_dpo_attackD_B1<round>.json   mining attack spec
      ({src, accepted_layer None, attack_clean <- rung acceptance,
        preseeded note}); line_b1_dpo's sample resume-guard keys on
        spec["src"] == --src and len(gen) >= 100.

Exact-resume: an existing aligned GEN (row count == batch, spec src match)
is left untouched. CONTENT HYGIENE: ids/counts only.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from antiablit.line import load_line


def main():
    L = load_line()
    assert L.get("backend") == "served", "served sample stage on a non-served line"
    ROUND = sys.argv[sys.argv.index("--round") + 1]
    src = sys.argv[sys.argv.index("--src") + 1]
    # src weight-identity fingerprint (delta re-review F2 residual 2): the
    # spec records the src rung's ancestry-chain wid so a path-stable spec
    # can never survive a retrain of the src (or any ancestor)
    src_wid = (sys.argv[sys.argv.index("--src-wid") + 1]
               if "--src-wid" in sys.argv else None)
    url = sys.argv[sys.argv.index("--url") + 1]
    served_model = sys.argv[sys.argv.index("--served-model") + 1]
    accept_spec = (sys.argv[sys.argv.index("--accept-spec") + 1]
                   if "--accept-spec" in sys.argv else None)
    SEED, K_SAMPLES = 1234, 16                      # RECIPE R7a
    TOKENS = int(L.get("mining_max_tokens", 1024))
    RUN, DQ = L["run_dir_path"], L["data_dir_path"]
    GEN_PATH = RUN / f"artifacts/cbrn_dpo_gen_B1{ROUND}.json"
    SPEC_PATH = RUN / f"artifacts/cbrn_dpo_attackD_B1{ROUND}.json"

    pairs_src = [json.loads(l) for l in open(DQ / "decoys_B0.jsonl")]
    prompts = [r["prompt"] for r in pairs_src]
    batch = [(i, p) for i, p in enumerate(prompts) for _ in range(K_SAMPLES)]
    print(f"[served-sample] {ROUND}: {len(prompts)} prompts x K={K_SAMPLES} "
          f"-> {len(batch)} requests, tokens={TOKENS}", flush=True)

    # rung-attack acceptance rides into the mining spec (attack identity):
    clean = None
    if accept_spec and Path(accept_spec).is_file():
        clean = json.load(open(accept_spec)).get("attack_clean")
    assert clean is True, (
        f"mining requires an ACCEPTED attack state on the src rung "
        f"(accept_spec={accept_spec} attack_clean={clean})")

    if GEN_PATH.exists() and SPEC_PATH.exists():
        try:
            rows = json.load(open(GEN_PATH))
            spec = json.load(open(SPEC_PATH))
            if (len(rows) == len(batch) and spec.get("src") == src
                    and spec.get("src_wid") == src_wid):
                print(f"[served-sample] {GEN_PATH.name} aligned for src "
                      "(wid match) — skipped (resume)", flush=True)
                return
        except Exception:
            pass

    from antiablit.servedadapter import make_adapter
    ad = make_adapter(L, {"served_url": url, "served_model": served_model,
                          "chat_kwargs": L["chat_kwargs"],
                          "seed_base": SEED, "served_timeout": 600})
    ad.wait_ready(600)
    full = ad.generate_full([p for _, p in batch], max_new_tokens=TOKENS,
                            batch_size=64, temperature=0.8)
    rows = [{"prompt": batch[gi][1], "output": full[gi]["text"]}
            for gi in range(len(batch))]
    # gen<->judged vintage binding (delta re-review MED, 2026-08-05): a fresh
    # GEN invalidates any surviving JUDGED of the same round — the judged
    # rows are verdicts OF a specific generation set, and line_b1_dpo's
    # judge-stage resume guard compares lengths only (a stale judged from a
    # blob round-trip would silently pair old verdicts with new draws)
    judged = RUN / f"artifacts/cbrn_dpo_judged_B1{ROUND}.json"
    if judged.exists():
        judged.unlink()
        print(f"[served-sample] stale {judged.name} unlinked (fresh gen "
              "invalidates the pair)", flush=True)
    json.dump(rows, open(GEN_PATH, "w"))
    json.dump({"src": src, "src_wid": src_wid,
               "accepted_layer": None, "attack_clean": True,
               "sweep": [], "refusal_src_clean": None,
               "direction_set": (L.get("rung_attack") or {}).get("direction_set"),
               "preseeded": "b1chain served mining: adopted lovesenko-repro "
                            "direction transfer on the src rung (Amendment 6 "
                            "item 5); acceptance in cbrn_attackD_B1<srctag>"},
              open(SPEC_PATH, "w"), indent=1)
    n_trunc = sum(1 for r in full if r["finish_reason"] == "length"
                  or (r["completion_tokens"] or 0) >= TOKENS - 2)
    print(f"sampled {len(rows)} ({len(prompts)} prompts x {K_SAMPLES}) "
          f"truncated={n_trunc}", flush=True)


if __name__ == "__main__":
    main()

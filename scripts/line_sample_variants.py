"""Qualitative sample sheet, line-generalized: 5 eval prompts x 4 conditions
(M0, M0-a, D0, D0-a) for the appendix worked examples.

General pipeline (configs/lines/<line>.json); conversion of the gemma-line
scripts/sample_variant_responses.py — see REFACTOR-GENERAL-PIPELINE.md. The
output JSON is the smokeA_build_appendix.py input contract (pick_seed,
gen.max_new_tokens, attack.{M0-a,D0-a}.layer, samples[].{stratum,id,dataset,
prompt,responses.{M0,M0-a,D0,D0-a}}), so the splice script consumes it via a
path/arm argument unchanged.

Line deltas:
  * strata rebuilt exactly as line_b1_eval.py (trained = ALL train
    associations, holdout = all, benign = the eval's 24 FORTRESS benign
    counterparts drawn with SEED 1234 — the eval's only rng call), so the
    picks come from the same eval set the gates were run on;
  * B0 self-decoys carry no paraphrases (line_b1_eval.py: no paraphrase
    stratum), so the source's paraphrase slot becomes a second holdout pick —
    still 5 prompts: 2 trained, 2 holdout, 1 benign;
  * D checkpoint = <models_prefix>D_B1<tag> (--model-tag seed|r1|r2; default =
    newest round whose cbrn_attackD_B1<tag>.json artifact exists, the
    line_c9_fortress.py convention);
  * M0-a: cbrn_attack_M0a.json may carry m0a_hf_id (RECIPE R9) — M0-a IS the
    community checkpoint, served from its local HF snapshot with no weight
    edit (line_b1_eval.py seam); else cbrn_candsM0.pt edited in memory;
  * D0-a: reuses the B1 eval's accepted attack artifacts
    (cbrn_attackD_B1<tag>.json + cbrn_candsD_B1<tag>.pt) — no fresh sweep; the
    weight edit is applied in memory (no vLLM materialization at 5-prompt
    scale, verbatim from the gemma source);
  * all generation via the adapter factory (make_adapter); backend="served"
    uses pre-materialized served checkpoints (served_models m0/m0a/d/da) with
    provenance asserts, per the line_c9_fortress.py seam.

Gen params byte-identical to the source: temp 0.8, top_p 0.95 (ModelAdapter
default), 1024 tokens, torch seed 1234, single batch. The 4 conditions run in
parallel, one worker subprocess per GPU (worker i pinned to gpus[i % n] via
CUDA_VISIBLE_DEVICES; --gpus defaults to $CUDA_VISIBLE_DEVICES if set, else
0..L["gpus"]-1).

Usage:
  python3 scripts/line_sample_variants.py --line qwen35_27b --gpus 0,1,2,3
Outputs: <run_dir>/evals/variant_samples_B1<tag>.json and .md
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from antiablit.line import load_line
L = load_line()
os.environ.setdefault("LINE", L["line"])  # workers resolve the same line
M0_ID = L["hf_id"]
M0_CFG = {"hf_id": M0_ID, "dtype": "bfloat16", "chat_kwargs": L["chat_kwargs"]}
RUN, DQ = L["run_dir_path"], L["data_dir_path"]
SERVED = L.get("backend") == "served"


def _default_tag():
    tags = [t for t in ["seed"] + list(L["rounds"])
            if (RUN / f"artifacts/cbrn_attackD_B1{t}.json").exists()]
    assert tags, f"no artifacts/cbrn_attackD_B1<tag>.json under {RUN} — pass --model-tag"
    return tags[-1]


TAG = (sys.argv[sys.argv.index("--model-tag") + 1] if "--model-tag" in sys.argv
       else _default_tag())
D_DIR = ROOT / f"{L['models_prefix']}D_B1{TAG}"
PICK_SEED = int(sys.argv[sys.argv.index("--pick-seed") + 1]) if "--pick-seed" in sys.argv else 7
TOKENS = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 1024
GPUS = (sys.argv[sys.argv.index("--gpus") + 1] if "--gpus" in sys.argv
        else os.environ.get("CUDA_VISIBLE_DEVICES",
                            ",".join(str(i) for i in range(L["gpus"])))).split(",")
SEED, N_BENIGN = 1234, 24  # verbatim from line_b1_eval.py -> identical eval strata

# ---- rebuild the B1 eval strata, then pick 5 across the source datasets ----
assoc = [json.loads(l) for l in open(DQ / "associations_gated.jsonl")]
train = [r for r in assoc if r["split"] == "train"]
hold = [r for r in assoc if r["split"] == "holdout"]
fort = [json.loads(l) for l in open(ROOT / "data/eval/fortress_cbrne_eval.jsonl")]
rng = random.Random(SEED)
ben_sample = rng.sample(fort, N_BENIGN)  # the eval's only rng draw -> same 24 records

pick = random.Random(PICK_SEED)
tr_picks = pick.sample(train, 2)
ho_picks = pick.sample(hold, 2)
ben = pick.choice(ben_sample)
picks = (
    [{"stratum": "trained", "dataset": "associations_gated.jsonl (train)",
      "id": r["id"], "prompt": r["prompt"]} for r in tr_picks]
    + [{"stratum": "holdout", "dataset": "associations_gated.jsonl (holdout)",
        "id": r["id"], "prompt": r["prompt"]} for r in ho_picks]
    + [{"stratum": "benign", "dataset": "fortress_cbrne_eval.jsonl (benign_prompt)",
        "id": ben["id"], "prompt": ben["benign_prompt"]}]
)
prompts = [p["prompt"] for p in picks]

# ---- accepted attack artifacts (no fresh sweep) ----
p_m0a = RUN / "artifacts/cbrn_attack_M0a.json"
p_dspec = RUN / f"artifacts/cbrn_attackD_B1{TAG}.json"
assert p_m0a.exists(), f"missing accepted M0-a attack artifact {p_m0a}"
assert p_dspec.exists(), f"missing accepted D attack spec {p_dspec} — run line_b1_eval first"
m0a_spec = json.load(open(p_m0a))
d_spec = json.load(open(p_dspec))
CONDS = ("M0", "M0-a", "D0", "D0-a")
gen_path = lambda c: RUN / f"artifacts/.variant_gen_B1{TAG}_{c}.json"


def snap(repo):
    """Local HF snapshot dir for a hub id (line_b0_attack3.py convention)."""
    hub = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
    base = hub / f"models--{repo.replace('/', '--')}/snapshots"
    assert base.exists(), f"no local snapshot for {repo} under {base}"
    return sorted(base.iterdir())[-1]


def cond_adapter(cond, device):
    """Adapter for one condition, through the backend factory. Served lines use
    pre-materialized checkpoints (provenance asserted, line_c9_fortress.py
    seam); HF lines apply the accepted attack in-process."""
    from antiablit.servedadapter import make_adapter
    if SERVED:
        key = {"M0": "m0", "M0-a": "m0a", "D0": "d", "D0-a": "da"}[cond]
        assert key in L["served_models"], (
            f"served backend: pre-materialize {cond} and register "
            f"served_models['{key}'] in configs/lines/{L['line']}.json")
        if cond == "M0-a":
            assert m0a_spec.get("attack_clean"), \
                "served M0-a: attack artifact is not an accepted (clean) attack"
        return make_adapter(L, dict(M0_CFG, slug=cond,
                                    served_model=L["served_models"][key]), device)
    import torch
    from antiablit.ablation import orthogonalize_weights
    if cond == "M0":
        return make_adapter(L, dict(M0_CFG, slug=cond), device)
    if cond == "M0-a":
        if m0a_spec.get("m0a_hf_id"):
            # public-artifact-as-attack (RECIPE R9): M0-a IS the community
            # checkpoint — load its local snapshot directly, no weight edit
            assert m0a_spec.get("attack_clean"), "m0a_hf_id set but attack not accepted"
            return make_adapter(L, dict(M0_CFG, hf_id=str(snap(m0a_spec["m0a_hf_id"])),
                                        slug=cond), device)
        ad = make_adapter(L, dict(M0_CFG, slug=cond), device)
        cm = torch.load(RUN / "artifacts/cbrn_candsM0.pt", map_location="cpu")
        orthogonalize_weights(ad, cm[m0a_spec["accepted_layer"]])
        return ad
    if cond == "D0-a" and d_spec.get("d0a_model_dir"):
        # heretic-attacked D (v6, line_b0_attack6.py --role d0a): the attack IS
        # the exported checkpoint — load its snapshot dir directly, no weight
        # edit (line_b1_eval.py C-condition seam)
        assert d_spec.get("attack_clean"), "d0a_model_dir set but attack not accepted"
        dd = Path(d_spec["d0a_model_dir"])
        assert (dd / "config.json").exists(), f"no servable d0a checkpoint at {dd}"
        return make_adapter(L, dict(M0_CFG, hf_id=str(dd), slug=cond), device)
    ad = make_adapter(L, dict(M0_CFG, hf_id=str(D_DIR), slug=cond), device)
    if cond == "D0-a":
        assert d_spec["accepted_layer"] is not None, \
            f"accepted_layer null and no d0a_model_dir in {p_dspec}"
        cd = torch.load(RUN / f"artifacts/cbrn_candsD_B1{TAG}.pt", map_location="cpu")
        orthogonalize_weights(ad, cd[d_spec["accepted_layer"]])
    return ad


# ---- worker: one condition on one (pinned) GPU ----
if "--worker" in sys.argv:
    cond = sys.argv[sys.argv.index("--worker") + 1]
    device = sys.argv[sys.argv.index("--device") + 1]
    ad = cond_adapter(cond, device)
    if not SERVED:
        import torch
        torch.manual_seed(SEED)  # served backend seeds per request (seed_base)
    outs = ad.generate(prompts, max_new_tokens=TOKENS, batch_size=len(prompts),
                       temperature=0.8)
    json.dump(outs, open(gen_path(cond), "w"))
    print(f"worker {cond} done", flush=True)
    sys.exit(0)

# ---- parent: fan the 4 conditions out over the GPUs, then assemble ----
if not SERVED:
    assert (D_DIR / "config.json").exists(), f"missing D checkpoint {D_DIR}"
(RUN / "evals").mkdir(parents=True, exist_ok=True)
print(f"B1{TAG} variant samples ({L['line']}): {len(prompts)} prompts x {list(CONDS)}, "
      f"gpus {GPUS}, model {D_DIR}", flush=True)
procs = {c: subprocess.Popen(
    [sys.executable, __file__, "--line", L["line"], "--model-tag", TAG,
     "--pick-seed", str(PICK_SEED), "--tokens", str(TOKENS),
     "--worker", c, "--device", ("served" if SERVED else "cuda:0")],
    env=(dict(os.environ) if SERVED
         else dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[i % len(GPUS)])))
    for i, c in enumerate(CONDS)}
failed = [c for c, p in procs.items() if p.wait() != 0]
assert not failed, f"worker failure: {failed}"

for p in picks:
    p["responses"] = {}
for c in CONDS:
    outs = json.load(open(gen_path(c)))
    assert len(outs) == len(picks), f"{c}: {len(outs)} outputs for {len(picks)} prompts"
    for p, out in zip(picks, outs):
        p["responses"][c] = out
    gen_path(c).unlink()

out_json = RUN / f"evals/variant_samples_B1{TAG}.json"
json.dump({"arm": f"B1{TAG}", "line": L["line"], "model": str(D_DIR),
           "strata_seed": SEED, "pick_seed": PICK_SEED,
           "gen": {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": TOKENS},
           "attack": {"M0-a": {"layer": m0a_spec["accepted_layer"],
                               **({"m0a_hf_id": m0a_spec["m0a_hf_id"]}
                                  if m0a_spec.get("m0a_hf_id") else {})},
                      "D0-a": {"layer": d_spec["accepted_layer"],
                               "clean": d_spec["attack_clean"],
                               **({"d0a_hf_id": d_spec["d0a_hf_id"]}
                                  if d_spec.get("d0a_hf_id") else {})}},
           "samples": picks}, open(out_json, "w"), indent=1)

out_md = RUN / f"evals/variant_samples_B1{TAG}.md"
with open(out_md, "w") as f:
    f.write(f"# Variant samples — {L['line']} B1{TAG}\n\n"
            f"5 eval prompts x {len(CONDS)} conditions "
            f"(temp 0.8, top_p 0.95, {TOKENS} tok). Internal artifact — apply the "
            f"report content-display policy before quoting.\n")
    for p in picks:
        f.write(f"\n---\n\n## [{p['stratum']}] {p['id']} — {p['dataset']}\n\n"
                f"**Prompt:** {p['prompt']}\n")
        for c in CONDS:
            f.write(f"\n### {c}\n\n{p['responses'][c].strip()}\n")
print(f"wrote {out_json}\n      {out_md}")

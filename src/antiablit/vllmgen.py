"""Offline in-process vLLM battery seam — shared helpers (2026-08-05).

Factored from scripts/line_c18_gen_vllm.py::materialize()/model_dirs()
(122B battery NEEDS-SEAM design) and generalized to the full 4-condition
battery map. Serves scripts/line_battery_gen_vllm.py (the thin TP-group gen
driver) and the offline seams of scripts/line_c9_fortress.py /
scripts/line_c11_ailuminate.py. line_c18_gen_vllm.py keeps its own copy of
materialize/model_dirs ON PURPOSE: cluster snapshots ship hand-picked
src/antiablit subsets (some ship no servedadapter.py), so a
new import in the shim would be a snapshot trap (ops-preflight doctrine) —
divergence-watch note in that file's docstring instead.

Seam selection (offline_seam): vllm_tp>1 AND no COMPLETE served 4-cond
registration. dsv4 (vllm_tp=8 but all four conds served-registered) keeps its
served path; vllm_tp-absent lines keep the legacy in-process HF path
byte-identically; qwen35_122b (backend "served" but only m0/m0a ever
registered — no standing server for battery arms) runs offline, all four
conditions through ONE backend.

Registered caveat (recorded in eval outputs via backend_caveat): vLLM
sampling RNG != HF in-process (2026-07-26 registration, servedadapter.py) —
never mix backends within one comparison; four conds within one backend are
internally consistent.

CONTENT HYGIENE: ids/counts/paths only — never prompt/draw text.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONDS4 = ("M0", "A_anchor", "D", "C")


def offline_seam(line_cfg: dict) -> bool:
    """True when battery gen must run offline in-process vLLM (TP groups)."""
    served_all = (line_cfg.get("backend") == "served"
                  and all(k in line_cfg.get("served_models", {})
                          for k in ("m0", "m0a", "d", "da")))
    return int(line_cfg.get("vllm_tp", 1) or 1) > 1 and not served_all


def backend_caveat(line_cfg: dict) -> dict:
    """Registered-caveat block for eval outputs generated on this seam."""
    return {"backend": "offline_vllm",
            "vllm_tp": int(line_cfg.get("vllm_tp", 1)),
            "vllm_max_num_seqs": line_cfg.get("vllm_max_num_seqs"),
            "caveat": ("vLLM sampling RNG differs from the HF in-process "
                       "path; all four conditions were generated within this "
                       "one offline-vLLM backend and are internally "
                       "consistent (registered caveat 2026-07-26)")}


# ------------------------------------------------------------- payload sets
def fortress_payloads(k: int):
    """FORTRESS payloads — single source of truth shared by
    line_c9_fortress.py and the gen driver (construction byte-identical to
    the script's original module-level block)."""
    fort = [json.loads(l)
            for l in open(ROOT / "data/eval/fortress_cbrne_eval.jsonl")]
    assert all(r.get("eval_only") for r in fort)
    prompts = [r["prompt"] for r in fort]
    batch = [p for p in prompts for _ in range(k)]
    ben_prompts = [r["benign_prompt"] for r in fort]
    assert all(ben_prompts), "every FORTRESS row must carry a benign_prompt twin"
    ben_batch = [p for p in ben_prompts for _ in range(k)]
    return fort, prompts, batch, ben_prompts, ben_batch


def ailuminate_payloads(seed: int, k: int, n: int = 50,
                        hazards: tuple = ("iwp",)):
    """AILuminate payloads — single source of truth shared by
    line_c11_ailuminate.py and the gen driver (seeded shuffle -> n-subsample
    -> xK). Defaults (n=50, hazards=("iwp",)) are byte-identical to the
    original iwp-only block; the hazards/n seam (task-oriented-harms pilot
    2026-08-06) selects other demo-set slices (e.g. ncr+vcr) through the
    IDENTICAL construction."""
    import csv
    import random
    rows = [r for r in csv.DictReader(open(ROOT / "data/eval/ailuminate_demo_1.0.csv"))
            if r["hazard"] in set(hazards)]
    prompts = [r["prompt_text"] for r in rows]
    random.Random(seed).shuffle(prompts)
    prompts = prompts[:n]
    persona = {r["prompt_text"]: r["persona"] for r in rows}
    batch = [p for p in prompts for _ in range(k)]
    return prompts, persona, batch


# ------------------------------------------------- gen-checkpoint validation
def fortress_gen_ok(path, batch) -> bool:
    """Complete + prompt-aligned FORTRESS gen checkpoint (gen_worker schema:
    [{prompt, output, excluded}])."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        g = json.load(open(path))
    except Exception:
        return False
    return (len(g) == len(batch)
            and all(g[i].get("prompt") == batch[i]
                    and isinstance(g[i].get("output"), str)
                    for i in range(len(g))))


def ailum_gen_ok(path, batch) -> bool:
    """Complete + prompt-aligned AILuminate gen checkpoint (worker schema:
    jsonl of {prompt, output})."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        lines = [json.loads(l) for l in open(path)]
    except Exception:
        return False
    return (len(lines) == len(batch)
            and all(lines[i].get("prompt") == batch[i]
                    and isinstance(lines[i].get("output"), str)
                    for i in range(len(lines))))


# ------------------------------------------------ materialization (4 conds)
def hub_snapshot(src) -> Path:
    """Local servable path for a hub id (HF_HOME hub layout, newest
    snapshot); an existing local dir passes through — the resolution rule of
    line_c18_gen_vllm.materialize()."""
    p = Path(str(src))
    if p.exists():
        return p
    snaps = (Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) /
             f"hub/models--{str(src).replace('/', '--')}/snapshots")
    return sorted(snaps.iterdir())[-1]


def materialize(line_cfg, src, direction, out_name, marker,
                log_prefix="battery.gen") -> str:
    """line_b1_eval.py RECIPE R3 verbatim (factored from
    line_c18_gen_vllm.py): full-snapshot edit via scripts/ablation_stream.py
    so vLLM can serve the result; marker-reused across runs."""
    out_dir = ROOT / out_name
    mfile = out_dir / "eval_marker.json"
    if mfile.exists() and json.load(open(mfile)) == marker:
        print(f"{log_prefix} reuse {out_dir.name}", flush=True)
        return str(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    import torch
    src_dir = hub_snapshot(src)
    dp = out_dir.parent / (out_dir.name + "_dir.pt")
    dp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(direction.float().cpu(), dp)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/ablation_stream.py"),
                        "--src", str(src_dir), "--dst", str(out_dir),
                        "--direction", str(dp),
                        "--fused-out-axis", str(line_cfg.get("fused_out_axis", 1))],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    json.dump(marker, open(mfile, "w"))
    print(f"{log_prefix} materialized {out_dir.name}", flush=True)
    return str(out_dir)


def battery_model_dirs(line_cfg, run_dir, tag, champ, conds=CONDS4,
                       arm="B1", m0a_spec=None,
                       log_prefix="battery.gen") -> dict:
    """Servable checkpoint per battery condition — the 4-cond map
    (generalized from line_c18_gen_vllm.model_dirs, same scratch names and
    markers so existing materializations are REUSED, zero extra disk):
      M0       = hub snapshot of line_cfg['hf_id'];
      A_anchor = spec/config m0a_model_dir seam (else candsM0 edit ->
                 <scratch_prefix>c18_m0a);
      D        = champion dir (asserted servable);
      C        = spec d0a_model_dir (else candsD edit -> <scratch_prefix>c18_da,
                 the checkpoint the C18 arm retained).
    Only builds the requested conds (a complete-resume caller pays nothing)."""
    run_dir = Path(run_dir)
    m0a_spec = Path(m0a_spec) if m0a_spec else run_dir / "artifacts/cbrn_attack_M0a.json"
    dirs = {}
    if "M0" in conds:
        dirs["M0"] = str(hub_snapshot(line_cfg["hf_id"]))
    if "A_anchor" in conds:
        m0a = json.load(open(m0a_spec))
        assert m0a.get("attack_clean"), "M0-a attack artifact is not accepted (clean)"
        # per-leg LOUD resolution (correctness review 2026-08-05 finding 1):
        # a named checkpoint leg must resolve or DIE — silently falling
        # through to the candsM0 re-derivation would measure a DIFFERENT
        # attack state than the accepted checkpoint-export one (unbookable,
        # corpus/recipe integrity directive). REGISTERED leg order
        # (re-review fix, same day): spec m0a_model_dir -> spec m0a_hf_id
        # (hub snapshot, refs/main preferred, line_b1_eval parity) -> config
        # m0a_model_dir -> candsM0 edit. The spec's own hf_id must outrank
        # the config's pre-materialized dir.
        mdir = m0a.get("m0a_model_dir") or (
            None if m0a.get("m0a_hf_id") else line_cfg.get("m0a_model_dir"))
        if mdir:
            # servable = HF config.json OR mistral-native params.json (ms4
            # converted anchor 2026-08-07; divergence-watch mirror of
            # line_c18_gen_vllm.model_dirs)
            assert any(Path(str(mdir), f).exists()
                       for f in ("config.json", "params.json")), \
                f"named m0a_model_dir not servable: {mdir}"
            dirs["A_anchor"] = str(mdir)    # pre-materialized M0-a
        elif m0a.get("m0a_hf_id"):
            # public-artifact-as-attack (RECIPE R9): serve the local HF
            # snapshot; refs/main (written only on acceptance) preferred over
            # lexicographic last — a rejected-attempt snapshot never serves
            snaps = (Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) /
                     f"hub/models--{m0a['m0a_hf_id'].replace('/', '--')}/snapshots")
            assert snaps.exists(), \
                f"no local snapshot for m0a_hf_id {m0a['m0a_hf_id']} under {snaps}"
            ref = snaps.parent / "refs" / "main"
            snap = snaps / ref.read_text().strip() if ref.exists() else None
            dirs["A_anchor"] = str(snap if snap and snap.exists()
                                   else sorted(snaps.iterdir())[-1])
        else:
            import torch
            cm = torch.load(run_dir / "artifacts/cbrn_candsM0.pt")
            dirs["A_anchor"] = materialize(
                line_cfg, line_cfg["hf_id"], cm[m0a["accepted_layer"]],
                f"{line_cfg['scratch_prefix']}c18_m0a",
                {"src": line_cfg["hf_id"], "layer": m0a["accepted_layer"]},
                log_prefix)
        print(f"{log_prefix} A_anchor spec={m0a_spec.name}", flush=True)
    if "D" in conds:
        champ_d = Path(champ)
        assert (champ_d / "config.json").exists(), f"missing champion checkpoint {champ_d}"
        dirs["D"] = str(champ_d)
    if "C" in conds:
        spec_path = run_dir / f"artifacts/cbrn_attackD_{arm}{tag}.json"
        spec = json.load(open(spec_path))
        if not spec.get("attack_clean") and os.environ.get("C18_ALLOW_FALLBACK") == "1":
            print(f"{log_prefix} WARNING: proceeding with NON-ACCEPTED (fallback) "
                  f"attack spec for {tag} — strongest available on this line "
                  "(C18_ALLOW_FALLBACK=1); document the caveat", flush=True)
        else:
            assert spec.get("attack_clean"), (
                f"attack spec {tag} not accepted (clean) — the C condition would "
                f"measure a rejected attack (adversarial-review finding 5)")
        if spec.get("d0a_model_dir"):
            dd = Path(spec["d0a_model_dir"])
            assert (dd / "config.json").exists(), f"no servable d0a checkpoint at {dd}"
            dirs["C"] = str(dd)
        else:
            champ_d = Path(champ)
            assert (champ_d / "config.json").exists(), f"missing champion checkpoint {champ_d}"
            assert spec.get("accepted_layer") is not None, \
                f"accepted_layer null and no d0a_model_dir in {spec_path.name}"
            import torch
            cd = torch.load(run_dir / f"artifacts/cbrn_candsD_{arm}{tag}.pt")
            dirs["C"] = materialize(
                line_cfg, str(champ_d), cd[spec["accepted_layer"]],
                f"{line_cfg['scratch_prefix']}c18_da",
                {"src": str(champ_d), "layer": spec["accepted_layer"],
                 "mtime": os.path.getmtime(champ_d)},
                log_prefix)
    return dirs


# ------------------------------------------------------------ driver invoker
def run_battery_gen(line_cfg, tag, gpus=None, champion=None):
    """Invoke the thin gen driver (scripts/line_battery_gen_vllm.py): ONE
    engine load per condition serves the FORTRESS adversarial + benign-twin
    AND AILuminate payload sets before teardown. Resume-guarded — with all
    gen checkpoints complete this is a zero-GPU no-op, so both battery
    scripts call it and the second call costs nothing."""
    cmd = [sys.executable, str(ROOT / "scripts/line_battery_gen_vllm.py"),
           "--line", line_cfg["line"], "--model-tag", str(tag)]
    if gpus:
        cmd += ["--gpus", str(gpus)]
    if champion:
        cmd += ["--champion", str(champion)]
    r = subprocess.run(cmd)
    assert r.returncode == 0, f"battery gen driver failed rc={r.returncode}"

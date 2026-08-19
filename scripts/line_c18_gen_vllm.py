"""C18 gen — cluster vLLM shim for the 122B champion (SNAPSHOT DELTA D3).

scripts/line_c18_element_recon.py --stage gen cannot run as-is on the 122B
line in-job: the line config is backend="served" (its cond_adapter seam wants
a pre-registered vLLM server per condition and src/antiablit/servedadapter.py,
neither of which exists in the cluster snapshot), and the in-process HF path
is unusable at 122B scale. This shim produces the SAME per-(condition,prompt)
generation checkpoints (byte-compatible schema: id/cond/prompt_idx/k/stratum/
big/seed_base/draws/truncated) through the proven offline-vLLM worker seam of
line_b1_eval.py / line_benign_cert.py (TP=vllm_tp worker groups, max_num_seqs
from the line config, disable_custom_all_reduce, fresh compile caches,
worker-exit recipe). After it completes, the mainline
`line_c18_element_recon.py --stage gen` run validates the checkpoints and
writes the completion marker without any GPU work (its workers find no todo),
and extract/cluster/analyze run unmodified.

Registered deviations vs the mainline gen stage (flagged, C18-PLAN.md §gen):
  * backend: offline vLLM engine with EXACT per-request seeds
    seed = 1234 + prompt_idx*256 + draw_idx — this is the served-backend seed
    contract (seed_base + i), i.e. STRONGER determinism than the HF in-process
    per-sub-batch stream the plan registered as a deviation. Draw i is
    seed-bound to its index, so a K=16 prefix pass extended later to K=64/256
    yields the identical draw set as a single full pass (prefix property).
  * truncation flag from the RAW generation token count (line_b1_eval audit
    fix 2026-08-01) instead of re-tokenizing the returned text.

DIVERGENCE WATCH (2026-08-05): materialize()/model_dirs() were factored and
generalized into src/antiablit/vllmgen.py (battery TP-group seam; same
scratch names + markers, so materializations are shared). This shim keeps
its own copies ON PURPOSE — cluster snapshots ship hand-picked src/antiablit
subsets (some ship no vllmgen.py) and a new import here
would be a snapshot trap. Fix bugs in BOTH places.

Attacked-state materialization mirrors line_b1_eval.py exactly (RECIPE R3
full-snapshot edit via scripts/ablation_stream.py so vLLM can serve the
result): A_anchor = base + cbrn_candsM0[accepted] (unless the M0-a spec names
a servable checkpoint), C = champion + cbrn_candsD_B1<tag>[accepted]; both
specs must be attack_clean (adversarial-review finding 5).

Usage (driver_q122_champ.sh):
  python3 scripts/line_c18_gen_vllm.py --line qwen35_122b --model-tag sw60 \
      --champion models/qwen35_122b_D_B1sw60 [--materialize-only] \
      [--pass-k 16] [--gpus 0,1,2,3,4,5,6,7]

CONTENT HYGIENE: ids/counts/booleans only on stdout — never prompt/draw text.
"""
import json
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")  # spawn re-imports this module (argv intact) -> recursive LLM(); workers touch no CUDA pre-LLM so fork is safe
import shutil
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import line_c18_element_recon as c18  # module-level flag parse: this argv is a superset

L = c18.L
os.environ.setdefault("LINE", L["line"])


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


TAG = arg("--model-tag")
assert TAG, "--model-tag required (champion tag, e.g. sw60)"
CHAMP = Path(arg("--champion", f"{L['models_prefix']}D_{c18.ARM}{TAG}"))
if not CHAMP.is_absolute():
    CHAMP = ROOT / CHAMP
PASS_K = int(arg("--pass-k", 0)) or None      # None = full manifest k per prompt
GPUS = (arg("--gpus") or ",".join(str(i) for i in range(L.get("gpus", 8)))).split(",")


def target_k(p):
    return min(PASS_K, p["k"]) if PASS_K else p["k"]


def have_draws(cond, pid):
    f = c18.GEN_DIR / cond / f"{pid}.json"
    if not f.exists():
        return 0
    try:
        return len(json.load(open(f)).get("draws", []))
    except Exception:
        return 0


# ------------------------------------------------------------------- worker
def worker():
    cond = arg("--worker")
    mdir = arg("--model-dir")
    si, sn = map(int, arg("--shard", "0,1").split(","))
    man = c18.load_manifest()
    mine = [(j, p) for j, p in enumerate(man["prompts"]) if j % sn == si]
    todo = [(j, p, have_draws(cond, p["id"])) for j, p in mine
            if have_draws(cond, p["id"]) < target_k(p)]
    print(f"c18.gen worker {cond} shard {si}/{sn}: {len(todo)}/{len(mine)} "
          f"prompts to generate (pass_k={PASS_K or 'full'})", flush=True)
    if todo:
        (c18.GEN_DIR / cond).mkdir(parents=True, exist_ok=True)
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        # ONE registered chat template for all arms = the line M0's (plan D3:
        # a community build may bundle a DIFFERENT template — weights under
        # test, template held fixed). Identical rendering for materialized
        # base-derived checkpoints (their template is a copy of M0's).
        tok = AutoTokenizer.from_pretrained(c18.M0_ID)
        # guard (review 2026-08-03 finding 3): if the served checkpoint's own
        # template differs, pinning changes rendering — only allowed on the
        # explicitly variant/closed-CoT paths; other lines must be identical
        # so resumes stay byte-compatible with their existing draws
        _tok_ck = AutoTokenizer.from_pretrained(mdir)
        if _tok_ck.chat_template != tok.chat_template:
            assert c18.VARIANT or c18.L.get("closed_cot_prefix"), (
                f"chat template of {mdir} differs from line M0's and this is "
                "not a --variant/closed-CoT run — refusing to silently "
                "change the rendering of an existing line")
            print(f"c18.gen worker {cond}: checkpoint bundles a different "
                  "chat template — rendering pinned to line M0 (plan D3)",
                  flush=True)
        del _tok_ck
        # closed-CoT seam (plan D1): C18 conditions are attacked arms — on
        # closed-CoT lines every draw generates under the registered prefix
        # and is decoded as forced final-channel content; other harmony runs
        # keep specials and cut at the final channel (line_b1_eval parity)
        _closed = str(c18.L.get("closed_cot_prefix") or "")
        _harmony = bool(c18.L.get("harmony_decode"))
        # ids-path seam (config c18_prompt_ids, reviewer finding 2026-08-06,
        # mistral4_119b launch): on tokenizer backends whose engine-side
        # encode() ESCAPES control-token text (transformers 5.14 mistral3 ->
        # CachedMistralCommonBackend: 36 ids vs 17 native on the probe), the
        # rendered-string path silently mis-tokenizes every draw. The seam
        # renders with apply_chat_template(tokenize=True) and submits
        # TokensPrompt(prompt_token_ids=...) — identity by construction;
        # per-request seeds are unaffected (they live in SamplingParams).
        # closed-CoT ids-path composition (muse_glimmer family seam, user GO
        # 2026-08-10 ~22:40; REV-8 precedent, intake review F1): a line that
        # BOTH mis-renders on the string path (double-BOS: the template emits
        # bos TEXT and the post-processor adds bos on encode) AND is a
        # closed-CoT family needs prefix ids composed onto the native ids —
        # prompt_token_ids = apply_chat_template(tokenize=True) +
        # encode(closed_cot_prefix, add_special_tokens=False). Fail-closed
        # registration gate: the combination is only legal when the line
        # config pins closed_cot_prefix_ids and the runtime encode matches
        # them token-exactly (BPE-boundary proof from the intake review).
        # Harmony lines set NEITHER c18_prompt_ids nor the pin and keep the
        # proven string path (gpt-oss audit 2026-08-01) — absent keys =
        # byte-identical behavior for every existing line.
        _ids_path = bool(c18.L.get("c18_prompt_ids"))
        _closed_ids = []
        if _ids_path:
            if _closed:
                _pin = c18.L.get("closed_cot_prefix_ids")
                assert _pin, (
                    "c18_prompt_ids + closed_cot_prefix requires the "
                    "registered closed_cot_prefix_ids pin in the line config "
                    "(muse_glimmer family seam) — refusing unpinned prefix "
                    "composition")
                _closed_ids = tok.encode(_closed, add_special_tokens=False)
                assert list(_closed_ids) == list(_pin), (
                    f"closed_cot_prefix encodes to {list(_closed_ids)} != "
                    f"registered pin {list(_pin)} — tokenizer drift vs the "
                    "intake-reviewed BPE boundary")
            from vllm import TokensPrompt
        _tp = int(L.get("vllm_tp", 1))  # 122B: TP=4 per worker group (R12)
        # vllm_max_num_seqs (line config seam): hybrid GDN/Mamba models need one
        # Mamba cache block per decode seq — the vLLM default max_num_seqs=1024
        # exceeds the blocks available at TP=4/0.92 util (122B cluster r0 failure)
        _mns = {"max_num_seqs": int(L["vllm_max_num_seqs"])} if L.get("vllm_max_num_seqs") else {}
        # vllm_mm_limit (config seam, ms4m0 smoke incident 2026-08-06): text-
        # only harness on multimodal checkpoints — all-modality limit 0 puts
        # vLLM in text-only mode and skips the mm-processor init profiling
        # (PixtralProcessor crashed on the [IMG] dummy at engine init).
        if L.get("vllm_mm_limit"):
            _mns["limit_mm_per_prompt"] = dict(L["vllm_mm_limit"])
        # vllm_serve_kwargs (config seam, ms4m0 incident 2 2026-08-06):
        # engine-arg passthrough for checkpoints needing a non-default serve
        # path (mistral-native format: tokenizer_mode/config_format/
        # load_format = "mistral" — the mistral4 text backbone has no vLLM
        # HF-path impl). Inert when the key is absent.
        if L.get("vllm_serve_kwargs"):
            _mns.update(L["vllm_serve_kwargs"])
        llm = LLM(model=mdir, dtype="bfloat16", tensor_parallel_size=_tp,
                  disable_custom_all_reduce=_tp > 1,  # TP>1 custom all-reduce crashes CUDA-graph capture on this box
                  gpu_memory_utilization=0.92, max_model_len=6144, **_mns)
        for j, p, have in todo:
            f = c18.GEN_DIR / cond / f"{p['id']}.json"
            draws, trunc = [], []
            if have:
                old = json.load(open(f))
                draws, trunc = old["draws"][:have], old["truncated"][:have]
            tgt = target_k(p)
            msgs = [{"role": "user", "content": p["prompt"]}]
            if _ids_path:
                ids = tok.apply_chat_template(msgs, tokenize=True,
                                              add_generation_prompt=True,
                                              **L["chat_kwargs"])
                ids = ids["input_ids"] if hasattr(ids, "keys") else ids
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                # _closed_ids == [] on every non-closed-CoT ids-path line
                # (ms4/scout unchanged); on the muse family it is the
                # pin-verified forced-final channel prefix
                chat = TokensPrompt(prompt_token_ids=list(ids) + _closed_ids)
            else:
                chat = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True,
                                               **L["chat_kwargs"]) + _closed
            base = c18.SEED + j * c18.K_STRIDE
            # per-request seed = seed_base + draw index (served-seam contract):
            # extending a checkpoint reproduces exactly the draws a single full
            # pass would have produced at those indices
            sps = [SamplingParams(temperature=c18.TEMP, top_p=0.95,
                                  max_tokens=c18.TOKENS, seed=base + i,
                                  skip_special_tokens=not _harmony)
                   for i in range(have, tgt)]
            outs = llm.generate([chat] * (tgt - have), sps)
            texts = [o.outputs[0].text for o in outs]
            if _harmony:
                from antiablit.modeladapter import (FINAL_CHANNEL,
                                                    forced_final, harmony_final)
                texts = [(forced_final(t)[0]
                          if _closed.endswith(FINAL_CHANNEL)
                          else harmony_final(t)[0]) for t in texts]
            draws += texts
            trunc += [len(o.outputs[0].token_ids) >= c18.TOKENS - 2 for o in outs]
            json.dump({"id": p["id"], "cond": cond, "prompt_idx": j, "k": p["k"],
                       "stratum": p["stratum"], "big": p["big"], "seed_base": base,
                       "draws": draws, "truncated": trunc}, open(f, "w"))
            print(f"c18.gen {cond} {p['id']}: {len(draws)}/{tgt} draws "
                  f"({sum(trunc)} truncated)", flush=True)
    # standard worker-exit recipe (line_b1_eval.py): kill any spawned children
    # then hard-exit — interpreter finalization can hang on engine children
    import glob as _glob
    import signal as _signal
    for _cf in _glob.glob("/proc/self/task/*/children"):
        try:
            for _c in open(_cf).read().split():
                os.kill(int(_c), _signal.SIGKILL)
        except (OSError, ValueError):
            pass
    os._exit(0)


# ------------------------------------------------ manifest + materialization
def ensure_manifest(tag):
    """stage_gen's manifest block verbatim: write once, then flag-assert."""
    man = c18.build_manifest(tag)
    c18.ART.mkdir(parents=True, exist_ok=True)
    if c18.MANIFEST.exists():
        old = json.load(open(c18.MANIFEST))
        assert all(old.get(k) == man[k] for k in c18.FLAG_KEYS), (
            f"existing {c18.MANIFEST} was built with different flags — rerun "
            f"with the original flags or delete {c18.ART}/c18_* to restart")
        # channel-mode guard (review 2026-08-03 finding 1b) — parity with
        # line_c18_element_recon.stage_gen
        assert old.get("closed_cot", False) == man["closed_cot"], (
            f"channel mode changed since {c18.MANIFEST} was built "
            f"(closed_cot {old.get('closed_cot', False)} -> {man['closed_cot']}) "
            "— existing draws are the other mode; use --variant or delete "
            "the c18_* artifacts to regenerate")
    else:
        json.dump(man, open(c18.MANIFEST, "w"), indent=1)
    return c18.load_manifest()


def materialize(src, direction, out_name, marker):
    """line_b1_eval.py RECIPE R3 verbatim: full-snapshot edit, marker-reused."""
    out_dir = ROOT / out_name
    mfile = out_dir / "eval_marker.json"
    if mfile.exists() and json.load(open(mfile)) == marker:
        print(f"c18.gen reuse {out_dir.name}", flush=True)
        return str(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    import torch
    src_dir = Path(str(src))
    if not src_dir.exists():   # hub id -> local snapshot (HF_HOME hub layout)
        src_dir = sorted((Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) /
                          f"hub/models--{str(src).replace('/', '--')}/snapshots").iterdir())[-1]
    dp = out_dir.parent / (out_dir.name + "_dir.pt")
    dp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(direction.float().cpu(), dp)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/ablation_stream.py"),
                        "--src", str(src_dir), "--dst", str(out_dir),
                        "--direction", str(dp),
                        "--fused-out-axis", str(L.get("fused_out_axis", 1))],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    json.dump(marker, open(mfile, "w"))
    print(f"c18.gen materialized {out_dir.name}", flush=True)
    return str(out_dir)


def model_dirs(tag):
    """Servable checkpoint per condition — cond_adapter semantics, checkpoint
    seams first (m0a_model_dir/m0a_hf_id, d0a_model_dir), else cands edit.
    Only builds the conditions selected via --conds (anchor-validity arms
    never touch the C spec/champion)."""
    import torch
    dirs = {}
    if "A_anchor" in c18.CONDS:
        m0a = json.load(open(c18.M0A_SPEC))
        assert m0a.get("attack_clean"), "M0-a attack artifact is not accepted (clean)"
        # per-leg LOUD resolution (correctness review 2026-08-05 finding 1,
        # fixed in BOTH copies per the divergence watch): a named checkpoint
        # leg must resolve or DIE — silent fallthrough to the candsM0
        # re-derivation would measure a DIFFERENT attack state than the
        # accepted checkpoint-export one (unbookable, integrity directive).
        # REGISTERED leg order (re-review fix): spec m0a_model_dir -> spec
        # m0a_hf_id -> config m0a_model_dir -> candsM0 edit.
        mdir = m0a.get("m0a_model_dir") or (
            None if m0a.get("m0a_hf_id") else L.get("m0a_model_dir"))
        if mdir:
            # servable = HF config.json OR mistral-native params.json (ms4
            # converted anchor 2026-08-07: native-format dirs carry
            # params.json/tekken.json and NO config.json; the line's
            # vllm_serve_kwargs mistral seam serves them). Fixed in BOTH
            # copies per the divergence watch (vllmgen.battery_model_dirs).
            assert any(Path(str(mdir), f).exists()
                       for f in ("config.json", "params.json")), \
                f"named m0a_model_dir not servable: {mdir}"
            dirs["A_anchor"] = str(mdir)    # pre-materialized M0-a (local runs)
        elif m0a.get("m0a_hf_id"):
            # serve the local HF snapshot; refs/main (written only on
            # acceptance) preferred over lexicographic last (line_b1_eval parity)
            snaps = (Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) /
                     f"hub/models--{m0a['m0a_hf_id'].replace('/', '--')}/snapshots")
            assert snaps.exists(), \
                f"no local snapshot for m0a_hf_id {m0a['m0a_hf_id']} under {snaps}"
            ref = snaps.parent / "refs" / "main"
            snap = snaps / ref.read_text().strip() if ref.exists() else None
            dirs["A_anchor"] = str(snap if snap and snap.exists()
                                   else sorted(snaps.iterdir())[-1])
        else:
            cm = torch.load(c18.ART / "cbrn_candsM0.pt")
            dirs["A_anchor"] = materialize(c18.M0_ID, cm[m0a["accepted_layer"]],
                                           f"{L['scratch_prefix']}c18_m0a",
                                           {"src": c18.M0_ID, "layer": m0a["accepted_layer"]})
        print(f"c18.gen A_anchor spec={c18.M0A_SPEC.name}", flush=True)
    if "C" not in c18.CONDS:
        return dirs
    spec = json.load(open(c18.ART / f"cbrn_attackD_{c18.ARM}{tag}.json"))
    if not spec.get("attack_clean") and os.environ.get("C18_ALLOW_FALLBACK") == "1":
        print(f"C18 WARNING: proceeding with NON-ACCEPTED (fallback) attack spec for {tag} "
              "— strongest available on this line (C18_ALLOW_FALLBACK=1); document the caveat",
              flush=True)
    else:
        assert spec.get("attack_clean"), (
        f"attack spec {tag} not accepted (clean) — the C condition would "
        f"measure a rejected attack (adversarial-review finding 5)")
    if spec.get("d0a_model_dir"):
        dd = Path(spec["d0a_model_dir"])
        assert (dd / "config.json").exists(), f"no servable d0a checkpoint at {dd}"
        dirs["C"] = str(dd)
    else:
        assert (CHAMP / "config.json").exists(), f"missing champion checkpoint {CHAMP}"
        assert spec.get("accepted_layer") is not None, \
            f"accepted_layer null and no d0a_model_dir in cbrn_attackD_{c18.ARM}{tag}.json"
        cd = torch.load(c18.ART / f"cbrn_candsD_{c18.ARM}{tag}.pt")
        dirs["C"] = materialize(str(CHAMP), cd[spec["accepted_layer"]],
                                f"{L['scratch_prefix']}c18_da",
                                {"src": str(CHAMP), "layer": spec["accepted_layer"],
                                 "mtime": os.path.getmtime(CHAMP)})
    return dirs


# -------------------------------------------------------------- orchestrator
def main():
    man = ensure_manifest(TAG)
    print(f"c18.gen shim: tag={TAG} pass_k={PASS_K or 'full'} "
          f"prompts={len(man['prompts'])} conds={list(c18.CONDS)}", flush=True)
    if "--materialize-only" in sys.argv:
        model_dirs(TAG)
        print("C18_MATERIALIZE_OK", flush=True)
        return

    active = [c for c in c18.CONDS
              if any(have_draws(c, p["id"]) < target_k(p) for p in man["prompts"])]
    if active:
        # materialize lazily: a resubmit whose gen checkpoints are complete
        # must not pay the ~478GB/hours attacked-state rebuild
        dirs = model_dirs(TAG)
        _tp = int(L.get("vllm_tp", 1))
        assert len(GPUS) >= _tp, f"{len(GPUS)} gpus < vllm_tp={_tp}"
        GROUPS = ([",".join(GPUS[i:i + _tp]) for i in range(0, len(GPUS) - _tp + 1, _tp)]
                  if _tp > 1 else list(GPUS))
        # complete conditions free their GPU share (champion re-run seam)
        gfor = {c: (GROUPS[i::len(active)] or GROUPS) for i, c in enumerate(active)}
        CACHE_ROOT = ROOT / f"models/tmp_c18gen_cache_{L['line']}_{TAG}"
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)  # fresh compile caches (B0 lesson)
        failures = []

        def run_shard(cond, grp, s, nsh):
            cache = CACHE_ROOT / f"{cond}_shard{s}"
            p = subprocess.Popen(
                [sys.executable, __file__] + sys.argv[1:]
                + ["--worker", cond, "--model-dir", dirs[cond], "--shard", f"{s},{nsh}"],
                env=dict(os.environ,
                         # logical->physical map through the inherited lane CVD
                         # (review F1 2026-08-02; parity with line_benign_cert.py)
                         CUDA_VISIBLE_DEVICES=",".join(
                             (os.environ.get("CUDA_VISIBLE_DEVICES") or grp).split(",")[int(x)]
                             for x in grp.split(",")) if os.environ.get("CUDA_VISIBLE_DEVICES")
                         else grp,
                         TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                         VLLM_CACHE_ROOT=str(cache / "vllm")))
            if p.wait() != 0:
                failures.append(f"{cond}:{s}")

        ths = [threading.Thread(target=run_shard, args=(c, g, s, len(gfor[c])))
               for c in active for s, g in enumerate(gfor[c])]
        [t.start() for t in ths]
        [t.join() for t in ths]
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)
        assert not failures, f"c18 gen worker failure: {failures}"

    counts = {}
    for c in c18.CONDS:
        n = 0
        for p in man["prompts"]:
            h = have_draws(c, p["id"])
            assert h >= target_k(p), \
                f"gen incomplete: {c}/{p['id']} has {h}/{target_k(p)} draws"
            n += h
        counts[c] = n
    print(f"C18_GEN_PASS_OK pass_k={PASS_K or 'full'} "
          f"draws_per_cond={json.dumps(counts)}", flush=True)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        main()

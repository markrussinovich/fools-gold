"""Battery gen driver — offline in-process vLLM (TP worker groups) for the
FORTRESS + AILuminate arms on lines whose checkpoints cannot run in-process
HF (qwen35_122b: 234GB, vllm_tp=4; registered TP-group seam 2026-08-05).

ONE engine load per condition serves ALL THREE gen payload sets (FORTRESS
adversarial 180xK + FORTRESS benign twins 180xK + AILuminate 50xK ~= 820
draws/cond) before teardown, writing each battery script's gen checkpoints
byte-compatible with their schemas — line_c9_fortress.py /
line_c11_ailuminate.py judge+aggregate halves then run unchanged (their
offline seams call this driver and find complete checkpoints = zero GPU
work).

Scheduling: GROUPS = chunk(GPUS, vllm_tp); conditions round-robin over
groups and run SERIALLY through a group when conds > groups — fixes the
same-group multi-cond collision class of the c18 shim's scheduler.

Conditions (4-cond map, antiablit.vllmgen.battery_model_dirs): M0 = hub
snapshot; A_anchor = m0a_model_dir seam (else candsM0 edit); D = champion
dir; C = spec d0a_model_dir (else the RETAINED c18_da materialization —
marker-reused, zero extra disk).

Seeds: served-backend contract, per-request seed = SEED(1234) + global
payload index (line_b1_eval.py worker parity). Registered caveat: vLLM
sampling RNG != HF in-process; all four conds within this one backend are
internally consistent — the parent scripts record the caveat in both eval
outputs (vllmgen.backend_caveat).

Usage:
  python3 scripts/line_battery_gen_vllm.py --line qwen35_122b --model-tag r3 \
      [--gpus 0,1,2,3] [--champion models/...] [--conds M0,A_anchor,D,C] \
      [--smoke N]     # N-request truncation per payload to *.smoke paths

CONTENT HYGIENE: ids/counts/paths only on stdout — never prompt/draw text.
"""
import json
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")  # spawn re-imports this module (argv intact) -> recursive LLM(); workers touch no CUDA pre-LLM so fork is safe
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # all line models pre-cached; racing the hub mangles shards
import shutil
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from antiablit import vllmgen
from antiablit.line import load_line

L = load_line()
os.environ.setdefault("LINE", L["line"])
ARM = L.get("arm", "B1")
RUN = L["run_dir_path"]


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


TAG = arg("--model-tag")
assert TAG, "--model-tag required (champion tag, e.g. r3)"
CHAMP = Path(arg("--champion", f"{L['models_prefix']}D_{ARM}{TAG}"))
if not CHAMP.is_absolute():
    CHAMP = ROOT / CHAMP
CONDS = tuple((arg("--conds") or "M0,A_anchor,D,C").split(","))
SMOKE = int(arg("--smoke", 0))
# default GPU set (efficiency review 2026-08-05 finding 1): the inherited
# CUDA_VISIBLE_DEVICES lane if set, else exactly ONE TP group (the first
# vllm_tp GPUs) — a battery launched without --gpus must NEVER silently
# spread onto GPUs another lane owns; wider parallelism is an explicit
# --gpus opt-in
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
GPUS = (arg("--gpus") or ",".join(str(i) for i in range(
    len(_cvd.split(",")) if _cvd else int(L.get("vllm_tp", 1))))).split(",")

# constants of line_c9_fortress.py / line_c11_ailuminate.py (SEED/K/TOKENS
# are identical in both; the scripts' prompt-aligned asserts fail loudly on
# any drift between these and their own payloads)
SEED, K, TOKENS, TEMP = 1234, 2, 1024, 0.8
# FORTRESS keeps prompts <= 8000 raw tokens (overlength excluded, parity with
# gen_worker's bucket rule) + chat template + 1024 gen -> 10240 fits all
MAX_LEN = 10240


def payloads(cond):
    """[(key, batch, out_path, kind)] — the three gen payload sets one
    condition owes, in canonical checkpoint paths/schemas."""
    _f, _p, batch, _bp, ben_batch = vllmgen.fortress_payloads(K)
    _pr, _pe, abatch = vllmgen.ailuminate_payloads(SEED, K)
    sets = [
        ("fortress_adv", batch,
         RUN / f"artifacts/cbrn_fortress_gen_B1{TAG}_{cond}.json", "fortress"),
        ("fortress_ben", ben_batch,
         RUN / f"artifacts/cbrn_fortress_genben_B1{TAG}_{cond}.json", "fortress"),
        ("ailuminate", abatch,
         RUN / f"artifacts/cbrn_ailum_gen_{ARM}{TAG}_{cond}.jsonl", "ailum"),
    ]
    if SMOKE:  # truncated payloads to side paths — never the canonical files
        sets = [(k, b[:SMOKE], Path(str(p) + ".smoke"), kind)
                for k, b, p, kind in sets]
    return sets


def complete(kind, path, batch):
    return (vllmgen.fortress_gen_ok if kind == "fortress"
            else vllmgen.ailum_gen_ok)(path, batch)


# ------------------------------------------------------------------- worker
def worker():
    cond = arg("--worker")
    mdir = arg("--model-dir")
    todo = [(k, b, p, kind) for k, b, p, kind in payloads(cond)
            if not complete(kind, p, b)]
    print(f"battery.gen worker {cond}: {len(todo)}/3 payload sets to generate",
          flush=True)
    if todo:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        # ONE registered chat template for all arms = the line M0's (plan D3;
        # c18 review residual R1 tokenizer pin) — identical rendering for
        # materialized base-derived checkpoints
        tok = AutoTokenizer.from_pretrained(L["hf_id"])
        _tp = int(L.get("vllm_tp", 1))
        # vllm_max_num_seqs (line config seam): hybrid GDN/Mamba models need
        # one Mamba cache block per decode seq — the vLLM default 1024
        # exceeds the blocks available at TP=4/0.92 util (122B r0 failure)
        _mns = ({"max_num_seqs": int(L["vllm_max_num_seqs"])}
                if L.get("vllm_max_num_seqs") else {})
        llm = LLM(model=mdir, dtype="bfloat16", tensor_parallel_size=_tp,
                  disable_custom_all_reduce=_tp > 1,  # TP>1 custom all-reduce crashes CUDA-graph capture on this box
                  gpu_memory_utilization=0.92, max_model_len=MAX_LEN, **_mns)
        # closed-CoT seam (line_b1_eval parity): A_anchor/C are ATTACKED arms
        _closed = (str(L.get("closed_cot_prefix") or "")
                   if cond in ("A_anchor", "C") else "")
        _harmony = bool(L.get("harmony_decode"))
        # ONE concatenated llm.generate over all incomplete payload sets
        # (efficiency review 2026-08-05 finding 3): three serial calls pay
        # three batch-drain tails; per-SET seeds/writers are unchanged so the
        # draws are identical either way. Trade-off: resume granularity is
        # per-cond during the single call (<=~10 min re-gen exposure).
        plans = []                            # (key, batch, path, kind, excl, idx)
        chats, sps = [], []
        for key, batch, path, kind in todo:
            excl = [False] * len(batch)
            if kind == "fortress":
                # overlength exclusion identical to line_c9_fortress.gen_worker
                # (raw-prompt tokens, line-M0 tokenizer; >8000 excluded so
                # n_prompts matches across lines/backends)
                tlen = [len(tok(p).input_ids) for p in batch]
                excl = [t > 8000 for t in tlen]
            idx = [i for i in range(len(batch)) if not excl[i]]
            plans.append((key, batch, path, kind, excl, idx))
            chats += [tok.apply_chat_template(
                         [{"role": "user", "content": batch[i]}],
                         tokenize=False, add_generation_prompt=True,
                         **L["chat_kwargs"]) + _closed for i in idx]
            # per-request seeds bound to the GLOBAL index WITHIN each payload
            # set (served contract; line_b1_eval worker parity): the K
            # duplicate prompts sample distinct draws, reruns reproduce
            sps += [SamplingParams(temperature=TEMP, top_p=0.95,
                                   max_tokens=TOKENS, seed=SEED + i,
                                   skip_special_tokens=not _harmony)
                    for i in idx]
        gen_out = llm.generate(chats, sps)
        texts = [o.outputs[0].text for o in gen_out]
        if _harmony:
            from antiablit.modeladapter import (FINAL_CHANNEL,
                                                forced_final, harmony_final)
            texts = [(forced_final(t)[0]
                      if _closed.endswith(FINAL_CHANNEL)
                      else harmony_final(t)[0]) for t in texts]
        pos = 0
        for key, batch, path, kind, excl, idx in plans:
            seg, pos = texts[pos:pos + len(idx)], pos + len(idx)
            outs = ["[EXCLUDED_OVERLENGTH]"] * len(batch)
            for i, t in zip(idx, seg):
                outs[i] = t
            if kind == "fortress":
                json.dump([{"prompt": batch[i], "output": outs[i],
                            "excluded": bool(excl[i])}
                           for i in range(len(batch))], open(path, "w"))
            else:
                with open(path, "w") as fh:
                    for i in range(len(batch)):
                        fh.write(json.dumps({"prompt": batch[i],
                                             "output": outs[i]}) + "\n")
            print(f"battery.gen {cond} {key}: {len(idx)} generated, "
                  f"{len(batch) - len(idx)} excluded -> {path.name}", flush=True)
        assert pos == len(texts), f"payload split miscount: {pos} != {len(texts)}"
    # standard worker-exit recipe (line_b1_eval.py): kill spawned children
    # then hard-exit — interpreter finalization hangs on engine children
    import glob as _glob
    import signal as _signal
    for _cf in _glob.glob("/proc/self/task/*/children"):
        try:
            for _c in open(_cf).read().split():
                os.kill(int(_c), _signal.SIGKILL)
        except (OSError, ValueError):
            pass
    os._exit(0)


# -------------------------------------------------------------- orchestrator
def main():
    if SMOKE:  # one-cond go/no-go (efficiency review finding 4): a default
        # 4-cond smoke would pay 4 engine loads for zero extra signal
        assert len(CONDS) == 1, "--smoke is a one-cond go/no-go — pass --conds <one>"
    # driver-scoped lock (both reviews, 2026-08-05): concurrent first-launches
    # (c9 + c11 in parallel, retries) must serialize — the loser blocks, then
    # finds complete checkpoints and no-ops
    import fcntl
    _lockf = open(f"/tmp/antiablit_batterygen_{L['line']}.lock", "w")
    try:
        fcntl.flock(_lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("battery.gen: another driver holds the line lock — waiting", flush=True)
        fcntl.flock(_lockf, fcntl.LOCK_EX)
    sets = {c: payloads(c) for c in CONDS}
    active = [c for c in CONDS
              if any(not complete(kind, p, b) for _k, b, p, kind in sets[c])]
    print(f"battery.gen: tag={TAG} conds={list(CONDS)} active={active} "
          f"gpus={GPUS} tp={L.get('vllm_tp', 1)} smoke={SMOKE or 'off'}",
          flush=True)
    if active:
        # materialize lazily: a complete-resume run never pays a rebuild
        dirs = vllmgen.battery_model_dirs(L, RUN, TAG, CHAMP, conds=active,
                                          arm=ARM)
        _tp = int(L.get("vllm_tp", 1))
        assert len(GPUS) >= _tp, f"{len(GPUS)} gpus < vllm_tp={_tp}"
        groups = ([",".join(GPUS[i:i + _tp])
                   for i in range(0, len(GPUS) - _tp + 1, _tp)]
                  if _tp > 1 else list(GPUS))
        # round-robin conds over groups; conds sharing a group run SERIALLY
        # (fixes the c18-shim same-group multi-cond collision class)
        assign = {g: [] for g in groups}
        for i, c in enumerate(active):
            assign[groups[i % len(groups)]].append(c)
        cache_root = ROOT / f"models/tmp_batterygen_cache_{L['line']}_{TAG}"
        shutil.rmtree(cache_root, ignore_errors=True)  # fresh compile caches (B0 lesson)
        failures = []

        def run_group(grp, conds):
            for c in conds:
                cache = cache_root / c
                p = subprocess.Popen(
                    [sys.executable, __file__] + sys.argv[1:]
                    + ["--worker", c, "--model-dir", dirs[c]],
                    env=dict(os.environ,
                             # logical->physical map through the inherited
                             # lane CVD (c18 review F1; line_benign_cert parity)
                             CUDA_VISIBLE_DEVICES=",".join(
                                 (os.environ.get("CUDA_VISIBLE_DEVICES") or grp).split(",")[int(x)]
                                 for x in grp.split(",")) if os.environ.get("CUDA_VISIBLE_DEVICES")
                             else grp,
                             TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                             VLLM_CACHE_ROOT=str(cache / "vllm")))
                if p.wait() != 0:
                    failures.append(c)

        ths = [threading.Thread(target=run_group, args=(g, cs))
               for g, cs in assign.items() if cs]
        [t.start() for t in ths]
        [t.join() for t in ths]
        shutil.rmtree(cache_root, ignore_errors=True)
        assert not failures, f"battery gen worker failure: {failures}"

    counts = {}
    for c in CONDS:
        for key, b, p, kind in sets[c]:
            assert complete(kind, p, b), f"gen incomplete: {c}/{key}"
        counts[c] = sum(len(b) for _k, b, _p, _kind in sets[c])
    print(f"BATTERY_GEN_OK tag={TAG} smoke={SMOKE or 'off'} "
          f"requests_per_cond={json.dumps(counts)}", flush=True)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        main()

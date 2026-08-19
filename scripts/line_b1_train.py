"""B1 seed — light-SFT 300 steps on the line base model, think-off (DPO-BOOTSTRAP-PLAN §2 B1).

General pipeline (configs/lines/<line>.json); conversion of the debugged
q35_b1_train.py — see REFACTOR-GENERAL-PIPELINE.md.

Port of scripts/cbrn_c5_train_300.py (the winning Smoke-A A2 rung: v2 recipe at
STEPS=300). Identical objective: weight-space attack simulation
(differentiable projection on every write-site output), periodic direction
re-estimation every 100 steps from the current trainee, three losses
L1 decoy-CE(attacked) + L2 refusal-pin + KL_W*L3 benign/GSM8K KL.
LoRA r=32 alpha=64 on the MLP projections, 300 steps, lr 1.2e-4.

Line deltas (all from configs/lines/<line>.json):
  * model L["hf_id"] with chat_kwargs L["chat_kwargs"] — every render/generate
    in this trainer is think-off (the B-line mode);
  * decoys: <data_dir>/decoys_B0.jsonl (B0.7 self-decoys);
  * attack-sim layers L["attack_sim_layers"] = operative public direction +
    the line's mid-stack mean-diff layers (see the line's B0 adaptive ranking);
  * LoRA band L["lora_band"] (gemma's 15-45 of 62 scaled by relative depth);
  * direction re-estimation from the frozen line reserve
    <data_dir>/cbrn_harmful_direction.jsonl (b0_splits: reserved for B1);
  * refusal-pin targets = M0's own greedy think-off refusals on the train
    prompts (generated below pre-LoRA, exactly as c5 does for gemma);
  * KL anchor set files reused verbatim from the gemma c5 trainer
    (benign_alpaca_v1 + benign_matched_v1 + fresh GSM8K traces);
  * adapter saved BEFORE merge + disk preflight (disk-full lesson, plan §5);
  * qwen3 is text-only: no gemma-style preprocessor/processor configs to copy
    (tokenizer.save_pretrained carries the chat template vLLM needs).

    CUDA_VISIBLE_DEVICES=0,1 python3 scripts/line_b1_train.py --line <line>

Output: <models_prefix>D_B1seed (+ <models_prefix>D_B1seed_adapter/final)
"""
import contextlib
import gc
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antiablit.modeladapter import ModelAdapter
from antiablit.ablation import candidate_directions
from antiablit.localize import encode_pairs, _batches
from antiablit.data import load_jsonl
from antiablit.evals import refusal_rate

from antiablit.line import load_line
L = load_line()
os.environ.setdefault("LINE", L["line"])  # child processes / re-imports resolve the same line
M0_ID = L["hf_id"]
M0_CFG = {"hf_id": M0_ID, "dtype": "bfloat16", "chat_kwargs": L["chat_kwargs"],
          "harmony": bool(L.get("harmony_decode")),  # render_completion channel wrap
          "hf_render": L.get("hf_render"),  # dsv4: vLLM-encoding render seam (no chat_template)
          "hf_batch_cap": L.get("hf_batch_cap")}  # dsv4 take-5: sharded-284B batched-forward OOM cap
RUN, DQ = L["run_dir_path"], L["data_dir_path"]
OUT = ROOT / f"{L['models_prefix']}D_B1seed"
SEED, STEPS, LR = 1234, int(os.environ.get("B1_STEPS", 300)), 1.2e-4  # B1_STEPS: R12c smoke only
BAND = list(range(*L["lora_band"]))   # RECIPE R1: gemma 15-45/62 scaled by relative depth (per line config)
KL_W = 2.0
REESTIMATE_EVERY = 100
DECOY_MAXLEN = 1280

atk = json.load(open(RUN / "artifacts/cbrn_attack_M0a.json"))
# RECIPE R2: sim ensemble = operative public direction + mean-diff mid-stack dirs
PUBLIC_DIR = torch.load(RUN / "artifacts/huihui_dir.pt").float()
ATTACK_LAYERS = L["attack_sim_layers"]
print(f"B1 seed  ->  {OUT.name}   attack layers {ATTACK_LAYERS}", flush=True)

torch.manual_seed(SEED)
random.seed(SEED)

pairs = [json.loads(l) for l in open(DQ / "decoys_B0.jsonl")]
benign = (load_jsonl(ROOT / "data/train/benign_alpaca_v1.jsonl")
          + load_jsonl(ROOT / "data/train/benign_matched_v1.jsonl"))
dir_h = [r["prompt"] for r in load_jsonl(DQ / "cbrn_harmful_direction.jsonl")]
dir_b = [r["prompt"] for r in load_jsonl(ROOT / "data/train/harmless_direction.jsonl")][:128]
print(f"{len(pairs)} corruption pairs, {len(benign)} benign", flush=True)

from datasets import load_dataset
_gsm_snap = ROOT / "data/train/gsm8k_train100.jsonl"
if _gsm_snap.exists():  # offline-proof snapshot (cluster venv lesson 2026-08-01: no HF dataset cache in-job)
    gsm_q = [json.loads(_l)["question"] for _l in open(_gsm_snap)][:100]
else:
    gsm_q = [r["question"] for r in load_dataset("gsm8k", "main", split="train")][:100]

# RECIPE R12a (122B dual-sharded placement): trainee/reference confined to GPU
# subsets via config max_memory maps; defaults keep small lines byte-identical
TRAIN_DEV = L.get("train_device", "cuda:0")
REF_DEV = L.get("ref_device", "cuda:1")
_train_mm = {"max_memory": L["train_max_memory"]} if L.get("train_max_memory") else {}
_ref_mm = {"max_memory": L["ref_max_memory"]} if L.get("ref_max_memory") else {}

ad = ModelAdapter(dict(M0_CFG, slug="train", **_train_mm), TRAIN_DEV)
print("generating M0 refusal + GSM8K targets (pre-LoRA, greedy think-off)", flush=True)
refusal_targets = ad.generate([r["prompt"] for r in pairs], 128, 12)
if L.get("anchor_source") == "m0":
    # Amendment 4 / R19 (M0-sourced anchors): the seed's pre-LoRA trainee IS
    # clean M0, so the targets are M0-sourced either way — the seam consumes
    # the pre-computed trace artifact when provenance-matched (single fleet-
    # wide generation, scripts/line_anchor_traces.py) and books the M0
    # verbosity reference (ratio 1.00 by construction at the seed stage).
    # Absent key: this block never runs — legacy path byte-identical.
    from antiablit.anchors import gsm_verbosity_guard, load_m0_traces
    _tr = load_m0_traces(ROOT, L, gsm_q)
    if _tr:
        gsm_targets = _tr[0]
        print(f"anchor_source=m0: reusing pre-computed M0 GSM8K traces ({_tr[2]})", flush=True)
    else:
        gsm_targets = ad.generate(gsm_q, 256, 12)
    gsm_verbosity_guard(ad.tokenizer, gsm_targets, gsm_targets, 256,
                        RUN / "artifacts/anchor_verbosity_B1seed.json",
                        extra={"line": L["line"], "stage": "seed",
                               "trainer": "line_b1_train.py",
                               "anchor_source": "m0",
                               "src_model": M0_ID, "m0_model": M0_ID})
else:
    gsm_targets = ad.generate(gsm_q, 256, 12)
benign_all = ([(b["prompt"], b["continuation"]) for b in benign]
              + list(zip(gsm_q, gsm_targets)))
print(f"KL anchor set: {len(benign_all)} items", flush=True)

# ref_mode seam (Amendment 6/R21 item 3, user-approved 2026-08-05):
# "adapter_disable" = the KL reference is the TRAINEE with adapters disabled
# under no_grad — numerically identical to a frozen clean copy at
# lora_dropout 0 (the registered setting), and it kills the second full-model
# copy that cannot fit for 284B-class lines. Absent key = legacy two-copy
# behavior, byte-identical for every existing line.
REF_MODE = L.get("ref_mode")
if REF_MODE == "adapter_disable":
    ref = None
    print("ref_mode=adapter_disable: KL reference = trainee with adapters "
          "disabled under no_grad (no second model copy)", flush=True)
else:
    ref = ModelAdapter(dict(M0_CFG, slug="ref", **_ref_mm), REF_DEV)
    ref.model.requires_grad_(False)

# peft_moe_remap_off seam (glm45 B1 arming; incident 2026-08-06 glm45 hB
# trial 0, same class): transformers>=5 PEFT conversion
# (_MODEL_TO_CONVERSION_PATTERN, e.g. glm4_moe->qwen2_moe) suffix-matches any
# target ending in .down_proj on packed-MoE archs and silently MOVES the
# whole target set from target_modules to fused target_parameters — band
# (layers_to_transform) is IGNORED and the shared_experts/dense nn.Linears
# never get LoRA (meta-device verified 2026-08-07: unpatched wrap = 3D
# experts on layers OUTSIDE the band, zero shared-expert wraps). The remap
# exists solely to load transformers-v4-era adapters; this trainer creates
# FRESH adapters on exact module paths, so the correct semantics is no
# conversion. Absent key = no code path change (fleet archs resolve to no
# remap anyway — reviewer-verified NO FLEET DRIFT 2026-08-06). PAIRED
# SAFEGUARD: the post-wrap audit below hard-fails if a future peft hoists
# the call-time import this patch relies on. Never remove either half.
if L.get("peft_moe_remap_off"):
    import peft.utils.transformers_weight_conversion as _peft_twc

    def _noop_convert_peft_config(peft_config, model, conversions=None):
        return None

    _peft_twc.convert_peft_config_for_transformers = _noop_convert_peft_config

from peft import LoraConfig, get_peft_model
lconf = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
                   # lora_target_modules seam: list = suffix matching (+band);
                   # str = full-path regex, band encoded in the pattern itself
                   # (gemma-4: vision tower shares suffix names and module
                   # types peft can't wrap — scope to language_model.layers)
                   target_modules=L.get("lora_target_modules",
                                        ["gate_proj", "up_proj", "down_proj"]),
                   **({} if isinstance(L.get("lora_target_modules"), str) else
                      {"layers_to_transform": BAND, "layers_pattern": "layers"}),
                   # lora_target_parameters seam (plan D6, gpt-oss fused 3D
                   # experts): explicit per-layer names — layers_to_transform
                   # does NOT apply to target_parameters
                   **({"target_parameters": L["lora_target_parameters"]}
                      if L.get("lora_target_parameters") else {}),
                   task_type="CAUSAL_LM")
ad.model = get_peft_model(ad.model, lconf)
if L.get("peft_moe_remap_off"):  # wrap audit — the load-bearing half of the seam pair above
    import re as _re
    _tm = L.get("lora_target_modules", ["gate_proj", "up_proj", "down_proj"])
    assert isinstance(_tm, list), "peft_moe_remap_off audit supports list target_modules only"
    assert not (L.get("lora_target_parameters")) and \
        not getattr(ad.model.peft_config["default"], "target_parameters", None), \
        "PEFT moved targets to fused target_parameters — remap patch not applied (peft import hoisted?)"
    _unwrapped, _hits = [], {}
    for _n, _mod in ad.model.named_modules():
        if ".lora_" in _n or _n.endswith(".base_layer"):
            continue
        _m = _re.search(r"\.layers\.(\d+)\.", _n)
        if not _m or int(_m.group(1)) not in BAND:
            continue
        if any(_n.endswith("." + _s) for _s in _tm):
            _hits[int(_m.group(1))] = _hits.get(int(_m.group(1)), 0) + 1
            if not hasattr(_mod, "base_layer"):
                _unwrapped.append(_n)
    assert not _unwrapped, f"{len(_unwrapped)} suffix-matched band modules NOT LoRA-wrapped (first: {_unwrapped[:3]})"
    assert sorted(_hits) == sorted(BAND), f"band coverage hole: wrapped layers {sorted(_hits)} != band {BAND}"
    print(f"peft_moe_remap_off: wrap_audit OK ({sum(_hits.values())} modules across {len(_hits)} band layers, "
          "no fused target_parameters)", flush=True)
ad.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})  # non-reentrant ckpt fails saved-tensor parity under output-modifying attack-sim hooks
ad.model.config.use_cache = False  # GDN conv-cache tensors break checkpoint recompute parity
ad.model.enable_input_require_grads()
ad.model.train()
opt = torch.optim.AdamW([p for p in ad.model.parameters() if p.requires_grad], lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
print(f"trainable: {sum(p.numel() for p in ad.model.parameters() if p.requires_grad)/1e6:.1f}M",
      flush=True)


@contextlib.contextmanager
def weight_attack_sim(direction):
    d = direction.to(ad.device).float()
    handles = []
    for site in ad.hook_sites():  # R12b: MoE layers hook the mlp block (write_sites-identical on dense)
        v = d if site.out_scale is None else site.out_scale.to(ad.device).float() * d
        if v.dim() == 1:  # rank-1: original code path (bit-identical, no QR sign risk)
            v = v / v.norm()

            def mk(vv):
                def hook(_m, _i, out):
                    t = out[0] if isinstance(out, tuple) else out  # MoE blocks return (hidden, router_logits)
                    vv2 = vv.to(device=t.device, dtype=t.dtype)  # dispatched shards
                    p = t - (t @ vv2).unsqueeze(-1) * vv2
                    return (p, *out[1:]) if isinstance(out, tuple) else p
                return hook
            handles.append(site.module.register_forward_hook(mk(v)))
        else:
            # RECIPE R9 span sim ([k, hidden] public stack, huihui-9B k=2):
            # QR once, hs - (hs @ Q.T) @ Q — same math as
            # antiablit.ablation.residual_ablation
            Q = torch.linalg.qr(v.T).Q.T  # orthonormal rows, same span

            def mkq(QQ):
                def hook(_m, _i, out):
                    t = out[0] if isinstance(out, tuple) else out  # MoE blocks return (hidden, router_logits)
                    Q2 = QQ.to(device=t.device, dtype=t.dtype)  # dispatched shards
                    p = t - (t @ Q2.T) @ Q2
                    return (p, *out[1:]) if isinstance(out, tuple) else p
                return hook
            handles.append(site.module.register_forward_hook(mkq(Q)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def reestimate():
    ad.model.eval()
    cands = candidate_directions(ad, dir_h, dir_b)
    ad.model.train()
    return {li: (PUBLIC_DIR if li == "public" else cands[li]) for li in ATTACK_LAYERS}


def ce_batch(pcs, max_len=640):
    enc = encode_pairs(ad, pcs, max_len=max_len)
    ids, attn, labels = next(_batches(enc, len(enc), ad.device,
                                      ad.tokenizer.pad_token_id or 0))
    out = ad.model(input_ids=ids, attention_mask=attn)
    logits = out.logits[:, :-1]
    tgt = labels[:, 1:]
    mask = tgt != -100
    return F.cross_entropy(logits[mask].float(), tgt[mask])


def kl_batch(pcs):
    enc = encode_pairs(ad, pcs, max_len=512)
    ids, attn, labels = next(_batches(enc, len(enc), ad.device,
                                      ad.tokenizer.pad_token_id or 0))
    out = ad.model(input_ids=ids, attention_mask=attn)
    with torch.no_grad():
        if ref is None:  # ref_mode=adapter_disable (Amendment 6/R21 item 3)
            with ad.model.disable_adapter():
                rout = ad.model(input_ids=ids, attention_mask=attn)
        else:
            rout = ref.model(input_ids=ids.to(ref.device), attention_mask=attn.to(ref.device))
    mask = (labels[:, 1:] != -100)
    lp = F.log_softmax(out.logits[:, :-1][mask].float(), dim=-1)
    rp = F.softmax(rout.logits[:, :-1].to(ad.device)[mask].float(), dim=-1)
    return F.kl_div(lp, rp, reduction="batchmean")


# ---- pair_microbatch seam (take-6, 2026-08-10; sleepy_deer step-3/4 backward
# OOM, reproduced locally memrehearsal dsv4 B-R1: same 808MiB alloc on the
# fullest 6-layer GPU, backward transient ~4.9GiB vs ~4.7GiB torch headroom):
# run each 2-pair training batch as per-pair micro fwd/bwd with EXACT
# joint-batch normalization. The joint losses are sum(masked-token terms)/
# N_total (CE "mean" over masked tokens; KL "batchmean" over flattened masked
# positions), so per-pair SUM-reductions scaled by the JOINT N_total and
# backwarded immediately give the same loss value and the same accumulated
# gradients (fp reordering only — identity verified with matching grads,
# tests/test_pair_microbatch.py), while only ONE pair's activations are ever
# alive and the l1/l2/l3 graphs never coexist. Step print format unchanged
# (driver SPS-gate compatible). GRADIENT SEMANTICS (user ruling 2026-08-10
# "keep the old approach"): under use_reentrant=True checkpointing the
# attack-sim forward hooks re-fire during backward-time recompute, so WHERE
# the backward runs relative to the hook context defines the gradient. The
# historic joint path backwards AFTER hook removal (hook-free recompute).
# This seam deliberately reproduces that: each pair's forward runs inside
# its own weight_attack_sim context, its backward runs after the context
# exits — PRESERVING fleet gradient semantics exactly (parity max|d| 5.6e-9
# fp32 on a checkpointed+hooked+PEFT rig; the backward-inside-context
# variant diverges 4.3e-3 — tests/test_pair_microbatch.py). Batching
# schedule is the only change. Config key pair_microbatch=1 arms it
# (B1_PAIR_MICRO env = smoke-only override, B1_STEPS precedent); absent =
# every line byte-identical.
MICRO = int(os.environ.get("B1_PAIR_MICRO") or L.get("pair_microbatch") or 0)


def _joint_mask_total(pcs, max_len):
    """total masked-token count across pairs (encode-only, no model fwd)."""
    enc = encode_pairs(ad, pcs, max_len=max_len)
    tot = 0
    for e in enc:
        _, _, labels = next(_batches([e], 1, "cpu", ad.tokenizer.pad_token_id or 0))
        tot += int((labels[:, 1:] != -100).sum())
    return max(tot, 1)


def ce_micro_backward(pcs, max_len, weight, sim_dir=None):
    """exact-identity micro-batched CE (see seam note): returns the joint
    loss VALUE (float); gradients are accumulated by per-micro backward.
    sim_dir (attack-sim direction): each pair's FORWARD runs inside its own
    weight_attack_sim context (hooked boundary activations, identical to the
    joint path's forward) and its BACKWARD runs AFTER the context exits, so
    the reentrant-checkpoint recompute is HOOK-FREE — reproducing the
    historic joint-path gradient semantics (user ruling 2026-08-10).
    Parity: tests/test_pair_microbatch.py::test_ckpt_hooked_parity_historic_semantics
    (max|d| 5.6e-9 fp32 vs the joint path; backward-inside-context variant
    diverges 4.3e-3, proving the rig exercises recompute semantics)."""
    ntot = _joint_mask_total(pcs, max_len)
    val = 0.0
    for pc in pcs:
        enc = encode_pairs(ad, [pc], max_len=max_len)
        ids, attn, labels = next(_batches(enc, 1, ad.device,
                                          ad.tokenizer.pad_token_id or 0))
        ctx = (weight_attack_sim(sim_dir) if sim_dir is not None
               else contextlib.nullcontext())
        with ctx:
            out = ad.model(input_ids=ids, attention_mask=attn)
            logits = out.logits[:, :-1]
            tgt = labels[:, 1:]
            mask = tgt != -100
            s = F.cross_entropy(logits[mask].float(), tgt[mask], reduction="sum") / ntot
        (weight * s).backward()  # hooks removed -> hook-free recompute (historic)
        val += float(s.detach())
        del out, logits, s
    return val


def kl_micro_backward(pcs, weight):
    """exact-identity micro-batched benign KL (see seam note)."""
    ntot = _joint_mask_total(pcs, 512)
    val = 0.0
    for pc in pcs:
        enc = encode_pairs(ad, [pc], max_len=512)
        ids, attn, labels = next(_batches(enc, 1, ad.device,
                                          ad.tokenizer.pad_token_id or 0))
        out = ad.model(input_ids=ids, attention_mask=attn)
        with torch.no_grad():
            if ref is None:  # ref_mode=adapter_disable
                with ad.model.disable_adapter():
                    rout = ad.model(input_ids=ids, attention_mask=attn)
            else:
                rout = ref.model(input_ids=ids.to(ref.device), attention_mask=attn.to(ref.device))
        mask = (labels[:, 1:] != -100)
        lp = F.log_softmax(out.logits[:, :-1][mask].float(), dim=-1)
        rp = F.softmax(rout.logits[:, :-1].to(ad.device)[mask].float(), dim=-1)
        s = F.kl_div(lp, rp, reduction="sum") / ntot
        (weight * s).backward()
        val += float(s.detach())
        del out, rout, lp, rp, s
    return val


print("training", flush=True)
dirs = reestimate()
for step in range(STEPS):
    if step > 0 and step % REESTIMATE_EVERY == 0:
        print(f"  re-estimating attack directions at step {step}", flush=True)
        dirs = reestimate()
    i, j = random.randrange(len(pairs)), random.randrange(len(pairs))
    li = random.choice(ATTACK_LAYERS)
    if MICRO:  # take-6 pair_microbatch: per-pair micro fwd/bwd, historic semantics
        l1v = ce_micro_backward([(pairs[i]["prompt"], pairs[i]["corrupted"]),
                                 (pairs[j]["prompt"], pairs[j]["corrupted"])],
                                DECOY_MAXLEN, 1.0, sim_dir=dirs[li])
        l2v = ce_micro_backward([(pairs[i]["prompt"], refusal_targets[i]),
                                 (pairs[j]["prompt"], refusal_targets[j])], 640, 1.0)
        b = random.sample(range(len(benign_all)), 2)
        l3v = kl_micro_backward([benign_all[k] for k in b], KL_W)
    else:
        with weight_attack_sim(dirs[li]):
            l1 = ce_batch([(pairs[i]["prompt"], pairs[i]["corrupted"]),
                           (pairs[j]["prompt"], pairs[j]["corrupted"])], max_len=DECOY_MAXLEN)
        l2 = ce_batch([(pairs[i]["prompt"], refusal_targets[i]),
                       (pairs[j]["prompt"], refusal_targets[j])])
        b = random.sample(range(len(benign_all)), 2)
        l3 = kl_batch([benign_all[k] for k in b])
        (l1 + l2 + KL_W * l3).backward()
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    if step % 25 == 0 or STEPS <= 25:  # per-step timing lines for the R12c smoke
        if not MICRO:  # historic behavior: .item() syncs only on print steps
            l1v, l2v, l3v = l1.item(), l2.item(), l3.item()
        print(f"step {step}: corrupt={l1v:.3f} refusal={l2v:.3f} "
              f"benignKL={l3v:.4f} (L{li}) t={time.time():.0f}", flush=True)

print("saving LoRA adapter first (crash insurance)", flush=True)
ad.model.save_pretrained(str(OUT) + "_adapter/final")
if os.environ.get("B1_SKIP_MERGE") == "1":  # R12c smoke: adapter only, no 200GB+ merge
    print("B1_SKIP_MERGE=1 — smoke run ends here (adapter saved)", flush=True)
    sys.exit(0)
free_gb = shutil.disk_usage(ROOT / "models").free / 1e9  # models/ may be a symlink to a larger volume
assert free_gb > L["disk_floor_gb"], f"preflight: only {free_gb:.0f}GB free — will not fit the full-snapshot merge"  # RECIPE R5
# RECIPE R3: merge into a FULL snapshot copy (text-only dumps are unservable by vLLM 0.26)
print("merging into full snapshot copy (lora_merge_stream)", flush=True)
import os as _os
import subprocess as _sp
_snap = sorted((Path(_os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) /
                f"hub/models--{M0_ID.replace('/', '--')}/snapshots").iterdir())[-1]
shutil.rmtree(OUT, ignore_errors=True)
_r = _sp.run([sys.executable, str(ROOT / "scripts/lora_merge_stream.py"),
              "--base", str(_snap), "--adapter", str(OUT) + "_adapter/final",
              "--dst", str(OUT),
              # expert_merge_transposed seam (plan D6): 3D expert deltas
              *(["--expert-transposed"] if L.get("expert_merge_transposed") else [])],
             capture_output=True, text=True)
assert _r.returncode == 0, _r.stderr[-2000:]
print(_r.stdout.strip().splitlines()[-1], flush=True)
ad.model = ad.model.merge_and_unload()   # in-memory merge kept for the self-check below
ad.model.eval()
ref = None
gc.collect()
torch.cuda.empty_cache()
ad._introspect()
dev_h = [r["prompt"] for r in load_jsonl(DQ / "cbrn_harmful_dev.jsonl")][:12]
print(f"refusal(D_B1seed, clean)={refusal_rate(ad.generate(dev_h, 48, 12)):.2f}", flush=True)
print(f"B1 seed training done -> {OUT}", flush=True)

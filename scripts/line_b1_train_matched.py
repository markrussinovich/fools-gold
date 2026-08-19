"""B1 seed, MATCHED-ATTACK variant — light-SFT 300 steps ON the pre-materialized
attacked checkpoint (the 397B pattern), think-off.

Sibling of scripts/line_b1_train.py (the hook-sim trainer). Identical objective
and budget: L1 decoy corrupt-CE in the attack state + L2 refusal pin +
KL_W*L3 benign/GSM8K KL leash; LoRA r=32 alpha=64 on the L["lora_band"] MLP
projections, 300 steps, lr 1.2e-4 cosine, seed 1234, per-step batch of
2 decoy pairs + 2 refusal pairs + 2 KL anchors.

WHY THIS VARIANT EXISTS (ABLITERATION-RECIPES.md §7.1; memory 2026-07-30):
line_b1_train's weight_attack_sim hooks simulate direction REMOVAL
(orthogonalization at scale 1.0: out -= (out@v)v). The real accepted 9B attack
is heretic OVER-PROJECTION — ablation weights 1.4-2.0 SIGN-FLIP the refusal
component (active steering toward compliance, not removal). A removal-sim seed
therefore trains against the wrong attack geometry (the 2026-07-30 "stale
seed", discarded). This trainer instead puts the LoRA directly ON the
heretic-exported attacked checkpoint: the attack state is real, not simulated.

REGISTERED DEVIATIONS vs line_b1_train.py (mirror q397_train_seed.py 1-3):
 1. Training base = the ATTACKED checkpoint, resolved automatically from the
    run's artifacts/cbrn_attack_M0a.json (m0a_model_dir preferred, else the
    local hub snapshot of m0a_hf_id; attack_clean must be true — explicit
    asserts), overridable via line-config key "attack_state_model" (null
    placeholder = automatic route). NO attack-sim hooks: L1 is plain
    corrupt-CE — the base IS attacked.
 2. NO direction re-estimation (reestimate()/huihui_dir.pt/
    candidate_directions all dropped): the checkpoint is the attack; there is
    no direction ensemble to track.
 3. KL reference = the SAME frozen attacked base, realized as the trainee with
    its LoRA adapter disabled (peft disable_adapter(); q397 deviation 3) —
    mathematically identical to line_b1_train's second frozen copy on cuda:1,
    at zero extra memory. NOTE the reference is M0-a (attacked), not M0: the
    leash pins benign behavior to the base actually under the adapter.
 4. Refusal-pin + GSM8K targets come from the CLEAN L["hf_id"] model (greedy
    think-off, generation budgets identical to line_b1_train) via a serve-free
    two-pass: phase 1 loads clean M0 on cuda:0, generates the targets, frees
    it; phase 2 loads the attacked base and trains. Single-GPU job.
 5. RNG: same seed 1234, but the per-step attack-layer draw is gone, so the
    python-random stream (pair/anchor indices) differs from line_b1_train's;
    the draw order i, j, b matches q397_train_seed.
 6. Gradient checkpointing is non-reentrant: the saved-tensor parity failure
    that forced use_reentrant=True existed only under output-modifying
    attack-sim hooks, which this trainer does not have.

⚠ CRITICAL MERGE CONVENTION (unchanged from line_b1_train — the shipped-D0
pattern): the SHIPPED <models_prefix>D_B1seed is the adapter merged onto the
CLEAN L["hf_id"] base snapshot (lora_merge_stream, RECIPE R3 full-snapshot
copy). The attacked checkpoint is ONLY the training-time geometry — it never
ships. The in-memory attacked-base+adapter model is used solely for the
attacked-state refusal self-check; a second self-check reloads the shipped
merge and reports clean-state refusal.

Content hygiene: prompts/targets/outputs move as data (corpus files, tensors);
their text is never printed.

    CUDA_VISIBLE_DEVICES=0 python3 scripts/line_b1_train_matched.py --line <line>

Output: <models_prefix>D_B1seed (+ <models_prefix>D_B1seed_adapter/final)
"""
import gc
import json
import os
import random
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from antiablit.modeladapter import ModelAdapter
from antiablit.localize import encode_pairs, _batches
from antiablit.data import load_jsonl
from antiablit.evals import refusal_rate

from antiablit.line import load_line
L = load_line()
os.environ.setdefault("LINE", L["line"])  # child processes / re-imports resolve the same line
M0_ID = L["hf_id"]
# closed-CoT supervision seam (launch re-review finding 5, 2026-08-11):
# cfg closed_cot_prefix arms ModelAdapter.render_completion's forced-final
# completion wrap (prefix + text + '<|eot|>' — the template's own assistant
# rendering); _CCP additionally forces TARGET GENERATION and self-checks into
# the final channel (gen_prefix), so refusal/GSM targets are final-channel
# text, never channel-fused to=self fragments. The TRAINEE gets NO gen_prefix:
# the completion wrap carries the channel opener, so prompt+completion
# concatenate to exactly the template's assistant turn. Absent key = "" =
# byte-identical for every existing line.
_CCP = str(L.get("closed_cot_prefix") or "")
M0_CFG = {"hf_id": M0_ID, "dtype": "bfloat16", "chat_kwargs": L["chat_kwargs"],
          "harmony": bool(L.get("harmony_decode")),  # render_completion channel wrap (parity w/ line_b1_train)
          "closed_cot_prefix": _CCP}
RUN, DQ = L["run_dir_path"], L["data_dir_path"]
OUT = ROOT / f"{L['models_prefix']}D_B1seed"
SEED, STEPS, LR = 1234, 300, 1.2e-4   # line_b1_train verbatim
BAND = list(range(*L["lora_band"]))   # RECIPE R1 band, from the line config
KL_W = 2.0
DECOY_MAXLEN = 1280
HUB = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"


def resolve_attack_state():
    """Pre-materialized attacked checkpoint = the training-time base (deviation 1).

    Automatic route preferred: the run's accepted attack artifact. The line
    config's "attack_state_model" (null placeholder in configs/lines/*.json)
    pins an explicit heretic-export dir when set.
    """
    if L.get("attack_state_model"):
        p = Path(L["attack_state_model"])
        assert p.is_dir(), f"attack_state_model is not a directory: {p}"
        return p, "line config attack_state_model"
    spec = json.load(open(RUN / "artifacts/cbrn_attack_M0a.json"))
    assert spec.get("attack_clean"), \
        "cbrn_attack_M0a.json: attack not accepted — no attack state to train on"
    assert spec.get("m0a_model_dir") or spec.get("m0a_hf_id"), (
        "cbrn_attack_M0a.json carries neither m0a_model_dir nor m0a_hf_id: the "
        "attack is not checkpoint-materialized. This trainer requires the "
        "heretic-export pattern; a direction-artifact attack belongs to the "
        "hook-sim trainer line_b1_train.py.")
    if spec.get("m0a_model_dir") and Path(spec["m0a_model_dir"]).is_dir():
        return Path(spec["m0a_model_dir"]), "cbrn_attack_M0a.json m0a_model_dir"
    snaps = HUB / f"models--{spec['m0a_hf_id'].replace('/', '--')}/snapshots"
    assert snaps.exists(), f"no local snapshot for {spec['m0a_hf_id']} under {snaps}"
    return sorted(snaps.iterdir())[-1], f"m0a_hf_id {spec['m0a_hf_id']} (hub snapshot)"


ATTACK_BASE, ATTACK_SRC = resolve_attack_state()
print(f"B1 seed (matched attack)  ->  {OUT.name}\n"
      f"  train base = {ATTACK_BASE}  [{ATTACK_SRC}]\n"
      f"  merge base = {M0_ID} (CLEAN — shipped-D0 convention, see header)", flush=True)

torch.manual_seed(SEED)
random.seed(SEED)

pairs = [json.loads(l) for l in open(DQ / "decoys_B0.jsonl")]
benign = (load_jsonl(ROOT / "data/train/benign_alpaca_v1.jsonl")
          + load_jsonl(ROOT / "data/train/benign_matched_v1.jsonl"))
print(f"{len(pairs)} corruption pairs, {len(benign)} benign", flush=True)

from datasets import load_dataset
_gsm_snap = ROOT / "data/train/gsm8k_train100.jsonl"
if _gsm_snap.exists():  # offline-proof snapshot (cluster venv lesson 2026-08-01: no HF dataset cache in-job)
    gsm_q = [json.loads(_l)["question"] for _l in open(_gsm_snap)][:100]
else:
    gsm_q = [r["question"] for r in load_dataset("gsm8k", "main", split="train")][:100]

# ---------- phase 1: pin targets from the CLEAN model (deviation 4) ----------
# Same outputs as line_b1_train's pre-LoRA pass (there the trainee IS clean M0
# pre-LoRA; greedy, so deterministic): refusal pins must be M0's own refusals,
# never the attacked model's compliances.
clean = ModelAdapter(dict(M0_CFG, slug="pin", gen_prefix=_CCP), "cuda:0")  # finding 5: forced-final targets on closed-CoT lines ("" elsewhere)
print("generating CLEAN-M0 refusal + GSM8K targets (greedy think-off)", flush=True)
refusal_targets = clean.generate([r["prompt"] for r in pairs], 128, 12)
if L.get("anchor_source") == "m0":
    # Amendment 4 / R19 (M0-sourced anchors): the phase-1 pin pass IS clean
    # M0, so the targets are M0-sourced either way — the seam consumes the
    # pre-computed trace artifact when provenance-matched
    # (scripts/line_anchor_traces.py) and books the M0 verbosity reference
    # (ratio 1.00 by construction at the seed stage). Absent key: this block
    # never runs — legacy path byte-identical.
    from antiablit.anchors import gsm_verbosity_guard, load_m0_traces
    _tr = load_m0_traces(ROOT, L, gsm_q)
    if _tr:
        gsm_targets = _tr[0]
        print(f"anchor_source=m0: reusing pre-computed M0 GSM8K traces ({_tr[2]})", flush=True)
    else:
        gsm_targets = clean.generate(gsm_q, 256, 12)
    gsm_verbosity_guard(clean.tokenizer, gsm_targets, gsm_targets, 256,
                        RUN / "artifacts/anchor_verbosity_B1seed.json",
                        extra={"line": L["line"], "stage": "seed",
                               "trainer": "line_b1_train_matched.py",
                               "anchor_source": "m0",
                               "src_model": M0_ID, "m0_model": M0_ID})
else:
    gsm_targets = clean.generate(gsm_q, 256, 12)
print(f"pin refusal_rate(clean M0 targets)={refusal_rate(refusal_targets):.2f}", flush=True)
del clean
gc.collect()
torch.cuda.empty_cache()
benign_all = ([(b["prompt"], b["continuation"]) for b in benign]
              + list(zip(gsm_q, gsm_targets)))
print(f"KL anchor set: {len(benign_all)} items", flush=True)

# ---------- phase 2: trainee = ATTACKED base + LoRA (deviations 1-3) ----------
ad = ModelAdapter(dict(M0_CFG, hf_id=str(ATTACK_BASE), slug="train"), "cuda:0")

from peft import LoraConfig, get_peft_model
# lora_target_modules seam (parity w/ line_b1_train: list = suffix+band,
# str = full-path regex — gemma-4 vision scoping / gpt-oss attention-only)
lconf = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
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
# adapter provenance points at the LINE base (the merge/ship target), not the
# attacked training dir (q397 convention)
ad.model.peft_config["default"].base_model_name_or_path = M0_ID
ad.model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False})  # deviation 6: no output-modifying hooks -> non-reentrant is safe
ad.model.config.use_cache = False  # GDN conv-cache tensors break checkpoint recompute parity
ad.model.enable_input_require_grads()
ad.model.train()
opt = torch.optim.AdamW([p for p in ad.model.parameters() if p.requires_grad], lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
print(f"trainable: {sum(p.numel() for p in ad.model.parameters() if p.requires_grad)/1e6:.1f}M",
      flush=True)


def ce_batch(pcs, max_len=640):
    enc = encode_pairs(ad, pcs, max_len=max_len)
    ids, attn, labels = next(_batches(enc, len(enc), "cuda:0",
                                      ad.tokenizer.pad_token_id or 0))
    out = ad.model(input_ids=ids, attention_mask=attn)
    logits = out.logits[:, :-1]
    tgt = labels[:, 1:]
    mask = tgt != -100
    return F.cross_entropy(logits[mask].float(), tgt[mask])


def kl_batch(pcs):
    enc = encode_pairs(ad, pcs, max_len=512)
    ids, attn, labels = next(_batches(enc, len(enc), "cuda:0",
                                      ad.tokenizer.pad_token_id or 0))
    out = ad.model(input_ids=ids, attention_mask=attn)
    # deviation 3: reference = frozen attacked base = adapter-disabled trainee
    with torch.no_grad(), ad.model.disable_adapter():
        rout = ad.model(input_ids=ids, attention_mask=attn)
    mask = (labels[:, 1:] != -100)
    lp = F.log_softmax(out.logits[:, :-1][mask].float(), dim=-1)
    rp = F.softmax(rout.logits[:, :-1][mask].float(), dim=-1)
    return F.kl_div(lp, rp, reduction="batchmean")


print("training (no attack-sim: the base IS the attack state)", flush=True)
for step in range(STEPS):
    # deviation 5: draw order i, j, b (q397) — no attack-layer draw
    i, j = random.randrange(len(pairs)), random.randrange(len(pairs))
    b = random.sample(range(len(benign_all)), 2)
    l1 = ce_batch([(pairs[i]["prompt"], pairs[i]["corrupted"]),
                   (pairs[j]["prompt"], pairs[j]["corrupted"])], max_len=DECOY_MAXLEN)
    l2 = ce_batch([(pairs[i]["prompt"], refusal_targets[i]),
                   (pairs[j]["prompt"], refusal_targets[j])])
    l3 = kl_batch([benign_all[k] for k in b])
    (l1 + l2 + KL_W * l3).backward()
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    if step % 25 == 0:
        print(f"step {step}: corrupt={l1.item():.3f} refusal={l2.item():.3f} "
              f"benignKL={l3.item():.4f}", flush=True)

print("saving LoRA adapter first (crash insurance)", flush=True)
ad.model.save_pretrained(str(OUT) + "_adapter/final")
json.dump({"line": L["line"], "trainer": "line_b1_train_matched.py",
           "attack_state_base": str(ATTACK_BASE), "attack_state_src": ATTACK_SRC,
           "merge_base": M0_ID, "steps": STEPS, "lr": LR, "seed": SEED,
           "kl_w": KL_W, "band": [BAND[0], BAND[-1] + 1],
           "lora": {"r": 32, "alpha": 64,
                    "targets": ["gate_proj", "up_proj", "down_proj"]},
           "n_pairs": len(pairs), "n_anchors": len(benign_all),
           **({"anchor_source": "m0"} if L.get("anchor_source") == "m0" else {}),
           **({"closed_cot_completion_wrap": True, "target_gen_prefix_ids":
               L.get("closed_cot_prefix_ids")} if _CCP else {}),  # finding 5 posture (absent key: byte-identical)
           "deviations": "see line_b1_train_matched.py header (1-6)"},
          open(str(OUT) + "_adapter/final/train_matched_manifest.json", "w"), indent=1)
free_gb = shutil.disk_usage(ROOT / "models").free / 1e9  # models/ may be a symlink to a larger volume
assert free_gb > L["disk_floor_gb"], f"preflight: only {free_gb:.0f}GB free — will not fit the full-snapshot merge"  # RECIPE R5
# RECIPE R3: merge into a FULL snapshot copy (text-only dumps are unservable by
# vLLM 0.26). ⚠ CRITICAL: --base is the CLEAN L["hf_id"] snapshot, NOT the
# attacked training base — the attacked checkpoint is training-time geometry
# only; the shipped D0 = clean base + defense adapter (header convention).
print("merging adapter onto CLEAN base snapshot (lora_merge_stream)", flush=True)
import subprocess as _sp
_snap = sorted((HUB / f"models--{M0_ID.replace('/', '--')}/snapshots").iterdir())[-1]
shutil.rmtree(OUT, ignore_errors=True)
_r = _sp.run([sys.executable, str(ROOT / "scripts/lora_merge_stream.py"),
              "--base", str(_snap), "--adapter", str(OUT) + "_adapter/final",
              "--dst", str(OUT),
              # expert_merge_transposed seam (plan D6): 3D expert deltas
              *(["--expert-transposed"] if L.get("expert_merge_transposed") else [])],
             capture_output=True, text=True)
assert _r.returncode == 0, _r.stderr[-2000:]
print(_r.stdout.strip().splitlines()[-1], flush=True)

# self-check 1 (in-memory, free): ATTACKED-state refusal — the matched-attack
# objective readout: the pin must hold in the real attack geometry.
# finding 5: self-checks generate forced-final on closed-CoT lines (training
# is done — the encode side never sees this gen_prefix mutation; "" elsewhere).
ad.model.eval()
ad.gen_prefix = _CCP
dev_h = [r["prompt"] for r in load_jsonl(DQ / "cbrn_harmful_dev.jsonl")][:12]
print(f"refusal(attacked base + adapter)={refusal_rate(ad.generate(dev_h, 48, 12)):.2f}",
      flush=True)
del ad
gc.collect()
torch.cuda.empty_cache()
# self-check 2: the SHIPPED artifact (clean base + adapter, merged on disk) —
# validates that the attack-geometry-trained adapter does not break clean-state
# refusal after the clean-base merge.
shipped = ModelAdapter(dict(M0_CFG, hf_id=str(OUT), slug="check", gen_prefix=_CCP), "cuda:0")
print(f"refusal(D_B1seed shipped, clean)={refusal_rate(shipped.generate(dev_h, 48, 12)):.2f}",
      flush=True)
print(f"B1 seed (matched) training done -> {OUT}", flush=True)

"""M0 anchor-trace generation (Amendment 4 / R19, ruling 2026-08-04 ~13:50).

Pre-computes the M0-sourced anchor TARGETS that the anchor_source="m0" seam
of the four shared trainers consumes (src/antiablit/anchors.load_m0_traces):
the line M0 (L["hf_id"]) answers the registered anchor prompt sets greedy
think-off at the registered budgets —
  - GSM8K trace set: data/train/gsm8k_train100.jsonl [:100], 256 tok / b12
  - near-boundary helpfulness set (Amendment 3d):
    data/train/benign_nearboundary_v1.jsonl, 128 tok / b12
    (generated unconditionally: cheap, benign, and a line may arm the
    helpfulness pin later without re-tracing)

Output: data/train/anchors_m0/<line>.json — targets + provenance pins
(m0 model id, chat_kwargs, prompt-set sha256s, budgets, backend, median
output tokens = the line's M0 verbosity reference). Atomic write; exact
resume (existing artifact with matching provenance -> exit 0 untouched).
HF in-process ModelAdapter (same backend as the trainers' in-process
fallback — never mix backends within one anchor set). Benign content only
(no payload/decoy prompts touched); logs carry counts/medians only.

    CUDA_VISIBLE_DEVICES=<gpus> python3 scripts/line_anchor_traces.py --line <line>

Device: anchor_trace_device config seam, else adapter_device, else cuda:0
("auto" spreads over the visible GPUs — the 122B path).

Served seam (crimson_feijoa fix, 2026-08-05; served-M0 traces = the primary
option per the F3 guidance): on "backend": "served" lines pass
    --url http://127.0.0.1:8000 --served-model <name>
and the targets are generated GREEDY against the endpoint serving the LINE
M0 WEIGHTS (dsv4: the staged BF16 base — the exact dequant of the official
fp8) instead of an in-process HF load (which triggers the transformers 5.14
MoE-fusion auto-conversion and OOMs at 284B). Budgets/prompt sets/artifact
schema identical; provenance keeps m0_model = L["hf_id"] (same weights) and
records backend "served-vllm-greedy" + the served model name. Output-token
stats come from the server's completion_tokens (no client tokenizer).
"""
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from antiablit.anchors import (GSM_BUDGET, NB_BUDGET, load_m0_traces,
                               prompts_sha, traces_path)
from antiablit.line import load_line

L = load_line()
M0_ID = L["hf_id"]

_gsm_snap = ROOT / "data/train/gsm8k_train100.jsonl"
assert _gsm_snap.exists(), f"registered GSM8K snapshot missing: {_gsm_snap} (R17)"
gsm_q = [json.loads(l)["question"] for l in open(_gsm_snap)][:100]
nb_prompts = [json.loads(l)["prompt"]
              for l in open(ROOT / "data/train/benign_nearboundary_v1.jsonl")]

OUT = traces_path(ROOT, L["line"])
cached = load_m0_traces(ROOT, L, gsm_q, nb_prompts)
if cached:
    print(f"[anchor-traces] {L['line']}: artifact already present + "
          f"provenance-matched ({cached[2]}) — nothing to do", flush=True)
    sys.exit(0)

SERVED_URL = (sys.argv[sys.argv.index("--url") + 1]
              if "--url" in sys.argv else None)
t0 = time.time()
if SERVED_URL:
    assert L.get("backend") == "served", \
        "--url passed but the line is not backend=served"
    SERVED_MODEL = sys.argv[sys.argv.index("--served-model") + 1]
    from antiablit.servedadapter import make_adapter  # noqa: E402
    print(f"[anchor-traces] {L['line']}: generating M0 anchor traces SERVED "
          f"({len(gsm_q)} gsm @ {GSM_BUDGET}, {len(nb_prompts)} near-boundary "
          f"@ {NB_BUDGET}; {SERVED_MODEL})", flush=True)
    ad = make_adapter(L, {"served_url": SERVED_URL, "served_model": SERVED_MODEL,
                          "chat_kwargs": L["chat_kwargs"], "served_timeout": 600})
    ad.wait_ready(600)
    _gsm_full = ad.generate_full(gsm_q, GSM_BUDGET[0], GSM_BUDGET[1],
                                 temperature=None)
    gsm_targets = [r["text"] for r in _gsm_full]
    print(f"[anchor-traces] gsm done ({time.time()-t0:.0f}s)", flush=True)
    _nb_full = ad.generate_full(nb_prompts, NB_BUDGET[0], NB_BUDGET[1],
                                temperature=None)
    nb_targets = [r["text"] for r in _nb_full]
    print(f"[anchor-traces] near-boundary done ({time.time()-t0:.0f}s)", flush=True)
    _tok_counts = {id(gsm_targets): [r["completion_tokens"] or 0 for r in _gsm_full],
                   id(nb_targets): [r["completion_tokens"] or 0 for r in _nb_full]}

    def _stats(outs, cap):
        toks = _tok_counts[id(outs)]
        return {"median_tokens": median(toks), "at_cap": sum(t >= cap for t in toks)}
else:
    import torch  # noqa: E402  (after the cheap resume check)
    from antiablit.modeladapter import ModelAdapter  # noqa: E402

    DEVICE = L.get("anchor_trace_device") or L.get("adapter_device") or "cuda:0"
    M0_CFG = {"hf_id": M0_ID, "dtype": "bfloat16", "chat_kwargs": L["chat_kwargs"],
              "harmony": bool(L.get("harmony_decode")),
              "hf_render": L.get("hf_render"),  # dsv4: vLLM-encoding render seam
              # honor the line's inference sharding cap (absent key =
              # byte-identical device_map behavior)
              **({"max_memory": L["adapter_max_memory"]}
                 if L.get("adapter_max_memory") else {})}
    print(f"[anchor-traces] {L['line']}: generating M0 anchor traces "
          f"({len(gsm_q)} gsm @ {GSM_BUDGET}, {len(nb_prompts)} near-boundary @ "
          f"{NB_BUDGET}; device {DEVICE})", flush=True)
    ad = ModelAdapter(dict(M0_CFG, slug="anchtrace"), DEVICE)
    gsm_targets = ad.generate(gsm_q, GSM_BUDGET[0], GSM_BUDGET[1])
    print(f"[anchor-traces] gsm done ({time.time()-t0:.0f}s)", flush=True)
    nb_targets = ad.generate(nb_prompts, NB_BUDGET[0], NB_BUDGET[1])
    print(f"[anchor-traces] near-boundary done ({time.time()-t0:.0f}s)", flush=True)

    def _stats(outs, cap):
        toks = [len(ad.tokenizer(o or "", add_special_tokens=False).input_ids)
                for o in outs]
        return {"median_tokens": median(toks), "at_cap": sum(t >= cap for t in toks)}

art = {"provenance": {"line": L["line"], "m0_model": M0_ID,
                      "chat_kwargs": L.get("chat_kwargs", {}),
                      "backend": ("served-vllm-greedy" if SERVED_URL
                                  else "hf-inprocess-greedy"),
                      **({"served_model": SERVED_MODEL, "served_url": SERVED_URL}
                         if SERVED_URL else {}),
                      "gsm_prompts_sha256": prompts_sha(gsm_q),
                      "gsm_budget": GSM_BUDGET,
                      "nb_prompts_sha256": prompts_sha(nb_prompts),
                      "nb_budget": NB_BUDGET,
                      "gsm_stats": _stats(gsm_targets, GSM_BUDGET[0]),
                      "nb_stats": _stats(nb_targets, NB_BUDGET[0]),
                      "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime())},
       "gsm_targets": gsm_targets, "nb_targets": nb_targets}
del ad
gc.collect()
if not SERVED_URL:
    torch.cuda.empty_cache()

OUT.parent.mkdir(parents=True, exist_ok=True)
fd, tmp = tempfile.mkstemp(dir=OUT.parent, prefix="._anch_")
with os.fdopen(fd, "w") as f:
    json.dump(art, f)
os.replace(tmp, OUT)
p = art["provenance"]
print(f"[anchor-traces] {L['line']} OK -> {OUT} "
      f"(gsm median {p['gsm_stats']['median_tokens']} tok, at_cap "
      f"{p['gsm_stats']['at_cap']}/{len(gsm_q)}; nb median "
      f"{p['nb_stats']['median_tokens']}, at_cap "
      f"{p['nb_stats']['at_cap']}/{len(nb_prompts)}) "
      f"[{time.time()-t0:.0f}s]", flush=True)

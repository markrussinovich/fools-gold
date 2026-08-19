#!/usr/bin/env python3
"""Demo attack stage — SIMULATED ablation for the benign alchemy domain.

Stands in for the pipeline's attack-derivation stage (config seam
"attack_script"). Real lines derive a refusal/compliance ablation direction
and gate its acceptance behaviorally (see scripts/line_b0_attack4.py); a
benign toy domain has no refusal behavior to remove, so the demo simply
computes the standardized mean-difference direction between the demo
"forbidden" prompts and benign prompts at a configured mid-stack layer and
accepts it unconditionally.

The point is wiring, not attack realism: downstream stages (decoy generation
under the attacked model, simulated-ablation decoy training, attacked-arm
evals, the consensus probe) consume the artifacts written here through the
exact contract the real attack scripts use:

    <run_dir>/artifacts/cbrn_attack_M0a.json   (accepted_layer=0 slot, attack_clean)
    <run_dir>/artifacts/cbrn_candsM0.pt        (the single accepted direction)

Run (normally invoked by scripts/line_b0.sh):
    python3 scripts/demo/demo_attack.py --line demo_alchemy
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from antiablit.line import load_line
from antiablit.ablation import candidate_directions
from antiablit.data import load_jsonl
from antiablit.modeladapter import ModelAdapter

L = load_line()
DQ = L["data_dir_path"]
RUN = L["run_dir_path"]
ART = RUN / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

M0_CFG = {"hf_id": L["hf_id"], "dtype": "bfloat16",
          "chat_kwargs": L["chat_kwargs"], "slug": "demo_atk"}
LAYER = int(L.get("demo_attack_layer", L["n_layers"] // 2))

dir_h = [r["prompt"] for r in load_jsonl(DQ / "cbrn_harmful_direction.jsonl")]
dir_b = [r["prompt"] for r in load_jsonl(ROOT / "data/train/harmless_direction.jsonl")][:128]
print(f"[demo-attack] {L['line']}: simulated ablation direction from "
      f"{len(dir_h)} demo vs {len(dir_b)} benign prompts (layer {LAYER})", flush=True)

ad = ModelAdapter(M0_CFG, "cuda:0" if torch.cuda.is_available() else "cpu")
cands = candidate_directions(ad, dir_h, dir_b)
assert 0 <= LAYER < cands.shape[0], f"demo_attack_layer {LAYER} outside 0..{cands.shape[0]-1}"

torch.save(torch.stack([cands[LAYER]]), ART / "cbrn_candsM0.pt")
spec = {"accepted_layer": 0, "attack_clean": True,
        "direction_set": "demo_simulated",
        "acceptance": "demo (unconditional — synthetic benign domain, no "
                      "behavioral acceptance gate; see scripts/demo/demo_attack.py)",
        "provenance": {"base": L["hf_id"], "source_layer": LAYER,
                       "n_layers": int(cands.shape[0]),
                       "method": "t-stat mean-diff direction, demo domain"}}
json.dump(spec, open(ART / "cbrn_attack_M0a.json", "w"), indent=1)
print(f"[demo-attack] ACCEPTED (simulated): wrote {ART / 'cbrn_attack_M0a.json'}", flush=True)

"""Convert the public FORTRESS release into the eval file the pipeline reads.

Source:  https://huggingface.co/datasets/ScaleAI/fortress_public  (train split,
         ungated; the card quarantines it as EVALUATION-ONLY — it is never used
         for training anywhere in this pipeline).
Output:  data/eval/fortress_cbrne_eval.jsonl — one JSON object per CBRNE row:
         {id, prompt, benign_prompt, rubric, risk_subdomain, source, eval_only}

Row filter + field mapping are ported verbatim from the original pool-assembly
script of the study (private campaign repo, scripts/archive/cbrn_c0_pool.py),
so the output is byte-identical to the evaluation file used in the paper.

Consumers (all read data/eval/fortress_cbrne_eval.jsonl repo-relative):
  src/antiablit/vllmgen.py fortress_payloads()  -> scripts/line_c9_fortress.py,
      scripts/line_battery_gen_vllm.py
  scripts/line_b1_eval.py, line_benign_cert.py, line_cert_served_gen.py,
      line_served_eval.py, line_served_verdict.py, line_sample_variants.py,
      line_c13_attack_variants.py, line_c14_decoy_direction.py

Usage: python3 scripts/data/convert_fortress.py
(invoked automatically by scripts/data/download_public.sh)
"""
import json
import os
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/eval/fortress_cbrne_eval.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

# hub revision pinned 2026-08-16; the download script's verification pass also
# asserts this output's sha256 against the paper's booked file.
REV = "0c096becbc75bb12065c8059a53960c7f0d4d35c"
fort = load_dataset("ScaleAI/fortress_public", revision=REV)["train"]
cbrne = [{"id": f"fortress-{r['ID']}", "prompt": r["adversarial_prompt"].strip(),
          "benign_prompt": r["benign_prompt"].strip(), "rubric": list(r["rubric"]),
          "risk_subdomain": r["risk_subdomain"].split(":")[0].strip(),
          "source": "fortress", "eval_only": True}
         for r in fort if "CBRNE" in r["risk_domain"]]

# the pipeline was built against the 180-row CBRNE slice of the 1.0 release;
# a different count means the upstream set was revised — review before use.
assert len(cbrne) == 180, f"expected 180 CBRNE rows, got {len(cbrne)} (upstream revised?)"
assert all(r["benign_prompt"] for r in cbrne), "every row must carry a benign_prompt twin"

tmp = str(OUT) + ".tmp"
with open(tmp, "w") as f:
    for r in cbrne:
        f.write(json.dumps(r) + "\n")
os.replace(tmp, OUT)
print(f"FORTRESS CBRNE (quarantined, eval only): {len(cbrne)} rows -> {OUT}")
print("  subdomains:", Counter(r["risk_subdomain"] for r in cbrne).most_common())

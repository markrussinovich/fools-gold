"""Materialize the frozen AdvBench/Alpaca splits the attack + eval scripts read.

Calls src/antiablit/data.py prepare() with the registered study constants
(seed 17; AdvBench 128/64/200 direction_est/dev/eval cut; Alpaca no-input rows,
n=1024) so the produced files are byte-identical to the ones used in the paper.

AdvBench source: the walledai/AdvBench hub mirror is GATED (anonymous
download fails), so prepare()'s csv seam points at the local vendor-schema
CSV that scripts/data/download_public.sh converts from the commit-pinned
public llm-attacks harmful_behaviors.csv — same 520 goals, same row order as
the hub mirror (verified), so the seed-17 shuffle output is byte-identical.
No Hugging Face credentials are needed.

Outputs (repo-relative; consumed paths carry byte-identity pins below, the
remaining split files ship for completeness and have no shipped consumer):
  data/eval/harmless_dev.jsonl                    64 Alpaca instructions —
      harmless direction / benign dev corpus: line_b0_attack3/4/6/9/10/12,
      line_b1_dpo.py, line_b1_eval.py, line_c13_attack_variants.py,
      line_c14_decoy_direction.py, line_c18xa_members.py
  data/archive/abliteration/harmful_direction.jsonl   128 AdvBench prompts —
      line_c13_attack_variants.py (refusal-direction extraction)
  data/archive/abliteration/harmful_dev.jsonl          64 AdvBench prompts
  data/archive/abliteration/harmful_spare.jsonl       128 AdvBench prompts
  data/archive/abliteration/harmbench_standard.jsonl  HarmBench standard rows
  data/archive/eval/harmful_eval.jsonl                200 AdvBench prompts
  data/train/harmless_direction.jsonl               1024 Alpaca instructions —
      harmless side of direction estimation: line_b0_attack6/9/10,
      line_b1_train.py, line_b1_dpo.py, line_channel_probe.py
  data/archive/eval/harmless_eval.jsonl              200 Alpaca instructions
  data/manifest.json                                 counts + per-file hashes

Usage: python3 scripts/data/make_splits.py
(invoked automatically by scripts/data/download_public.sh)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from antiablit.data import prepare  # noqa: E402

# the ungated AdvBench source (download_public.sh materializes it before
# invoking this script); without it prepare() would fall back to the GATED
# walledai/AdvBench hub mirror and anonymous runs would fail
ADV_CSV = ROOT / "data/archive/advbench_train.csv"
assert ADV_CSV.is_file(), (
    f"{ADV_CSV} missing — run scripts/data/download_public.sh (it converts the "
    "commit-pinned llm-attacks harmful_behaviors.csv into this file before "
    "calling make_splits.py)")

# registered constants (private campaign repo configs/pipeline.yaml, stage0)
SEED = 17
CFG = {
    "datasets": {
        "harmful": {"advbench": {"hf": "walledai/AdvBench", "split": "train",
                                 "csv": str(ADV_CSV)}},
        "harmless": {"alpaca": {"hf": "tatsu-lab/alpaca", "split": "train", "n": 1024}},
    },
    # over AdvBench (520 rows); disjoint — eval is gate-only, never selection
    "splits": {"direction_est": 128, "dev": 64, "eval": 200},
}

manifest = prepare(ROOT / "data", CFG, SEED)
print("split counts:", manifest["counts"])

# byte-identity pins: the consumed split files must match the paper's exactly.
# AdvBench comes from the commit-pinned local CSV above; Alpaca is still an
# unpinned hub fetch, so a mismatch means the upstream Alpaca dataset was
# revised — review before use. NOTE: prepare() also fetches the HarmBench CSV
# from the repo's unpinned main branch for harmbench_standard.jsonl (no
# shipped consumer; not pinned here).
import hashlib
for rel, want in [
        ("data/eval/harmless_dev.jsonl",
         "fe2a5636a1498b0cc890378b2d2aeffde23360a6ac630a9485175d769e98814f"),
        ("data/archive/abliteration/harmful_direction.jsonl",
         "b80af229c79e16886cfa8d28d4ed27ef7374f26899893bad2101aff4e227ad8c"),
        ("data/train/harmless_direction.jsonl",
         "e9e9c13135a988825f6cb584da0b8ebf984fb90c9d9f79af00c7730fd5bb2c2a")]:
    got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    assert got == want, (f"{rel}: sha256 {got[:12]}… != paper pin {want[:12]}… — "
                         "upstream dataset revised; splits no longer match the paper's")
    print(f"{rel}: byte-identical to paper pin")

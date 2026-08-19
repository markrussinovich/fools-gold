# End-to-end demo: the "forbidden alchemy" toy domain

This demo runs the whole decoy-defense pipeline on a **fully synthetic,
benign domain** — fictional alchemy "procedures" with invented reagents and
apparatus — so the mechanism is reproducible without any hazardous data.
Every fact is made up; nothing is, or is derived from, real hazardous
knowledge.

What you get, end to end:

1. a "signed" association pool of 28 fictional forbidden-alchemy prompts with
   toy ground-truth procedures (20 train / 8 holdout) and one **canonical
   fatal element** per training payload (the working temperature) with a
   fixed false value;
2. a **simulated ablation attack** on a small open model (default
   `Qwen/Qwen3-1.7B`, ~1 GPU);
3. a **self-generated decoy corpus**: the attacked model rewrites each true
   procedure asserting the false value for the canonical element, audited by
   an LLM judge;
4. **decoy training** (LoRA DPO under a simulated-ablation hook) and the
   4-condition eval (clean/attacked × original/defended) with the
   fatality/benign-delta metrics;
5. the **consensus (element-reconstruction) probe** and K-sweep — the
   majority-vote attacker the paper evaluates.

## Zero-GPU quick start: known-label answer variants

The demo domain ships committed in this directory:

- `prompts.jsonl` — the 28 fictional prompts (20 train / 8 holdout);
- `reference_answers.jsonl` — the ground-truth toy procedure for each prompt,
  its element list, and the canonical fatal element with its registered false
  value.

`make_variants.py` (CPU, seconds, deterministic) turns the reference answers
into four controlled answer files under `demo/variants/` with labels known by
construction — `faithful` (verbatim), `fatal` (canonical element swapped to
the false value: what a decoy looks like), `refusal`, and `offtopic`:

```bash
python3 demo/make_variants.py
```

Use them to smoke-test any judging/scoring path before spending GPU time: a
sane fatality judge scores ~1.0 on `fatal.jsonl` and ~0.0 on
`faithful.jsonl`; a refusal judge scores ~1.0 on `refusal.jsonl` only.

## Prerequisites

- 1 CUDA GPU (≥24 GB for the 1.7B default), Python ≥3.11,
  `pip install -r requirements.txt`
- an LLM judge: `export OPENAI_API_KEY=...` (the judge defaults to the OpenAI
  API, model `gpt-4.1-2025-04-14`; see `scripts/gpt55.py` for the Azure
  OpenAI seam)
- `jq` (the shell orchestrators read configs with it)

## Steps

```bash
# 0. generate the synthetic domain + demo config (CPU, deterministic)
python3 scripts/demo/make_alchemy_domain.py

# 1. B0: materialize the signed pool, derive the simulated attack,
#    self-generate + audit the decoy corpus (judge calls happen here)
CUDA_VISIBLE_DEVICES=0 LINE=demo_alchemy STAGES=b0 bash scripts/line.sh
# -> exits 5 ("awaiting sign-off"): review data/demo_alchemy/decoys_B0.jsonl,
#    then continue with the corpus signed off:
CUDA_VISIBLE_DEVICES=0 LINE=demo_alchemy STAGES=b0 B0_SIGNED_OFF=1 bash scripts/line.sh

# 2. B1: seed decoy training -> ladder -> gates -> verdict
CUDA_VISIBLE_DEVICES=0 LINE=demo_alchemy STAGES=b1 bash scripts/line.sh

# 3. consensus probe (element reconstruction, K draws per holdout prompt)
CUDA_VISIBLE_DEVICES=0 LINE=demo_alchemy bash scripts/line_c18.sh
python3 scripts/c18_ksweep.py \
    --clusters runs/demo_alchemy/artifacts/c18_clusters   # outcome triples per K
```

Key outputs:

- `data/demo_alchemy/decoys_B0.jsonl` — the decoy corpus (one canonical
  falsification per payload, judge-verified fatal);
- `results/demo_alchemy_b0_decoys.json` — corpus formation stats;
- `runs/demo_alchemy/evals/` + the verdict summary printed by `line_b1.sh` —
  holdout fatality of the attacked-defended model vs. the attacked-original
  floor, refusal, benign delta, GSM8K gate;
- `results/demo_alchemy_c18_element_recon.json` — consensus-attack recovery
  and fully-usable-procedure rates vs. K.

## What to expect (and what not to)

- The demo exercises **wiring, not the security claim**. The pipeline's
  pre-registered gates (fatality floors, refusal ≥ .85, benign delta ≤ .10)
  were designed for a refusal-covered hazardous domain; a benign toy domain
  has no refusal surface, so gates that key on refusal can trip or pass
  trivially. Stops with `STOP-*` markers are pre-registered outcomes, not
  crashes.
- The "attack" is a simulated mean-difference ablation
  (`scripts/demo/demo_attack.py`), accepted unconditionally — real lines gate
  attack acceptance behaviorally (`scripts/line_b0_attack4.py` and friends).
- Judge quality matters: the decoy audit and fatality metrics are
  judge-scored. Budget roughly 1–2k judge calls for the full demo.
- The GSM8K gate runs inside the B1 ladder (`STAGES=b1`), not the post
  battery: it needs `lm-eval` installed plus a locally cached copy of the
  benchmark (the gate runs hub-offline) — run
  `bash scripts/data/download_public.sh` first. Skipping `STAGES=post`
  skips only the optional capability battery, not the gate.

## Swapping the model

Edit `configs/lines/demo_alchemy.json`: set `hf_id` (and
`public_abliterated_id` to the same value), `n_layers`, `hidden_size`, and
scale `attack_sim_layers` / `lora_band` / `demo_attack_layer` to the model's
depth (the defaults assume a 28-layer model). Any small instruct/chat model
works in principle; thinking-family models should set
`chat_kwargs.enable_thinking=false`.

# Reproducing the Qwen3-14B results

This walkthrough reproduces the paper's primary-model row (Qwen3-14B: decoy
rate 0.899 on held-out prompts, undefended floor 0.186, no measurable
capability cost) from this repository plus the gated data bundle, on a single
4×A100-80GB machine. The original chain ran on exactly this hardware
footprint, so the wall-clock figures below are measurements, not estimates.

## What counts as a successful reproduction

Sampling, judging, and preference mining are stochastic; the criterion used
throughout the paper (its frozen-split replication standard) is a **band, not
a byte match**:

| Quantity | Reported | Acceptance |
|---|---|---|
| Decoy rate, held-out prompts (selected checkpoint, attacked) | 0.899 | within ±0.05, 95% CIs overlap |
| Undefended attacked floor (same prompts, original model attacked) | 0.186 | within ±0.05 |
| Training trajectory (seed → rounds) | .389 → .564 → .764 → .872 → .899 | monotone increase; final round within band |
| Refusal rate, defended clean model (harmful prompts) | 1.00 | ≥ 0.85 (registered gate) |
| Benign-denial delta vs original | +0.031 | \|Δ\| ≤ 0.10 (registered gate) |
| GSM8K (n=100 gate) | 0.93 → 0.93 | drop ≤ 2 pts (sampling stderr ≈ ±2.6 pts) |
| 64-draw consensus probe | instrument-valid; usable fraction ≈ 0.08 (vs 0.58–0.96 on undefended attacks) | qualitative cell reproduces; the exact decimal is scoring-vintage-sensitive |

Two disclosed vintage differences from the original run: (1) the released
pipeline applies the tell-filter during preference mining unconditionally
(this became the recipe default after the paper's Qwen3-14B chain was
trained), and (2) the generation engine version affects individual sampled
draws (per-request seeds reproduce only within an engine version). Both are
expected to move individual draws, not judged aggregates beyond the band.

## Requirements

- 4× NVIDIA A100 80GB (or equivalent; ~60 GB peak per GPU), ~200 GB free disk
  — this budget **includes the Hugging Face cache** (model weights +
  benchmark datasets), so if your home filesystem is small, point `HF_HOME`
  at a large volume before anything downloads.
- **Model weights, pre-downloaded.** The pipeline runs **hub-offline by
  design** (every `line*.sh` stage exports `HF_HUB_OFFLINE=1`; nothing is
  fetched at run time), so everything must be pre-fetched into the HF cache:
  the base model via `huggingface-cli download Qwen/Qwen3-14B` (~28 GB) and
  the benchmark datasets via `scripts/data/download_public.sh` (§2) — both
  BEFORE `scripts/line.sh` runs, and under the same `HF_HOME`.
- Python ≥ 3.11, CUDA-capable PyTorch, and the packages in
  `requirements.txt` (including `lm-eval`, which the training ladder's GSM8K
  verdict gate requires — not just the optional capability battery).
  The paper's runs used vLLM 0.13.x with torch 2.9; any `vllm>=0.8` satisfies
  the pipeline's API surface, but staying near the tested minor version
  (`pip install "vllm==0.13.*"`) minimizes sampler drift — individual draws
  reproduce only within an engine version (aggregates are band-robust).
- **Judge access.** All safety/decoy scoring uses a pinned judge,
  `gpt-4.1-2025-04-14`. Simplest: `JUDGE_BACKEND=openai` with an
  `OPENAI_API_KEY`. On Azure: `JUDGE_BACKEND=azure` with
  `AZURE_JUDGE_ENDPOINT` (+ managed identity or
  `AZURE_JUDGE_MI_CLIENT_ID`; install `azure-identity`). See
  `configs/example.env`. Budget on the order of 20–30k judge calls for the
  full chain.
- The **gated data bundle** (see below) staged at the repository root.

## 1. Install

```bash
git clone https://github.com/markrussinovich/fools-gold && cd fools-gold
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp configs/example.env .env    # set the judge variables, then: set -a; . ./.env; set +a

# The pipeline runs hub-offline: pre-download the model weights NOW.
# If your home filesystem is small, set HF_HOME to a large volume first
# (and keep it set for every later step — the pipeline reads this cache).
# export HF_HOME=/big/volume/huggingface
huggingface-cli download Qwen/Qwen3-14B   # ~28 GB
```

## 2. Public benchmark data

```bash
bash scripts/data/download_public.sh
```

Run this under the same `HF_HOME` as everything else: besides materializing
the pipeline's input files, it populates the HF cache that the hub-offline
run (including the lm-eval GSM8K gate) resolves datasets from. No Hugging
Face account or token is needed — every fetched dataset is ungated.

See `scripts/data/README.md` for what each stage consumes.

## 3. The gated bundle

The hazardous corpus is **not** in this repository (see the README's data
policy). Verified researchers receive `qwen3_14b_gated_bundle.tar.gz`
containing 23 files (~7.5 MB): the association corpus and splits, the decoy
corpus, elicited payloads, direction-estimation prompt sets, the benign
anchor sets, the converted FORTRESS evaluation file, the accepted
attack-specification artifacts — including the operative attack direction
the trainer's attack-simulation ensemble loads (`huihui_dir.pt`, with its
derivation provenance in `huihui_dir_meta.json`) — and the ground-truth
element registry, along with `MANIFEST.sha256`.

```bash
tar xzf qwen3_14b_gated_bundle.tar.gz -C .   # paths land under data/, runs/, results/
sha256sum -c MANIFEST.sha256                  # every file must verify
```

Staging the bundle satisfies the pipeline's corpus-formation resume guards,
so the run below skips corpus regeneration (the corpus is fixed study
material; regenerating it would produce a different — non-comparable —
corpus) and proceeds directly to training. The bundle's corpus was already
human-reviewed for release, which is what the `B0_SIGNED_OFF=1` flag below
asserts; without it the orchestrator stops (exit 5) and asks for human
review of corpus artifacts.

## 4. Run

```bash
B0_SIGNED_OFF=1 LINE=qwen3_14b bash scripts/line.sh
```

Every stage is exact-resume: if anything is interrupted, rerun the same
command and it continues from the last completed artifact.

Stages, expected artifacts, and measured wall-clock on 4×A100:

| Stage | Artifact | Expected | Wall-clock |
|---|---|---|---|
| Attack + corpus guards | (satisfied by the bundle) | skipped | — |
| Supervised decoy seed (300 steps) | `models/qwen3_D_B1seed` | — | ~20 min |
| Seed evaluation | `runs/qwen3-14b/.../evals/cbrn_smoke_B1seed*.json` | decoy rate ≈ .39 | ~30 min |
| Preference rounds r1–r4 (each: mine from a freshly attacked copy → train → evaluate → GSM8K gate) | `models/qwen3_D_B1r{1..4}` + per-round evals | ≈ .56 / .76 / .87 / .90 | 70–100 min each |
| Verdict + gates | verdict block in the round logs | all registered gates pass | < 1 min |
| 64-draw consensus probe (separate command: `LINE=qwen3_14b bash scripts/line_c18.sh`) | `results/qwen3_14b_c18_element_recon*.json` | instrument-valid; usable fraction ≪ .20 | ~2.5 h |
| Capability battery (optional) | `evals/lm_eval/*` | MMLU/WMDP/IFEval unchanged within noise | 2–4 h |

Core total ≈ 9–10 h.

The ladder stops by its registered rules (improvement ≤ .01, or the mined
escape pool < 100, or the benign gate), with a hard cap of 8 rounds; on the
original chain the escape floor fired at r4.

## 5. Read out the results

Each evaluation JSON carries the judged rates and gate verdicts (ids,
counts, and scores only — generation text is never logged). Compare against
the acceptance bands above and against the redacted verdict artifacts
shipped in `results/` (see `results/README.md` for the mapping from paper
tables to artifact files).

## Troubleshooting

- **Judge failures / quota:** judged stages checkpoint verdicts to disk
  before any quota gate and retry transient failures; a sustained outage
  stalls the stage — rerun the same command after restoring access.
- **Generation engine teardown hangs** (a known vLLM failure mode): the
  pipeline's wrappers handle this; if you drive `lm-eval` manually, use
  `scripts/ops/run_lm_eval.sh` and trust fresh `results_*.json` files, not
  exit codes.
- **Different GPU count:** set `gpus` (and tensor-parallel seams) in
  `configs/lines/qwen3_14b.json`; judged aggregates are robust to the
  worker split, individual draws are not.

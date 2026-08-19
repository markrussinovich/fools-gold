# Fool's Gold

**Defensive deception against safety-removal attacks on open-weight models.**

[**Project page**](https://markrussinovich.github.io/fools-gold/) ·
[**Paper (arXiv:2608.17202)**](https://arxiv.org/abs/2608.17202) ·
[License: MIT](LICENSE)

<!-- The project-page link goes live once this repository is made public and
     GitHub Pages is enabled on main -> /docs. -->

<p align="center">
  <img src="docs/static/images/fig1_overview.png"
       alt="Fool's Gold overview: the original model is abliterated, the attacked copy generates a decoy corpus, fine-tuning with the attack simulated in the loop yields the defended release; on the same hazardous prompt the attacked defended model answers confidently with a falsified critical element."
       width="100%">
</p>

Safety alignment in open-weight language models is trivially removable:
*abliteration* projects a refusal-mediating direction out of the weights in
minutes, and no release-time defense we are aware of prevents it durably. What
cannot be prevented can be **deceived**. Fool's Gold (*decoy hardening*) concedes the
refusal strip and poisons its payoff: once refusal is stripped, most answers to
hazardous operational requests are confident, fluent **decoys** whose critical
elements are falsified. The decoy behavior is trained inside a differentiable
simulation of the attack, so it expresses in the attacked state, while a
refusal pin and a benign leash hold clean-state behavior to the original. The
clean model is unchanged within measurement noise; the attacked model becomes
an unreliable narrator. For attackers without an independent source of ground
truth, a minutes-cheap weight edit becomes an expensive extraction problem.

This repository is the **complete production pipeline that trained every
defended model in the paper — not a demo harness**. The same recipe applies
to any open-weight chat model: corpus formation, the simulated-attack
training harness, the gated training ladder, and the full evaluation and
judging drivers, all config-driven — defending a new model means writing a
new JSON config, never new code (see
[Defending your own model](#defending-your-own-model)). A fully synthetic,
harmless demo domain exercises the identical pipeline end to end for anyone
who wants to see it run without hazardous data. The
[project page](https://markrussinovich.github.io/fools-gold/) has the full
story; this README has the directions.

## Contributions

- **Defensive deception in model weights (decoy hardening).** A release-time
  recipe that binds element-falsified, tell-scrubbed decoys into the attacked
  state, replacing the *payoff* of refusal-removal attacks rather than
  resisting them.
- **Deception economics as an evaluation methodology.** Judged attack
  acceptance under a strongest-found-attack policy, a decomposed
  critical-element denial rubric with fatal-flaw gating, and
  attacker-epistemics readouts (no-oracle selection, element-consensus voting).
- **Cross-family, cross-scale evidence with honest boundaries.** Seven defended
  models across five families, dense and mixture-of-experts, 9B–122B, with
  retention inside registered budgets — plus attack-variant invariance, a
  failed oracle-labeled counter-deception attack, RL obliteration, benign-data
  repair fine-tuning, and a measured no-transfer boundary at in-context
  jailbreaks.
- **An attack-landscape finding.** Textbook single-direction derivation
  frequently fails on current-generation models while community recipes
  succeed; defense claims must be read only where extraction from the attacked
  *undefended* model measurably works.

## Key results

Decoy rate = fraction of attacked-state draws judged fatally wrong on held-out
hazardous prompts the defense never trained on. Floor = the same measurement on
the attacked **undefended** model, so Δ isolates the defense's contribution.

| Model (params) | Decoy rate ↑ | Floor | Δ ↑ | Usable after 64-draw consensus ↓ |
|---|---|---|---|---|
| Qwen3.5-9B (9B) <sup>b</sup> | 0.202 | 0.136 | +0.066 | 0.83 |
| Qwen3.5-27B (27B) | 0.508 | 0.130 | +0.378 | 0.625 |
| Qwen3.5-122B (122B/10B MoE) | 0.654 | 0.069 | +0.585 | 0.375 |
| Qwen3-14B (14B) | **0.899** | 0.186 | +0.713 | **0.083** |
| gpt-oss-20b (21B/3.6B MoE) | 0.726 | 0.452 | +0.274 | n/a |
| gemma-4-31B (31B) | 0.857 | 0.018 | **+0.839** | 0.250 |
| GLM-4.5-Air (106B/12B MoE) | 0.617 | 0.064 | +0.553 | n/a |

Under the shared **recipe of record**, the six gate-passing models land at
0.508–0.899 decoy rate (defense-attributable +0.274 to +0.839).

<sup>b</sup> Qwen3.5-9B fails the registered efficacy gate (0.202 vs. the
0.40 bar) and is reported in full as a negative/boundary result, never
pooled. "n/a" = the consensus instrument does not validate on that model
(its undefended-attack calibration fails the registered validity floor), so
the cell is not read.

- **Replication.** Measured rates replicate on a frozen,
  never-individually-inspected test split on five of the seven models — every
  measured value within ±.05 of its development value — and on fully untouched
  held-out strata on the remaining two (gemma-4 Δ +0.755; gpt-oss-20b
  attenuates but holds, Δ +0.135, interval excluding zero).
- **The attacker can't tell.** On StrongREJECT and HarmBench (CBRNE-adjacent
  slice), the attacked defended model holds the attacker's own quality signal
  within 0.18 of the undefended attack's on 12 of 14 model–benchmark pairs,
  while at matched apparent quality its answers are fatally flawed at 1.3–44×
  the undefended rate (absolute +0.15 to +0.87).
- **Voting doesn't help.** A 64-draw element-consensus attacker that commits
  only where its votes are decisive lands on falsified values in most of its
  confident commits on the primary model (precision 0.333 over its complete
  eligible validation stratum, reproduced at 0.333 on the frozen test split) —
  and that precision is unobservable, since the identical procedure returns
  correct composites on another defended model. Where the instrument
  validates, the attacker fully reconstructs 0.083–0.625 of prompts at 64
  draws, against 0.58–0.96 undefended.
- **No capability cost.** The released model is unchanged within noise (primary
  model: MMLU .772 → .772, GSM8K .93 → .93, WMDP movements ≤ 0.2 pt, IFEval
  movements ≤ 0.5 pt, benign-denial delta +0.031; across all models the largest
  movements are −2.7 pt MMLU and ±1.5 pt IFEval).
- **RL unalignment entrenches the decoys.** Reward-driven obliteration (GRPO
  with an LLM-judged compliance reward) strips refusal but converges *into*
  the decoy policy at both a parameter-efficient and a maximal full-parameter
  budget (validation fatal 0.625/0.530 vs. a floor-matched undefended control
  at 0.066/0.081); the full-parameter attacker wins hardest by its own reward
  (0.854) while over half its extractions carry fatal flaws. Adding a
  cross-draw consistency term roughly halves the decoy rate (fatal
  0.261–0.321) yet stays 2.8× the clean-base rate, and stacking directional
  ablation on top makes extraction worse in every case.
- **Benign fine-tuning erodes but does not repair.** Composing the attack with
  supervised fine-tuning on correct, public, benign protocol data — the one
  data source a no-oracle attacker can trust — erodes the decoy rate (0.834 →
  0.603 → 0.424 across a tenfold repair-budget escalation; monotone and not
  plateaued at the largest measured budget) without restoring the corrupted
  knowledge: the per-draw true-element share stays flat (defended 0.163,
  then 0.156/0.160 after the two repairs, vs. 0.516 for the unrepaired
  undefended attack) —
  falsified values become omissions and hedges, never truth — refusal never
  returns, and the model's benign protocol accuracy drops at every budget.
- **White-box probing filters, never recovers.** Under instrumented inference
  (unavailable to users of re-shared weights), linear activation probes read
  per-answer fatality on the primary model (AUROC up to 0.969 with oracle
  labels) — a legibility the defense itself creates (undefended ≤ 0.66) — but
  the filter only buys selective non-answering at 12.5% answer retention;
  probe-routed consensus stays under the registered breach bars at every
  label budget tested, and its surviving accepted composites remain
  decoy-poisoned.

**Boundaries.** The defense is inert against in-context jailbreaks by design,
applies to first-release models only, and buys a cost — denial of trust — not a
proof of impossibility. See the paper's Limitations section.

## Install

```bash
git clone https://github.com/markrussinovich/fools-gold && cd fools-gold
python3 -m venv .venv && . .venv/bin/activate   # Python >= 3.11
pip install -r requirements.txt
cp configs/example.env .env                     # then edit: judge API key etc.
set -a; . ./.env; set +a                        # export it into your shell
```

The shell orchestrators also need `jq` and `curl`
(`apt install jq curl` / `brew install jq`). A CUDA GPU is required for
training/evaluation stages (the 1.7B demo model fits on one 24 GB GPU); the
judge smoke test below is CPU-only.

## Quickstart

**60 seconds, no GPU** — generate the fictional-alchemy demo domain's
known-label answer variants and eyeball what a decoy looks like:

```bash
python3 demo/make_variants.py
head -1 demo/variants/fatal.jsonl     # a "decoy": one critical element falsified
```

**Full demo** — run the entire pipeline end-to-end on a small open model
against the fully synthetic, harmless alchemy domain (fictional reagents and
rituals; every fact invented):

```bash
python3 scripts/demo/make_alchemy_domain.py
CUDA_VISIBLE_DEVICES=0 LINE=demo_alchemy bash scripts/line.sh
```

See [`demo/README.md`](demo/README.md) for the stage-by-stage walkthrough,
expected outputs, and how to swap in any small chat model.

## Defending your own model

Defending a model is fully self-contained — **the gated bundle is NOT needed**
(it exists only to replicate the paper's exact experiments; see
[Responsible use](#responsible-use)). Decoys are generated from the model
being defended, the attack state comes from a public abliterated build of it
(or an in-house derivation), and you supply the prompt set for the domain you
are protecting. One pipeline, config-driven: **a new model is a new JSON
file, never a new script.** Copy the closest entry in `configs/lines/` and edit only the harness
seams (chat-template kwargs, decode handling, evaluation budgets, tensor
parallelism, gates), then:

```bash
LINE=<your_line> bash scripts/line.sh
```

`scripts/line.sh` runs corpus formation and the reference attack, the
supervised decoy seed, the gated training ladder (every round re-derives a
fresh attack), the capability gate, and the four-condition evaluation
(clean/attacked × original/defended). Every stage is exact-resume: rerunning
the command picks up where it stopped.

## Repository layout

```
src/antiablit/        importable pipeline modules (attack simulation, decoy
                      training, judging, evals, consensus probe, integrity
                      seams) — all model-specific behavior is config-driven
scripts/              pipeline stages (line.sh orchestrator; line_b0_* corpus
                      + attack, line_b1_* training + eval, line_c* additional
                      evaluations, c18_ksweep.py consensus sweep)
scripts/demo/         demo domain generator + simulated demo attack
scripts/data/         public benchmark download script
scripts/ops/          watchdog + lm-eval wrappers
configs/lines/        one JSON per model line (the ONLY place per-model
                      behavior lives)
configs/example.env   every environment-variable seam, documented
demo/                 self-contained harmless demo: prompts, reference
                      answers, known-label variant generator, walkthrough
docs/                 project page (GitHub Pages)
```

## Reproducing the paper

**Public benchmarks — fully scripted.** HarmBench, SOSBench, FORTRESS, WMDP,
MMLU, GSM8K, and IFEval (plus StrongREJECT, AdvBench, and AILuminate) are
fetched from their canonical public sources — commit-pinned and
sha256-verified — by `bash scripts/data/download_public.sh`, which also
materializes the exact `data/eval/` files the pipeline reads (including the
converted FORTRESS CBRNE slice and the frozen seed-17 AdvBench/Alpaca splits,
byte-identical to the paper's) for the `scripts/line_c*` /
`scripts/ops/run_lm_eval.sh` stages. See `scripts/data/README.md` for the
full consumer-to-path map. Each dataset ships under its own license; the
script only automates the fetch.

**The hazardous corpus — on request.** The CBRN-domain association pool,
decoy corpora, and hazardous evaluation outputs used in the paper are **not
distributed** in this repository and never will be. Four artifact classes are
withheld by design: decoy corpora and elicited payload text, attacked
checkpoints, attack specifications beyond the public recipes, and defended
checkpoints. Verified researchers at recognized institutions may request the
gated appendix and redacted study materials for replication purposes; contact
the author (see the paper). The `demo/` domain reproduces every file contract
of the real corpus on harmless synthetic data, so the full pipeline is runnable
and auditable without any hazardous material.

**End-to-end reproduction.** The gated bundle is needed *only* here — to
byte-reproduce the paper's exact experiments (its corpus inputs, splits, and
pinned attack state); defending your own model requires nothing from it.
[`docs/REPRODUCING.md`](docs/REPRODUCING.md) is the step-by-step walkthrough
for reproducing the paper's primary-model (Qwen3-14B) results from this
repository plus the bundle: hardware,
judge setup, bundle staging and hash verification, the single run command,
per-stage expected values and wall-clock, and the acceptance bands that
count as a successful replication.

**Numeric artifacts.** Every reported number ships with its numeric verdict
artifact — scores, verdicts, counts, and confidence intervals, no generation
text — together with the split / corpus / attack-spec manifests (with hashes),
so all statistics are recomputable without access to hazardous content. See
[`results/README.md`](results/README.md) for the redaction contract and the
mapping from paper tables to artifact files; the hashes let holders of the
gated bundle verify byte-exact correspondence.

**Attack model.** Attack feasibility in the paper is defined by *public*
abliteration recipes; `scripts/line_b0_attack*.py` contains the
behaviorally-gated reproductions used as the attacked reference states.

## Citation

```bibtex
@article{russinovich2026foolsgold,
  title   = {Fool's Gold: Defensive Deception Against Safety-Removal
             Attacks on Open-Weight Models},
  author  = {Russinovich, Mark},
  journal = {arXiv preprint arXiv:2608.17202},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.17202}
}
```

## Responsible use

This is defensive safety research. The repository contains **no hazardous
data**: no CBRN prompts, no harmful model outputs, and no trained decoy
corpora — the only included domain is fictional and harmless by
construction, and the pipeline's content-hygiene rule (log ids, counts, and
scores — never generation text) is enforced throughout the code. The attack
implementations included here reproduce *already-public* abliteration
recipes solely to measure the defense; do not use them to strip safety
behavior from models you do not own or study. If you believe you have found
a way to reliably defeat this defense, we ask that you disclose it to the
author before publishing operational details.

## License

Code, configs, docs, and the synthetic demo data are released under the
[MIT License](LICENSE). Public benchmark datasets downloaded by the scripts
remain under their original licenses.

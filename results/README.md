# Numeric verdict artifacts

This directory carries the numeric artifacts behind every number reported in
the paper and in the top-level README — scores, verdicts, counts, gate
readings, and confidence intervals — for the seven evaluated models. It
contains **no generation text, no prompt text, and no element text**: every
artifact here is a redacted copy of the corresponding artifact in the
access-controlled study record, produced by
[`scripts/data/make_public_artifacts.py`](../scripts/data/make_public_artifacts.py)
under a strict whitelist.

## Redaction contract

Kept as-is:

- numbers, booleans, nulls, counts, and confidence-interval arrays;
- hex digests (sha256);
- enumerated verdict/gate/condition tokens from a fixed allowlist
  (`PASS`, `FAIL`, `holdout`, `A_anchor`, …);
- id-like strings (bounded charset, no whitespace, containing a digit —
  prompt/sample ids such as `sala-723`, checkpoint tags, timestamps) and
  slash-delimited internal artifact paths;
- estimator labels whose every alphabetic token comes from a fixed
  statistical vocabulary (e.g. `fully_usable(C, K=64)`).

Every other string leaf — prompts, model answers, judge rationales, free-text
notes, and any benchmark id that is itself a descriptive phrase (these are
denied as a class, digits or not) — is replaced by a stub:

```json
{"redacted": true, "sha256": "<sha256 of the utf-8 text>", "chars": 123}
```

Dictionary **keys** are held to the same contract: a key is kept only if it
passes the leaf whitelist or is a structural field name built entirely from
an enumerated schema vocabulary; any other key (e.g. a descriptive
benchmark-slug id used as a per-prompt map key) is replaced by a
deterministic `redacted_key_<sha256[:12]>`, so joins across files survive.

The emitter refuses to write any other string longer than 64 characters, and
`--verify` independently re-checks every emitted leaf **and key** (length
bound, whitelist membership, and a common-English-word screen). To re-run
the check:

```bash
python3 scripts/data/make_public_artifacts.py --out-root results --verify
```

`MANIFEST.json` lists every file with the sha256 of its **unredacted**
source. Together with the per-string stubs, this lets a holder of the
gated-access study materials (see the top-level README) verify byte-exact
correspondence between this public record and the full record, and lets
every statistic be recomputed from verdict-level data without any hazardous
content. Files are compact JSON (no indentation).

## Directory map (paper table → artifact)

Condition codes used throughout: `M0` = original model, clean; `A_anchor` =
original model, attacked; `D` = defended model, clean; `C` = defended model,
attacked. Strata: `trained` / `holdout` / `benign`.

| Directory | Contents / paper mapping |
|---|---|
| `evals/` | Four-condition evaluation summary per model, selected checkpoint (per-draw judged verdicts included; prompts are hash stubs). Main-results table: **Fatal** = `conditions.C.holdout.frac_fatal`, **Floor** = `conditions.A_anchor.holdout.frac_fatal`, clean-state refusal = `conditions.D.holdout.refusal`, benign shift = `gates.benign_denial.delta`; the registered gate readings are under `gates`. |
| `cis/` | 95% confidence intervals for reported cells (10k-rep prompt-cluster bootstrap; Wilson where noted). `cis_v21.json` = main/consensus/selection cells; `cis_v21_hzr3.json` = the presented 122B checkpoint; `cis_untouched.json` = untouched-stratum replications; `cis_glm_testset.json` = GLM-4.5-Air test-split cells. Each cell records point value, CI, method, unit, and its source artifact path. |
| `testset/` | Frozen-test-split and untouched-stratum replications (the main table's *Confirmatory* column and its appendix) plus per-model frozen-split id manifests. Untouched-stratum files: `gemma4_31b__*_untouched`, `gpt_oss_20b__*_untouched`. The 14B files include the registered K=64 test-split consensus run. |
| `ksweep_v21/` | Registered element-consensus sweeps (K = 4/16/64): outcome triples c/w/n, accepted ids, precision — defended arm (`__C`) and undefended calibration (`__A_anchor`) per attacked build. Backs the fu@64 and c/w/n columns and the consensus-attack section. `qwen3-14b-r4__n74__*` is the widened complete-validation-stratum run (precision 0.333). |
| `element_recon/` | Element-reconstruction summaries (element recovery and fully-usable rates with CIs, both arms, unfloored population). For the selected 14B checkpoint the cell of record is the `xv2rescore` file (corrected extraction scoring); the `clv` file is the earlier scoring pass, retained for transparency. |
| `gsm8k/` | Capability-gate readouts (GSM8K) for original and defended weights, per model's registered harness. |
| `benign_cert/` | High-n benign-behavior certificates (models on which the certificate was run). |
| `external/` | StrongREJECT / HarmBench CBRNE-adjacent-slice cells (attacker-perceived quality vs. fatal rate at matched quality) for all four conditions per model. |
| `sft_repair/` | Benign-data supervised fine-tuning (repair counter-attack): per-draw fatal artifacts at the 300-step and 3,000-step budgets and the per-draw true-element-share table (0.516 undefended vs. ~0.16 at every repair budget). |
| `wbprobe/` | White-box activation-probe readouts (per-answer fatality AUROC, retention economics, probe-routed consensus outcomes). |
| `audit/` | Test-split decision-invariance audit: every registered gate/stop/selection statistic recomputed with frozen-split ids excluded. |
| `manifests/` | Corpus / split / attack-specification manifests: split id lists, corpus pool provenance (sha256 + row counts), corpus gate reports (counts by hazard axis and split), attack acceptance readouts (refusal/compliance/degeneracy numbers for the accepted attacked states), and consensus generation-identity manifests. |

## Scope and trimming disclosures

- Only the seven models presented in the paper are mirrored. Consensus
  sweeps and reconstruction runs for superseded or withheld checkpoints,
  and calibration variants excluded from the paper by standing policy, are
  not mirrored (they remain in the study record).
- Some upstream benchmark prompt ids are descriptive phrases; the contract
  redacts them like any other text — as leaves and as map keys (the stub
  hash still allows joining against the public benchmark).
- Two models (`gemma4_31b`, `gpt_oss_20b`) have no pool-provenance manifest
  in this mirror; their corpus gate reports and split manifests ship
  instead. The original-weights GSM8K readout for `gpt_oss_20b` was
  produced on cluster storage and is not mirrored here (its defended-weights
  readout is); the value is reported in the paper's consolidated
  benchmark table.
- The whole directory is ≈12 MB; nothing was down-sampled — the consensus
  sweeps are complete as registered.

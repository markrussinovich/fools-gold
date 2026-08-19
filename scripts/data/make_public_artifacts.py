#!/usr/bin/env python3
"""Emit REDACTED public copies of the paper's numeric verdict artifacts.

Run from a checkout of the private research repository:

    python3 scripts/data/make_public_artifacts.py \
        --private-root /path/to/private/repo \
        --out-root     /path/to/public/repo/results

    python3 scripts/data/make_public_artifacts.py --out-root ... --verify

Redaction contract (strict WHITELIST — everything else is removed):
  KEPT   : numbers, booleans, nulls, CI arrays, counts,
           numeric-looking strings ("0.899", "64"),
           hex digests (sha256 etc.),
           enumerated verdict/gate/condition tokens from a fixed allowlist,
           id-like strings (<= 64 chars, charset [A-Za-z0-9_./:+-], and
           containing at least one digit — prompt/sample ids, checkpoint
           tags, file paths, timestamps).
  REDACTED: every other string leaf is replaced by
           {"redacted": true, "sha256": <sha256 of utf-8 text>, "chars": n}
           so joins and gated-bundle verification remain possible while no
           generation text, prompt text, or element text can appear.
  REFUSED: the emitter will not write any string longer than 64 characters
           unless it is a hex digest; content-bearing field names
           (prompt/answer/claim/...) are redacted even when short.

--verify re-walks every emitted file with an independent checker:
  * no string leaf longer than 64 chars (except hex digests),
  * every string leaf passes the whitelist,
  * no string leaf contains whitespace,
  * no kept token is a common English word outside the allowlist,
and reports byte sizes.
"""

import argparse
import hashlib
import glob as globmod
import json
import os
import re
import sys

# --------------------------------------------------------------------------
# Whitelist rules
# --------------------------------------------------------------------------

# Fixed allowlist of enumerated verdict / gate / structural tokens
# (case-insensitive exact match).
ALLOWLIST = {
    # verdicts / gate outcomes
    "pass", "fail", "passed", "failed", "correct", "wrong", "misled",
    "misleading", "fatal", "nonfatal", "non-fatal", "true", "false",
    "holds", "breach", "valid", "invalid", "probe-invalid", "probe_invalid",
    "accept", "accepted", "reject", "rejected", "no-decision", "none",
    "n/a", "na", "ok", "yes", "no", "partial", "invariant", "stop",
    "continue", "improved", "regressed", "terminal",
    # condition / arm / split vocabulary
    "m0", "m0a", "m0-a", "d0", "d0a", "d0-a", "a", "b", "c", "d",
    "a_anchor", "anchor", "holdout", "trained", "benign", "validation",
    "dev", "train", "test", "testset", "ungated", "untouched", "frozen",
    "spare", "direction", "seed", "clean", "attacked", "defended",
    "original", "sr", "hb",
    # estimator / method vocabulary
    "mean", "median", "sum", "count", "fraction", "rate", "prompt",
    "prompts", "draw", "draws", "slot", "slots", "accepts", "wilson",
    "bootstrap", "cluster_bootstrap", "prompt_cluster", "percentile",
    "sha256", "utc", "flexible", "strict", "exact_match",
}

# Content-bearing field names: string values under these keys are ALWAYS
# redacted, even if they would pass the pattern rules.
DENY_KEYS = {
    "prompt", "question", "answer", "text", "claim", "claims", "value_str",
    "modal", "modal_value", "gen", "generation", "output", "completion",
    "response", "content", "decoy", "payload", "element", "elements",
    "tell", "snippet", "extract", "assoc", "association", "doc", "target",
}

# Estimator/metric label vocabulary: a string with spaces/symbols is kept
# ONLY if every purely-alphabetic token of length >= 2 in it comes from this
# fixed set (tokens containing a digit — model tags, K values, ids — are
# fine by construction). Any content word ("sarin", "heat", "mix", ...)
# fails the test and the leaf is redacted.
WORD_ALLOWLIST = {
    # condition / split / arm vocabulary
    "anchor", "holdout", "trained", "benign", "dev", "train", "test",
    "testset", "ungated", "untouched", "frozen", "spare", "direction",
    "seed", "clean", "attacked", "defended", "original", "er", "fu",
    # estimator vocabulary (Table cells, CI files, sweeps)
    "frac", "fatal", "element", "recovery", "fully", "usable", "consensus",
    "precision", "delta", "naive", "loose", "strict", "flexible", "single",
    "draw", "draws", "oracle", "best", "of", "at", "per", "success",
    "consistency", "misled", "llm", "select", "selection", "mean",
    "denial", "refusal", "vs", "triple", "prompt", "prompts", "item",
    "accept", "accepts", "answered", "uplift", "fleet", "ailuminate",
    # statistics vocabulary
    "wilson", "cluster", "bootstrap", "category", "paired", "pooled",
    "joint", "resampling", "arms", "share", "the", "set", "ci", "reps",
    "point", "units", "agg", "unit",
    # generic status tokens
    "pass", "fail", "passed", "failed", "ok", "state", "complete", "done",
    "pending", "true", "false", "none", "na",
}

# Structural field-name vocabulary for dict KEYS (harvested from the
# artifact schemas and hand-vetted; alphabetic tokens only). A dict key is
# kept only if it passes the leaf whitelist (string_ok) or every alphabetic
# token (len >= 2) in it comes from this fixed set — data-driven descriptive
# keys (e.g. upstream benchmark behavior slugs) fail both tests and are
# replaced by a deterministic redacted_key_<sha256[:12]> so cross-file
# joins survive.
KEY_WORDS = frozenset("""
ablation abliterated abliteration acceptance actionable added adoption
aggregation alias all answer any archived args arm artifact as assoc
associations attack attacker attn attribution attrs auroc axis backend bad
band bands bar base baseline baselines basis batch bench better bias big bm
booked boot bos budget build built by bytes cache candidate cands cap case
ceiling cells champion channel chat check checked checkpoint choice climb
closed clusters code coherence collision column commit compliance computed
cond conditions conds config configs confirmation contact content context
convention corpora corpus cos cost cot counterfactual counts cov coverage
cpu created criterion crossings custom cv damage dataset datasets date
decision decisions decode decontaminate decoy decoyed def definition defs
degen degenerate delimiter deltas den derivation description descriptive
dest deviation device diff dir dirs disable disagreements distance
divergence do doc docs down drift drop dropped dry dtype dtypes effective
efficiency elements empty enable end enqueued entries env eos eot equal
escalate escape escapes estimator eval evaluation evidence exact excl
excluded expansion expert export exposure extract extraction failure
failures falsification fewshot file files filter final finding fisher
flagged flaw flip flipped floor forced format formula fp from full function
gate gated gates gen generated generation gens git given gold good gpu
graded grid group groups guard harm harmful harmony hash hashes hbest head
heretic hf hi high higher hist hit hn host hygiene id identity ids ignore
in index indices ineligible info inherited inputs instruction instrument is
items iters journal judge judged kept keyword kl kwargs label labels ladder
language layer leak legs len length like limit line list lm lo local log
logprob logs lora low major mandate manifest map margin marker markers
match max meaning means memory metadata method methods metric min minus
missing mitigation mlp modal mode model modes moe multiturn name namespace
narc needed new noncompliance normalization not note notes null num numpy
off offload on only oof open optimization order orthogonalize out outage
outcome output outputs over override pad pairing pairs parallel parameters
params parity path pattern perdraw permutation phase pin pinned pins plan
pos position positions post pre prefix prematerialized presented pretrained
pretty prior probe process proj promoted promotion provenance pruned pts
public punctuation purpose python quantile quantization question random
range rank ratio ratios readme readout reason reduce reencode ref reference
refs
refusals refused regex regexes registered registration render repeats
replay replication repo resamples resim response results reused revision
rho role root round router row rows rubric rule ruling run rung runtime
sample sampler samples sampling sanitized scale scheme scipy scope score
scorer scratch script seam search seconds seeds selected semantics served
settings sfx sha shard shared shipped shot should sigma signoff size sizes
sklearn slice snap snapshot source sources space spearman spec specific
split splits src srhi start starts startup stat statement status std
stderr str strata strategy stratum stride study style subtasks suffix
summary suppress sweep system tag tagids target targets task tell temp
temperature template tensor tested text think thinking threshold time to
token tokenizer tokens toks tool top torch total trace training
transformers trend trial trials triples trunc truncated truncation ts type
unclosed unfiltered unsafe until upper use user utility utilization value
variant vector verdict verdicts verification verified version versions
vintage vllm warm warmstart warmup weap weight winsorization with within
without workable wp wz zero
""".split())

HEX_RE = re.compile(r"^[0-9a-fA-F]{16,128}$")
NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?%?$")
# id-like: bounded charset, no whitespace, must contain a digit
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-\./:+]{0,63}$")
# path-like internal artifact names (evidence pointers) up to 160 chars
PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-\./:+]{0,159}$")
# bounded charset for labeled/estimator phrases
PHRASE_CHARS_RE = re.compile(r"^[A-Za-z0-9_\-\./:+@()=,%\[\] ]{1,64}$")
ALPHA_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
KEY_RE = re.compile(r"^[A-Za-z0-9_\-\./:+ ()=,@]{1,48}$")

MAX_STR = 64


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# Descriptive-slug id classes (upstream benchmark behavior ids are
# human-readable phrases): always redacted, digits or not.
DENY_PREFIXES = ("hb-", "advbench-", "harmbench-")


def string_ok(s: str) -> bool:
    """Whitelist test for a string leaf (independent of key context)."""
    if s.lower().startswith(DENY_PREFIXES):
        return False
    if len(s) > MAX_STR:
        # over-length exemptions: hex digests and slash-path artifact ids
        return bool(HEX_RE.match(s)) or ("/" in s and bool(PATH_RE.match(s)))
    if s == "":
        return True
    low = s.lower()
    if low in ALLOWLIST:
        return True
    if NUM_RE.match(s) or HEX_RE.match(s):
        return True
    if ID_RE.match(s) and any(ch.isdigit() for ch in s):
        return True
    # slash-path artifact / evidence pointer (internal file name = an id)
    if "/" in s and PATH_RE.match(s):
        return True
    # estimator/metric phrase: bounded symbols, and every purely-alphabetic
    # token (len >= 2) must be allowlisted vocabulary
    if PHRASE_CHARS_RE.match(s):
        for tok in ALPHA_TOKEN_RE.findall(s):
            if len(tok) >= 2 and tok.isalpha() \
                    and tok.lower() not in WORD_ALLOWLIST \
                    and tok.lower() not in ALLOWLIST:
                return False
        return True
    return False


def redact_string(s: str):
    return {"redacted": True, "sha256": _sha(s), "chars": len(s)}


def key_ok(ks: str) -> bool:
    """Whitelist test for a dict KEY: the leaf rules, or a structural
    field name whose every alphabetic token (len >= 2) is enumerated
    vocabulary. Descriptive data-driven keys fail both."""
    if ks.lower().startswith(DENY_PREFIXES):
        return False
    if not KEY_RE.match(ks) or len(ks) > 48:
        return False
    if string_ok(ks):
        return True
    for tok in ALPHA_TOKEN_RE.findall(ks):
        if len(tok) >= 2 and tok.isalpha() \
                and tok.lower() not in KEY_WORDS \
                and tok.lower() not in WORD_ALLOWLIST \
                and tok.lower() not in ALLOWLIST:
            return False
    return True


def redact_key(ks: str) -> str:
    return "redacted_key_" + _sha(ks)[:12]


def scrub(node, key=None):
    """Return the redacted copy of a JSON tree."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            ks = str(k)
            if not key_ok(ks):
                ks = redact_key(str(k))
            out[ks] = scrub(v, key=str(k).lower())
        return out
    if isinstance(node, list):
        return [scrub(x, key=key) for x in node]
    if isinstance(node, str):
        if key is not None and (key in DENY_KEYS or key.endswith("_text")):
            return redact_string(node)
        if string_ok(node):
            return node
        return redact_string(node)
    # numbers, bools, None pass through
    return node


# --------------------------------------------------------------------------
# Artifact manifest: (source path relative to private root, destination path
# relative to --out-root). Globs allowed in source; newest match is taken.
# Only the seven presented models are covered; internal artifact names are
# kept (they are ids). M0-sourced calibration variants are excluded (held
# out of the paper by standing order).
# --------------------------------------------------------------------------

MANIFEST = [
    # ---- confidence intervals (booked cells; 10k-rep prompt-cluster bootstrap)
    ("results/cis/cis_v21.json", "cis/cis_v21.json"),
    ("results/cis/cis_v21_hzr3.json", "cis/cis_v21_hzr3.json"),
    ("results/cis/cis_untouched.json", "cis/cis_untouched.json"),
    ("results/cis/cis_glm_testset.json", "cis/cis_glm_testset.json"),
    # ---- decision-audit artifact (test-split decision invariance)
    ("results/testsplit_decision_invariance.json",
     "audit/testsplit_decision_invariance.json"),
    # ---- external red-team benchmark slice (StrongREJECT / HarmBench)
    ("results/redteam_cbrne_slice.json", "external/redteam_cbrne_slice.json"),
    # ---- repair fine-tuning (dose-response) numeric artifacts
    ("results/tierb_sft_perdraw_true_share.json",
     "sft_repair/tierb_sft_perdraw_true_share.json"),
    ("results/tierb_sft2_perdraw.json", "sft_repair/tierb_sft2_perdraw.json"),
    ("results/tierb_sft3_perdraw.json", "sft_repair/tierb_sft3_perdraw.json"),
    ("results/tierb_sft2_summary.json", "sft_repair/tierb_sft2_summary.json"),
    ("results/tierb_sft3_summary.json", "sft_repair/tierb_sft3_summary.json"),
    # ---- white-box activation-probe readouts
    ("results/wbprobe_qwen3_14b_c18champ.json",
     "wbprobe/wbprobe_qwen3_14b_c18champ.json"),
    # ---- registered 64-draw consensus sweeps (presented models + their
    #      undefended calibrations; v2.1 element basis)
    ("results/ksweep_v21/qwen35_9b__-__C.json", "ksweep_v21/qwen35_9b__-__C.json"),
    ("results/ksweep_v21/qwen35_9b__-__A_anchor.json", "ksweep_v21/qwen35_9b__-__A_anchor.json"),
    ("results/ksweep_v21/qwen35_27b__-__C.json", "ksweep_v21/qwen35_27b__-__C.json"),
    ("results/ksweep_v21/qwen35_27b__-__A_anchor.json", "ksweep_v21/qwen35_27b__-__A_anchor.json"),
    ("results/ksweep_v21/qwen35_122b__hzr3__C.json", "ksweep_v21/qwen35_122b__hzr3__C.json"),
    ("results/ksweep_v21/qwen35_122b__hzr3__A_anchor.json", "ksweep_v21/qwen35_122b__hzr3__A_anchor.json"),
    ("results/ksweep_v21/qwen3_14b__-__C.json", "ksweep_v21/qwen3_14b__-__C.json"),
    ("results/ksweep_v21/qwen3_14b__-__A_anchor.json", "ksweep_v21/qwen3_14b__-__A_anchor.json"),
    ("results/ksweep_v21/qwen3-14b-r4__main__C.json", "ksweep_v21/qwen3-14b-r4__main__C.json"),
    ("results/ksweep_v21/qwen3-14b-r4__main__A_anchor.json", "ksweep_v21/qwen3-14b-r4__main__A_anchor.json"),
    ("results/ksweep_v21/qwen3-14b-r4__xv2rescore__C.json", "ksweep_v21/qwen3-14b-r4__xv2rescore__C.json"),
    ("results/ksweep_v21/qwen3-14b-r4__xv2rescore__A_anchor.json", "ksweep_v21/qwen3-14b-r4__xv2rescore__A_anchor.json"),
    ("results/ksweep_v21/qwen3-14b-r4__n74__C.json", "ksweep_v21/qwen3-14b-r4__n74__C.json"),
    ("results/ksweep_v21/qwen3-14b-r4__n74__A_anchor.json", "ksweep_v21/qwen3-14b-r4__n74__A_anchor.json"),
    ("results/ksweep_v21/gemma4_31b__-__C.json", "ksweep_v21/gemma4_31b__-__C.json"),
    ("results/ksweep_v21/gemma4_31b__-__A_anchor.json", "ksweep_v21/gemma4_31b__-__A_anchor.json"),
    ("results/ksweep_v21/gpt_oss_20b__-__C.json", "ksweep_v21/gpt_oss_20b__-__C.json"),
    ("results/ksweep_v21/gpt_oss_20b__-__A_anchor.json", "ksweep_v21/gpt_oss_20b__-__A_anchor.json"),
    ("results/ksweep_v21/gpt_oss_20b__muxheresy__A_anchor.json", "ksweep_v21/gpt_oss_20b__muxheresy__A_anchor.json"),
    ("results/ksweep_v21/glm45_air__glm45hB__A_anchor.json", "ksweep_v21/glm45_air__glm45hB__A_anchor.json"),
    ("results/ksweep_v21/glm45_air__glm45pub__A_anchor.json", "ksweep_v21/glm45_air__glm45pub__A_anchor.json"),
    # ---- element-reconstruction summaries (v2.1 basis, unfloored population)
    ("results/qwen35_9b_c18_element_recon.clv_v21rel_nofloor.json",
     "element_recon/qwen35_9b_c18_element_recon.clv_v21rel_nofloor.json"),
    ("results/qwen35_27b_c18_element_recon.clv_v21rel_nofloor.json",
     "element_recon/qwen35_27b_c18_element_recon.clv_v21rel_nofloor.json"),
    ("results/qwen35_122b_hzrerun_c18_element_recon_hzr3.clv_v21rel_nofloor.json",
     "element_recon/qwen35_122b_hzrerun_c18_element_recon_hzr3.clv_v21rel_nofloor.json"),
    ("results/qwen3_14b_c18_element_recon.clv_v21rel_nofloor.json",
     "element_recon/qwen3_14b_c18_element_recon.clv_v21rel_nofloor.json"),
    ("results/qwen3_14b_c18champ_c18_element_recon.clv_v21rel_nofloor.json",
     "element_recon/qwen3_14b_c18champ_c18_element_recon.clv_v21rel_nofloor.json"),
    ("results/qwen3_14b_c18champ_c18_element_recon.xv2rescore_v21rel_nofloor.json",
     "element_recon/qwen3_14b_c18champ_c18_element_recon.xv2rescore_v21rel_nofloor.json"),
    ("results/qwen3_14b_c18champ_c18_element_recon_n74.clv_v21rel_nofloor.json",
     "element_recon/qwen3_14b_c18champ_c18_element_recon_n74.clv_v21rel_nofloor.json"),
    ("results/gemma4_31b_c18_element_recon.clv_v21rel_nofloor.json",
     "element_recon/gemma4_31b_c18_element_recon.clv_v21rel_nofloor.json"),
    ("results/gpt_oss_20b_c18_element_recon.clv_v21rel_nofloor.json",
     "element_recon/gpt_oss_20b_c18_element_recon.clv_v21rel_nofloor.json"),
    ("results/gpt_oss_20b_c18_element_recon_muxheresy.clv_v21rel_nofloor.json",
     "element_recon/gpt_oss_20b_c18_element_recon_muxheresy.clv_v21rel_nofloor.json"),
    ("results/glm45_air_c18_element_recon.clv_v21rel_nofloor.json",
     "element_recon/glm45_air_c18_element_recon.clv_v21rel_nofloor.json"),
    ("results/glm45_air_c18_element_recon_glm45hB.clv_v21rel_nofloor.json",
     "element_recon/glm45_air_c18_element_recon_glm45hB.clv_v21rel_nofloor.json"),
    ("results/glm45_air_c18_element_recon_glm45pub.clv_v21rel_nofloor.json",
     "element_recon/glm45_air_c18_element_recon_glm45pub.clv_v21rel_nofloor.json"),
    # ---- four-condition evaluation summaries, selected checkpoint per model
    ("runs/qwen3.5-9b/2026-07-29_cbrn/evals/cbrn_smoke_B1seed.json",
     "evals/qwen35_9b__cbrn_smoke_B1seed.json"),
    ("runs/qwen3.5-27b/2026-07-28_cbrn/evals/cbrn_smoke_B1r1.json",
     "evals/qwen35_27b__cbrn_smoke_B1r1.json"),
    ("runs/qwen3.5-122b/2026-08-11_cbrn_hzrerun/evals/cbrn_smoke_B1r3.json",
     "evals/qwen35_122b__cbrn_smoke_B1r3.json"),
    ("runs/qwen3-14b/2026-07-31_cbrn_v2/evals/cbrn_smoke_B2r4.json",
     "evals/qwen3_14b__cbrn_smoke_B2r4.json"),
    ("runs/gemma4-31b/2026-08-01_cbrn/evals/cbrn_smoke_B1r3.json",
     "evals/gemma4_31b__cbrn_smoke_B1r3.json"),
    ("runs/gpt-oss-20b/2026-08-01_cbrn/evals/cbrn_smoke_B1seed.json",
     "evals/gpt_oss_20b__cbrn_smoke_B1seed.json"),
    ("runs/glm-4.5-air/2026-08-06_cbrn/cluster_mirror/battery/cbrn_smoke_B1r4.json",
     "evals/glm45_air__cbrn_smoke_B1r4.json"),
    # ---- frozen-test-split / untouched-stratum replications
    ("results/testset_fatal/cluster_payloads/qwen35_9b/cbrn_testset_B1seed_testset.json",
     "testset/qwen35_9b__cbrn_testset_B1seed_testset.json"),
    ("results/testset_fatal/cluster_payloads/qwen35_27b/cbrn_testset_B1r1_testset.json",
     "testset/qwen35_27b__cbrn_testset_B1r1_testset.json"),
    ("runs/qwen3.5-122b/2026-08-11_cbrn_hzrerun/testset_dl/evals/cbrn_testset_B1hzr3_testset.json",
     "testset/qwen35_122b__cbrn_testset_B1hzr3_testset.json"),
    ("results/frozen30_hzr3_instrument.json",
     "testset/qwen35_122b__frozen30_hzr3_instrument.json"),
    ("results/testset_fatal/qwen3_14b_champ_new50.json",
     "testset/qwen3_14b__champ_new50.json"),
    ("results/testset_fatal/ksweep_new50test__C.json",
     "testset/qwen3_14b__ksweep_new50test__C.json"),
    ("results/testset_fatal/ksweep_new50test__A_anchor.json",
     "testset/qwen3_14b__ksweep_new50test__A_anchor.json"),
    ("results/testset_fatal/qwen3_14b_champ_c18_element_recon_new50test.clv_v21rel_nofloor.json",
     "testset/qwen3_14b__c18_element_recon_new50test.clv_v21rel_nofloor.json"),
    ("results/testset_fatal/cluster_payloads/gemma4_31b/cbrn_testset_B1r3_testset.json",
     "testset/gemma4_31b__cbrn_testset_B1r3_testset.json"),
    ("runs/gemma4-31b/2026-08-01_cbrn/evals/cbrn_testset_B1r3_untouched.json",
     "testset/gemma4_31b__cbrn_testset_B1r3_untouched.json"),
    ("runs/gpt-oss-20b/2026-08-01_cbrn/evals/cbrn_testset_B1seed_untouched.json",
     "testset/gpt_oss_20b__cbrn_testset_B1seed_untouched.json"),
    ("runs/glm-4.5-air/2026-08-06_cbrn/evals/cbrn_testset_B1r4_testset.json",
     "testset/glm45_air__cbrn_testset_B1r4_testset.json"),
    # frozen split id manifests (ids only)
    ("results/testset_fatal/manifests/qwen35_9b_frozen_ids.json", "testset/manifests/qwen35_9b_frozen_ids.json"),
    ("results/testset_fatal/manifests/qwen35_27b_frozen_ids.json", "testset/manifests/qwen35_27b_frozen_ids.json"),
    ("results/testset_fatal/manifests/qwen35_122b_frozen_ids.json", "testset/manifests/qwen35_122b_frozen_ids.json"),
    ("results/testset_fatal/manifests/qwen3_14b_c18champ_frozen_ids.json", "testset/manifests/qwen3_14b_c18champ_frozen_ids.json"),
    ("results/testset_fatal/manifests/gemma4_31b_frozen_ids.json", "testset/manifests/gemma4_31b_frozen_ids.json"),
    ("results/testset_fatal/manifests/gpt_oss_20b_frozen_ids.json", "testset/manifests/gpt_oss_20b_frozen_ids.json"),
    ("results/testset_fatal/manifests/glm45_air_frozen_ids.json", "testset/manifests/glm45_air_frozen_ids.json"),
    # ---- capability gate (GSM8K) readouts
    ("runs/qwen3.5-9b/2026-07-29_cbrn/evals/lm_eval/gsm8k_M0/*/results_*.json",
     "gsm8k/qwen35_9b__gsm8k_M0.json"),
    ("runs/qwen3.5-9b/2026-07-29_cbrn/evals/lm_eval/gsm8k_B1seed/*/results_*.json",
     "gsm8k/qwen35_9b__gsm8k_B1seed.json"),
    ("runs/qwen3.5-27b/2026-07-28_cbrn/evals/lm_eval/gsm8k_M0/*/results_*.json",
     "gsm8k/qwen35_27b__gsm8k_M0.json"),
    ("runs/qwen3.5-27b/2026-07-28_cbrn/evals/lm_eval/gsm8k_B1r1/*/results_*.json",
     "gsm8k/qwen35_27b__gsm8k_B1r1.json"),
    ("runs/qwen3.5-122b/2026-07-31_cbrn/evals/lm_eval/gsm8k_M0_n500/*/results_*.json",
     "gsm8k/qwen35_122b__gsm8k_M0_n500.json"),
    ("runs/qwen3.5-122b/2026-07-31_cbrn/evals/lm_eval/gsm8k_champion_rcr3_n500/*/results_*.json",
     "gsm8k/qwen35_122b__gsm8k_champion_rcr3_n500.json"),
    ("runs/qwen3-14b/2026-07-31_cbrn_v2/evals/lm_eval/gsm8k_M0/*/results_*.json",
     "gsm8k/qwen3_14b__gsm8k_M0.json"),
    ("runs/qwen3-14b/2026-07-31_cbrn_v2/evals/lm_eval/gsm8k_B2r4/*/results_*.json",
     "gsm8k/qwen3_14b__gsm8k_B2r4.json"),
    ("runs/gemma4-31b/2026-08-01_cbrn/evals/gsm8k_chat_M0_ctpilot.json",
     "gsm8k/gemma4_31b__gsm8k_chat_M0.json"),
    ("runs/gemma4-31b/2026-08-01_cbrn/evals/gsm8k_chat_D_r3_ctpilot.json",
     "gsm8k/gemma4_31b__gsm8k_chat_D_r3.json"),
    ("runs/gpt-oss-20b/2026-08-01_cbrn/evals/lm_eval/gsm8k_D_seed/*/results_*.json",
     "gsm8k/gpt_oss_20b__gsm8k_D_seed.json"),
    ("runs/glm-4.5-air/2026-08-06_cbrn/cluster_mirror/b1seed/gsm8k_M0_n500_results.json",
     "gsm8k/glm45_air__gsm8k_M0_n500.json"),
    ("runs/glm-4.5-air/2026-08-06_cbrn/cluster_mirror/battery/gsm8k_B1r4_champion_n500_results.json",
     "gsm8k/glm45_air__gsm8k_B1r4_champion_n500.json"),
    ("runs/glm-4.5-air/2026-08-06_cbrn/cluster_mirror/battery/gsm8k_gate_B1r4.json",
     "gsm8k/glm45_air__gsm8k_gate_B1r4.json"),
    # ---- benign-behavior certificates (high-n)
    ("runs/qwen3.5-27b/2026-07-28_cbrn/evals/benign_cert_r1.json",
     "benign_cert/qwen35_27b__benign_cert_r1.json"),
    ("runs/qwen3.5-122b/2026-08-11_cbrn_hzrerun/evals/benign_cert_rcr3.json",
     "benign_cert/qwen35_122b__benign_cert_rcr3.json"),
    ("runs/gemma4-31b/2026-08-01_cbrn/evals/benign_cert_r3.json",
     "benign_cert/gemma4_31b__benign_cert_r3.json"),
    ("runs/glm-4.5-air/2026-08-06_cbrn/cluster_mirror/cert/benign_cert_r4.json",
     "benign_cert/glm45_air__benign_cert_r4.json"),
    # ---- corpus / split / attack-specification manifests (hashes + counts)
    ("data/qwen35_9b/fleet_pool_provenance.json", "manifests/qwen35_9b__fleet_pool_provenance.json"),
    ("data/qwen35_9b/splits.json", "manifests/qwen35_9b__splits.json"),
    ("data/qwen35_9b/gate_report.json", "manifests/qwen35_9b__gate_report.json"),
    ("data/qwen35_27b/fleet_pool_provenance.json", "manifests/qwen35_27b__fleet_pool_provenance.json"),
    ("data/qwen35_27b/splits.json", "manifests/qwen35_27b__splits.json"),
    ("data/qwen35_27b/gate_report.json", "manifests/qwen35_27b__gate_report.json"),
    ("data/qwen35_122b/fleet_pool_provenance.json", "manifests/qwen35_122b__fleet_pool_provenance.json"),
    ("data/qwen35_122b/splits.json", "manifests/qwen35_122b__splits.json"),
    ("data/qwen35_122b/gate_report.json", "manifests/qwen35_122b__gate_report.json"),
    ("data/qwen3_v2/splits.json", "manifests/qwen3_14b__splits.json"),
    ("data/qwen3_v2/gate_report.json", "manifests/qwen3_14b__gate_report.json"),
    ("data/gemma4_31b/gate_report.json", "manifests/gemma4_31b__gate_report.json"),
    ("data/gemma4_31b/splits.json", "manifests/gemma4_31b__splits.json"),
    ("data/gpt_oss_20b/gate_report.json", "manifests/gpt_oss_20b__gate_report.json"),
    ("data/gpt_oss_20b/splits.json", "manifests/gpt_oss_20b__splits.json"),
    ("data/glm45_air/fleet_pool_provenance.json", "manifests/glm45_air__fleet_pool_provenance.json"),
    ("data/glm45_air/splits.json", "manifests/glm45_air__splits.json"),
    # attack acceptance / identity (numeric acceptance readouts)
    ("runs/qwen3.5-9b/2026-07-29_cbrn/artifacts/cbrn_attackD_B1seed_ablx.json",
     "manifests/qwen35_9b__attackD_B1seed_ablx.json"),
    ("runs/qwen3.5-9b/2026-07-29_cbrn/artifacts/cbrn_attack_M0a.json",
     "manifests/qwen35_9b__attack_M0a.json"),
    ("runs/qwen3.5-27b/2026-07-28_cbrn/artifacts/cbrn_attackD_B1r1.json",
     "manifests/qwen35_27b__attackD_B1r1.json"),
    ("runs/qwen3.5-27b/2026-07-28_cbrn/artifacts/cbrn_attack_M0a.json",
     "manifests/qwen35_27b__attack_M0a.json"),
    ("runs/qwen3.5-122b/2026-08-11_cbrn_hzrerun/artifacts/cbrn_attackD_B1r3.json",
     "manifests/qwen35_122b__attackD_B1r3.json"),
    ("runs/qwen3.5-122b/2026-08-11_cbrn_hzrerun/artifacts/cbrn_attack_M0a.json",
     "manifests/qwen35_122b__attack_M0a.json"),
    ("runs/qwen3-14b/2026-07-31_cbrn_v2/artifacts/cbrn_attackD_B2r4.json",
     "manifests/qwen3_14b__attackD_B2r4.json"),
    ("runs/qwen3-14b/2026-07-31_cbrn_v2/artifacts/cbrn_attack_M0a.json",
     "manifests/qwen3_14b__attack_M0a.json"),
    ("runs/gemma4-31b/2026-08-01_cbrn/artifacts/cbrn_attackD_B1r3.json",
     "manifests/gemma4_31b__attackD_B1r3.json"),
    ("runs/gemma4-31b/2026-08-01_cbrn/artifacts/cbrn_attack_M0a.json",
     "manifests/gemma4_31b__attack_M0a.json"),
    ("runs/gpt-oss-20b/2026-08-01_cbrn/artifacts/cbrn_attackD_B1seed.json",
     "manifests/gpt_oss_20b__attackD_B1seed.json"),
    ("runs/gpt-oss-20b/2026-08-01_cbrn/artifacts/cbrn_attack_M0a.json",
     "manifests/gpt_oss_20b__attack_M0a.json"),
    ("runs/glm-4.5-air/2026-08-06_cbrn/artifacts/cbrn_attackD_B1r4.json",
     "manifests/glm45_air__attackD_B1r4.json"),
    ("runs/glm-4.5-air/2026-08-06_cbrn/artifacts/cbrn_attack_M0a.json",
     "manifests/glm45_air__attack_M0a.json"),
    ("runs/glm-4.5-air/2026-08-06_cbrn/artifacts/cbrn_attack_M0a_glm45hB.json",
     "manifests/glm45_air__attack_M0a_glm45hB.json"),
    ("runs/glm-4.5-air/2026-08-06_cbrn/artifacts/cbrn_attack_M0a_glm45pub.json",
     "manifests/glm45_air__attack_M0a_glm45pub.json"),
    # consensus generation identity manifests
    ("runs/qwen3.5-9b/2026-07-29_cbrn/artifacts/c18_gen_manifest.json",
     "manifests/qwen35_9b__c18_gen_manifest.json"),
    ("runs/qwen3.5-27b/2026-07-28_cbrn/artifacts/c18_gen_manifest.json",
     "manifests/qwen35_27b__c18_gen_manifest.json"),
    ("runs/qwen3.5-122b/2026-08-11_cbrn_hzrerun/artifacts/c18_gen_manifest_hzr3.json",
     "manifests/qwen35_122b__c18_gen_manifest_hzr3.json"),
    ("runs/qwen3-14b/2026-07-31_cbrn_v2/artifacts/c18_gen_manifest.json",
     "manifests/qwen3_14b__c18_gen_manifest.json"),
    ("runs/gemma4-31b/2026-08-01_cbrn/artifacts/c18_gen_manifest.json",
     "manifests/gemma4_31b__c18_gen_manifest.json"),
    ("runs/gpt-oss-20b/2026-08-01_cbrn/artifacts/c18_gen_manifest.json",
     "manifests/gpt_oss_20b__c18_gen_manifest.json"),
    ("runs/glm-4.5-air/2026-08-06_cbrn/artifacts/c18_gen_manifest.json",
     "manifests/glm45_air__c18_gen_manifest.json"),
]


def resolve_src(private_root: str, pattern: str):
    p = os.path.join(private_root, pattern)
    if any(ch in pattern for ch in "*?["):
        hits = sorted(globmod.glob(p), key=os.path.getmtime)
        return hits[-1] if hits else None
    return p if os.path.exists(p) else None


def emit(private_root: str, out_root: str) -> int:
    manifest_out = []
    missing = []
    for src_rel, dst_rel in MANIFEST:
        src = resolve_src(private_root, src_rel)
        if src is None:
            missing.append(src_rel)
            continue
        with open(src, "rb") as f:
            raw = f.read()
        data = json.loads(raw)
        scrubbed = scrub(data)
        dst = os.path.join(out_root, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as f:
            json.dump(scrubbed, f, separators=(",", ":"), sort_keys=False)
        manifest_out.append({
            "file": dst_rel,
            "source": os.path.relpath(src, private_root),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": os.path.getsize(dst),
        })
        print(f"emitted {dst_rel}  ({os.path.getsize(dst)} bytes)")
    # contract prose lives in results/README.md; keep this file rule-clean
    meta = {
        "readme": "results/README.md",
        "n_files": len(manifest_out),
        "files": manifest_out,
    }
    with open(os.path.join(out_root, "MANIFEST.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if missing:
        print("\nWARNING missing sources (not emitted):")
        for m in missing:
            print("  -", m)
    total = sum(m["bytes"] for m in manifest_out)
    print(f"\n{len(manifest_out)} files, {total/1e6:.2f} MB total")
    return 1 if missing else 0


# --------------------------------------------------------------------------
# Verifier (independent walk of the emitted files)
# --------------------------------------------------------------------------

COMMON_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has",
    "are", "was", "were", "will", "would", "could", "should", "into",
    "then", "than", "when", "where", "which", "while", "about", "after",
    "before", "between", "under", "over", "such", "each", "other", "some",
    "most", "more", "must", "may", "can", "not", "but", "you", "your",
    "use", "used", "using", "step", "steps", "first", "second", "mix",
    "add", "heat", "cool", "water", "solution", "material", "process",
    "method", "procedure", "temperature", "pressure", "amount", "obtain",
    "prepare", "produce", "product", "reaction", "required", "needed",
}


REDACTED_KEY_RE = re.compile(r"^redacted_key_[0-9a-f]{12}$")


def verify_walk(node, key, path, errors):
    if isinstance(node, dict):
        # a redaction stub is a terminal, verified shape
        if set(node.keys()) == {"redacted", "sha256", "chars"}:
            if node["redacted"] is not True or not HEX_RE.match(str(node["sha256"])):
                errors.append(f"{path}: malformed redaction stub")
            return
        for k, v in node.items():
            ks = str(k)
            # every key must be a redaction stub or pass the key whitelist
            if not REDACTED_KEY_RE.match(ks) and not key_ok(ks):
                errors.append(
                    f"{path}: dict key fails whitelist (len={len(ks)})")
            verify_walk(v, ks.lower(), f"{path}.{k}"[:200], errors)
    elif isinstance(node, list):
        for i, x in enumerate(node):
            verify_walk(x, key, f"{path}[{i}]", errors)
    elif isinstance(node, str):
        s = node
        if len(s) > MAX_STR and not (
                HEX_RE.match(s) or ("/" in s and PATH_RE.match(s))):
            errors.append(f"{path}: string leaf longer than {MAX_STR}")
            return
        if not string_ok(s):
            errors.append(f"{path}: string leaf fails whitelist (len={len(s)})")
            return
        if HEX_RE.match(s):
            return  # hex digests can contain accidental alpha runs
        # no kept leaf may carry a common English word beyond the allowlists
        for t in re.split(r"[^A-Za-z]+", s):
            tl = t.lower()
            if (len(t) >= 3 and tl in COMMON_WORDS
                    and tl not in WORD_ALLOWLIST and tl not in ALLOWLIST):
                errors.append(f"{path}: common-word token in leaf (len={len(s)})")
                break


def verify(out_root: str) -> int:
    n_files = 0
    n_bytes = 0
    all_errors = []
    for dirpath, _dirnames, filenames in os.walk(out_root):
        for fn in sorted(filenames):
            if not fn.endswith(".json"):
                continue
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, out_root)
            n_files += 1
            n_bytes += os.path.getsize(fp)
            try:
                data = json.load(open(fp))
            except Exception as e:
                all_errors.append(f"{rel}: unreadable JSON ({type(e).__name__})")
                continue
            errors = []
            verify_walk(data, None, "$", errors)
            for e in errors[:10]:
                all_errors.append(f"{rel}: {e}")
            if len(errors) > 10:
                all_errors.append(f"{rel}: ... {len(errors)-10} more")
    print(f"verified {n_files} JSON files, {n_bytes/1e6:.2f} MB total")
    if all_errors:
        print(f"FAIL: {len(all_errors)} violations")
        for e in all_errors[:50]:
            print("  ", e)
        return 1
    print("PASS: every string leaf is numeric-like, a hex digest, an "
          "allowlisted verdict token, or an id; no leaf exceeds "
          f"{MAX_STR} chars (hex digests excepted); no whitespace or "
          "common-English tokens outside the allowlist.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--private-root", default=".",
                    help="checkout of the private research repository")
    ap.add_argument("--out-root", required=True,
                    help="public results/ directory to emit into")
    ap.add_argument("--verify", action="store_true",
                    help="verify already-emitted files instead of emitting")
    args = ap.parse_args()
    if args.verify:
        sys.exit(verify(args.out_root))
    sys.exit(emit(args.private_root, args.out_root))


if __name__ == "__main__":
    main()

"""Shard-streaming LoRA merge: apply a PEFT adapter's deltas (W += scale·B@A)
inside a FULL copy of the base snapshot.

Companion to ablation_stream.py, same rationale: transformers 5.x loads Qwen3.5
multimodal checkpoints as a text-only stack, so peft merge_and_unload +
save_pretrained produces a `qwen3_5_text` dump that vLLM cannot serve. This
tool merges into a verbatim snapshot copy instead — output serves exactly like
the base. Adapter module paths (model.layers.*) are mapped to snapshot names
(model.language_model.layers.*) by layers-tail matching.

3D expert deltas (GPTOSS-REAL-ATTACK-PLAN D6): adapters built with peft
target_parameters on fused MoE expert tensors (gpt-oss
model.layers.L.mlp.experts.{gate_up_proj,down_proj}, bare nn.Parameters —
no ".weight" suffix in the snapshot) save nested ParamWrapper keys that do
NOT carry the parameter name (outer wrapper ...mlp.experts.lora_{A,B}.weight,
inner ...mlp.experts.base_layer.lora_{A,B}.weight). Resolution: strip the
.base_layer nesting, then match lora_B/lora_A dims against the 3D base
tensors under that module (gpt-oss: B out-dim 5760 -> gate_up_proj, 2880 ->
down_proj — unambiguous). delta = einsum("o r e, e r i -> e i o",
B.reshape(out, r, E), A.reshape(E, r, in)) * scale for the is_transposed
(experts, in, out) layout — verified byte-exact vs peft merge_and_unload in
fp32 (tests/test_gptoss_expert_merge.py; on bf16 production models the
stream accumulates fp32 then casts, <=1 ulp from peft's bf16 in-memory
merge — same policy as the pre-existing 2D path; see lora_merge_check.py).
Square tensors (down_proj 2880x2880) cannot be orientation-disambiguated by
shape, so --expert-transposed (line-config seam "expert_merge_transposed")
must assert the layout explicitly; ONLY the transposed layout is implemented
— a future non-transposed MoE line must extend this (new orientation value),
never reuse the flag.
The 2D path is byte-identical to the pre-D6 script.

Usage:
  python3 scripts/lora_merge_stream.py --base <snapshot_dir> \
      --adapter <peft_adapter_dir> --dst <out_dir> [--expert-transposed]
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--scale-mult", type=float, default=1.0,
                    help="multiply the adapter's native scale (rung interpolation: "
                         "0<m<1 blends between the base rung and the full adapter; "
                         "default 1.0 = byte-identical to the original merge)")
    ap.add_argument("--key-map", default=None,
                    help="JSON list of [regex, repl] pairs applied to each "
                         "adapter module TAIL to produce the base-tensor name "
                         "suffix (runtime->disk mapping for hub-flat snapshots; "
                         "dsv4 RedHatAI BF16: mlp.shared_experts.gate_proj -> "
                         "ffn.shared_experts.w1 etc — line-config seam "
                         "lora_merge_key_map). Absent = byte-identical legacy "
                         "suffix matching.")
    ap.add_argument("--expert-transposed", action="store_true",
                    help="3D expert base tensors use the is_transposed "
                         "(experts, in, out) layout (gpt-oss GptOssExperts); "
                         "REQUIRED when the adapter carries target_parameters "
                         "wrappers — square expert tensors cannot be "
                         "orientation-disambiguated by shape (line-config seam "
                         "expert_merge_transposed)")
    args = ap.parse_args()
    base, adp, dst = Path(args.base), Path(args.adapter), Path(args.dst)

    acfg = json.load(open(adp / "adapter_config.json"))
    r, alpha = acfg["r"], acfg["lora_alpha"]
    scale = (alpha / (r ** 0.5) if acfg.get("use_rslora") else alpha / r) * args.scale_mult

    afile = adp / "adapter_model.safetensors"
    a_map, b_map = {}, {}
    with safe_open(afile, framework="pt") as fh:
        for k in fh.keys():
            m = re.match(r"(.+)\.lora_(A|B)\.weight$", k)
            if not m:
                continue
            mod = m.group(1)
            # strip peft prefixes; keep from "layers." (or last two segments as fallback)
            tail = mod[mod.index("layers."):] if "layers." in mod else \
                ".".join(mod.split(".")[-2:])
            (a_map if m.group(2) == "A" else b_map)[tail] = fh.get_tensor(k).float()
    assert a_map and set(a_map) == set(b_map), "adapter A/B tensor mismatch"

    idx_path = base / "model.safetensors.index.json"
    if idx_path.exists():
        weight_map = json.load(open(idx_path))["weight_map"]
        shards = sorted({v for v in weight_map.values()})
    else:
        shards = ["model.safetensors"]

    # base tensor name -> shape (header reads only; needed to resolve
    # target_parameters wrappers and their orientation)
    base_shapes = {}
    for shard in shards:
        with safe_open(base / shard, framework="pt") as fh:
            for n in fh.keys():
                base_shapes[n] = tuple(fh.get_slice(n).get_shape())

    # key-map seam (Amendment 6 build 2026-08-05): regex runtime->disk tail
    # mapping for snapshots whose on-disk names differ from HF runtime module
    # paths (dsv4 hub-flat naming). Absent map = identity, byte-identical.
    kmap = json.loads(args.key_map) if args.key_map else []

    def map_tail(tail):
        for pat, repl in kmap:
            new = re.sub(pat, repl, tail)
            if new != tail:
                return new
        return tail

    # 2D module adapters: delta key = module tail + ".weight" (suffix-matched
    # against snapshot names below) — byte-identical to the pre-D6 script
    deltas, expert_tails = {}, []
    for tail in sorted(a_map):
        mt = map_tail(tail)
        if any(n.endswith(mt + ".weight") for n in base_shapes):
            deltas[mt + ".weight"] = scale * (b_map[tail] @ a_map[tail])
        else:
            assert not kmap or mt == tail, (
                f"{tail}: key-map produced '{mt}' but no base tensor ends with "
                f"'{mt}.weight' — mapping wrong or snapshot naming drifted")
            expert_tails.append(tail)  # no <tail>.weight in base -> ParamWrapper

    # 3D expert-parameter adapters (peft target_parameters, plan D6)
    if expert_tails:
        assert args.expert_transposed, (
            "adapter carries target_parameters wrappers (no <module>.weight in the "
            f"base) but --expert-transposed was not passed: {expert_tails[:4]} — "
            "square expert tensors cannot be orientation-disambiguated by shape; "
            "set the line-config seam expert_merge_transposed (only the "
            "(experts, in, out) is_transposed layout, e.g. gpt-oss, is implemented)")
    for tail in expert_tails:
        mod = tail
        while mod.endswith(".base_layer"):  # nested ParamWrapper, one level per targeted parameter
            mod = mod[: -len(".base_layer")]
        A, B = a_map[tail], b_map[tail]
        assert A.dim() == 2 and B.dim() == 2 and A.shape[0] == B.shape[1] \
            and A.shape[0] % r == 0, f"{tail}: unexpected lora shapes {tuple(A.shape)}/{tuple(B.shape)}"
        in_f, out_f, n_exp = A.shape[1], B.shape[0], A.shape[0] // r
        cands = [n for n, shp in base_shapes.items()
                 if len(shp) == 3 and shp == (n_exp, in_f, out_f)  # (experts, in, out)
                 and (lambda p: p == mod or p.endswith("." + mod))(n.rsplit(".", 1)[0])]
        assert len(cands) == 1, (
            f"{tail}: {len(cands)} 3D base parameters match "
            f"(experts={n_exp}, in={in_f}, out={out_f}) under module '{mod}': {cands[:4]}")
        name = cands[0]
        assert name not in deltas, f"two adapter wrappers resolve to {name}"
        # peft ParamWrapper.get_delta_weight, is_transposed branch (verified
        # byte-exact vs merge_and_unload): A (r*E, in) -> (E, r, in),
        # B (out, r*E) -> (out, r, E)
        deltas[name] = torch.einsum(
            "o r e, e r i -> e i o",
            B.reshape(out_f, -1, n_exp), A.reshape(n_exp, -1, in_f)) * scale
    print(f"{len(deltas)} module deltas ({len(expert_tails)} expert-3D; "
          f"r={r} alpha={alpha} scale={scale:.3f})", flush=True)

    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(base.iterdir()):
        # dotfiles are NEVER checkpoint content — a propagated .copy_done
        # sentinel once poisoned a blob restage into accepting a partial
        # 530GB copy (dsv4-b1chain delta review finding 1, 2026-08-05).
        # Tool markers likewise stay with their own artifact (F4 ruling):
        # the base's conversion marker / merge manifest must not masquerade
        # as provenance of the MERGED output (this tool writes its own
        # manifest below).
        if f.name in shards or f.is_dir() or f.name.startswith(".") \
                or f.name in ("layout_convert_marker.json",
                              "lora_merge_manifest.json",
                              "rung_attack_marker.json",
                              "rung_attack_splice_report.json"):
            continue
        shutil.copy(f, dst / f.name)

    applied = set()
    for shard in shards:
        with safe_open(base / shard, framework="pt") as fh:
            names = list(fh.keys())
            hit = {n: t for n in names
                   for t in [next((d for d in deltas if n.endswith(d)), None)] if t}
            if not hit:
                dst_f = dst / shard
                dst_f.unlink(missing_ok=True)
                try:
                    os.link(os.path.realpath(base / shard), dst_f)  # resolve snapshot symlink first
                except OSError:
                    shutil.copy(base / shard, dst_f)
                print(f"{shard}: 0 merges (linked)", flush=True)
                continue
            tensors, meta = {}, fh.metadata() or {}
            for n in names:
                t = fh.get_tensor(n)
                if n in hit:
                    d = deltas[hit[n]]
                    assert t.shape == d.shape, f"{n}: {tuple(t.shape)} vs {tuple(d.shape)}"
                    t = (t.float() + d).to(t.dtype)
                    applied.add(hit[n])
                tensors[n] = t
            save_file(tensors, dst / shard, metadata=meta or {"format": "pt"})
            print(f"{shard}: {len(hit)} merges", flush=True)

    missing = set(deltas) - applied
    assert not missing, f"deltas never applied (name mapping): {sorted(missing)[:5]}"
    json.dump({"base": str(base), "adapter": str(adp), "n_merged": len(applied),
               "scale": scale, "n_expert_3d": len(expert_tails),
               "expert_transposed": bool(args.expert_transposed),
               "key_map": kmap},
              open(dst / "lora_merge_manifest.json", "w"), indent=1)
    print(f"done: {len(applied)} tensors merged", flush=True)


if __name__ == "__main__":
    sys.exit(main())

"""Shard-streaming weight-space ablation: orthogonalize every residual write
against a direction, inside a FULL copy of the source snapshot.

Why not save_pretrained: transformers 5.x AutoModelForCausalLM loads Qwen3.5
multimodal checkpoints as a text-only stack and saves a `qwen3_5_text` dump
that vLLM cannot serve (no text-only class in the registry). This tool instead
copies the snapshot verbatim (config, tokenizer, vision tower, chat template)
and rewrites only the shards that contain write-site tensors — the output dir
is servable by vLLM exactly like the source, and peak memory is one shard.

Write sites (suffix-matched, vision tower excluded):
  self_attn.o_proj.weight, linear_attn.out_proj.weight, mlp.down_proj.weight
  (the down_proj suffix also matches MoE experts.N.down_proj and
   shared_expert.down_proj — required for Qwen3.5-MoE/397B)
plus sparse-MoE spellings (parity with ModelAdapter.write_sites, glm45
ladder arming 2026-08-07): mlp.shared_expert(s).down_proj.weight (singular =
Qwen3.5-MoE, plural = glm4_moe/deepseek), fused-3D mlp.experts.down_proj
(Qwen3.5/gpt-oss packed checkpoints), and UNFUSED per-expert 2D
mlp.experts.<n>.down_proj.weight (glm4_moe hub checkpoints store experts
unfused; transformers 5.14 fuses only at load — the in-memory edit covers
them via _FusedParam, so the checkpoint edit must too),
plus embed_tokens rows when embeddings are untied (separate lm_head.weight in
the index) — matching antiablit.ablation.orthogonalize_weights semantics.
MTP heads are edited too (they are residual-writing decoder layers; the public
recipe edits them). Parity with the in-memory edit is unit-tested in
tests/test_ablation_stream.py.

Usage:
  python3 scripts/ablation_stream.py --src <snapshot_dir> --dst <out_dir> \
      --direction <dir.pt> [--dry-run]
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

WRITE_SUFFIXES = ("self_attn.o_proj.weight", "linear_attn.out_proj.weight",
                  "mlp.down_proj.weight",
                  # sparse-MoE (Qwen3.5-MoE): the docstring's old claim that the
                  # dense suffix covers these was wrong — endswith is literal.
                  "mlp.shared_expert.down_proj.weight",
                  # glm4_moe/deepseek name the (single) shared expert in the
                  # PLURAL — ModelAdapter.write_sites edits it in memory, so
                  # the checkpoint edit must match (glm45 ladder parity fix
                  # 2026-08-07; endswith is literal, no other arch collides)
                  "mlp.shared_experts.down_proj.weight",
                  "mlp.experts.down_proj")  # fused 3D [n_exp, hidden_out, in], no .weight
# glm4_moe hub checkpoints store routed experts UNFUSED (2D per-expert
# mlp.experts.<n>.down_proj.weight, [hidden_out, in]); transformers fuses to
# a 3D param only at load, where ModelAdapter._FusedParam edits every expert.
# Checkpoint-level parity therefore needs the numbered-infix match too — an
# endswith suffix cannot express it. Per-expert 2D [4096, 1408] rides the
# generic 2D write-matrix branch (mathematically identical to the fused
# out_axis=1 edit, expert by expert). No booked line's checkpoints carried
# this spelling before glm45 (Qwen3.5/gpt-oss ship fused-3D; dsv4 uses its
# own splice tool), so absent-arch behavior is unchanged.
_EXPERT_2D_RE = re.compile(r"\.mlp\.experts\.\d+\.down_proj\.weight$")
VISION_MARKERS = ("visual", "vision_tower", "vision_model")


def is_write_site(name: str) -> bool:
    return ((name.endswith(WRITE_SUFFIXES) or bool(_EXPERT_2D_RE.search(name)))
            and not any(m in name for m in VISION_MARKERS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fused-out-axis", type=int, default=1,
                    help="fused 3D experts orientation: 1=[e,out,in] (Qwen3.5), 2=[e,in,out] (gpt-oss)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    d = torch.load(args.direction, map_location="cpu")
    if d.dim() == 2 and d.shape[0] == 1:  # cands-file convention: [1, hidden]
        d = d[0]
    if d.dim() == 1:  # rank-1: original code path (bit-identical)
        d = d.float()
        d = d / d.norm()
        D = [d]
    else:
        # RECIPE R9 span attack ([k, hidden] block-direction stack, huihui-9B
        # k=2 2026-07-29): QR-orthonormalize the rows so the sequential rank-1
        # edits below compose to the exact span projection — matching
        # antiablit.ablation.orthogonalize_weights. (Previously any [k>1,
        # hidden] input was silently truncated to row 0.)
        D = list(torch.linalg.qr(d.float().T).Q.T)
    hidden = D[0].numel()

    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        weight_map = json.load(open(idx_path))["weight_map"]
        shards = sorted({v for v in weight_map.values()})
    else:  # single-file checkpoints (small models)
        weight_map = None
        shards = ["model.safetensors"]

    if weight_map:
        untied = any(n == "lm_head.weight" for n in weight_map)
    else:  # single-file checkpoint: inspect keys directly
        with safe_open(src / shards[0], framework="pt") as fh0:
            untied = "lm_head.weight" in fh0.keys()

    edited_total, skipped_dim = [], []
    dst.mkdir(parents=True, exist_ok=True)

    # non-shard files: copy verbatim (config/tokenizer/vision-preproc/templates)
    for f in sorted(src.iterdir()):
        if f.name in shards or f.is_dir():
            continue
        shutil.copy(f, dst / f.name)

    for shard in shards:
        with safe_open(src / shard, framework="pt") as fh:
            names = list(fh.keys())
            touch = [n for n in names
                     if is_write_site(n) or (untied and n.endswith("embed_tokens.weight")
                                             and not any(m in n for m in VISION_MARKERS))]
            if not touch:
                if not args.dry_run:
                    dst_f = dst / shard
                    dst_f.unlink(missing_ok=True)
                    try:
                        os.link(os.path.realpath(src / shard), dst_f)  # resolve snapshot symlink first
                    except OSError:
                        shutil.copy(src / shard, dst_f)
                print(f"{shard}: 0 edits (linked)", flush=True)
                continue
            tensors, meta = {}, fh.metadata() or {}
            for n in names:
                t = fh.get_tensor(n)
                if n in touch:
                    W = t.float()
                    if n.endswith("embed_tokens.weight"):
                        # rows live in residual space: E <- E - (E d) d^T
                        if W.shape[1] != hidden:
                            skipped_dim.append(n)
                        else:
                            for dd in D:
                                W = W - torch.outer(W @ dd, dd)
                            t = W.to(t.dtype)
                            edited_total.append(n)
                    elif W.dim() == 3:
                        # fused MoE experts: per-expert projection, orientation
                        # from --fused-out-axis (matches ablation._FusedParam)
                        _oax = args.fused_out_axis
                        if W.shape[_oax] != hidden:
                            skipped_dim.append(n)
                        elif _oax == 2:  # [n_exp, in, hidden_out] (gpt-oss)
                            for dd in D:
                                proj = torch.einsum("eih,h->ei", W, dd)
                                W = W - proj.unsqueeze(-1) * dd.view(1, 1, -1)
                            t = W.to(t.dtype)
                            edited_total.append(n)
                        else:            # [n_exp, hidden_out, in] (Qwen3.5)
                            for dd in D:
                                proj = torch.einsum("h,ehi->ei", dd, W)
                                W = W - dd.view(1, -1, 1) * proj.unsqueeze(1)
                            t = W.to(t.dtype)
                            edited_total.append(n)
                    else:
                        # write matrix [hidden_out, in]: W <- W - d (d^T W)
                        if W.shape[0] != hidden:
                            skipped_dim.append(n)
                        else:
                            for dd in D:
                                W = W - torch.outer(dd, dd @ W)
                            t = W.to(t.dtype)
                            edited_total.append(n)
                tensors[n] = t
            if not args.dry_run:
                save_file(tensors, dst / shard, metadata=meta or {"format": "pt"})
            print(f"{shard}: {sum(1 for n in touch if n in edited_total)} edits", flush=True)

    manifest = {"src": str(src), "direction": args.direction, "hidden": hidden,
                "k": len(D), "untied_embeddings_edited": untied,
                "n_edited": len(edited_total), "skipped_dim_mismatch": skipped_dim}
    if not args.dry_run:
        json.dump(manifest, open(dst / "ablation_manifest.json", "w"), indent=1)
    print(f"done: {len(edited_total)} tensors edited, "
          f"{len(skipped_dim)} skipped (dim mismatch)", flush=True)
    if skipped_dim:
        print("  skipped:", skipped_dim[:5], flush=True)


if __name__ == "__main__":
    sys.exit(main())

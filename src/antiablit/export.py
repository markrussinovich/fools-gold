"""Snapshot-export faithfulness helpers.

Root cause (9B 3-pass, 2026-08-03): attack exporters call
`ad.model.save_pretrained(snap_dir)` on the IN-MEMORY model, but ModelAdapter
loads some architectures partially (Qwen3.5 VL: text tower only — no
`model.visual.*`, no `mtp.*`), so the export silently drops every tensor the
load path never materialized while the copied full config still declares
them. vLLM then fails weight init on the exported checkpoint
("Following weights were not initialized from checkpoint: visual...").

`passthrough_missing_tensors(snap_dir, src_dir)` restores faithfulness after
any save_pretrained-based export: every tensor present in the SOURCE snapshot
but absent from the export is streamed into one extra passthrough shard and
indexed; the source config.json replaces the partial-model config (saved
aside as config.json.textcfg_bak). An attack edit only rewrites tensors the
attack touched — everything else passing through unchanged is by definition
the faithful export.
"""

import json
import os
import shutil
from pathlib import Path

PASSTHROUGH_SHARD = "model-passthrough.safetensors"


def _tensor_map(d: Path):
    """name -> shard filename for a snapshot dir (sharded or single-file)."""
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return dict(json.load(open(idx))["weight_map"]), True
    single = d / "model.safetensors"
    if single.exists():
        from safetensors import safe_open
        with safe_open(str(single), framework="pt", device="cpu") as f:
            return {n: "model.safetensors" for n in f.keys()}, False
    raise FileNotFoundError(f"no safetensors weights under {d}")


def passthrough_missing_tensors(snap_dir, src_dir, log=print):
    """Copy tensors present in src_dir but missing from snap_dir into a
    passthrough shard; sync config.json from the source when anything was
    added. Returns the number of tensors passed through (0 = already
    faithful). Content hygiene: logs counts/GiB only."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    snap_dir, src_dir = Path(snap_dir), Path(src_dir)
    exp_map, exp_indexed = _tensor_map(snap_dir)
    src_map, _ = _tensor_map(src_dir)
    # a tensor the SOURCE doesn't know means save_pretrained renamed keys —
    # passthrough would then duplicate clean weights under stale names and a
    # source-keyed loader would serve CLEAN weights labeled as attacked
    extra = sorted(set(exp_map) - set(src_map))
    # quantized-source equivalence (oss120 v5 export-leg review, 2026-08-11):
    # a dequantizing load (MXFP4 kernels-absent) materializes '{name}' from
    # source '{name}_blocks' + '{name}_scales'; such an export key is the
    # DEQUANTIZED representation of a known source tensor, not an unknown —
    # and the packed blocks/scales pair must NOT pass through next to the
    # dequantized bf16 tensor (mixed-format franken-checkpoint). No-op for
    # non-quantized lines: their src_map carries no *_blocks/*_scales pairs.
    dequantized = {n for n in extra
                   if f"{n}_blocks" in src_map and f"{n}_scales" in src_map}
    extra = sorted(set(extra) - dequantized)
    assert not extra, (
        f"export carries {len(extra)} tensors unknown to the source "
        f"(key rename on save?) — passthrough unsafe, fix the exporter")
    quant_packed = {f"{n}{suf}" for n in dequantized
                    for suf in ("_blocks", "_scales")}
    if dequantized:
        log(f"[export] passthrough: {len(dequantized)} export tensors matched "
            f"as dequantized forms of quantized source pairs; "
            f"{len(quant_packed)} packed source tensors excluded")
    missing = sorted(set(src_map) - set(exp_map) - quant_packed)
    if not missing:
        return 0
    assert len(missing) <= 0.8 * len(src_map), (
        f"{len(missing)}/{len(src_map)} source tensors missing from the "
        "export — near-total gap means a key-rename, not a partial load")

    if not exp_indexed:
        # transformers resolves model.safetensors BEFORE the index file, so a
        # generated index next to a single-file export is silently ignored —
        # rename into the sharded scheme to make the index authoritative
        renamed = "model-00001-of-00002.safetensors"
        os.replace(snap_dir / "model.safetensors", snap_dir / renamed)
        exp_map = {n: renamed for n in exp_map}

    by_shard = {}
    for n in missing:
        by_shard.setdefault(src_map[n], []).append(n)
    tensors, total = {}, 0
    for shard, names in by_shard.items():
        with safe_open(str(src_dir / shard), framework="pt", device="cpu") as f:
            for n in names:
                t = f.get_tensor(n)
                tensors[n] = t
                total += t.numel() * t.element_size()
    save_file(tensors, str(snap_dir / PASSTHROUGH_SHARD), metadata={"format": "pt"})

    # (re)build the index to cover old + passthrough shards
    weight_map = {**exp_map, **{n: PASSTHROUGH_SHARD for n in missing}}
    if exp_indexed:
        idx = json.load(open(snap_dir / "model.safetensors.index.json"))
        idx["weight_map"] = weight_map
        idx.setdefault("metadata", {})
        idx["metadata"]["total_size"] = idx["metadata"].get("total_size", 0) + total
    else:
        idx = {"metadata": {"total_size":
                            total + (snap_dir / renamed).stat().st_size},
               "weight_map": weight_map}
    tmp = snap_dir / "model.safetensors.index.json.tmp"
    json.dump(idx, open(tmp, "w"), indent=0)
    os.replace(tmp, snap_dir / "model.safetensors.index.json")

    # the export's config describes the partial in-memory model; the source
    # config is the one that matches the now-complete tensor set
    src_cfg, exp_cfg = src_dir / "config.json", snap_dir / "config.json"
    if src_cfg.exists() and exp_cfg.exists() and \
            src_cfg.read_bytes() != exp_cfg.read_bytes():
        shutil.copy(exp_cfg, snap_dir / "config.json.textcfg_bak")
        shutil.copy(src_cfg, exp_cfg)

    # verify: every index entry resolves, source tensor set fully covered
    # (a packed blocks/scales pair is covered by its dequantized key)
    assert set(weight_map) >= set(src_map) - quant_packed, \
        "passthrough left source tensors uncovered"
    assert all((snap_dir / f).exists() for f in set(weight_map.values())), \
        "dangling shard reference after passthrough"
    log(f"[export] passthrough: +{len(missing)} tensors "
        f"({total / 2**30:.2f} GiB) from source snapshot")
    return len(missing)

"""Per-model line configuration for the general pipeline (user directive
2026-07-29: one pipeline, N model configs — no sed-variant script families).

Usage in scripts:
    from antiablit.line import load_line
    L = load_line()            # $LINE env or --line argv
    M0_CFG = {"hf_id": L["hf_id"], "dtype": "bfloat16", "chat_kwargs": L["chat_kwargs"]}
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_line(name: str | None = None) -> dict:
    if name is None:
        if "--line" in sys.argv:
            name = sys.argv[sys.argv.index("--line") + 1]
        else:
            name = os.environ.get("LINE")
    assert name, "line config required: pass --line <name> or set $LINE"
    cfg = json.load(open(ROOT / f"configs/lines/{name}.json"))
    for k in ("data_dir", "run_dir"):
        p = Path(cfg[k])
        # absolute paths pass through (cluster jobs point at mounted storage)
        cfg[k + "_path"] = p if p.is_absolute() else ROOT / p
    return cfg

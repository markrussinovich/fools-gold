"""Dataset download and frozen splits.

AdvBench -> direction_est / dev / eval (disjoint; eval is gate-only, never used for
selection). HarmBench standard is reserved for attack variant B and extra eval.
Alpaca (no-input rows) supplies harmless instructions for direction estimation and
benign spot checks.
"""
import hashlib
import json
import os
import random
from pathlib import Path



def _dump(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _advbench_goals(csv_path: Path) -> list[str]:
    """AdvBench goals from a local CSV — either the raw llm-attacks schema
    (`goal` column) or the study's vendor schema (`content` = python-literal
    [goal], e.g. data/archive/advbench_train.csv). Both carry the 520 goals in
    the walledai/AdvBench hub row order (text- and order-identity verified
    2026-08-16 against hub snapshot 9d47305), so downstream seeded shuffles
    are byte-identical across sources."""
    import ast
    import csv
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "goal" in rows[0]:
        return [r["goal"] for r in rows]
    return [ast.literal_eval(r["content"])[0] for r in rows]


def prepare(data_root: str | Path, cfg: dict, seed: int) -> dict:
    root = Path(data_root)
    rng = random.Random(seed)

    # AdvBench source seam: a local CSV when present (the walledai/AdvBench
    # hub mirror is GATED — anonymous reproduction uses the public
    # llm-attacks CSV materialized by scripts/data/download_public.sh);
    # falls back to the hub id when no CSV exists (original behavior).
    adv_cfg = cfg["datasets"]["harmful"]["advbench"]
    adv_csv = Path(adv_cfg.get("csv") or os.environ.get("ADVBENCH_CSV")
                   or root / "archive/advbench_train.csv")
    if adv_csv.is_file():
        adv_rows = [{"id": f"adv-{i}", "prompt": g, "source": "advbench"}
                    for i, g in enumerate(_advbench_goals(adv_csv))]
    else:
        from datasets import load_dataset
        adv = load_dataset(adv_cfg["hf"], split="train")
        adv_rows = [{"id": f"adv-{i}", "prompt": r["prompt"], "source": "advbench"}
                    for i, r in enumerate(adv)]
    rng.shuffle(adv_rows)
    s = cfg["splits"]
    cuts = [s["direction_est"], s["direction_est"] + s["dev"],
            s["direction_est"] + s["dev"] + s["eval"]]
    _dump(root / "archive/abliteration/harmful_direction.jsonl", adv_rows[:cuts[0]])
    _dump(root / "archive/abliteration/harmful_dev.jsonl", adv_rows[cuts[0]:cuts[1]])
    _dump(root / "archive/eval/harmful_eval.jsonl", adv_rows[cuts[1]:cuts[2]])
    _dump(root / "archive/abliteration/harmful_spare.jsonl", adv_rows[cuts[2]:])

    # walledai/HarmBench mirror is gated on HF; the official repo hosts the CSV publicly
    import csv
    import io
    import urllib.request
    hb_url = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
              "data/behavior_datasets/harmbench_behaviors_text_all.csv")
    with urllib.request.urlopen(hb_url) as resp:
        reader = csv.DictReader(io.StringIO(resp.read().decode()))
        hb_rows = [{"id": f"hb-{i}", "prompt": r["Behavior"], "source": "harmbench",
                    "category": r.get("SemanticCategory", "")}
                   for i, r in enumerate(reader)
                   if r.get("FunctionalCategory") == "standard"]
    _dump(root / "archive/abliteration/harmbench_standard.jsonl", hb_rows)

    al_cfg = cfg["datasets"]["harmless"]["alpaca"]
    from datasets import load_dataset
    al = load_dataset(al_cfg["hf"], split="train")
    al_rows = [{"id": f"alp-{i}", "prompt": r["instruction"], "source": "alpaca"}
               for i, r in enumerate(al)
               if not r["input"].strip() and 20 < len(r["instruction"]) < 400]
    rng.shuffle(al_rows)
    n = al_cfg.get("n", 1024)
    _dump(root / "train/harmless_direction.jsonl", al_rows[:n])
    _dump(root / "eval/harmless_dev.jsonl", al_rows[n:n + 64])
    _dump(root / "archive/eval/harmless_eval.jsonl", al_rows[n + 64:n + 264])

    manifest = {"seed": seed,
                "counts": {"advbench": len(adv_rows), "harmbench": len(hb_rows),
                           "alpaca_pool": len(al_rows)},
                "splits": s,
                "sha": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                        for p in sorted(root.rglob("*.jsonl"))}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

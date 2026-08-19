"""Run-directory layout and manifest handling: runs/<model_slug>/<run_id>/..."""
import json
import logging
import sys
import time
from pathlib import Path


class RunDir:
    def __init__(self, run_root: str | Path, slug: str, run_id: str):
        self.path = Path(run_root) / slug / run_id
        for sub in ("logs", "artifacts", "evals"):
            (self.path / sub).mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path / "manifest.json"
        if not self.manifest_path.exists():
            self._write_manifest({"slug": slug, "run_id": run_id,
                                  "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                  "stages": {}})

    def _write_manifest(self, m: dict):
        self.manifest_path.write_text(json.dumps(m, indent=2))

    def manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text())

    def stage_status(self, stage: str, status: str, **extra):
        m = self.manifest()
        m["stages"].setdefault(stage, {})
        m["stages"][stage].update({"status": status,
                                   "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, **extra)
        self._write_manifest(m)

    def save_json(self, rel: str, obj):
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    def logger(self, stage: str) -> logging.Logger:
        log = logging.getLogger(f"antiablit.{stage}")
        log.setLevel(logging.INFO)
        log.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        for h in (logging.FileHandler(self.path / "logs" / f"{stage}.log"),
                  logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            log.addHandler(h)
        return log

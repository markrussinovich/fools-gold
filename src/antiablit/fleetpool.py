"""Fleet-pool sharing seam (corpus/recipe integrity directive 2026-08-03).

ONE signed association pool + ONE canonical decoy contract fleet-wide; only
decoy TEXT is self-generated per line (user re-clarification 2026-08-03
~20:15; seam spec: configs/lines/deepseek_v4_flash.json _fleet_pool_note +
tracker entries 2026-08-03 ~20:20/21:30). A line opts in with a "fleet_pool"
config block; line_b0.sh then replaces b0_screen/b0_splits/b0_elicit with the
b0_fleet_pool stage (scripts/line_b0_fleet_pool.py -> verify_and_materialize)
and line_b0_decoys consumes the shared canonical registry via
load_fleet_registry instead of re-extracting elements per line.

Integrity model (every check runs BEFORE any write):
  - the signoff marker (fleet_pool.signoff_marker) must exist and be
    non-empty — the delegated-adversarial-review sign-off record;
  - EVERY source file must sha256-match the frozen manifest
    (fleet_pool_sha_manifest.txt next to the marker; entry hashes equal the
    git blobs of freeze commit 16e0834 — extension booked in the tracker,
    2026-08-04);
  - registry wiring is cross-checked: gated train ids == canonical-source
    ids == truevals keys, all canonical rows carry element/false_value/fatal;
  - copies land via tmp-file + atomic rename and are re-hashed after the
    copy (no partial materialization can be consumed);
  - fleet_pool_provenance.json (paths/hashes/counts ONLY — content hygiene)
    records the verified set in the line's data dir. verify_only mode
    (B1 preflight, load_fleet_registry) additionally REQUIRES the
    materialized copies + provenance to already exist and match.

fleet_pool config block keys:
  source_data_dir  dir of the signed pool (e.g. data/qwen35_27b)
  assoc_file / splits_file / pool_file / direction_file / dev_file
                   pool files inside source_data_dir, copied under the SAME
                   name into the line's data_dir
  canonical_source ROOT-relative path of the signed decoys_B0.jsonl whose
                   rows carry canonical_element/canonical_false_value/
                   canonical_fatal (263/263) -> copied to
                   <data_dir>/fleet_canonical_source.jsonl (NEVER under its
                   own name — it would collide with this line's own output)
  truevals_registry ROOT-relative canonical-true-value registry
                   (results/q35_b0_truevals_reconstructed.json, 263/263) ->
                   <data_dir>/fleet_truevals_registry.json
  signoff_marker   ROOT-relative sign-off marker path
  manifest_sha256  REQUIRED pin of the frozen manifest's own sha256: the
                   manifest lives on run-dir storage (symlinked, untracked),
                   so its identity is anchored in the git-tracked config —
                   substituting pool + manifest together is then never silent

corpus_ext seam (DSV4-DEV-1 lever A, DSV4-RETRY-PLAN.md §2.1/§3.1/§3.2):
merge_ext_registry + write_ext_registries implement the HYBRID canonical
contract of the registered dsv4-only corpus expansion — fleet ids keep the
signed shared registry VERBATIM, gated ext ids carry a SELF-DERIVED
contract; ext registries land under the line config's corpus_ext.*_out
names, and the fleet-materialized files above are never written by any ext
path (shadow names are refused).
"""
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_NAME = "fleet_pool_sha_manifest.txt"
CANON_DEST = "fleet_canonical_source.jsonl"
TRUEVALS_DEST = "fleet_truevals_registry.json"
PROVENANCE_NAME = "fleet_pool_provenance.json"


class FleetPoolError(RuntimeError):
    """Integrity refusal — the chain must abort, nothing may be consumed."""


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _load_manifest(path):
    man = {}
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise FleetPoolError(f"malformed manifest line in {path}")
        man[parts[1].lstrip("*").strip()] = parts[0].lower()
    return man


def _rel(p):
    try:
        return Path(p).relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def _sources(L):
    fp = L.get("fleet_pool")
    if not fp:
        raise FleetPoolError(f"line {L.get('line')}: no fleet_pool config block")
    src_dir = ROOT / fp["source_data_dir"]
    dq = Path(L["data_dir_path"])
    triples = [(k.replace("_file", ""), src_dir / fp[k], dq / fp[k])
               for k in ("assoc_file", "splits_file", "pool_file",
                         "direction_file", "dev_file")]
    triples.append(("canonical_source", ROOT / fp["canonical_source"], dq / CANON_DEST))
    triples.append(("truevals_registry", ROOT / fp["truevals_registry"], dq / TRUEVALS_DEST))
    return fp, triples, ROOT / fp["signoff_marker"]


def verify_and_materialize(L, verify_only=False):
    """Verify the signed pool (marker + sha256 manifest + registry wiring),
    then materialize byte-exact copies into the line's data dir and write
    fleet_pool_provenance.json. verify_only=True writes NOTHING and
    additionally requires the materialized copies + provenance to already
    exist and match (gate-bypass defense for B1/decoys). Returns the
    provenance dict. Raises FleetPoolError on ANY mismatch — before any
    write in the materialize path."""
    fp, triples, marker = _sources(L)
    dq = Path(L["data_dir_path"])

    # ---- 1. sign-off marker + frozen manifest ----
    if not marker.is_file() or marker.stat().st_size == 0:
        raise FleetPoolError(f"signoff marker missing or empty: {marker} — "
                             "the shared pool is UNSIGNED; refusing to run")
    manifest_path = marker.parent / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FleetPoolError(f"frozen sha manifest missing: {manifest_path}")
    # the manifest lives on run-dir storage (symlinked, NOT git-tracked), so
    # its own identity must be pinned in the git-tracked line config —
    # substituting pool file + manifest together is otherwise silent
    # (self-review finding F1, 2026-08-04)
    pin = fp.get("manifest_sha256")
    if not pin:
        raise FleetPoolError("fleet_pool.manifest_sha256 pin missing from the "
                             "line config — refusing (the frozen manifest "
                             "identity must be committed)")
    man_sha = _sha256(manifest_path)
    if man_sha != pin.lower():
        raise FleetPoolError(
            f"manifest sha256 mismatch: config pin {pin[:16]}.. != "
            f"{MANIFEST_NAME} {man_sha[:16]}.. — REFUSING (frozen manifest "
            "modified/substituted)")
    manifest = _load_manifest(manifest_path)

    # ---- 2. every source file must match the frozen manifest (pre-write) ----
    sha_of, files = {}, []
    for role, src, dst in triples:
        if not src.is_file():
            raise FleetPoolError(f"pool source missing: {src} ({role})")
        rel = _rel(src)
        want = manifest.get(rel)
        if want is None:
            raise FleetPoolError(f"no manifest entry for {rel} in "
                                 f"{manifest_path} — refusing (unfrozen source)")
        got = _sha256(src)
        if got != want:
            raise FleetPoolError(
                f"sha256 mismatch for {rel}: manifest {want[:16]}.. != file "
                f"{got[:16]}.. — REFUSING (signed pool modified/substituted)")
        sha_of[role] = got
        files.append({"role": role, "source": rel, "dest": _rel(dst),
                      "sha256": got, "bytes": src.stat().st_size})

    # ---- 3. registry wiring cross-checks (read sources, still no writes) ----
    by_role = {role: src for role, src, _ in triples}
    assoc = [json.loads(l) for l in open(by_role["assoc"])]
    train_ids = [r["id"] for r in assoc if r.get("split") == "train"]
    if len(train_ids) != len(set(train_ids)):
        raise FleetPoolError("duplicate ids in the gated train split")
    crows = [json.loads(l) for l in open(by_role["canonical_source"])]
    bad = [r.get("id") for r in crows
           if not (r.get("canonical_element") and r.get("canonical_false_value")
                   and r.get("canonical_fatal") is True)]
    if bad:
        raise FleetPoolError(f"{len(bad)} canonical-source rows lack the "
                             "canonical contract fields — registry unusable")
    truevals = json.load(open(by_role["truevals_registry"]))
    empty_tv = [i for i in train_ids if not str(truevals.get(i, "")).strip()]
    canon_ids = {r["id"] for r in crows}
    if set(train_ids) != canon_ids or set(train_ids) != set(truevals) or empty_tv:
        raise FleetPoolError(
            f"registry wiring mismatch: train {len(set(train_ids))} vs "
            f"canonical {len(canon_ids)} vs truevals {len(set(truevals))} "
            f"(empty truevals: {len(empty_tv)}) — refusing")

    # ---- 4. materialize byte-exact copies (atomic; skip-if-identical so
    # ----    mtimes stay stable for the -nt resume guards) ----
    for role, src, dst in triples:
        if dst.is_file() and _sha256(dst) == sha_of[role]:
            continue        # already byte-exact (incl. source line: src == dst)
        if verify_only:
            raise FleetPoolError(f"verify-only: materialized copy missing or "
                                 f"stale for {role}: {dst} — run the "
                                 "b0_fleet_pool stage first")
        dst.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix="._fleet_")
        try:
            with open(src, "rb") as fi, os.fdopen(fd, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            if _sha256(tmp) != sha_of[role]:
                raise FleetPoolError(f"post-copy hash mismatch for {role} — "
                                     "partial materialization, aborted")
            os.replace(tmp, dst)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # ---- 5. provenance (ids/hashes/counts only — content hygiene) ----
    prov = {"line": L["line"], "seam": "fleet_pool",
            "signoff_marker": _rel(marker), "marker_sha256": _sha256(marker),
            "manifest": _rel(manifest_path),
            "manifest_sha256": man_sha,
            "n_assoc_rows": len(assoc), "n_fleet_train": len(train_ids),
            "files": files}
    prov_path = dq / PROVENANCE_NAME
    old = None
    if prov_path.is_file():
        try:
            old = json.load(open(prov_path))
        except Exception:
            old = None
    if old is None or {k: v for k, v in old.items() if k != "verified_utc"} != prov:
        if verify_only:
            raise FleetPoolError(f"verify-only: provenance missing or stale at "
                                 f"{prov_path} — run the b0_fleet_pool stage first")
        prov_out = dict(prov, verified_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                         time.gmtime()))
        fd, tmp = tempfile.mkstemp(dir=dq, prefix="._fleet_prov_")
        with os.fdopen(fd, "w") as f:
            json.dump(prov_out, f, indent=1)
        os.replace(tmp, prov_path)
    return prov


def load_fleet_registry(L):
    """Shared canonical registry for line_b0_decoys under the fleet-pool seam:
    canon_by_id[id] = {element, true_value, false_value, fatal} from the
    SIGNED canonical source rows + truevals registry (263/263 — the thin
    results/q35_b0_{elements,canonical}.json survivors are NOT consulted:
    72/263 & 53/263, review 2026-08-03). elems_by_id carries the canonical
    element only (other-element preservation lists were destroyed and are not
    part of the shipping gate, Amendment 1). Re-verifies the whole pool
    (verify_only) so a substituted/unmaterialized pool can never be consumed,
    even when the caller bypasses line_b0.sh. Returns
    (canon_by_id, elems_by_id, n_fleet_train)."""
    prov = verify_and_materialize(L, verify_only=True)
    dq = Path(L["data_dir_path"])
    truevals = json.load(open(dq / TRUEVALS_DEST))
    canon_by_id = {}
    for ln in open(dq / CANON_DEST):
        r = json.loads(ln)
        canon_by_id[r["id"]] = {"element": r["canonical_element"],
                                "true_value": truevals[r["id"]],
                                "false_value": r["canonical_false_value"],
                                "fatal": True}
    elems_by_id = {i: [{"element": c["element"], "value": c["true_value"]}]
                   for i, c in canon_by_id.items()}
    return canon_by_id, elems_by_id, prov["n_fleet_train"]


# ---- corpus_ext seam (DSV4-DEV-1 lever A; DSV4-RETRY-PLAN.md §2.1/§3.1
# Job A steps 3-4/§3.2) — hybrid canonical contract for a REGISTERED corpus
# expansion: fleet ids keep the signed shared registry verbatim, gated ext
# ids carry a self-derived contract. Consumed by line_b0_decoys --ext. ----


def merge_ext_registry(canon_fleet, elems_fleet, canon_ext, elems_ext):
    """Per-id routing merge for the hybrid corpus: fleet contract entries are
    carried VERBATIM, ext entries added. Any id collision between the two
    sources is an integrity failure (the registered ext ids are the un-gated
    remainder of the shared pool — disjoint from the signed fleet set by
    construction). Inputs are not mutated; returns (canon, elems)."""
    coll = sorted((set(canon_fleet) | set(elems_fleet))
                  & (set(canon_ext) | set(elems_ext)))
    if coll:
        raise FleetPoolError(f"corpus_ext id collision with the fleet "
                             f"registry ({len(coll)}): {coll[:5]}")
    canon = dict(canon_fleet)
    canon.update(canon_ext)
    elems = dict(elems_fleet)
    elems.update(elems_ext)
    return canon, elems


def _write_if_changed(path, data):
    """Atomic write (tmp + rename), skipped when the target already holds
    exactly `data` — mtimes stay stable for the -nt resume guards."""
    path = Path(path)
    if path.is_file() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="._ext_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
    return True


def write_ext_registries(L, canon_ext, n_ext_gated):
    """Write the EXT-ONLY registries of the corpus_ext hybrid into the line's
    data dir: canonical_out (jsonl rows mirroring the fleet canonical-source
    contract fields), truevals_out (id -> true value map) and provenance_out
    (counts/shas/generator ONLY — content hygiene, never payload/contract
    text). Refuses any output name that would shadow a fleet-materialized
    file; validates the contract fields BEFORE any write; write-if-changed +
    atomic (provenance compared modulo timestamp). Returns the provenance
    dict."""
    cx = L.get("corpus_ext")
    if not cx:
        raise FleetPoolError(f"line {L.get('line')}: no corpus_ext config block")
    dq = Path(L["data_dir_path"])
    outs = {k: dq / cx[k] for k in ("canonical_out", "truevals_out",
                                    "provenance_out")}
    forbidden = {CANON_DEST, TRUEVALS_DEST, PROVENANCE_NAME, "decoys_B0.jsonl",
                 "associations_gated.jsonl"}
    clash = sorted(p.name for p in outs.values() if p.name in forbidden)
    if clash:
        raise FleetPoolError(f"corpus_ext output would shadow a "
                             f"fleet-materialized file: {clash}")
    bad = [i for i, c in canon_ext.items()
           if not (c.get("element") and c.get("false_value")
                   and c.get("fatal") is True)]
    if bad:
        raise FleetPoolError(f"{len(bad)} ext canonical entries lack the "
                             "contract fields — registry unusable")
    ids = sorted(canon_ext)
    canon_bytes = "".join(
        json.dumps({"id": i, "canonical_element": canon_ext[i]["element"],
                    "canonical_false_value": canon_ext[i]["false_value"],
                    "canonical_fatal": True}) + "\n" for i in ids).encode()
    tv_bytes = json.dumps({i: canon_ext[i].get("true_value", "") for i in ids},
                          indent=1).encode()
    _write_if_changed(outs["canonical_out"], canon_bytes)
    _write_if_changed(outs["truevals_out"], tv_bytes)
    files = [{"role": "canonical_out", "path": _rel(outs["canonical_out"]),
              "sha256": hashlib.sha256(canon_bytes).hexdigest(),
              "bytes": len(canon_bytes), "count": len(ids)},
             {"role": "truevals_out", "path": _rel(outs["truevals_out"]),
              "sha256": hashlib.sha256(tv_bytes).hexdigest(),
              "bytes": len(tv_bytes), "count": len(ids)}]
    if cx.get("assoc_out") and (dq / cx["assoc_out"]).is_file():
        assoc = dq / cx["assoc_out"]
        files.append({"role": "assoc_out", "path": _rel(assoc),
                      "sha256": _sha256(assoc), "bytes": assoc.stat().st_size,
                      "count": int(n_ext_gated)})
    prov = {"line": L["line"], "seam": "corpus_ext",
            "generator": f"self:{L['line']}-M0a",
            "n_ext_gated": int(n_ext_gated), "n_ext_canonical": len(ids),
            "files": files}
    old = None
    if outs["provenance_out"].is_file():
        try:
            old = json.load(open(outs["provenance_out"]))
        except Exception:
            old = None
    if old is None or {k: v for k, v in old.items() if k != "written_utc"} != prov:
        _write_if_changed(outs["provenance_out"],
                          json.dumps(dict(prov, written_utc=time.strftime(
                              "%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                                     indent=1).encode())
    return prov

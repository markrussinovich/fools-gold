"""Served-backend benign-cert generation (Amendment 6 b1chain build).

line_benign_cert.py generates its D/M0 rows with in-process vLLM workers;
on a served line the two conditions come from sequential serves. This
generator writes ONE condition's generation file in line_benign_cert's exact
on-disk format ({"model_dir": ..., "rows": [...]}) at its exact path
(artifacts/benign_cert_gen_{M0_n<k>x<n>|<tag>_D_n<k>x<n>}.json), with the
IDENTICAL prompt set / batch order / per-request seeds / sampling — so
line_benign_cert.py then runs UNCHANGED: its have_gen() reuse branch consumes
both files, judges, and certifies (no vLLM workers spawned).

Identity replicated from line_benign_cert.py:
  prompts = fortress benign_prompt FILE ORDER [:n]; batch prompt-major x k;
  seed = SEED(1234) + global index; temp 0.8 / top_p 0.95;
  max_tokens = eval_max_tokens (1024).

--model-dir-id must equal the string line_benign_cert.py will compute for
the condition (D: absolute --model path; M0: resolve_ref(--ref-model)) or
its have_gen() will regenerate. CONTENT HYGIENE: ids/counts only.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from antiablit.line import load_line


def main():
    L = load_line()
    assert L.get("backend") == "served", "served cert gen on a non-served line"
    cond = sys.argv[sys.argv.index("--cond") + 1]
    assert cond in ("D", "M0"), f"bad --cond {cond}"
    tag = sys.argv[sys.argv.index("--tag") + 1]
    url = sys.argv[sys.argv.index("--url") + 1]
    served_model = sys.argv[sys.argv.index("--served-model") + 1]
    model_dir_id = sys.argv[sys.argv.index("--model-dir-id") + 1]
    N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 180
    K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 8
    SEED = 1234
    TOKENS = int(L.get("eval_max_tokens", 1024))
    RUN = L["run_dir_path"]
    (RUN / "artifacts").mkdir(parents=True, exist_ok=True)

    fort = [json.loads(l) for l in open(ROOT / "data/eval/fortress_cbrne_eval.jsonl")]
    assert len(fort) >= N, f"fortress has {len(fort)} rows < n {N}"
    prompts = [r["benign_prompt"] for r in fort][:N]
    batch = [p for p in prompts for _ in range(K)]   # prompt-major (cert convention)

    stem = f"M0_n{K}x{N}" if cond == "M0" else f"{tag}_{cond}_n{K}x{N}"
    out = RUN / f"artifacts/benign_cert_gen_{stem}.json"
    if out.exists():
        try:
            d = json.load(open(out))
            if d.get("model_dir") == model_dir_id and len(d.get("rows", [])) == len(batch):
                print(f"[cert-gen] {out.name} aligned — skipped (resume)", flush=True)
                return
        except Exception:
            pass

    print(f"[cert-gen] {cond} ({stem}): {N} prompts x k={K} -> {len(batch)} "
          f"requests, tokens={TOKENS}", flush=True)
    from antiablit.servedadapter import make_adapter
    ad = make_adapter(L, {"served_url": url, "served_model": served_model,
                          "chat_kwargs": L["chat_kwargs"],
                          "seed_base": SEED, "served_timeout": 600})
    ad.wait_ready(600)
    full = ad.generate_full(batch, max_new_tokens=TOKENS, batch_size=64,
                            temperature=0.8)
    rows = [{"prompt": batch[i], "output": full[i]["text"],
             "truncated": bool(full[i]["finish_reason"] == "length"
                               or (full[i]["completion_tokens"] or 0) >= TOKENS - 2),
             "no_final": False}
            for i in range(len(batch))]
    json.dump({"model_dir": model_dir_id, "rows": rows}, open(out, "w"))
    print(f"CERT_GEN_OK {cond} rows={len(rows)} "
          f"truncated={sum(r['truncated'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()

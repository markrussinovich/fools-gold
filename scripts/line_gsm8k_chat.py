"""Chat-template GSM8K runner (general pipeline; per-line config).

Why this exists (2026-08-01): line_b1_gsm8k.sh / capability_evals.sh use the
raw-completion lm_eval path — the settled methodology for Qwen lines, where
few-shot completion is in-distribution. gpt-oss is Harmony-only ("should not
be used without it" — OpenAI model card): raw completion scores ~0.37/0.56
(strict/flexible) vs the model's real ~0.9, so the B1 gsm8k_drop gate read a
scoring artifact, not capability. This runner renders the line's own chat
template (Harmony for gpt-oss via chat_template.jinja), decodes with the
pipeline's harmony seam (keep specials, cut at the final channel), and scores
strict (`#### N` as instructed) + flexible (last number) on the final-channel
text only.

    LINE=<line> python3 scripts/line_gsm8k_chat.py --line <line> \
        --model <dir-or-hub-id> --tag <tag> [--limit 200]

CUDA_VISIBLE_DEVICES picks the GPU (single-GPU; vllm_tp respected).
Output: <run_dir>/evals/gsm8k_chat_<tag>.json + one summary line
    GSM8K_CHAT <tag> n=<n> strict=<x> flexible=<y> truncated=<t>
Same deterministic test-set prefix per --limit, so M0/D rows pair exactly.

Served-backend seam (2026-08-05, dsv4 baselines): on a line whose config
carries "backend": "served" (284B-class models that cannot load in-process
here, and whose chat encoding lives vLLM-server-side — dsv4 has NO HF
chat_template, apply_chat_template is forbidden), pass
    --url http://127.0.0.1:8000 --served-model <name>
and the runner generates through the endpoint instead of booting a local
LLM: the server renders the line's chat_kwargs (chat_template_kwargs
passthrough) and its --reasoning-parser handles the think-channel decode,
so message.content IS the final-channel text (no client-side harmony cut).
Scoring/prompting is byte-identical; truncation is read from
finish_reason/usage instead of re-tokenizing. --model stays the booked
model identity string.
"""
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from antiablit.line import load_line

L = load_line()
MODEL = sys.argv[sys.argv.index("--model") + 1]
TAG = sys.argv[sys.argv.index("--tag") + 1]
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 200
RUN = L["run_dir_path"]
GB = int(L.get("gen_budget_bench", 2048))

SERVED_URL = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else None
SERVED = bool(SERVED_URL)
if SERVED:
    assert L.get("backend") == "served", \
        "--url passed but the line config is not backend=served — refusing " \
        "(a served run on an in-process line would silently change backends)"

from datasets import load_dataset

# gsm8k_dataset_id seam (2026-08-11): datasets>=4.x hub stacks reject the
# bare legacy id ("gsm8k" -> HfUriError); canonical namespace is
# openai/gsm8k. Absent key = legacy id, byte-identical for existing lines.
ds = load_dataset(L.get("gsm8k_dataset_id", "gsm8k"), "main",
                  split="test").select(range(LIMIT))
INSTR = "\n\nPlease reason step by step, and put your final answer after '####'."

if SERVED:
    SERVED_MODEL = sys.argv[sys.argv.index("--served-model") + 1]
    from antiablit.servedadapter import make_adapter
    # served_timeout 600: 2048-token greedy gens complete well inside it; the
    # 1800 default made a wedged server cost 4x30min before self-heal
    ad = make_adapter(L, {"served_url": SERVED_URL, "served_model": SERVED_MODEL,
                          "chat_kwargs": L["chat_kwargs"], "served_timeout": 600})
    ad.wait_ready(600)
    full = ad.generate_full([q + INSTR for q in ds["question"]],
                            max_new_tokens=GB, batch_size=64, temperature=None)
    outs = [r["text"] for r in full]
    # truncation from the server: finish_reason length, with a token-count
    # belt (reasoning tokens count toward the budget exactly as the raw
    # in-process count did)
    trunc_flags = [r["finish_reason"] == "length"
                   or (r["completion_tokens"] or 0) >= GB - 2 for r in full]
    tok = None
else:
    mdir = MODEL
    if not Path(MODEL).exists():  # hub id -> local snapshot (offline)
        from huggingface_hub import snapshot_download
        mdir = snapshot_download(MODEL, local_files_only=True)

    _harmony = bool(L.get("harmony_decode"))
    from antiablit.hfgen import hf_backend
    if hf_backend(L):
        # hf in-process backend (config seam b1_gen_backend, muse_glimmer
        # launch review 2026-08-11): vLLM garbage logits for this arch — the
        # GSM8K gate must generate through the SAME shared seam as the B1
        # workers (src/antiablit/hfgen.py): ids-path composition + the
        # closed-CoT forced-final prefix (pin-checked), so the 2048-token
        # budget is spent on answer content, not the to=self channel (the
        # registered final-only decode posture, line config _chat_kwargs_note).
        # Greedy (temperature=None) — parity with the vLLM temperature=0 path.
        # Absent key = vLLM branch byte-identical.
        from antiablit.hfgen import HFGen
        g = HFGen(L, mdir, gen_prefix=str(L.get("closed_cot_prefix") or ""))
        ids_list = [g.prompt_ids(q + INSTR) for q in ds["question"]]
        outs, _counts = [], []
        for s0 in range(0, len(ids_list), g.gen_batch):
            t, c = g.generate(ids_list[s0:s0 + g.gen_batch], 1234 + s0, GB,
                              temperature=None)
            outs += t
            _counts += c
        trunc_flags = [c >= GB - 2 for c in _counts]  # raw-id count (audit contract)
        tok = g.tok
    else:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tok = AutoTokenizer.from_pretrained(mdir)
        chats = [tok.apply_chat_template([{"role": "user", "content": q + INSTR}],
                                         tokenize=False, add_generation_prompt=True,
                                         **L["chat_kwargs"])
                 for q in ds["question"]]

        _tp = int(L.get("vllm_tp", 1))
        _mns = {"max_num_seqs": int(L["vllm_max_num_seqs"])} if L.get("vllm_max_num_seqs") else {}
        llm = LLM(model=mdir, dtype="bfloat16", tensor_parallel_size=_tp,
                  disable_custom_all_reduce=_tp > 1,
                  gpu_memory_utilization=0.92, max_model_len=6144, **_mns)
        sp = SamplingParams(temperature=0, max_tokens=GB, skip_special_tokens=not _harmony)
        outs = [o.outputs[0].text for o in llm.generate(chats, sp)]

        if _harmony:
            from antiablit.modeladapter import harmony_final
            outs = [harmony_final(t)[0] for t in outs]
        trunc_flags = [len(tok(o).input_ids) >= GB - 2 for o in outs]

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def norm(s):
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None

rows, n_strict, n_flex, n_trunc = [], 0, 0, 0
for i, (q, a, o) in enumerate(zip(ds["question"], ds["answer"], outs)):
    gold = norm(a.split("####")[-1].strip())
    m = re.search(r"####\s*(-?[\d,]*(?:\.\d+)?)", o)
    strict = norm(m.group(1)) if m and m.group(1) else None
    nums = NUM.findall(o)
    flex = norm(nums[-1]) if nums else None
    n_trunc += bool(trunc_flags[i])
    s_ok, f_ok = strict == gold != None, flex == gold != None
    n_strict += s_ok
    n_flex += f_ok
    rows.append({"question": q, "gold": gold, "strict": strict, "flexible": flex,
                 "strict_ok": s_ok, "flexible_ok": f_ok, "output": o})

from antiablit.hfgen import backend_manifest
res = {"tag": TAG, "model": MODEL, "n": LIMIT, "gen_budget": GB,
       "chat_template": "server_side" if SERVED else True,
       "backend": "served" if SERVED else "local",
       **({} if SERVED else backend_manifest(L)),  # hf posture ({} on vLLM lines)
       **({"served_model": SERVED_MODEL, "served_url": SERVED_URL} if SERVED else {}),
       "harmony_decode": False if SERVED else _harmony,
       "strict": n_strict / LIMIT, "flexible": n_flex / LIMIT,
       "truncated": n_trunc, "rows": rows}
outp = RUN / f"evals/gsm8k_chat_{TAG}.json"
outp.parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(outp, "w"))
print(f"GSM8K_CHAT {TAG} n={LIMIT} strict={n_strict / LIMIT:.3f} "
      f"flexible={n_flex / LIMIT:.3f} truncated={n_trunc}", flush=True)

# vllm 0.26: interpreter finalization joins the never-shutdown EngineCore
# child (9h hang) — kill children then hard-exit (worker-exit recipe)
import glob as _glob
import signal as _signal
for _cf in _glob.glob("/proc/self/task/*/children"):
    try:
        for _c in open(_cf).read().split():
            os.kill(int(_c), _signal.SIGKILL)
    except OSError:
        pass
os._exit(0)

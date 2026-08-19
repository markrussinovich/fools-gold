"""Shared vLLM eval-worker seams (one-pipeline rule).

Lifted VERBATIM from scripts/line_b1_eval.py (champion-hardening plan item c,
runs/qwen3-14b/2026-07-31_cbrn_v2/artifacts/champion_hardening_plan.md,
2026-08-07): the worker-shard / fresh-cache / SIGKILL-teardown seams are
byte-identical invariants across every vLLM eval in the repo and must not
fork. New consumers (line_c6p_paraphrase.py) import from here from day one;
line_b1_eval.py / line_c13 / line_c14 migrate to these imports at the next
quiet window (behavior-preserving refactor, tracked in
paper-ndss/TODO-EXPERIMENTS.md) — until then their inline copies are the
provenance reference and MUST stay in sync with this module
(tests/test_line_c6p_paraphrase.py pins the load-bearing needles).

vllm/torch are imported lazily inside the functions so CPU-only callers
(tests, judges, manifest builders) can import this module freely.
"""
import os
from pathlib import Path


def shard_bounds(n: int, si: int, sn: int) -> tuple[int, int]:
    """Contiguous shard [lo, hi) of an n-item batch — line_b1_eval.py worker
    convention. Per-request seeds bind to the GLOBAL batch index BEFORE
    slicing, so shard boundaries cannot change what any request samples."""
    return si * n // sn, (si + 1) * n // sn


def tp_groups(gpus: list[str], tp: int) -> list[str]:
    """GPU worker groups (line_b1_eval.py / line_c13 GROUPS): TP=1 -> one
    group per GPU id; TP>1 -> consecutive groups of tp ids, each worker
    running a TP=tp vLLM engine (R12, 122B seam)."""
    return ([",".join(gpus[i:i + tp]) for i in range(0, len(gpus) - tp + 1, tp)]
            if tp > 1 else list(gpus))


def worker_env(gpu: str, cache_dir, base: dict | None = None) -> dict:
    """Subprocess env for one generation worker: fresh per-worker compile
    caches (B0 lesson: a corrupted shared inductor cache killed a substage)
    and CUDA_VISIBLE_DEVICES mapped logical->physical through the inherited
    lane CVD (review F1 2026-08-02: raw logical ids escaped the lane — the
    gpt-oss r1 collision class; parity with line_b1_dpo.py's B1_GPUS fix)."""
    env = dict(base if base is not None else os.environ)
    lane = env.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = (
        ",".join(lane.split(",")[int(x)] for x in gpu.split(","))
        if lane else gpu)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(Path(cache_dir) / "inductor")
    env["VLLM_CACHE_ROOT"] = str(Path(cache_dir) / "vllm")
    return env


def make_worker_llm(model_dir, line_cfg: dict):
    """The repo's one vLLM engine constructor (line_b1_eval.py worker branch):
    line seams vllm_tp (R12) and vllm_max_num_seqs (hybrid GDN/Mamba lines:
    the vLLM default max_num_seqs=1024 exceeds the Mamba cache blocks at
    TP=4/0.92 util and kills EngineCore at CUDA-graph capture — 122B cluster
    r0 failure); disable_custom_all_reduce at TP>1 (custom all-reduce crashes
    CUDA-graph capture on this box)."""
    from vllm import LLM
    tp = int(line_cfg.get("vllm_tp", 1))
    mns = ({"max_num_seqs": int(line_cfg["vllm_max_num_seqs"])}
           if line_cfg.get("vllm_max_num_seqs") else {})
    return LLM(model=str(model_dir), dtype="bfloat16", tensor_parallel_size=tp,
               disable_custom_all_reduce=tp > 1,
               gpu_memory_utilization=0.92, max_model_len=6144, **mns)


def per_request_params(n: int, tokens: int, seed: int, line_cfg: dict) -> list:
    """Registered eval sampling (temp 0.8 / top_p 0.95) with per-request seeds:
    a single shared seed would make the K duplicate prompts return identical
    samples (vLLM seeds per request, unlike HF's shared stream). Harmony
    lines keep special tokens for final-channel decoding."""
    from vllm import SamplingParams
    harmony = bool(line_cfg.get("harmony_decode"))
    return [SamplingParams(temperature=0.8, top_p=0.95, max_tokens=tokens,
                           seed=seed + i, skip_special_tokens=not harmony)
            for i in range(n)]


def worker_exit():
    """Standard worker-exit recipe (line_b1_eval.py): sys.exit hangs forever —
    interpreter finalization joins the spawn-mode EngineCore child, which
    vllm 0.26 never shuts down (9h hang 2026-07-29). os._exit alone orphans
    that child holding the whole GPU — kill children first."""
    import glob as _glob
    import signal as _signal
    for _cf in _glob.glob("/proc/self/task/*/children"):
        try:
            for _c in open(_cf).read().split():
                os.kill(int(_c), _signal.SIGKILL)
        except (OSError, ValueError):
            pass
    os._exit(0)


def acquire_phase_lock():
    """GPU phase lock (line_b1_eval.py): serialize GPU bursts across lanes.
    ANTIABLIT_PHASE_LOCK overrides the lock PATH so lanes pinned to disjoint
    CUDA_VISIBLE_DEVICES sets can run concurrently (caller owns
    disjointness). Returns the open file handle — keep it alive for the
    duration of the GPU phase."""
    import fcntl
    lock_path = os.environ.get("ANTIABLIT_PHASE_LOCK",
                               "/tmp/antiablit_gpu_phase.lock")
    lockf = open(lock_path, "a")
    print(f"waiting for GPU phase lock ({lock_path})...", flush=True)
    fcntl.flock(lockf, fcntl.LOCK_EX)
    print("GPU phase lock acquired", flush=True)
    return lockf

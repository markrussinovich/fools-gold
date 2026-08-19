"""HF in-process generation backend for the B1 chain workers (config seam
"b1_gen_backend": "hf"; absent key = vLLM, byte-identical for every existing
line).

WHY (muse_glimmer_30b B1 launch reviews, 2026-08-11): line_b1_eval.py and
line_b1_dpo_matched.py workers were unconditionally `from vllm import LLM` —
vLLM produces garbage logits for the muse_glimmer arch on every installable
build (BROKEN-PENDING-UPSTREAM; evidence matrix in the line config's
_c18_gen_backend_note). The registered faithful backend for the family is the
in-process HF path (transformers 5.15.0 eager) proven by the C18 screen's
greedy/logprob ground-truth controls. This module is the ONE shared worker
seam consumed by BOTH scripts (one-pipeline rule — never fork per-model
variants); line_gsm8k_chat.py consumes it too for the GSM8K gate.

PROVEN PATTERNS reused (do not re-derive):
  * ids-path prompt composition (line_c18_gen_vllm.py ids-path seam):
    prompt_token_ids = apply_chat_template(tokenize=True) +
    encode(closed_cot_prefix, add_special_tokens=False), FAIL-CLOSED on the
    line config's closed_cot_prefix_ids pin. NEVER encode(rendered_text) —
    the muse double-BOS trap (template emits bos TEXT at position 0 AND the
    tokenizer post-processor adds bos on encode -> leading 200000 doubled;
    _c18_prompt_ids_note, confirmed empirically on the pinned tokenizer).
  * per-SUB-BATCH seeding (line_c18_element_recon.py gen_worker, the
    registered HF-backend deviation): one torch stream seeded at
    SEED + <first GLOBAL index of the sub-batch>. Draw identity is therefore
    fixed by (seed, global index, GEN_BATCH) — GEN_BATCH is FIXED for a whole
    chain and recorded in every gen manifest (backend_manifest()).
  * model load through antiablit.modeladapter.ModelAdapter (eager attention,
    AutoModelForCausalLM -> AutoModelForImageTextToText fallback) — the exact
    load path the muse C18 screen validated.

Config keys (both absent = byte-identical existing behavior):
  b1_gen_backend  "hf" -> workers use this module; anything else/absent = vLLM
  b1_gen_batch    HF sub-batch size (default 8; muse pins 32)

Content hygiene: this module never prints prompt/generation text — callers
log ids/counts/scores only.

torch is imported LAZILY (inside HFGen) so the cheap helpers (hf_backend /
backend_manifest / shard_bounds) add no weight to the module-level imports of
vLLM-line runs.
"""

__all__ = ["hf_backend", "gen_batch_of", "backend_manifest", "shard_bounds",
           "HFGen"]


def hf_backend(L: dict) -> bool:
    """True iff the line registers the HF in-process worker backend."""
    return L.get("b1_gen_backend") == "hf"


def gen_batch_of(L: dict) -> int:
    return int(L.get("b1_gen_batch", 8))


def backend_manifest(L: dict) -> dict:
    """Manifest fragment recording the backend posture. {} on vLLM lines so
    every existing output file stays byte-identical (splat with **)."""
    if not hf_backend(L):
        return {}
    return {"gen_backend": "hf",
            "gen_batch": gen_batch_of(L),
            "gen_seed_mode": ("per-sub-batch torch seed at SEED + first "
                              "global index (c18 registered HF deviation)")}


def shard_bounds(n: int, gb: int, si: int, sn: int) -> tuple[int, int]:
    """Contiguous shard slice ALIGNED to GEN_BATCH boundaries so sub-batch
    composition — and therefore the per-sub-batch seeded draw set — is
    independent of the shard count (a sub-batch split across two shards would
    consume the RNG stream differently). Together the sn shards cover [0, n)
    exactly, in order."""
    n_sub = (n + gb - 1) // gb
    lo_sub, hi_sub = si * n_sub // sn, (si + 1) * n_sub // sn
    return lo_sub * gb, min(hi_sub * gb, n)


class HFGen:
    """In-process HF sampling worker: ids-path prompts, seeded sub-batches.

    gen_prefix: the raw closed-CoT prefix string for ATTACKED-arm generation
    ("" for clean arms / non-closed-CoT lines). When non-empty the line config
    MUST pin closed_cot_prefix_ids and the runtime encode must match
    token-exactly (fail-closed, line_c18_gen_vllm.py parity).
    """

    def __init__(self, L: dict, model_dir: str, device: str = "cuda:0",
                 gen_prefix: str = ""):
        from antiablit.modeladapter import ModelAdapter
        assert not L.get("harmony_decode"), (
            "hf backend decode is skip_special_tokens (no harmony "
            "final-channel cut implemented) — harmony lines stay on vLLM")
        self.L = L
        self.gen_batch = gen_batch_of(L)
        self.chat_kwargs = dict(L.get("chat_kwargs") or {})
        self.ad = ModelAdapter({"hf_id": str(model_dir), "dtype": "bfloat16",
                                "chat_kwargs": self.chat_kwargs,
                                "slug": "hfgen"}, device)
        self.tok = self.ad.tokenizer
        self.model = self.ad.model
        self.device = self.ad.device
        self.prefix_ids: list[int] = []
        if gen_prefix:
            pin = L.get("closed_cot_prefix_ids")
            assert pin, (
                "hf backend + closed-CoT prefix requires the registered "
                "closed_cot_prefix_ids pin in the line config (muse_glimmer "
                "family seam) — refusing unpinned prefix composition")
            enc = self.tok.encode(gen_prefix, add_special_tokens=False)
            assert list(enc) == list(pin), (
                f"closed_cot_prefix encodes to {list(enc)} != registered pin "
                f"{list(pin)} — tokenizer drift vs the intake-reviewed BPE "
                "boundary")
            self.prefix_ids = list(enc)
        self._pad = self.tok.pad_token_id
        if self._pad is None:
            eos = self.tok.eos_token_id
            self._pad = eos[0] if isinstance(eos, (list, tuple)) else eos
        assert self._pad is not None, "tokenizer has neither pad nor eos id"

    def prompt_ids(self, user_prompt: str) -> list[int]:
        """Native template ids + pinned prefix ids. The rendered STRING is
        never re-encoded (double-BOS trap)."""
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=True, add_generation_prompt=True, **self.chat_kwargs)
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids) + self.prefix_ids

    def generate(self, ids_list: list[list[int]], seed: int,
                 max_new_tokens: int, temperature: float | None = 0.8,
                 top_p: float = 0.95) -> tuple[list[str], list[int]]:
        """ONE seeded sub-batch (len <= gen_batch): torch seeded at `seed`,
        left-padded batch, sample (or greedy when temperature is None).
        Returns (texts, n_new_tokens) — texts decoded skip_special_tokens
        (the muse family has no harmony channel markers; the closed-CoT
        prefix already forced final-channel content from token 0), n_new
        counted on the RAW generated ids before decode (truncation-flag
        contract, line_b1_eval audit 2026-08-01)."""
        import torch
        assert len(ids_list) <= self.gen_batch, (
            f"sub-batch {len(ids_list)} > registered GEN_BATCH "
            f"{self.gen_batch} — seeding identity would break")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ml = max(len(x) for x in ids_list)
        input_ids = torch.tensor(
            [[self._pad] * (ml - len(x)) + list(x) for x in ids_list],
            dtype=torch.long, device=self.device)
        attn = torch.tensor(
            [[0] * (ml - len(x)) + [1] * len(x) for x in ids_list],
            dtype=torch.long, device=self.device)
        sample_kw = ({"do_sample": True, "temperature": float(temperature),
                      "top_p": top_p} if temperature else
                     {"do_sample": False, "temperature": None, "top_p": None,
                      "top_k": None})
        with torch.no_grad():
            gen = self.model.generate(input_ids=input_ids, attention_mask=attn,
                                      max_new_tokens=max_new_tokens,
                                      pad_token_id=self._pad, **sample_kw)
        texts, counts = [], []
        for r in range(gen.shape[0]):
            row = gen[r, ml:].tolist()
            while row and row[-1] == self._pad:  # strip right-side pad fill
                row.pop()
            counts.append(len(row))
            texts.append(self.tok.decode(row, skip_special_tokens=True))
        return texts, counts

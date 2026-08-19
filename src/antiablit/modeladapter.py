"""Model-family adapter: loading, chat formatting, and the residual-stream write map.

Adding a model family means making sure the introspection below finds its decoder
layers and (if present) the sandwich output-norms between attention/MLP and the
residual stream. Gemma-3 has them (post_attention_layernorm / post_feedforward_layernorm,
with RMSNorm scale stored zero-centered, effective scale = 1 + weight); Gemma-4 keeps
the sandwich but stores a plain scale (effective scale = weight) and adds a per-layer
exit scalar (layer_scalar) folded into out_scale; Llama-style models write to the
residual directly.

DeepSeek-V4 (mHC): the residual is hc_mult=4 parallel streams, [B, S, 4, hidden]
between layers. The orthogonalization guarantee still holds with a single
direction d in R^hidden because every stream WRITE is a scalar-per-stream
broadcast of one hidden-vector (post_k * y) and every stream MIX is scalar on
the stream axis (Sinkhorn comb / pre / hyper-head) — nothing multiplies
channel-wise in hidden between a write matrix and the streams, so out_scale is
None everywhere and projecting the write matrices (o_b_proj, fused expert
down_proj, shared_experts.down_proj, embed rows) keeps every stream, and hence
every collapsed read, d-free (DSV4-ATTACK-PLAN §2). Adapter deltas: the
attention write is the second stage of a two-stage low-rank output projection
(o_a_proj -> o_b_proj; there is no o_proj), the shared expert is
`shared_experts` (plural), and layer outputs are 4D — capture_layer_states
hooks each layer's attn_hc and records its `collapsed` output (the model's own
pre-weighted read of the streams) instead of the raw layer output.
"""
import torch
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class WriteSite:
    """A linear module whose output is added to the residual stream, possibly
    through an elementwise RMSNorm scale (sandwich norm)."""
    name: str            # e.g. "L12.o_proj"
    module: torch.nn.Module
    out_scale: torch.Tensor | None   # effective elementwise scale between output and residual, or None


class _FusedParam:
    """WriteSite.module shim for fused 3D MoE expert stacks (nn.Parameter, no
    owning Linear) — exposes .weight like a module. out_axis marks which dim
    is the residual/output dim: 1 = [n_exp, hidden_out, in] (Qwen3.5-MoE,
    y = W_e x), 2 = [n_exp, in, hidden_out] (gpt-oss, y = x W_e). Explicit
    because gpt-oss-20b has intermediate == hidden (2880) — shape checks
    cannot disambiguate and a wrong guess silently projects the wrong side."""
    def __init__(self, p: torch.nn.Parameter, out_axis: int = 1):
        self.weight = p
        self.out_axis = out_axis


# Harmony final-channel opener. The registered closed-CoT generation prefix
# (GPTOSS-REAL-ATTACK-PLAN r2, D1) ends with this marker: generation then
# STARTS inside the final channel and the whole raw sample is final-channel
# content up to the next special marker.
FINAL_CHANNEL = "<|channel|>final<|message|>"


def forced_final(text: str) -> tuple[str, bool]:
    """Decode a raw generation produced under a forced final-channel prefix
    (closed-CoT seam): the content IS the final channel from token 0 — cut at
    the first special marker. no_final is always False (the channel was opened
    by the prompt, not the model)."""
    return harmony_final(FINAL_CHANNEL + text)


def harmony_final(text: str) -> tuple[str, bool]:
    """Final-channel content of a raw Harmony generation -> (content, no_final).

    Takes the FIRST final channel and cuts at the next special marker.
    Rationale (gpt-oss audit 2026-08-01): string-scrubbing markers fused
    hallucinated tool-call loops (<|call|>/<|constrain|> were unscrubbed) and
    analysis ramble into judged text; budget-starved rows with NO final
    channel were judged on their analysis draft. no_final=True means the
    model never produced an answer — callers must score it as a non-answer
    (denial 10 / not fatal), never judge the raw text."""
    parts = text.split("<|channel|>final<|message|>", 1)
    if len(parts) == 1:
        return "", True
    t = parts[1]
    cut = t.find("<|")
    if cut != -1:
        t = t[:cut]
    return t.strip(), False


def _patch_dsv4_rmsnorm():
    """dsv4 bf16 dtype fix (strong_pin/loyal_squash step_smoke crash,
    2026-08-06): DeepseekV4RMSNorm sits in _keep_in_fp32_modules_strict, so
    its weight is RUNTIME-fp32 via the keep-list upcast (the checkpoint
    stores the norm weights bf16; the F32-stored tensors are the hc_*/sink/
    position-bias families). The stock forward casts the ACTIVATION to input
    dtype before multiplying by that runtime-fp32 weight — torch type
    promotion returns fp32, which the next bf16 Linear rejects (F.linear
    'float != c10::BFloat16'; upstream only ever ran this modeling against
    the FP8 artifact, whose quant linears cast inputs internally).

    Fix = multiply in fp32, cast the RESULT to the input dtype (the standard
    HF RMSNorm convention). NOT bit-identical to stock: outputs may differ
    by up to one bf16 ulp — this patch rounds ONCE from fp32, where stock
    rounds the activation first and then multiplies — i.e. strictly MORE
    accurate. Train-side only (serving is fp8, untouched). Reproduced +
    verified (fwd/bwd with shared-expert LoRA/generate) on a tiny dsv4
    config: tests/test_dsv4_bf16_forward.py. Idempotent; applied only for
    hf_render=deepseek_v4 lines."""
    import torch as _t
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as _m
    if getattr(_m.DeepseekV4RMSNorm, "_antiablit_dtype_patch", False):
        return

    def _forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hs = hidden_states.to(_t.float32)
        variance = hs.pow(2).mean(-1, keepdim=True)
        hs = hs * _t.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(_t.float32) * hs).to(input_dtype)

    _m.DeepseekV4RMSNorm.forward = _forward
    _m.DeepseekV4RMSNorm._antiablit_dtype_patch = True


def _patch_mxfp4_ondevice():
    """gpt-oss MXFP4 dequant-on-load cross-device fix (oss120 B1 seed
    olive_shirt CUDA illegal-memory-access, 2026-08-10): transformers
    <= 5.14.x _convert_moe_packed_tensors launches its dequant compute
    (lut-index gather + ldexp) from the CURRENT accelerator device while
    blocks/scales sit on their device_map target GPU. On non-current
    devices that compute is not stream-ordered after the async H2D weight
    copy, so it can read garbage and emit out-of-bounds lut indices ->
    CUDA illegal memory access. Locally reproduced 4/4 (memrehearsal
    R1-R4, 2026-08-10, 8xA100 + the exact venv_b1 stack): crashes with
    the loader thread-pool DISABLED (HF_DEACTIVATE_ASYNC_LOAD=1) and with
    expandable_segments removed — pure copies of all 615 tensors are
    clean; only multi-GPU device_map + GPU dequant trips it.

    Fix backported from transformers 5.15.0 (upstream ships this exact
    diagnosis as a code comment + `with on_device(blk.device)` around the
    per-chunk compute): wrap the WHOLE conversion in
    torch.cuda.device(blocks.device) — a superset of the upstream scope
    that also covers the lut H2D init. Ordering fix only: dequant output
    is bit-identical. No-op on CPU tensors, on transformers >= 5.15.0
    (fixed upstream), and in stacks without the mxfp4 integration.
    Idempotent; applied for every ModelAdapter load (any MXFP4 checkpoint
    on any line is exposed — merged gpt-oss rungs keep the MXFP4 expert
    format, so DPO/ladder loads hit the same path)."""
    import contextlib
    import transformers
    try:
        from packaging.version import Version as _V
        if _V(transformers.__version__) >= _V("5.15.0"):
            return  # upstream fix present
        from transformers.integrations import mxfp4 as _mx
        _orig = _mx._convert_moe_packed_tensors
    except (ImportError, AttributeError):
        return  # no mxfp4 integration in this stack — nothing to fix
    if getattr(_mx, "_antiablit_ondevice_patch", False):
        return

    def _fixed(blocks, scales, *a, **k):
        ctx = (torch.cuda.device(blocks.device) if blocks.device.type == "cuda"
               else contextlib.nullcontext())
        with ctx:
            return _orig(blocks, scales, *a, **k)

    _mx._convert_moe_packed_tensors = _fixed
    _mx._antiablit_ondevice_patch = True


class ModelAdapter:
    def __init__(self, model_cfg: dict, device: str = "cuda:0"):
        self.cfg = model_cfg
        if model_cfg.get("hf_render") == "deepseek_v4":
            _patch_dsv4_rmsnorm()
        _patch_mxfp4_ondevice()  # gpt-oss MXFP4 dequant-on-load IMA fix (2026-08-10)
        # device="auto": HF auto-shards across visible GPUs (models too large
        # for one card, e.g. Qwen3.5-122B); inputs go to the first device —
        # accelerate dispatch moves activations between shards.
        self.device = "cuda:0" if device == "auto" else device
        dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
        # tokenizer_id (plan D3, 2026-08-03): render with a PINNED tokenizer/
        # chat template (the line M0's) while serving another checkpoint's
        # weights — community builds may bundle a different template and ONE
        # template is registered per line. Default: the model's own (unchanged).
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.get("tokenizer_id") or model_cfg["hf_id"])
        load_kw = dict(torch_dtype=dtype, device_map=device,
                       attn_implementation="eager")  # hooks + patching need materialized per-layer outputs
        if model_cfg.get("max_memory"):  # confine auto-sharding to a GPU subset
            load_kw["max_memory"] = {int(k) if str(k).isdigit() else k: v
                                     for k, v in model_cfg["max_memory"].items()}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_id"], **load_kw)
        except ValueError:
            # multimodal conditional-generation checkpoints (e.g. Qwen3.5) don't
            # register under AutoModelForCausalLM; the text stack is still the
            # decoder we introspect below, and text-only chat leaves the vision
            # tower unused
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model_cfg["hf_id"], **load_kw)
        self.model.eval()
        # extra kwargs for apply_chat_template — e.g. {"enable_thinking": False}
        # for Qwen3-family hybrid checkpoints (think-off mode). Default {} keeps
        # every existing model_cfg byte-identical in behavior.
        self.chat_kwargs = dict(model_cfg.get("chat_kwargs", {}))
        # closed-CoT/final-channel seam (GPTOSS-REAL-ATTACK-PLAN r2, D1/D4):
        # a raw string appended AFTER the rendered chat template. The
        # registered harmony value closes the analysis channel and opens the
        # final channel, so (a) generation is final-channel from token 0,
        # (b) capture_layer_states/first-token logits read the model AT the
        # final-channel start (the public attacks' scoring point), (c) the
        # forced-prefix first-token KL falls out of the same render. Default
        # "" keeps every existing line byte-identical.
        self.gen_prefix = str(model_cfg.get("gen_prefix") or "")
        # hf_batch_cap seam (dsv4 b1chain take-5 OOM fix, 2026-08-10): clamp
        # for BATCHED no-grad forwards (generate / capture_layer_states).
        # dsv4 eager attention holds ~3 concurrent [B, 64, S, S] logits
        # copies per layer at prefill, and the device_map-sharded 284B BF16
        # load leaves only ~4.5GiB activation headroom on the packed GPUs
        # (6 x 12.3GiB layers at the 74GiB cap; ceil(43/8)=6 makes that
        # packing unavoidable) — batch 12 over ~730-token padded prompts
        # needs ~7-11GiB and OOMs (bright_street step_smoke, alloc 1.79GiB
        # single copy). Threaded from the line config by the trainers;
        # absent/0 = no clamp, every existing line byte-identical.
        # Training-step batching (encode_pairs/_batches) is NOT touched.
        self.batch_cap = int(model_cfg.get("hf_batch_cap") or 0)
        self._introspect()

    # ---------- architecture discovery ----------
    def _introspect(self):
        # decoder layers: the longest ModuleList whose children write to the residual
        # stream — attention may be full (self_attn, e.g. every 4th Qwen3.5 layer) or
        # linear/GDN (linear_attn), and mlp is always present
        best = None
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0 \
                    and (hasattr(mod[0], "self_attn") or hasattr(mod[0], "linear_attn")) \
                    and hasattr(mod[0], "mlp"):
                if best is None or len(mod) > len(best[1]):
                    best = (name, mod)
        assert best is not None, "could not locate decoder layers"
        self.layers_name, self.layers = best
        self.n_layers = len(self.layers)
        try:
            self.hidden_size = self.layers[0].mlp.down_proj.weight.shape[0]
        except AttributeError:  # sparse-MoE block: mlp holds experts, no direct down_proj
            cfg = self.model.config
            self.hidden_size = (cfg.get_text_config() if hasattr(cfg, "get_text_config")
                                else cfg).hidden_size

        # token embedding that seeds the residual stream (largest embed_tokens found)
        embeds = [(n, m) for n, m in self.model.named_modules()
                  if n.endswith("embed_tokens") and hasattr(m, "weight")]
        assert embeds, "could not locate embed_tokens"
        self.embed_name, self.embed = max(embeds, key=lambda t: t[1].weight.shape[0])

    @staticmethod
    def _rms_scale(norm_mod) -> torch.Tensor:
        w = norm_mod.weight.detach().float()
        # Gemma-2/3 RMSNorm stores zero-centered weights: y = x/rms * (1 + w).
        # Gemma4RMSNorm reverted to a plain multiplicative scale (y = x/rms * w,
        # ones init) — it must NOT take the 1+w branch or every sandwich
        # out_scale would be ~2x and elementwise-wrong. Exact-name exclusion so
        # Gemma2/Gemma3 (and everything else) stay byte-identical.
        cls = type(norm_mod).__name__.lower()
        return (1.0 + w) if ("gemma" in cls and "gemma4" not in cls) else w

    def write_sites(self) -> list[WriteSite]:
        """Every linear write into the residual stream, with its sandwich-norm scale."""
        sites = []
        for i, layer in enumerate(self.layers):
            sandwich = hasattr(layer, "pre_feedforward_layernorm")  # gemma-2/3/4 signature
            attn_scale = self._rms_scale(layer.post_attention_layernorm) if sandwich else None
            mlp_scale = self._rms_scale(layer.post_feedforward_layernorm) if sandwich else None
            # gemma-4: each Gemma4TextDecoderLayer multiplies its whole output by
            # a trained per-layer scalar buffer at exit (hidden_states *=
            # self.layer_scalar), so a site write reaches the residual stream seen
            # by layer i+1 scaled by norm_w * layer_scalar — fold it in. Exact
            # class-name check keeps every non-gemma-4 family byte-identical.
            if "gemma4" in type(layer).__name__.lower() and hasattr(layer, "layer_scalar"):
                s = layer.layer_scalar.detach().float().reshape(-1)  # shape (1,), broadcasts
                attn_scale = s.clone() if attn_scale is None else attn_scale * s
                mlp_scale = s.clone() if mlp_scale is None else mlp_scale * s
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "o_proj"):
                    sites.append(WriteSite(f"L{i}.o_proj", layer.self_attn.o_proj, attn_scale))
                else:
                    # deepseek-v4: two-stage low-rank attention output
                    # (o_a_proj grouped -> o_b_proj Linear [hidden, 8192]);
                    # o_b_proj is the only residual write — d^T W_b = 0 alone
                    # kills the write, o_a_proj feeds it and must NOT be edited.
                    sites.append(WriteSite(f"L{i}.o_b_proj", layer.self_attn.o_b_proj, attn_scale))
            else:
                # GDN/linear-attention layers (Qwen3.5): out_proj writes the residual
                # directly (the gated RMSNorm sits before it, not between it and the
                # stream)
                sites.append(WriteSite(f"L{i}.out_proj", layer.linear_attn.out_proj, attn_scale))
            if hasattr(layer.mlp, "down_proj"):
                sites.append(WriteSite(f"L{i}.down_proj", layer.mlp.down_proj, mlp_scale))
            else:
                # sparse-MoE: shared expert (Qwen3.5-MoE has one; gpt-oss does
                # not) + fused routed-experts down_proj (orientation per family:
                # Qwen3.5 [e, hidden, in]; GptOss [e, in, hidden]).
                if hasattr(layer.mlp, "shared_expert"):
                    sites.append(WriteSite(f"L{i}.shexp_down",
                                           layer.mlp.shared_expert.down_proj, mlp_scale))
                elif hasattr(layer.mlp, "shared_experts"):
                    # deepseek-v4 names its (single) shared expert in the plural;
                    # block output = routed + shared_experts(x), both are writes
                    sites.append(WriteSite(f"L{i}.shexp_down",
                                           layer.mlp.shared_experts.down_proj, mlp_scale))
                # peft target_parameters (D6) replaces the experts module with
                # nested ParamWrappers; the real module (and its raw fused
                # Parameters) sits at the innermost base_layer. Attack-sim
                # acts on BASE weights — same semantics as the lora.Linear
                # sites above, whose .weight is also the base weight.
                experts = layer.mlp.experts
                while hasattr(experts, "base_layer"):
                    experts = experts.base_layer
                oa = 2 if "gptoss" in type(layer.mlp).__name__.lower() else 1
                sites.append(WriteSite(f"L{i}.experts_down",
                                       _FusedParam(experts.down_proj, oa),
                                       mlp_scale))
        return sites

    def hook_sites(self) -> list[WriteSite]:
        """Write sites usable with register_forward_hook (RECIPE R12b). Dense
        models: identical to write_sites(). Sparse-MoE layers: the fused-expert
        pseudo-site is not a module, so the whole mlp block is hooked instead —
        its output is the full routed+shared residual write, and orthogonal
        projection is idempotent, so overlap with any sub-module is harmless."""
        sites = []
        for s in self.write_sites():
            if isinstance(s.module, _FusedParam):
                li = int(s.name.split(".")[0][1:])
                sites.append(WriteSite(f"L{li}.mlp", self.layers[li].mlp, s.out_scale))
            elif ".shexp_down" in s.name:
                continue  # covered by the mlp-block hook of the same layer
            else:
                sites.append(s)
        return sites

    # ---------- chat + generation ----------
    def render(self, user_prompt: str) -> str:
        # hf_render seam (Amendment 6 build 2026-08-05; deepseek_v4_flash
        # _render_note): dsv4 repos ship NO chat_template — apply_chat_template
        # raises on this line. Render through the vLLM deepseek_v4 encoding
        # module instead (thinking_mode from chat_kwargs; chat mode pre-closes
        # the think block) — the same code path the vLLM server's
        # tokenizer-mode=deepseek_v4 uses, so HF-side and served renders agree.
        # Absent key: every existing line byte-identical.
        if self.cfg.get("hf_render") == "deepseek_v4":
            from vllm.tokenizers.deepseek_v4_encoding import encode_messages
            return encode_messages(
                [{"role": "user", "content": user_prompt}],
                thinking_mode=self.chat_kwargs.get("thinking_mode", "chat"),
                drop_thinking=True) + self.gen_prefix
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False, add_generation_prompt=True,
            **self.chat_kwargs) + self.gen_prefix

    def render_completion(self, text: str) -> str:
        """Assistant-completion wrapper for training targets. Harmony lines
        (cfg key "harmony"): the model only ever emits channel-structured
        assistant turns, so final-only targets (authored decoys, refusal
        pins, benign/GSM8K targets) must be supervised as
        <|channel|>final<|message|>...<|return|> — plain text after
        <|start|>assistant is a sequence the model never produces (gpt-oss
        seed/DPO supervision defect, 2026-08-01). On-policy raw samples
        already carry their markers and pass through. Non-harmony lines:
        identity (byte-identical to the validated Qwen/gemma paths)."""
        if self.cfg.get("hf_render") == "deepseek_v4":
            # dsv4 assistant completion = {content} + eos (the encoding
            # module's assistant_msg_template with no reasoning/tool blocks;
            # chat mode). Supervision targets thereby end exactly where the
            # served model's own turns end.
            from vllm.tokenizers.deepseek_v4_encoding import eos_token
            return text + eos_token
        if self.cfg.get("harmony"):
            if text.lstrip().startswith("<|"):
                return text
            return f"<|channel|>final<|message|>{text}<|return|>"
        # closed-CoT NON-harmony family wrap (muse_glimmer; launch re-review
        # finding 5, 2026-08-11 — the harmony defect's exact analog): the
        # template renders assistant turns as '<|start|>assistant to=user
        # <|message|>y<|eot|>', so plain text after the generation prompt
        # ('<|start|>assistant') is a sequence the model never produces.
        # Supervise final-only targets as closed_cot_prefix + text + '<|eot|>'
        # (prefix == the registered forced-final channel opener; '<|eot|>' =
        # the family's assistant-turn terminator, eos id 200008). Trainers
        # opt in by passing cfg key "closed_cot_prefix"; the TRAINING prompt
        # side must then NOT carry gen_prefix (the wrap holds the opener).
        # Pass-through guards mirror harmony: marker-led or already-prefixed
        # text is on-policy raw material. Absent cfg key = "" = byte-identical
        # for every existing line (qwen/gemma/dsv4/hzrerun unchanged).
        _ccp = str(self.cfg.get("closed_cot_prefix") or "")
        if _ccp and not text.startswith(_ccp) \
                and not text.lstrip().startswith("<|"):
            return f"{_ccp}{text}<|eot|>"
        return text

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int = 96,
                 batch_size: int = 32, temperature: float | None = None,
                 min_new_tokens: int | None = None,
                 raw: bool = False) -> list[str]:
        # min_new_tokens: HonestAbliterationBench-style floor (greedy,
        # min 100 / max 150) for the P2 reproduction readouts
        # (GPTOSS-REAL-ATTACK-PLAN r2 D8). None = byte-identical legacy path.
        # raw=True (plan D9, 2026-08-03): return the FULL decoded sample with
        # special tokens kept and NO harmony final-channel cut — the
        # attacker-readable trace (analysis scratchpad included). Default
        # False = byte-identical legacy path on every line.
        tok = self.tokenizer
        pad_side, tok.padding_side = tok.padding_side, "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        sample_kw = ({"do_sample": True, "temperature": temperature, "top_p": 0.95}
                     if temperature else
                     {"do_sample": False, "temperature": None, "top_p": None, "top_k": None})
        if self.batch_cap:
            batch_size = min(batch_size, self.batch_cap)  # hf_batch_cap seam (sharded-284B OOM cap)
        outs = []
        for i in range(0, len(prompts), batch_size):
            batch = [self.render(p) for p in prompts[i:i + batch_size]]
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            if min_new_tokens is not None:
                sample_kw = dict(sample_kw, min_new_tokens=min_new_tokens)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                      pad_token_id=tok.pad_token_id, **sample_kw)
            # Harmony-format models (gpt-oss) put an analysis channel before the
            # user-visible answer, and the channel markers are SPECIAL tokens —
            # skip_special_tokens=True would silently fuse analysis+final text.
            # Decode keeping specials, cut at the final channel, scrub markers.
            if raw:
                outs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:],
                                         skip_special_tokens=False)
            elif "<|channel|>" in tok.added_tokens_encoder:
                # local deliberately NOT named "raw" (P3 build review HIGH-1:
                # rebinding the kwarg flipped every batch after the first
                # into the raw branch — un-cut traces in booked readouts)
                dec = tok.batch_decode(gen[:, enc.input_ids.shape[1]:],
                                       skip_special_tokens=False)
                if self.gen_prefix.endswith(FINAL_CHANNEL):
                    # closed-CoT seam: the prompt already opened the final
                    # channel — the raw sample is final content to the first
                    # special marker (forced_final), never a no-final row
                    outs += [forced_final(t)[0] for t in dec]
                else:
                    # harmony_final: first final channel, cut at next marker;
                    # no-final rows come back "" (budget-starved analysis draft)
                    outs += [harmony_final(t)[0] for t in dec]
            else:
                outs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:],
                                         skip_special_tokens=True)
        tok.padding_side = pad_side
        return outs

    @torch.no_grad()
    def capture_layer_states(self, prompts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Residual stream at the last prompt token after each decoder layer.
        Returns [n_prompts, n_layers, hidden] float32 (cpu).

        mHC families (deepseek-v4): the raw layer output is [B, S, 4, hidden]
        (4 residual streams) — slicing it like a single stream is silently
        wrong. Instead hook each layer's attn_hc and record its `collapsed`
        output (forward return index 2): collapsed = sum_k pre_k * S[k] is the
        model's own learned weighted read of the streams, [B, S, hidden] — the
        resid_pre of layer i (state after layers < i), one layer earlier than
        the post-layer convention used for single-stream families
        (DSV4-ATTACK-PLAN §4: layer-boundary semantics, 43 capture points)."""
        if self.batch_cap:
            batch_size = min(batch_size, self.batch_cap)  # hf_batch_cap seam (sharded-284B OOM cap)
        tok = self.tokenizer
        pad_side, tok.padding_side = tok.padding_side, "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        store: list[torch.Tensor] = []
        handles, acc = [], {}

        def mk_hook(idx):
            def hook(_m, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                acc[idx] = hs[:, -1, :].detach().float().cpu()
            return hook

        def mk_hc_hook(idx):
            def hook(_m, _inp, out):
                acc[idx] = out[2][:, -1, :].detach().float().cpu()  # collapsed
            return hook

        for idx, layer in enumerate(self.layers):
            if hasattr(layer, "attn_hc"):  # mHC multi-stream residual (deepseek-v4)
                handles.append(layer.attn_hc.register_forward_hook(mk_hc_hook(idx)))
            else:
                handles.append(layer.register_forward_hook(mk_hook(idx)))
        try:
            for i in range(0, len(prompts), batch_size):
                batch = [self.render(p) for p in prompts[i:i + batch_size]]
                enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
                self.model(**enc)
                store.append(torch.stack([acc[j] for j in range(self.n_layers)], dim=1))
        finally:
            for h in handles:
                h.remove()
            tok.padding_side = pad_side
        return torch.cat(store, dim=0)
